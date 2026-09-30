"""
cri/lca/engine.py — generic cradle-to-gate life cycle inventory (LCI) calculator.

ISO 14067-aligned product carbon footprint (CFP) for one declared unit:

  inventory (activity data per declared unit)
      × emission factors (sourced, see cri/data/lca_factors.json)
      → CO2e by life-cycle stage, with
        • fossil and biogenic CO2 reported separately (ISO 14067 §6.4.9.4-style
          separate documentation; biogenic CO2 is NOT added to the total)
        • GWP100 from IPCC AR6 for CH4 and N2O
        • CO2 captured and stored deducted only where the user states a capture rate
        • Monte Carlo uncertainty: every activity datum and factor is lognormal with a
          coefficient of variation set by its data-quality score (1 = metered … 5 =
          rough estimate), 5,000 draws → P5 / P50 / P95
        • declared exclusions (cut-off) listed in the result
  "Dynamic": grid electricity intensity follows NGFS Phase 5 pathways for the plant's
  region (so the same product's footprint moves year by year with the grid), and
  templates can interpolate the owner's decarbonisation plan (see cement.py).

Stages
  A1  purchased materials (upstream production of inputs, user-supplied factors)
  A2  inbound transport (tonne-km × DESNZ factors incl. well-to-tank)
  A3  manufacturing: process emissions, fuel combustion, fuel supply chain (WTT),
      electricity (lifecycle grid factor)

This is a screening-grade calculation engine, not a verified CFP study: a claim under
ISO 14067 still needs the study report and critical review the standard requires.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DATA = Path(__file__).resolve().parent.parent / "data" / "lca_factors.json"
CV_BY_DQ = {1: 0.02, 2: 0.05, 3: 0.10, 4: 0.20, 5: 0.35}   # coefficient of variation per data-quality score
LOW_CARBON_FLOOR_G_KWH = 25.0   # lifecycle floor of a decarbonised grid: IPCC AR5 WG3 Annex III medians
                                # (wind 11, nuclear 12, hydro 24, solar PV 41–48 gCO2e/kWh)


@lru_cache(maxsize=1)
def factors() -> dict:
    return json.loads(_DATA.read_text())


# ── inventory model ───────────────────────────────────────────────────────────

@dataclass
class FuelUse:
    fuel: str                 # key in factors()["fuels"]
    gj: float                 # GJ (net calorific value) per declared unit
    dq: int = 2
    stage: str = "A3"


@dataclass
class ElectricityUse:
    kwh: float
    country: Optional[str] = None          # ISO3 → Ember grid factor
    g_per_kwh: Optional[float] = None      # supplier-specific / PPA factor (overrides grid)
    dq: int = 2
    label: str = "Electricity"


@dataclass
class ProcessEmission:
    name: str
    kg_co2e: float
    dq: int = 2
    capturable: bool = True                # CO2 that a capture unit could remove


@dataclass
class Material:
    name: str
    t: float
    kgco2e_per_t: float
    source: str
    dq: int = 3


@dataclass
class TransportLeg:
    mode: str                 # truck | rail | ship
    t: float
    km: float
    dq: int = 3
    label: str = ""


@dataclass
class Inventory:
    declared_unit: str
    fuels: list[FuelUse] = field(default_factory=list)
    electricity: list[ElectricityUse] = field(default_factory=list)
    process: list[ProcessEmission] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    transport: list[TransportLeg] = field(default_factory=list)
    capture_rate: float = 0.0              # share of capturable fossil/process CO2 captured and stored
    exclusions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ── factor lookups ────────────────────────────────────────────────────────────

def grid_factor(country: Optional[str]) -> tuple[float, str]:
    g = factors()["grid"]
    c = g["countries"].get((country or "").upper())
    if c:
        return float(c["g_per_kwh"]), f"Ember {c['year']} {c['name']} (lifecycle)"
    return float(g["world_g_per_kwh"]), "Ember world average (lifecycle) — plant country not given"


def grid_index_paths(country: Optional[str]) -> dict:
    """NGFS Phase 5 direct grid-intensity index (2025 = 1) per scenario → median of models."""
    from ..finance.geo import _ngfs_regions
    F = factors()["grid_index"]
    regions = _ngfs_regions().get((country or "").upper())
    out: dict = {}
    for model, regs in F["data"].items():
        reg = regions[model] if regions else "World"
        for scen, vals in (regs.get(reg) or {}).items():
            if vals and all(v is not None for v in vals):
                out.setdefault(scen, []).append(vals)
    from ..finance.transition import LABEL
    return {"years": F["years"],
            "scenarios": {LABEL.get(s, s): np.median(np.array(v), axis=0).tolist() for s, v in out.items()},
            "region_basis": "country's NGFS model regions" if regions else "World (country not mapped)"}


# ── deterministic calculation ────────────────────────────────────────────────

def _fuel_terms(fu: FuelUse) -> dict:
    f = factors()["fuels"][fu.fuel]
    gwp = factors()["gwp"]
    co2 = fu.gj * f["co2_kg_per_gj"]
    ch4 = fu.gj * f["ch4_kg_per_gj"] * (gwp["CH4_nonfossil"] if f["biogenic"] else gwp["CH4_fossil"])
    n2o = fu.gj * f["n2o_kg_per_gj"] * gwp["N2O"]
    wtt = fu.gj * (f["wtt_kgco2e_per_gj"] or 0.0)
    return {"fossil_co2": 0.0 if f["biogenic"] else co2, "biogenic_co2": co2 if f["biogenic"] else 0.0,
            "ch4_n2o": ch4 + n2o, "wtt": wtt, "wtt_missing": f["wtt_kgco2e_per_gj"] is None}


def calculate(inv: Inventory, grid_multiplier: float = 1.0) -> dict:
    """Deterministic CFP for one declared unit (kg CO2e). grid_multiplier scales grid factors (dynamic years)."""
    F = factors()
    stages = {"A1 purchased materials": 0.0, "A2 inbound transport": 0.0, "A3 process emissions": 0.0,
              "A3 fuel combustion (fossil CO2, CH4, N2O)": 0.0, "A3 fuel supply chain (well-to-tank)": 0.0,
              "A3 electricity (lifecycle)": 0.0}
    detail = []
    biogenic = 0.0
    capturable = 0.0
    exclusions = list(inv.exclusions)
    for m in inv.materials:
        v = m.t * m.kgco2e_per_t
        stages["A1 purchased materials"] += v
        detail.append({"stage": "A1", "item": m.name, "kg_co2e": v, "basis": f"{m.t:g} t × {m.kgco2e_per_t:g} kg/t ({m.source})"})
    for t in inv.transport:
        fr = F["freight"][t.mode]
        v = t.t * t.km * (fr["kgco2e_per_tkm"] + (fr["wtt_kgco2e_per_tkm"] or 0.0))
        stages["A2 inbound transport"] += v
        detail.append({"stage": "A2", "item": t.label or t.mode, "kg_co2e": v,
                       "basis": f"{t.t:g} t × {t.km:g} km × {fr['label']} (DESNZ 2025, incl. WTT)"})
    for p in inv.process:
        stages["A3 process emissions"] += p.kg_co2e
        if p.capturable:
            capturable += p.kg_co2e
        detail.append({"stage": "A3", "item": p.name, "kg_co2e": p.kg_co2e, "basis": "process chemistry"})
    for fu in inv.fuels:
        t = _fuel_terms(fu)
        stages["A3 fuel combustion (fossil CO2, CH4, N2O)"] += t["fossil_co2"] + t["ch4_n2o"]
        stages["A3 fuel supply chain (well-to-tank)"] += t["wtt"]
        biogenic += t["biogenic_co2"]
        capturable += t["fossil_co2"]
        lab = F["fuels"][fu.fuel]["label"]
        detail.append({"stage": "A3", "item": f"{lab} combustion", "kg_co2e": t["fossil_co2"] + t["ch4_n2o"],
                       "basis": f"{fu.gj:.3g} GJ × IPCC 2006 defaults" + (" (CO2 biogenic, reported separately)" if t["biogenic_co2"] else "")})
        if t["wtt"]:
            detail.append({"stage": "A3", "item": f"{lab} supply chain", "kg_co2e": t["wtt"], "basis": "DESNZ 2025 well-to-tank"})
        elif t["wtt_missing"] and f"{lab} supply chain" not in " ".join(exclusions):
            exclusions.append(f"{lab} supply chain (no well-to-tank factor available)")
    for e in inv.electricity:
        if e.g_per_kwh is not None:
            g, src = float(e.g_per_kwh), "supplier-specific factor"
        else:
            g, src = grid_factor(e.country)
            g = LOW_CARBON_FLOOR_G_KWH + (g - LOW_CARBON_FLOOR_G_KWH) * grid_multiplier if g > LOW_CARBON_FLOOR_G_KWH else g
        v = e.kwh * g / 1000.0
        stages["A3 electricity (lifecycle)"] += v
        detail.append({"stage": "A3", "item": e.label, "kg_co2e": v, "basis": f"{e.kwh:,.4g} kWh × {g:.0f} g/kWh ({src})" if e.kwh < 10000 else f"{e.kwh:,.0f} kWh × {g:.0f} g/kWh ({src})"})
    gross = sum(stages.values())
    captured = inv.capture_rate * capturable
    total = gross - captured
    return {"declared_unit": inv.declared_unit, "total_kg_co2e": total, "gross_kg_co2e": gross,
            "captured_kg_co2": captured, "biogenic_co2_kg": biogenic, "by_stage": stages, "detail": detail,
            "exclusions": exclusions}


# ── Monte Carlo uncertainty ───────────────────────────────────────────────────

def _ln(rng, mean, cv, n):
    if mean == 0:
        return np.zeros(n)
    s = np.sqrt(np.log(1 + cv ** 2))
    return np.abs(mean) * rng.lognormal(-s * s / 2, s, n) * np.sign(mean)


def uncertainty(inv: Inventory, n: int = 5000, seed: int = 11, grid_multiplier: float = 1.0) -> dict:
    """P5/P50/P95 of the total by perturbing each flow (activity × factor) lognormally by its data quality."""
    rng = np.random.default_rng(seed)
    base = calculate(inv, grid_multiplier)
    tot = np.zeros(n)
    capt = np.zeros(n)
    cv = lambda dq: CV_BY_DQ.get(int(dq), 0.2)                     # noqa: E731
    for m in inv.materials:
        tot += _ln(rng, m.t * m.kgco2e_per_t, np.hypot(cv(m.dq), 0.10), n)   # + 10 % factor uncertainty
    for t in inv.transport:
        fr = factors()["freight"][t.mode]
        tot += _ln(rng, t.t * t.km * (fr["kgco2e_per_tkm"] + (fr["wtt_kgco2e_per_tkm"] or 0)), np.hypot(cv(t.dq), 0.15), n)
    for p in inv.process:
        d = _ln(rng, p.kg_co2e, cv(p.dq), n); tot += d
        if p.capturable:
            capt += d
    for fu in inv.fuels:
        a = _ln(rng, 1.0, cv(fu.dq), n)                                    # activity (GJ) shared across terms
        t = _fuel_terms(fu)
        co2 = t["fossil_co2"] * a * _ln(rng, 1.0, 0.03, n)                # IPCC CO2 factors ≈ ±3 %
        tot += co2 + t["ch4_n2o"] * a * _ln(rng, 1.0, 0.5, n) + t["wtt"] * a * _ln(rng, 1.0, 0.2, n)
        capt += co2
    for e in inv.electricity:
        if e.g_per_kwh is not None:
            g = float(e.g_per_kwh)
        else:
            g, _ = grid_factor(e.country)
            g = LOW_CARBON_FLOOR_G_KWH + (g - LOW_CARBON_FLOOR_G_KWH) * grid_multiplier if g > LOW_CARBON_FLOOR_G_KWH else g
        tot += _ln(rng, e.kwh * g / 1000.0, np.hypot(cv(e.dq), 0.15), n)   # grid factors ≈ ±15 %
    tot -= inv.capture_rate * capt
    q = np.percentile(tot, [5, 50, 95])
    return {"p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2]), "mean": float(tot.mean()),
            "deterministic": base["total_kg_co2e"], "n_draws": n,
            "method": "lognormal per flow; CV by data quality (1:2%, 2:5%, 3:10%, 4:20%, 5:35%) combined with factor "
                      "uncertainty (IPCC CO2 ±3%, CH4/N2O ±50%, WTT ±20%, grid ±15%, materials ±10%, freight ±15%)"}


def data_quality_summary(inv: Inventory) -> dict:
    """Emission-weighted data-quality score (1 best – 5 worst) and the share of emissions from primary data (DQ ≤ 2)."""
    contrib: list[tuple[float, int]] = []
    for m in inv.materials:
        contrib.append((abs(m.t * m.kgco2e_per_t), m.dq))
    for p in inv.process:
        contrib.append((abs(p.kg_co2e), p.dq))
    for fu in inv.fuels:
        t = _fuel_terms(fu)
        contrib.append((t["fossil_co2"] + t["ch4_n2o"] + t["wtt"], fu.dq))
    for e in inv.electricity:
        g = e.g_per_kwh if e.g_per_kwh is not None else grid_factor(e.country)[0]
        contrib.append((e.kwh * g / 1000, e.dq))
    for t in inv.transport:
        fr = factors()["freight"][t.mode]
        contrib.append((t.t * t.km * fr["kgco2e_per_tkm"], t.dq))
    w = sum(v for v, _ in contrib)
    if not w:
        return {"weighted_score": None, "rating": None, "primary_data_share": None}
    score = sum(v * q for v, q in contrib) / w
    return {"weighted_score": round(score, 2),
            "rating": "good" if score <= 2 else "fair" if score <= 3 else "poor",
            "primary_data_share": round(sum(v for v, q in contrib if q <= 2) / w, 3)}
