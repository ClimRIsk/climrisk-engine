"""
cri/lca/cement.py — cradle-to-gate carbon footprint of cement and clinker (ISO 14067 methodology),
with a dynamic 2025–2050 trajectory and the EU CBAM view.

Method (per tonne of cement; clinker reported per tonne of clinker)
  Calcination   clinker × (CaO × 44.01/56.08 + MgO × 44.01/40.30) from carbonate sources
                (IPCC 2006 Vol.3 Eq. 2.4 / GCCA-CSI Tier 2); if composition is unknown the
                GCCA-CSI default 525 kg CO2/t clinker is used and flagged
                + cement kiln dust (CKD) leaving the kiln, at its calcination degree
                + organic carbon in raw meal (GCCA-CSI default 0.2 % TOC, raw meal 1.55 t/t clinker)
  Kiln fuels    thermal energy (GJ/t clinker) × fuel mix → IPCC 2006 combustion factors;
                biomass CO2 reported separately (biogenic), DESNZ 2025 well-to-tank upstream
  Electricity   kiln + raw-mill kWh/t clinker and cement-grinding kWh/t cement × Ember lifecycle
                grid factor of the plant's country (or a supplier-specific factor)
  Constituents  gypsum, limestone, slag, fly ash, calcined clay … with the factor the user
                supplies; any constituent without a factor is listed as an exclusion (cut-off)
  Transport     inbound tonne-km by truck / rail / ship (DESNZ 2025 incl. WTT)
  CCS           optional capture rate on process + fossil combustion CO2

Dynamic view: electricity follows the NGFS Phase 5 grid pathway of the plant's region, and
the owner's plan (clinker ratio, thermal efficiency, biomass share, capture) is phased in
linearly to its target year — per scenario, 2025–2050.

CBAM view (Regulation (EU) 2023/956): direct embedded emissions (calcination + kiln and
mill fuel CO2) and indirect (electricity) per tonne; indicative certificate cost using the
EU ETS price and the CBAM phase-in schedule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .engine import (ElectricityUse, FuelUse, Inventory, Material, ProcessEmission, TransportLeg,
                     calculate, data_quality_summary, factors, grid_factor, grid_index_paths, uncertainty)


@dataclass
class Constituent:
    name: str
    t_per_t_cement: float
    kgco2e_per_t: Optional[float] = None
    source: str = ""
    dq: int = 3


@dataclass
class CementPlant:
    name: str = "Cement plant"
    country: Optional[str] = None                 # ISO3; or use lat/lon
    lat: Optional[float] = None
    lon: Optional[float] = None
    clinker_ratio: float = 0.75                   # t clinker per t cement (REQUIRED in practice)
    kiln_thermal_gj_per_t_clinker: float = 3.5
    fuel_mix: dict = field(default_factory=lambda: {"coal_bituminous": 1.0})   # share of thermal energy
    clinker_cao: Optional[float] = None           # mass fraction of CaO from carbonates in clinker (e.g. 0.65)
    clinker_mgo: Optional[float] = None
    ckd_t_per_t_clinker: float = 0.0              # dust leaving the kiln system
    ckd_calcination: float = 1.0
    raw_meal_toc: Optional[float] = None          # None → GCCA-CSI default 0.2 %
    raw_meal_per_t_clinker: Optional[float] = None
    kwh_per_t_clinker: float = 60.0               # kiln + raw mill
    kwh_per_t_cement_grinding: float = 40.0
    electricity_g_per_kwh: Optional[float] = None # supplier-specific / PPA factor
    grinding_fuels_gj_per_t_cement: dict = field(default_factory=dict)   # e.g. drying fuel
    constituents: list[Constituent] = field(default_factory=list)
    transport: list[TransportLeg] = field(default_factory=list)
    capture_rate: float = 0.0
    dq: dict = field(default_factory=dict)        # data quality per input, 1–5
    # plan (dynamic)
    plan_target_year: Optional[int] = None
    plan_clinker_ratio: Optional[float] = None
    plan_thermal_gj_per_t_clinker: Optional[float] = None
    plan_biomass_share: Optional[float] = None    # share of thermal energy from biomass by target year
    plan_capture_rate: Optional[float] = None
    plan_capture_start_year: Optional[int] = None
    plan_electricity_g_per_kwh: Optional[float] = None   # e.g. renewable PPA from target year

    def resolved_country(self) -> Optional[str]:
        if self.country:
            return self.country.upper()
        if self.lat is not None and self.lon is not None:
            from ..finance.geo import admin_at
            return admin_at(self.lat, self.lon).get("iso3")
        return None


def _dq(p: CementPlant, k: str, default: int) -> int:
    return int(p.dq.get(k, default))


def calcination_kg_per_t_clinker(p: CementPlant) -> tuple[float, str, int]:
    P = factors()["process"]
    if p.clinker_cao is not None:
        base = p.clinker_cao * 1000 * P["cao_to_co2"]["value"] + (p.clinker_mgo or 0.0) * 1000 * P["mgo_to_co2"]["value"]
        basis, dq = f"Tier 2 from clinker CaO {p.clinker_cao:.3f} / MgO {(p.clinker_mgo or 0):.3f}", _dq(p, "composition", 2)
    else:
        base = P["clinker_default_kgco2_per_t"]["value"]
        basis, dq = "GCCA-CSI default 525 kg CO2/t clinker (composition not provided)", 4
    ckd = p.ckd_t_per_t_clinker * base * p.ckd_calcination
    toc = p.raw_meal_toc if p.raw_meal_toc is not None else P["raw_meal_toc_default"]["value"]
    rm = p.raw_meal_per_t_clinker if p.raw_meal_per_t_clinker is not None else P["raw_meal_to_clinker_default"]["value"]
    org = toc * rm * 1000 * 44.01 / 12.011
    return base + ckd + org, basis, dq


def inventory(p: CementPlant, clinker_ratio=None, thermal=None, fuel_mix=None, capture=None,
              g_per_kwh=None) -> Inventory:
    cr = p.clinker_ratio if clinker_ratio is None else clinker_ratio
    th = p.kiln_thermal_gj_per_t_clinker if thermal is None else thermal
    mix = p.fuel_mix if fuel_mix is None else fuel_mix
    country = p.resolved_country()
    if not 0 < cr <= 1:
        raise ValueError("clinker_ratio must be between 0 and 1")
    tot_share = sum(mix.values())
    if abs(tot_share - 1.0) > 0.02:
        raise ValueError(f"fuel_mix shares must sum to 1 (got {tot_share:.3f})")
    for k in mix:
        if k not in factors()["fuels"]:
            raise ValueError(f"unknown fuel '{k}'")
    inv = Inventory(declared_unit="1 t cement", capture_rate=p.capture_rate if capture is None else capture)
    calc, basis, dq_c = calcination_kg_per_t_clinker(p)
    inv.process.append(ProcessEmission(f"Calcination & raw-meal carbon ({basis})", calc * cr, dq_c))
    for fuel, share in mix.items():
        if share > 0:
            inv.fuels.append(FuelUse(fuel, th * share * cr, _dq(p, "fuels", 2)))
    for fuel, gj in (p.grinding_fuels_gj_per_t_cement or {}).items():
        inv.fuels.append(FuelUse(fuel, gj, _dq(p, "fuels", 2)))
    g = g_per_kwh if g_per_kwh is not None else p.electricity_g_per_kwh
    inv.electricity.append(ElectricityUse(p.kwh_per_t_clinker * cr, country, g, _dq(p, "electricity", 2), "Kiln & raw mill electricity"))
    inv.electricity.append(ElectricityUse(p.kwh_per_t_cement_grinding, country, g, _dq(p, "electricity", 2), "Cement grinding electricity"))
    for c in p.constituents:
        if c.kgco2e_per_t is None:
            inv.exclusions.append(f"{c.name} ({c.t_per_t_cement:g} t/t cement): no emission factor supplied — excluded")
        else:
            inv.materials.append(Material(c.name, c.t_per_t_cement, c.kgco2e_per_t, c.source or "user-supplied", c.dq))
    inv.transport = list(p.transport)
    if p.clinker_cao is None:
        inv.notes.append("Calcination uses the GCCA-CSI default; provide clinker CaO/MgO for a Tier 2 result.")
    if not country and g is None:
        inv.notes.append("Plant country not given: world-average grid factor used.")
    return inv


def _lerp(a, b, t):
    return a + (b - a) * t


def _has_plan(p: CementPlant) -> bool:
    gradual = [p.plan_clinker_ratio, p.plan_thermal_gj_per_t_clinker, p.plan_biomass_share, p.plan_electricity_g_per_kwh]
    if any(v is not None for v in gradual) and not p.plan_target_year:
        raise ValueError("plan_target_year is required when plan targets are given")
    if (p.plan_capture_rate is not None) != (p.plan_capture_start_year is not None):
        raise ValueError("plan_capture_rate and plan_capture_start_year must be given together")
    return bool(p.plan_target_year or p.plan_capture_rate)


def trajectory(p: CementPlant, scenarios: Optional[list] = None, use_plan: bool = True) -> dict:
    if not use_plan:
        from dataclasses import replace
        p = replace(p, plan_target_year=None, plan_clinker_ratio=None, plan_thermal_gj_per_t_clinker=None,
                    plan_biomass_share=None, plan_capture_rate=None, plan_capture_start_year=None,
                    plan_electricity_g_per_kwh=None)
    country = p.resolved_country()
    paths = grid_index_paths(country)
    years = list(range(2025, 2051))
    grid_years = np.array(paths["years"])
    out = {}
    for s, idx in paths["scenarios"].items():
        if scenarios and s not in scenarios:
            continue
        idx_y = np.interp(years, grid_years, idx)
        vals = []
        for y, gi in zip(years, idx_y):
            t = 0.0
            if p.plan_target_year and p.plan_target_year > 2025:
                t = float(np.clip((y - 2025) / (p.plan_target_year - 2025), 0, 1))
            cr = _lerp(p.clinker_ratio, p.plan_clinker_ratio, t) if p.plan_clinker_ratio else p.clinker_ratio
            th = _lerp(p.kiln_thermal_gj_per_t_clinker, p.plan_thermal_gj_per_t_clinker, t) if p.plan_thermal_gj_per_t_clinker else p.kiln_thermal_gj_per_t_clinker
            mix = dict(p.fuel_mix)
            if p.plan_biomass_share is not None:
                cur_bio = sum(v for k, v in mix.items() if factors()["fuels"][k]["biogenic"])
                target_bio = _lerp(cur_bio, p.plan_biomass_share, t)
                fossil = {k: v for k, v in mix.items() if not factors()["fuels"][k]["biogenic"]}
                fs = sum(fossil.values()) or 1.0
                mix = {k: v / fs * (1 - target_bio) for k, v in fossil.items()}
                mix["biomass_other"] = mix.get("biomass_other", 0.0) + target_bio
            cap = p.capture_rate
            if p.plan_capture_rate and p.plan_capture_start_year and y >= p.plan_capture_start_year:
                cap = p.plan_capture_rate
            g = p.plan_electricity_g_per_kwh if (p.plan_electricity_g_per_kwh is not None and t >= 1.0) else p.electricity_g_per_kwh
            r = calculate(inventory(p, cr, th, mix, cap, g), grid_multiplier=float(gi))
            vals.append(round(r["total_kg_co2e"], 1))
        out[s] = vals
    return {"years": years, "scenarios": out, "grid_basis": paths["region_basis"],
            "plan_applied": _has_plan(p)}


def cbam_view(p: CementPlant, res: dict, year: int = 2026) -> dict:
    """Specific embedded emissions (t CO2 / t cement) and indicative CBAM certificate cost."""
    from ..finance.policy import EU_CBAM_FREE_SHARE, applicable
    from ..finance.asset_value import cost_key
    st = res["by_stage"]
    fossil_combustion_co2 = sum(d["kg_co2e"] for d in res["detail"]
                                if d["stage"] == "A3" and d["item"].endswith("combustion"))
    direct = (st["A3 process emissions"] + fossil_combustion_co2 - res["captured_kg_co2"]) / 1000.0
    indirect = st["A3 electricity (lifecycle)"] / 1000.0
    from ..finance.policy import _wb
    wb = _wb()
    eu = next((i for i in wb["instruments"] if i["id"] == "ETS_EU"), None)
    ets_price = next((x for x in reversed(eu["prices"]) if x is not None), None) if eu else None
    free = EU_CBAM_FREE_SHARE.get(year, 0.0 if year > 2034 else 1.0)
    country = p.resolved_country()
    pol = applicable(country, None, "cement", "integrated dry", cost_key("cement", "integrated dry")) if country else None
    paid = 0.0
    if pol and pol["charged_share"] and pol["effective_price_usd_t"]:
        cs = pol["charged_share"]
        paid = pol["effective_price_usd_t"] * (cs.get("mode") if "mode" in cs else 1.0)
    per_t = None
    if ets_price is not None:
        per_t = max(0.0, (direct + indirect) * (1 - free) * (ets_price - paid))
    eu_member = bool(country) and country in set(wb["eu27"]) | {"ISL", "NOR", "LIE"}
    return {"direct_tco2_per_t_cement": round(direct, 4), "indirect_tco2_per_t_cement": round(indirect, 4),
            "year": year, "cbam_phase_in_share": round(1 - free, 3), "eu_ets_price_usd_t": ets_price,
            "carbon_price_paid_usd_t": round(paid, 2),
            "indicative_cbam_cost_usd_per_t_cement": None if per_t is None or eu_member else round(per_t, 2),
            "applies": not eu_member,
            "notes": ["Cement: CBAM counts direct and indirect emissions (Regulation (EU) 2023/956).",
                      "Indirect here uses a lifecycle grid factor; CBAM uses the grid's direct-combustion factor, "
                      "which is lower.",
                      "Indicative cost = embedded emissions × EU ETS price × phase-in share, less carbon price "
                      "effectively paid at origin; the legal calculation adjusts with EU free-allocation benchmarks."]
            + (["Plant is inside the EU ETS: CBAM does not apply; the EU ETS applies directly."] if eu_member else [])}


def assess(p: CementPlant, n: int = 5000, cbam_year: int = 2026) -> dict:
    has_plan = _has_plan(p)
    inv = inventory(p)
    res = calculate(inv)
    unc = uncertainty(inv, n=n)
    dq = data_quality_summary(inv)
    cr = p.clinker_ratio
    calc, basis, _ = calcination_kg_per_t_clinker(p)
    clinker_only = Inventory("1 t clinker", capture_rate=p.capture_rate)
    clinker_only.process.append(ProcessEmission("Calcination", calc, 2))
    for fuel, share in p.fuel_mix.items():
        if share > 0:
            clinker_only.fuels.append(FuelUse(fuel, p.kiln_thermal_gj_per_t_clinker * share))
    clinker_only.electricity.append(ElectricityUse(p.kwh_per_t_clinker, p.resolved_country(), p.electricity_g_per_kwh))
    clk = calculate(clinker_only)
    g, gsrc = (p.electricity_g_per_kwh, "supplier-specific") if p.electricity_g_per_kwh is not None else grid_factor(p.resolved_country())
    return {
        "product": "cement", "plant": p.name, "country": p.resolved_country(), "declared_unit": "1 t cement",
        "footprint_kg_co2e_per_t": round(res["total_kg_co2e"], 1),
        "uncertainty": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in unc.items()},
        "by_stage": {k: round(v, 1) for k, v in res["by_stage"].items()},
        "detail": [dict(d, kg_co2e=round(d["kg_co2e"], 2)) for d in res["detail"]],
        "biogenic_co2_kg_per_t": round(res["biogenic_co2_kg"], 1),
        "captured_co2_kg_per_t": round(res["captured_kg_co2"], 1),
        "clinker": {"footprint_kg_co2e_per_t_clinker": round(clk["total_kg_co2e"], 1),
                    "calcination_kg_co2_per_t_clinker": round(calc, 1), "calcination_basis": basis,
                    "clinker_ratio": cr},
        "electricity_factor": {"g_per_kwh": g, "source": gsrc},
        "data_quality": dq,
        "exclusions": res["exclusions"],
        "notes": inv.notes,
        "trajectory": trajectory(p, use_plan=False),
        "trajectory_with_plan": trajectory(p) if has_plan else None,
        "cbam": cbam_view(p, res, cbam_year),
        "standard": "ISO 14067:2018 methodology, partial CFP cradle-to-gate (A1–A3); GWP100 IPCC AR6; biogenic CO2 reported separately. Run POST /lca/study for distribution, use stage, end of life, allocation, cut-off, sensitivity, data quality and the study report.",
        "factor_sources": ["IPCC 2006 Guidelines Vol.2 (combustion) & Vol.3 Ch.2 (cement)", "GCCA/WBCSD-CSI Cement CO2 & "
                           "Energy Protocol v3.1 defaults", "Ember Yearly Electricity Data (CC BY 4.0)",
                           "UK DESNZ GHG Conversion Factors 2025 (OGL)", "NGFS Phase 5 via IIASA", "IPCC AR6 GWP100"],
    }
