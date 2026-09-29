"""
finance/transition.py — transition-risk drivers for an asset under NGFS Phase 5 scenarios.

For each scenario and each of the three NGFS models (GCAM 6.0, MESSAGEix-GLOBIOM 2.0,
REMIND-MAgPIE 3.3 — the model spread is real uncertainty and is sampled, not averaged
away) we take, for the asset's model region:

  carbon price       Price|Carbon, US$2010 → US$2024 (× 1.438, US CPI)
  activity index     the sector's output relative to 2025 (coal / gas / oil power
                     generation, primary coal / oil / gas, steel, cement production)

and combine them with today's policy (finance/policy.py):

  effective price(t) = explicit price today (held flat in real terms)
                       + max(0, NGFS price(t) − explicit price)    ← future policy
  carbon cost(t)     = emissions × [explicit × charged share today
                                    + future increment × future charged share]
  future charged share  0.3 / 0.6 / 1.0 — how much of a future economy-wide carbon
                        price reaches a given plant (free allocation, benchmarks and
                        exemptions usually shield part of it)
Emissions are held at today's level: this is the cost of NOT abating, which is what
a lender or investor needs to see; abatement is an action, not an assumption.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DATA = Path(__file__).resolve().parent.parent / "data"
YEARS = np.arange(2026, 2051)
SCENARIOS = ["Net Zero 2050", "Below 2?C", "Delayed transition", "Nationally Determined Contributions (NDCs)",
             "Current Policies", "Fragmented World", "Low demand"]
LABEL = {"Below 2?C": "Below 2°C", "Nationally Determined Contributions (NDCs)": "NDCs"}
FAMILY = {"Net Zero 2050": "Orderly", "Below 2?C": "Orderly", "Low demand": "Orderly",
          "Delayed transition": "Disorderly", "Nationally Determined Contributions (NDCs)": "Hot house world",
          "Current Policies": "Hot house world", "Fragmented World": "Too little, too late"}
FUTURE_CS = (0.3, 0.6, 1.0)

ACTIVITY_VAR = {
    "power:coal": "Secondary Energy|Electricity|Coal", "power:gas": "Secondary Energy|Electricity|Gas",
    "power:oil": "Secondary Energy|Electricity|Oil", "coal-mining": "Primary Energy|Coal",
    "oil-and-gas": "Primary Energy|Oil", "gas": "Primary Energy|Gas", "refining": "Primary Energy|Oil",
    "cement": "Production|Cement", "steel": "Production|Steel",
}


def label(s: str) -> str:
    return LABEL.get(s, s)


@lru_cache(maxsize=1)
def _ngfs() -> dict:
    return json.loads((_DATA / "ngfs_phase5.json").read_text())


def activity_key(sector: str, atype: str, cost_key: Optional[str]) -> Optional[str]:
    s = (sector or "").lower()
    if cost_key in ("power:coal", "power:gas", "power:oil"):
        return cost_key
    if s == "coal-mining":
        return "coal-mining"
    if s in ("oil-and-gas-production", "oil-and-gas-transport") or s.startswith("ghgrp-211") or s == "ghgrp-486":
        return "oil-and-gas"
    if cost_key == "refining" or s == "ghgrp-324":
        return "refining"
    if cost_key == "cement" or s == "ghgrp-327":
        return "cement"
    if (cost_key or "").startswith("steel") or s == "ghgrp-331":
        return "steel"
    if s == "ghgrp-221":
        return "power:gas"
    return None


def _series(model: str, region: str, var: str, scen: str) -> Optional[np.ndarray]:
    d = _ngfs()
    v = (((d["data"].get(model) or {}).get(region) or {}).get(var) or {}).get(scen)
    if not v:
        return None
    yrs = np.array(d["years"], float)
    arr = np.array([np.nan if x is None else x for x in v], float)
    ok = np.isfinite(arr)
    if ok.sum() < 2:
        return None
    return np.interp(YEARS, yrs[ok], arr[ok])


def drivers(iso3: Optional[str], sector: str, atype: str, cost_key: Optional[str]) -> dict:
    """Carbon price (US$2024/t) and activity index paths per scenario × model."""
    d = _ngfs()
    regions = d["country_region"].get(iso3 or "")
    world = not regions
    k = d["usd2010_to_usd2024"]
    act = activity_key(sector, atype, cost_key)
    out = {"years": YEARS.tolist(), "models": [], "regions": {}, "price": {}, "activity": {},
           "activity_variable": ACTIVITY_VAR.get(act) if act else None, "world_fallback": world}
    for model in d["data"]:
        reg = "World" if world else regions[model]
        out["models"].append(model)
        out["regions"][model] = reg
        for s in SCENARIOS:
            p = _series(model, reg, "Price|Carbon", s)
            if p is None:
                continue
            out["price"].setdefault(s, {})[model] = np.clip(p * k, 0, None)
            if act:
                v = ACTIVITY_VAR[act]
                a = _series(model, reg, v, s)
                raw = (((d["data"][model].get(reg) or {}).get(v) or {}).get(s) or [])
                v2025 = raw[1] if len(raw) > 1 and raw[1] is not None else (raw[0] if raw else None)
                if a is not None and v2025 and v2025 > 0:          # index: output relative to 2025
                    out["activity"].setdefault(s, {})[model] = np.clip(a / v2025, 0, 3)
    return out


def central_paths(drv: dict, explicit_price: float) -> dict:
    """Median-across-models effective carbon price and activity index per scenario (for display)."""
    res = {}
    for s, per_model in drv["price"].items():
        P = np.array([np.maximum(explicit_price, p) for p in per_model.values()])
        A = np.array(list(drv["activity"].get(s, {}).values())) if drv["activity"].get(s) else None
        res[s] = {"label": label(s), "family": FAMILY.get(s, ""),
                  "price_median": np.round(np.median(P, axis=0), 1).tolist(),
                  "price_min": np.round(P.min(axis=0), 1).tolist(), "price_max": np.round(P.max(axis=0), 1).tolist(),
                  "activity_median": None if A is None else np.round(np.median(A, axis=0), 3).tolist()}
    return res
