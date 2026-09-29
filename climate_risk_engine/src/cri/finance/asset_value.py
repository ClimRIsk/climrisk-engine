"""
finance/asset_value.py — replacement value of a physical asset (USD 2024), as a range.

Physical damage is a share of replacement value, so every money figure downstream
depends on this. Preference order:
  1. value supplied by the user (book, insured or appraised value)            → exact
  2. capacity × unit capital cost for the asset type and region                → range
  3. emissions × median value-per-tonne of comparable register assets          → wide range
Ranges are triangular (low, mode, high) and are sampled in the Monte Carlo, so the
value uncertainty flows into the loss distribution.

Unit costs — indicative public benchmarks for NEW build in high-income economies,
2024 USD (inflated from source years with US CPI):
  power       EIA AEO2022 Table 1 (USC coal $4,074/kW, CCGT $1,062–1,201/kW,
              combustion turbine $785–1,294/kW, biomass $4,525/kW, 2021$) and IEA
              WEO power-cost assumptions for the low end
  industry    per tonne of annual capacity: cement, BF-BOF / EAF / DRI steel,
              aluminium smelting and alumina refining, ammonia, methanol, steam
              cracking, pulp; refining per barrel/day — ranges from IEA technology
              roadmaps and published project costs
Regional factor (construction cost relative to US/EU): China 0.35–0.5, India 0.5–0.6,
other emerging economies 0.7 — IEA WEO shows coal plants at ~$800/kW in China vs
~$2,000/kW in the EU/US.
These are screening values; replace with the owner's asset values for decisions.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

# (unit, low, mode, high) in USD 2024 per unit of Climate TRACE capacity
_UNIT_COST = {
    "power:coal": ("MW", 2.0e6, 3.2e6, 4.7e6),
    "power:gas": ("MW", 0.9e6, 1.2e6, 1.45e6),
    "power:oil": ("MW", 0.8e6, 1.1e6, 1.5e6),
    "power:biomass": ("MW", 3.0e6, 4.5e6, 5.3e6),
    "power:waste": ("MW", 4.0e6, 6.0e6, 8.0e6),
    "power:other": ("MW", 1.0e6, 1.5e6, 2.5e6),
    "cement": ("t/yr", 150, 200, 300),
    "steel:BF/BOF": ("t/yr", 700, 1000, 1400),
    "steel:EAF": ("t/yr", 300, 450, 700),
    "steel:DRI": ("t/yr", 600, 900, 1200),
    "aluminium:smelting": ("t/yr", 3000, 4500, 6000),
    "aluminium:refinery": ("t/yr", 800, 1200, 1800),
    "refining": ("bbl/d", 15000, 25000, 35000),
    "chemicals:ammonia": ("t/yr", 800, 1200, 1800),
    "chemicals:methanol": ("t/yr", 500, 800, 1200),
    "chemicals:other": ("t/yr", 500, 1000, 1500),
    "petrochemicals": ("t/yr", 1500, 2000, 3000),
    "pulp": ("t/yr", 1000, 1500, 2000),
}
_REGION_FACTOR = {"CHN": (0.35, 0.45, 0.55), "IND": (0.5, 0.55, 0.65)}
_HIGH_INCOME = set("""USA CAN GBR IRL FRA DEU NLD BEL LUX AUT CHE LIE NOR SWE FIN DNK ISL ITA ESP PRT GRC CYP MLT SVN
SVK CZE POL HUN EST LVA LTU HRV JPN KOR TWN SGP HKG AUS NZL ISR ARE QAT KWT SAU BHR OMN""".split())


def cost_key(sector: str, atype: str) -> Optional[str]:
    s, a = (sector or "").lower(), (atype or "").lower()
    if s == "electricity-generation":
        first = a.split(",")[0].strip()
        if first in ("coal", "gas", "oil", "biomass", "waste"):
            return "power:" + first
        return "power:other"
    if s == "cement":
        return "cement"
    if s == "iron-and-steel":
        return "steel:DRI" if "dri" in a else "steel:BF/BOF" if "bf" in a else "steel:EAF"
    if s == "aluminum":
        return "aluminium:refinery" if "refin" in a else "aluminium:smelting"
    if s == "oil-and-gas-refining":
        return "refining"
    if s == "chemicals":
        return "chemicals:ammonia" if "ammonia" in a else "chemicals:methanol" if "methanol" in a else "chemicals:other"
    if s == "petrochemical-steam-cracking":
        return "petrochemicals"
    if s == "pulp-and-paper":
        return "pulp"
    return None


def region_factor(iso3: Optional[str]) -> tuple:
    if iso3 in _REGION_FACTOR:
        return _REGION_FACTOR[iso3]
    if iso3 in _HIGH_INCOME:
        return (1.0, 1.0, 1.0)
    return (0.6, 0.7, 0.85)


@lru_cache(maxsize=1)
def _value_per_tonne() -> dict:
    """Median capacity-based value per tCO2e by sector, from register assets with both numbers."""
    from ..data.asset_register import _load
    R = _load()
    sec = R["sectors"][R["sector"]]
    out: dict = {}
    vals = []
    for i in range(len(sec)):
        cap, em = float(R["capacity"][i]), float(R["emissions_t"][i])
        if not (np.isfinite(cap) and np.isfinite(em) and cap > 0 and em > 1000):
            continue
        k = cost_key(str(sec[i]), str(R["atype"][i]))
        if not k:
            continue
        v = cap * _UNIT_COST[k][2] * region_factor(str(R["countries"][R["country"][i]]))[1]
        out.setdefault(k, []).append(v / em)
        vals.append(v / em)
    res = {k: (float(np.percentile(v, 25)), float(np.median(v)), float(np.percentile(v, 75)))
           for k, v in out.items() if len(v) >= 10}
    res["_all"] = (float(np.percentile(vals, 25)), float(np.median(vals)), float(np.percentile(vals, 75)))
    return res


_NAICS_PROXY = {"221": "power:gas", "324": "refining", "327": "cement", "331": "steel:EAF",
                "325": "chemicals:other", "322": "pulp"}


def estimate(sector: str, atype: str = "", capacity: Optional[float] = None, emissions_t: Optional[float] = None,
             iso3: Optional[str] = None, value_usd: Optional[float] = None) -> dict:
    if value_usd and value_usd > 0:
        v = float(value_usd)
        return {"low": v, "mode": v, "high": v, "basis": "user-supplied value", "quality": "exact"}
    k = cost_key(sector, atype)
    rf = region_factor(iso3)
    if k and capacity and np.isfinite(capacity) and capacity > 0:
        unit, lo, mo, hi = _UNIT_COST[k]
        return {"low": capacity * lo * rf[0], "mode": capacity * mo * rf[1], "high": capacity * hi * rf[2],
                "basis": f"{capacity:,.0f} {unit} × unit capital cost for {k.replace(':', ' ')} "
                         f"(regional factor {rf[1]:.2f})", "quality": "benchmark"}
    if emissions_t and emissions_t > 0:
        vpt = _value_per_tonne()
        if not k and str(sector).startswith("ghgrp-"):
            k = _NAICS_PROXY.get(str(sector)[6:9])
        lo, mo, hi = vpt.get(k or "", vpt["_all"])
        return {"low": emissions_t * lo * 0.6, "mode": emissions_t * mo, "high": emissions_t * hi * 1.6,
                "basis": f"emissions × median value per tCO2e of comparable assets ({k or 'all sectors'}) — "
                         "no capacity data; supply the asset value for a firmer result",
                "quality": "proxy"}
    return {"low": None, "mode": None, "high": None, "basis": "unknown — supply the asset value", "quality": "none"}
