"""
finance/policy.py — which carbon-pricing policies apply to an asset today, and how hard they bite.

Data: World Bank Carbon Pricing Dashboard (release May 2026, prices on 1 April,
nominal US$/tCO2e) — every ETS and carbon tax worldwide with its jurisdiction,
sector coverage, allocation approach and status (cri/data/carbon_pricing_wb.json).
Jurisdiction comes from the asset's coordinates (FLOPROS admin-1 layer), so state
and provincial systems (California, RGGI, Alberta, Chinese pilots, Mexican states…)
are matched as well as national ones.

"Charged share" = share of the asset's emissions that actually pays the price.
Free allocation, output-based benchmarks and tax-free allowances mean it is usually
well below 1 for industry. Ranges (low, mode, high) are Monte-Carlo inputs:
  auctioning / plain carbon tax           1.0
  EU ETS power                            1.0
  EU ETS CBAM sectors (cement, steel,     1 − free share, free share falling
    aluminium, fertiliser, hydrogen)        97.5 % (2026) → 0 % (2034) — CBAM Reg. (EU)
                                            2023/956 phase-in schedule
  EU ETS other industry                   0.2 / 0.35 / 0.5 (benchmark shortfall)
  intensity / output-based benchmarks     0.05 / 0.1 / 0.25 (pay only above benchmark)
  grandfathering                          0.05 / 0.15 / 0.3
  South Africa carbon tax                 0.05 / 0.2 / 0.4 (60–95 % tax-free allowances)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DATA = Path(__file__).resolve().parent.parent / "data"

EU_CBAM_FREE_SHARE = {2025: 1.0, 2026: 0.975, 2027: 0.95, 2028: 0.90, 2029: 0.775, 2030: 0.515,
                      2031: 0.39, 2032: 0.265, 2033: 0.14, 2034: 0.0}
CBAM_KEYS = ("cement", "steel", "aluminium", "chemicals:ammonia", "power")
_EU_GRID_NEIGHBOURS = {"GBR", "CHE", "UKR", "MDA", "SRB", "BIH", "MNE", "MKD", "ALB", "XKX", "TUR", "MAR", "TUN"}


@lru_cache(maxsize=1)
def _wb() -> dict:
    return json.loads((_DATA / "carbon_pricing_wb.json").read_text())


def policy_sector(sector: str, atype: str = "") -> str:
    s = (sector or "").lower()
    if s == "electricity-generation":
        return "el"
    if s.startswith("ghgrp-"):
        n = s[6:9]
        if n == "221":
            return "el"
        if n in ("211", "212", "213", "486"):
            return "min"
        if n == "562":
            return "was"
        return "ind"
    if "mining" in s or s.startswith("oil-and-gas-production") or s.startswith("oil-and-gas-transport"):
        return "min"
    return "ind"


def _price_latest(inst: dict) -> tuple[Optional[float], Optional[int]]:
    yrs = _wb()["price_years"]
    for y, p in reversed(list(zip(yrs, inst.get("prices") or []))):
        if p is not None:
            return float(p), y
    return None, None


def _matches(inst: dict, iso3: Optional[str], admin1: Optional[str]) -> bool:
    wb = _wb()
    isos = set()
    for c in inst.get("iso3") or []:
        isos |= set(wb["eu27"]) if c == "EU27" else {c}
    if iso3 not in isos:
        return False
    adm = inst.get("admin1")
    if not adm:
        return True
    names = set(wb["rggi_states"]) if adm == ["RGGI"] else set(adm)
    return (admin1 or "") in names


def _charged_share(inst: dict, psec: str, cost_key: Optional[str]) -> dict:
    alloc = (inst.get("allocation") or "").lower()
    if inst["id"] == "ETS_EU":
        if psec == "el":
            return {"low": 1.0, "mode": 1.0, "high": 1.0, "rule": "EU ETS power: fully auctioned"}
        if cost_key and cost_key.startswith(CBAM_KEYS):
            return {"schedule": "eu_cbam", "rule": "EU ETS CBAM sector: free allocation phases out 2026→2034"}
        return {"low": 0.2, "mode": 0.35, "high": 0.5, "rule": "EU ETS industry: pays benchmark shortfall"}
    if inst["id"] == "Tax_ZA":
        return {"low": 0.05, "mode": 0.2, "high": 0.4, "rule": "South Africa: 60–95 % tax-free allowances"}
    if "intensity" in alloc or "benchmark" in alloc or "output-based" in alloc or "baseline" in alloc:
        return {"low": 0.05, "mode": 0.1, "high": 0.25, "rule": "benchmark system: pays only above the intensity benchmark"}
    if "grandfather" in alloc:
        return {"low": 0.05, "mode": 0.15, "high": 0.3, "rule": "grandfathered free allowances"}
    if "free" in alloc and "auction" in alloc:
        return {"low": 0.1, "mode": 0.3, "high": 0.6, "rule": "mostly free allocation, part auctioned"}
    if "free" in alloc:
        return {"low": 0.05, "mode": 0.15, "high": 0.3, "rule": "free allocation"}
    return {"low": 1.0, "mode": 1.0, "high": 1.0, "rule": "auctioned or set-price tax: all covered emissions pay"}


def applicable(iso3: Optional[str], admin1: Optional[str], sector: str, atype: str = "",
               cost_key: Optional[str] = None) -> dict:
    wb = _wb()
    psec = policy_sector(sector, atype)
    now, pipeline, other = [], [], []
    for inst in wb["instruments"]:
        if not _matches(inst, iso3, admin1):
            continue
        codes = [c for c in (inst.get("sectors") or "").split(",") if c]
        covers = (psec in codes) if codes else None
        price, py = _price_latest(inst)
        row = {"id": inst["id"], "name": inst["name"], "type": inst["type"], "status": inst["status"],
               "price_usd_t": price, "price_year": py, "prices_2021_2026": inst.get("prices"),
               "covers_asset_sector": covers, "share_of_jurisdiction_emissions": inst.get("share"),
               "allocation": inst.get("allocation"), "recent": inst.get("recent")}
        if inst["status"] == "Implemented" and covers is not False:
            row["charged_share"] = _charged_share(inst, psec, cost_key)
            now.append(row)
        elif inst["status"] in ("Under development", "Under consideration") and covers is not False:
            pipeline.append(row)
        else:
            other.append(row)
    priced = [r for r in now if r["price_usd_t"] is not None and r["covers_asset_sector"]]
    main = max(priced, key=lambda r: r["price_usd_t"]) if priced else None
    price = main["price_usd_t"] if main else 0.0
    # Known stacking: UK Carbon Price Support adds to UK ETS for power generators
    if main and iso3 == "GBR" and psec == "el":
        cps = next((r for r in now if r["id"] == "Tax_UK"), None)
        ets = next((r for r in now if r["id"] == "ETS_UK"), None)
        if cps and ets and cps["price_usd_t"] and ets["price_usd_t"]:
            price = cps["price_usd_t"] + ets["price_usd_t"]
    unclear = [r for r in now if r["covers_asset_sector"] is None]
    eu_ets_area = set(wb["eu27"]) | {"ISL", "NOR", "LIE"}
    cbam_exporter = bool(cost_key and cost_key.startswith(CBAM_KEYS) and iso3 not in eu_ets_area
                         and (not cost_key.startswith("power") or iso3 in _EU_GRID_NEIGHBOURS))
    # Where the EU ETS applies alongside a national floor (NL levy, Norway tax), the ETS
    # allocation rules decide how much of the emissions pay.
    eu = next((r for r in priced if r["id"] == "ETS_EU"), None)
    if main and eu and main is not eu:
        main = dict(main, charged_share=eu["charged_share"])
    notes = []
    if not now:
        notes.append("No carbon price currently applies to this asset's sector in its jurisdiction.")
    if unclear:
        notes.append("Sector coverage not reported for: " + ", ".join(r["name"] for r in unclear) + ".")
    if cbam_exporter:
        notes.append("Product is in scope of the EU Carbon Border Adjustment Mechanism (definitive phase from "
                     "January 2026): exports to the EU pay the EU carbon price less any carbon price paid at home.")
    if main and main.get("charged_share", {}).get("schedule") == "eu_cbam":
        notes.append("EU ETS free allocation for this sector falls from 97.5 % in 2026 to zero in 2034 as CBAM "
                     "phases in — the effective carbon cost rises sharply over the decade.")
    return {"policy_sector": psec, "jurisdiction": {"iso3": iso3, "admin1": admin1},
            "in_force": now, "pipeline": pipeline, "not_covering": other,
            "effective_price_usd_t": round(price, 1), "price_setting_instrument": main["name"] if main else None,
            "charged_share": main["charged_share"] if main else None, "cbam_exporter": cbam_exporter,
            "notes": notes, "source": wb["source"], "url": wb["url"]}


def charged_share_path(cs: Optional[dict], years) -> tuple:
    """(low, mode, high) arrays of charged share by year."""
    import numpy as np
    y = np.asarray(years)
    if not cs:
        return (np.zeros(len(y)),) * 3
    if cs.get("schedule") == "eu_cbam":
        f = np.array([EU_CBAM_FREE_SHARE.get(int(t), 0.0 if t > 2034 else 1.0) for t in y])
        s = 1.0 - f
        return s, s, s
    return tuple(np.full(len(y), cs[k]) for k in ("low", "mode", "high"))
