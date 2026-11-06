"""Official national grid factors for Australia, Canada, Germany, China and Japan (data/national_grids.json, built by scripts/build_national_grids.py).

resolve() returns the same shape as carbon.factors.electricity_location(): g_per_kwh, source, basis, dq, gases, gwp_basis, note, plus the extras
each publisher gives (scope 3, consumption intensity, CO2-only, upstream). Nothing is guessed: an unknown region falls back to the national value and says so.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_PATH = Path(__file__).resolve().parent.parent / "data" / "national_grids.json"


@lru_cache(maxsize=1)
def data() -> dict:
    return json.loads(_PATH.read_text())


def has(iso: Optional[str]) -> bool:
    return bool(iso) and iso in data()["countries"]


def countries() -> list:
    return sorted(data()["countries"])


def _norm(s: str) -> str:
    return " ".join(str(s).strip().upper().replace("_", " ").split())


def _region(c: dict, region: Optional[str]) -> tuple:
    """-> (code, matched). Unknown or missing region -> (default, False)."""
    regs = c["regions"]
    if region is None or str(region).strip() == "":
        return c["default_region"], False
    r = str(region).strip()
    for cand in (_norm(r), r):
        if cand in regs:
            return cand, True
        al = c["aliases"].get(cand) or c["aliases"].get(_norm(cand))
        if al:
            return al, True
    for code, v in regs.items():
        if _norm(v["name"]) == _norm(r) or v.get("name_zh") == r:
            return code, True
    return c["default_region"], False


def _year(series: dict, year) -> tuple:
    ys = sorted(series)
    if year is not None and str(year)[:4] in series:
        return str(year)[:4], True
    return ys[-1], False


def resolve(iso: str, region: Optional[str] = None, year=None) -> dict:
    c = data()["countries"][iso]
    code, matched = _region(c, region)
    reg = c["regions"][code]
    notes: list = []
    if region and not matched:
        notes.append(f"region '{region}' is not one this table holds; the national value is used (regions: {', '.join(sorted(c['regions']))})")
    elif not region and len(c["regions"]) > 1:
        notes.append(f"national value used; give the {c['kind']} (region=) for a more exact factor")
    if code in c.get("region_notes", {}) and region:
        notes.append(c["region_notes"][code])
    out = {"basis": f"national ({c['label']})" if code == c["default_region"] else f"{c['kind'].split(' or ')[0]} ({c['label']})", "dq": 2 if matched or len(c["regions"]) == 1 else 3,
           "gases": c["gases"], "gwp_basis": c["gwp_basis"], "region": code, "region_name": reg["name"], "edition": c["label"], "publisher": c["publisher"]}
    base = f"{c['publisher'].split(' (')[0]}, {c['label']}"
    if iso == "AUS":
        out.update(g_per_kwh=float(reg["g_per_kwh"]), scope3_g_per_kwh=float(reg["scope3_g_per_kwh"]), period=c["period"],
                   source=f"{base}, Table 1, {reg['name']} (Scope 2, location-based)")
    elif iso == "CAN":
        y = str(year)[:4] if year is not None and str(year)[:4] in reg["g_per_kwh"] else str(c["default_year"])
        if year is not None and str(year)[:4] != y:
            notes.append(f"year {year} is not held; {y} used (held: {', '.join(sorted(reg['g_per_kwh']))})")
        g, cons = reg["g_per_kwh"][y], reg["consumption_g_per_kwh"][y]
        out.update(g_per_kwh=float(g), consumption_g_per_kwh=float(cons), year=y, period=c["period"] if y == "2024" else "calendar year",
                   source=f"{base}, Annex 7, {reg['name']}, generation intensity {y}{' (preliminary)' if y == '2024' else ''}")
        if abs(cons - g) > max(25.0, 0.25 * g):
            notes.append(f"consumption intensity ({cons:g} g) differs a lot from generation intensity ({g:g} g) because of imports and losses; consumers in {reg['name']} may prefer the consumption figure (consumption_g_per_kwh)")
    elif iso == "DEU":
        y, exact = _year(reg["series"], year)
        s = reg["series"][y]
        if year is not None and not exact:
            notes.append(f"year {year} is not held; {y} used (held: {', '.join(sorted(reg['series']))})")
        if s["status"] != "inventory":
            notes.append(f"UBA marks {y} as {s['status']}; a final value replaces it in next year's publication")
        out.update(g_per_kwh=float(s["ghg_g_per_kwh"]), co2_only_g_per_kwh=float(s["co2_g_per_kwh"]), with_upstream_g_per_kwh=float(s["ghg_with_upstream_g_per_kwh"]),
                   inland_consumption_co2_g_per_kwh=float(s["inland_consumption_co2_g_per_kwh"]), year=y, status=s["status"], period=c["period"],
                   source=f"{base}, Table 1, direct GHG of the German electricity mix {y} ({s['status']})")
    elif iso == "CHN":
        out.update(g_per_kwh=float(reg["g_per_kwh"]), year="2023", period=c["period"], level=reg["level"],
                   source=f"{base}, {reg['name']} average electricity CO2 emission factor", residual_g_per_kwh=float(c["residual_mix"]["g_per_kwh"]),
                   fossil_electricity_g_per_kwh=float(c["extra"]["fossil_electricity_g_per_kwh"]))
        if year is not None and str(year)[:4] != "2023":
            notes.append(f"only the 2023 data year is held (published Dec 2025); {year} is not available")
    elif iso == "JPN":
        out.update(g_per_kwh=float(reg["g_per_kwh"]), period=c["period"], dq=3,
                   source=f"{base}: substitute value (代替値) of the SHK electricity factor list")
        notes.append("Japan publishes no single national grid factor; this is the scheme's substitute value. Use your retailer's published basic/adjusted factor (supplier_g_per_kwh or location_g_per_kwh) when you know it")
    if c.get("gwp_note") and iso == "CAN":
        notes.append(c["gwp_note"])
    out["note"] = "; ".join(notes) or None
    return out


def residual(iso: str) -> Optional[dict]:
    c = data()["countries"].get(iso or "")
    r = c and c.get("residual_mix")
    if not r:
        return None
    return {"g_per_kwh": float(r["g_per_kwh"]), "source": r["source"], "gases": c["gases"], "gwp_basis": c["gwp_basis"]}
