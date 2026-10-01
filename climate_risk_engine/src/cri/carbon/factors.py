"""
Emission-factor library for the GHG inventory (built by scripts/build_carbon_factors.py).

Every factor lookup returns the value *and* its provenance (source, year, GWP basis, data-quality hint), so each
line of an inventory can be traced back to a published table.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

_PATH = Path(__file__).resolve().parents[1] / "data" / "carbon_factors.json"
AR5 = {"CH4": 28.0, "N2O": 265.0}
AR6 = {"CH4_fossil": 29.8, "CH4_nonfossil": 27.0, "N2O": 273.0}


@lru_cache(maxsize=1)
def lib() -> dict:
    return json.loads(_PATH.read_text())


@lru_cache(maxsize=1)
def _desnz_index() -> dict:
    return {f["id"]: f for f in lib()["desnz"]["factors"]}


def _norm_gas(name: str) -> str:
    n = name.upper().replace(" ", "")
    m = re.fullmatch(r"R-?(\d{3})([A-Z]?)", n)
    if m:
        return f"R-{m.group(1)}{m.group(2)}"
    return name.replace(" ", "")


def desnz(factor_id: str, gwp: str = "AR6") -> dict:
    """A DESNZ 2026 factor, optionally restated from AR5 to AR6 GWPs."""
    f = _desnz_index().get(factor_id)
    if f is None:
        raise KeyError(f"unknown factor id '{factor_id}' — search with GET /carbon/factors?q=")
    out = {"id": f["id"], "label": " › ".join(f["path"]) + (f" ({f['variant']})" if f["variant"] else ""),
           "unit": f["unit"], "desnz_scope": f["scope"], "source": lib()["desnz"]["source"], "year": 2026}
    ar5 = f["kgco2e"]
    split = {k: f.get(k) for k in ("co2", "ch4", "n2o")}
    biogenic = f["path"][0] in ("Bioenergy", "WTT- bioenergy")
    if gwp == "AR6" and all(v is not None for v in split.values()):
        ch4 = split["ch4"] / AR5["CH4"] * (AR6["CH4_nonfossil"] if biogenic else AR6["CH4_fossil"])
        n2o = split["n2o"] / AR5["N2O"] * AR6["N2O"]
        out.update(kgco2e_per_unit=split["co2"] + ch4 + n2o, gases={"CO2": split["co2"], "CH4": ch4, "N2O": n2o},
                   gwp="AR6 (restated from DESNZ AR5 per-gas split)")
    elif gwp == "AR6" and f["path"][0] == "Refrigerant & other" and len(f["path"]) >= 3:
        gas = _norm_gas(f["path"][2])
        ar6 = lib()["gwp"]["refrigerants_ar6"].get(gas)
        if ar6 is not None and f["unit"] == "kg":
            out.update(kgco2e_per_unit=float(ar6), gases={gas: float(ar6)}, gwp="AR6 (IPCC AR6 WG1 Table 7.SM.7)")
        else:
            out.update(kgco2e_per_unit=ar5, gases={gas: ar5}, gwp="AR5 (no AR6 value in library — flagged)",
                       gwp_flag=True)
    else:
        out.update(kgco2e_per_unit=ar5, gases={"CO2e": ar5},
                   gwp="AR5 (DESNZ)" + ("" if gwp == "AR5" else " — no per-gas split to restate"),
                   gwp_flag=gwp == "AR6" and f["path"][0] not in ("Material use", "Waste disposal", "Water supply",
                                                                   "Water treatment", "Hotel stay", "Homeworking"))
    return out


def wtt_for(factor_id: str) -> Optional[dict]:
    """Matching well-to-tank factor for a DESNZ fuel/vehicle factor (Scope 3 category 3)."""
    f = _desnz_index().get(factor_id)
    if not f:
        return None
    first = f["path"][0]
    wtt_first = {"Fuels": "WTT- fuels", "Bioenergy": "WTT- bioenergy", "Passenger vehicles": "WTT- pass vehs & travel- land",
                 "Delivery vehicles": "WTT- delivery vehs & freight", "Heat and steam": "WTT- heat and steam"}.get(first)
    if not wtt_first:
        return None
    for g in lib()["desnz"]["factors"]:
        if g["path"][0] == wtt_first and g["path"][1:] == f["path"][1:] and g["unit"] == f["unit"] and g["variant"] == f["variant"]:
            return {"id": g["id"], "kgco2e_per_unit": g["kgco2e"], "source": lib()["desnz"]["source"], "gwp": "AR5 (DESNZ)"}
    return None


def search(q: str = "", scope: Optional[str] = None, limit: int = 50) -> list[dict]:
    terms = [t for t in re.split(r"\s+", q.lower().strip()) if t]
    out = []
    for f in lib()["desnz"]["factors"]:
        if scope and f["scope"] != scope:
            continue
        hay = " ".join(f["path"] + [f["variant"], f["unit"]]).lower()
        if all(t in hay for t in terms):
            out.append({"id": f["id"], "scope": f["scope"], "label": " › ".join(f["path"]), "variant": f["variant"],
                        "unit": f["unit"], "kgco2e_per_unit_ar5": f["kgco2e"]})
            if len(out) >= limit:
                break
    return out


# ── electricity ──────────────────────────────────────────────────────────────

def _iso3(country: Optional[str]) -> Optional[str]:
    if not country:
        return None
    c = country.upper()
    if len(c) == 2:
        return json.loads((_PATH.parent / "iso2_iso3.json").read_text()).get(c)
    return c


def electricity_location(country: str, us_state: Optional[str] = None, egrid_subregion: Optional[str] = None) -> dict:
    """Location-based grid factor (g CO2e/kWh, direct generation) with provenance."""
    E = lib()["electricity"]
    iso = _iso3(country)
    if iso == "GBR":
        f = E["GBR"]["desnz_generation"][0]
        g = (f["co2"] + f["ch4"] / AR5["CH4"] * AR6["CH4_fossil"] + f["n2o"] / AR5["N2O"] * AR6["N2O"]) * 1000
        return {"g_per_kwh": g, "source": "UK DESNZ 2026 UK electricity (generation), CH4/N2O restated to AR6",
                "basis": "national", "dq": 2, "gases": "CO2+CH4+N2O (AR6)"}
    if iso == "USA":
        eg = E["us_egrid"]
        sub = (egrid_subregion or "").upper() or None
        note = None
        if not sub and us_state:
            ss = eg["state_subregion"].get(us_state.upper())
            if ss:
                sub = ss["dominant"]
                note = (f"eGRID subregion inferred from state {us_state.upper()} ({ss['share']:.0%} of in-state generation"
                        f" in {sub}); give the subregion for exactness")
        if sub and sub in eg["subregions"]:
            return {"g_per_kwh": eg["subregions"][sub]["g_per_kwh_ar6"], "source": f"US EPA eGRID2023 subregion {sub}",
                    "basis": "eGRID subregion", "subregion": sub, "dq": 2 if not note else 3, "note": note,
                    "gases": "CO2+CH4+N2O (AR6)"}
        return {"g_per_kwh": eg["us_average_g_per_kwh_ar6"], "source": "US EPA eGRID2023 US average", "basis": "national",
                "dq": 3, "note": "No state/subregion given — US average used", "gases": "CO2+CH4+N2O (AR6)"}
    aib = E["europe_aib"]["countries"].get(iso or "")
    if aib and aib.get("production_g_co2_per_kwh") is not None:
        return {"g_per_kwh": aib["production_g_co2_per_kwh"], "source": "AIB European production mix 2025 (direct CO2)",
                "basis": "national", "dq": 2, "gases": "CO2 only"}
    em = E["ember_derived"]["countries"].get(iso or "")
    if em:
        return {"g_per_kwh": em["direct_g_per_kwh"], "source": f"Ember {em['year']} (direct share derived from IPCC AR5)",
                "basis": "national (derived)", "dq": 3, "gases": "CO2e (derived)",
                "note": "No official national factor in the library — supply the national/utility factor if available"}
    raise KeyError(f"no grid factor for country '{country}' — supply a location-based factor")


def electricity_residual(country: str, egrid_subregion: Optional[str] = None) -> Optional[dict]:
    E = lib()["electricity"]
    iso = _iso3(country)
    if iso == "USA" and egrid_subregion:
        v = E["us_egrid"]["residual_mix"]["g_co2_per_kwh"].get(egrid_subregion.upper())
        if v is not None:
            return {"g_per_kwh": v, "source": E["us_egrid"]["residual_mix"]["source"] + f", {egrid_subregion.upper()}",
                    "gases": "CO2 only"}
    aib = E["europe_aib"]["countries"].get(iso or "")
    if aib and aib.get("residual_g_co2_per_kwh") is not None:
        return {"g_per_kwh": aib["residual_g_co2_per_kwh"], "source": "AIB European residual mix 2025 (direct CO2)",
                "gases": "CO2 only"}
    return None


def td_loss_share(country: str, egrid_subregion: Optional[str] = None) -> Optional[dict]:
    E = lib()["electricity"]
    iso = _iso3(country)
    if iso == "USA":
        ic = E["us_egrid"]["subregion_interconnect"].get((egrid_subregion or "").upper(), "U.S.")
        v = E["us_egrid"]["grid_gross_loss"].get(ic) or E["us_egrid"]["grid_gross_loss"]["U.S."]
        return {"share": v, "source": f"eGRID2023 grid gross loss ({ic})"}
    r = E["td_losses"]["countries"].get(iso or "")
    return {"share": r["loss_pct"] / 100, "source": f"World Bank WDI {r['year']}"} if r else None


def upstream_generation_share(country: str) -> Optional[float]:
    """(lifecycle − direct) / direct for the national grid — upstream (WTT) emissions of generation."""
    em = lib()["electricity"]["ember_derived"]["countries"].get(_iso3(country) or "")
    if not em or not em.get("lifecycle_g_per_kwh") or not em["direct_g_per_kwh"]:
        return None
    return max(0.0, em["lifecycle_g_per_kwh"] - em["direct_g_per_kwh"]) / em["direct_g_per_kwh"]


# ── spend ────────────────────────────────────────────────────────────────────

def spend_factor(naics: str, spend_year: int) -> dict:
    S = lib()["spend"]
    code = str(naics).strip()
    f = S["naics"].get(code)
    if f is None:
        raise KeyError(f"NAICS {code} not in USEEIO v1.3 — use a 6-digit 2017 NAICS code")
    cpi = S["cpi_u"]["values"]
    y = str(spend_year)
    if y not in cpi:
        raise KeyError(f"no CPI for {spend_year}")
    deflator = cpi["2022"] / cpi[y]
    return {"kgco2e_per_usd_2022": f["kgco2e_per_usd"], "title": f["title"], "deflator_to_2022": deflator,
            "kgco2e_per_usd_nominal": f["kgco2e_per_usd"] * deflator, "source": S["source"], "gwp": "AR5 (USEEIO)"}
