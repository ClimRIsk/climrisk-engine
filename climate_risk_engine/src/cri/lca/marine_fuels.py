"""
Marine fuel well-to-wake (WtW) intensity and ship GHG intensity, from the FuelEU Maritime default values (Regulation (EU) 2023/1805, Annex II) as reproduced in the European Commission's guidance.

What is computed (formulas as printed in the Commission guidance, Annex I of the Regulation)
  WtW [gCO2e/MJ] = WtT + TtW
  TtW per MJ     = [ (1 - Cslip/100) x (CfCO2 x 1 + CfCH4 x GWP_CH4 + CfN2O x GWP_N2O)  +  Cslip/100 x GWP_CH4 ] / LCV
  Cslip is the share of fuel mass lost unburned as methane; slip is counted as CH4 with factor 1.
  Ship GHG intensity = sum(fuel energy x fuel WtW) / sum(fuel energy)   (on-shore power and wind assistance are not modelled here)
  Compliance balance (gCO2e) = (limit - actual intensity) x energy used. The penalty rate is not encoded: it was not in the text retrieved.

What is deliberately NOT here
  * Defaults for biofuels, bio-LNG, e-fuels, e-ammonia: the regulation sets none; the supplier's certified WtT value must be passed in (wtt_override). Without it the result is null.
  * Cslip for LPG, ammonia and others: Annex II gives none ("not available"); a certified value must be passed in. Without it the result says the slip is not included.
  * Anything outside FuelEU's scope or fuel list. This is a regulatory calculation (FuelEU accounting), not a measurement of a particular ship's emissions.
  * GWP: FuelEU uses AR4 values (CH4 25, N2O 298) per the guidance; the EU MRV/ETS uses AR5. The values are a parameter.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_PATH = Path(__file__).resolve().parents[1] / "data" / "marine_fuels_fueleu.json"


class MarineFuelError(ValueError):
    pass


@lru_cache(maxsize=1)
def lib() -> dict:
    return json.loads(_PATH.read_text(encoding="utf-8"))


def fuels() -> dict:
    return lib()["fuels"]


NO_SLIP_DEFAULT = ("lpg_butane", "lpg_propane", "nh3_ng_fuel_cell", "nh3_ng_ice")      # Annex II gives no slip value (certified value required)


def wtw(fuel: str, wtt_override: Optional[float] = None, cslip_override_pct: Optional[float] = None, gwp: Optional[dict] = None,
        cf_overrides: Optional[dict] = None) -> dict:
    """WtW gCO2e/MJ for one fuel on one consumer. Everything is returned as a component so the answer can be traced to the table."""
    F = fuels()
    if fuel not in F:
        raise MarineFuelError(f"unknown fuel {fuel!r}; defaults exist for {sorted(F)}. For biofuels and e-fuels give the supplier's certified values through a custom entry.")
    f = dict(F[fuel])
    if cf_overrides:
        f.update(cf_overrides)
    g = gwp or lib()["gwp100_fueleu"]
    ch4, n2o = float(g["CH4"]), float(g["N2O"])
    slip = f["cslip_pct"] if cslip_override_pct is None else float(cslip_override_pct)
    s = slip / 100.0
    lcv = f["lcv_mj_per_g"]
    co2 = (1 - s) * f["cf_co2_g_per_g"] / lcv
    ch4_c = (1 - s) * f["cf_ch4_g_per_g"] * ch4 / lcv
    n2o_c = (1 - s) * f["cf_n2o_g_per_g"] * n2o / lcv
    slip_c = s * ch4 / lcv
    ttw = co2 + ch4_c + n2o_c + slip_c
    wtt = f["wtt_gco2e_per_mj"] if wtt_override is None else float(wtt_override)
    notes = []
    if fuel in NO_SLIP_DEFAULT and cslip_override_pct is None:
        notes.append("Annex II gives no methane/fuel slip default for this fuel: slip is NOT included. Pass a certified cslip_override_pct.")
    if wtt_override is not None:
        notes.append("WtT is the value you supplied; it should be a certified pathway value.")
    return {"fuel": fuel, "label": f["label"], "wtt": wtt, "ttw_co2": co2, "ttw_ch4": ch4_c, "ttw_n2o": n2o_c, "ttw_slip": slip_c, "ttw": ttw, "wtw": wtt + ttw,
            "unit": "gCO2e/MJ", "lcv_mj_per_g": lcv, "cslip_pct": slip, "gwp": {"CH4": ch4, "N2O": n2o}, "notes": notes,
            "source": lib()["source"]}


def custom_fuel(wtt_gco2e_per_mj: float, lcv_mj_per_g: float, cf_co2_g_per_g: float, cf_ch4_g_per_g: float = 0.0, cf_n2o_g_per_g: float = 0.0, cslip_pct: float = 0.0, label: str = "custom") -> dict:
    """A fuel with the supplier's or laboratory's certified values (biofuel, RFNBO, e-fuel). Returns a dict to pass as `custom` entries in ship_intensity."""
    for k, v in dict(w=wtt_gco2e_per_mj, l=lcv_mj_per_g).items():
        if v is None or float(v) <= 0 and k == "l":
            raise MarineFuelError("lcv_mj_per_g must be positive")
    return {"label": label, "lcv_mj_per_g": float(lcv_mj_per_g), "wtt_gco2e_per_mj": float(wtt_gco2e_per_mj), "cf_co2_g_per_g": float(cf_co2_g_per_g),
            "cf_ch4_g_per_g": float(cf_ch4_g_per_g), "cf_n2o_g_per_g": float(cf_n2o_g_per_g), "cslip_pct": float(cslip_pct)}


def ship_intensity(consumption: list, year: Optional[int] = None) -> dict:
    """consumption: [{fuel: key | custom: {...custom_fuel...}, tonnes: fuel mass}]. Returns energy, WtW and TtW emissions, intensity, and, if a year is given, the FuelEU limit and compliance balance."""
    if not consumption:
        raise MarineFuelError("no fuel consumption given")
    E = W = T = 0.0
    lines, notes = [], []
    for c in consumption:
        t = float(c.get("tonnes") or 0)
        if t < 0:
            raise MarineFuelError("tonnes must be >= 0")
        if c.get("custom"):
            key = "_custom"
            lib()["fuels"]  # noqa
            cust = c["custom"]
            tmp = dict(cust)
            F = fuels()
            F[key] = tmp                                                       # transient entry in the cached dict
            try:
                r = wtw(key)
            finally:
                F.pop(key, None)
        else:
            r = wtw(c["fuel"], c.get("wtt_override"), c.get("cslip_override_pct"))
        e_mj = t * 1e6 * r["lcv_mj_per_g"]                                      # tonnes -> g -> MJ
        E += e_mj
        W += e_mj * r["wtw"]
        T += e_mj * r["ttw"]
        notes += r["notes"]
        lines.append({"fuel": r["label"], "tonnes": t, "energy_mj": e_mj, "wtw_gco2e_per_mj": r["wtw"], "wtw_tco2e": e_mj * r["wtw"] / 1e6, "ttw_tco2e": e_mj * r["ttw"] / 1e6})
    if E <= 0:
        raise MarineFuelError("total energy is zero")
    out = {"energy_mj": E, "wtw_tco2e": W / 1e6, "ttw_tco2e": T / 1e6, "ghg_intensity_gco2e_per_mj": W / E, "lines": lines, "notes": sorted(set(notes)),
           "scope": "Fuel energy only. On-shore power, wind assistance, RFNBO multipliers and pooling are not modelled. A regulatory calculation, not a measurement."}
    if year is not None:
        lim = limit_for(year)
        out["fueleu_limit_gco2e_per_mj"] = lim
        out["compliance_balance_tco2e"] = (lim - W / E) * E / 1e6
        out["compliance"] = "surplus" if out["compliance_balance_tco2e"] >= 0 else "deficit"
        out["penalty"] = "not encoded: the penalty rate was not in the text retrieved"
    return out


def limit_for(year: int) -> float:
    """FuelEU GHG-intensity limit in force in a year (5-year steps from 2025)."""
    L = lib()["ghg_intensity_limits_gco2e_per_mj"]
    if year < 2025:
        raise MarineFuelError("FuelEU limits start in 2025")
    steps = [2025, 2030, 2035, 2040, 2045, 2050]
    y = max(s for s in steps if s <= min(year, 2050))
    return float(L[str(y)])
