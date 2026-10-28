"""
Regulatory and price scenarios for one voyage: years x bunker prices x allowance prices.
The regime moves with the year (ETS phase-in, FuelEU intensity limit); the prices are YOUR scenario values: the engine has no price forecast and does not make one.
"""
from __future__ import annotations

from . import voyage as V


def grid(vessel: dict, distance_nm, speed_kn, fuel, origin, dest, years: list, bunker_prices: list, allowance_prices: list, currency: str = "EUR",
         charter_per_day: float = None, **kw) -> dict:
    if not years or not bunker_prices or not allowance_prices:
        raise V.VoyageInputError("give at least one year, one bunker price and one allowance price", ["years", "bunker prices", "allowance prices"])
    out = []
    for y in years:
        base = V.assess_voyage(vessel, distance_nm, speed_kn, fuel, origin, dest, y, {"eua_price": 0.0, "currency": currency}, speed_options_kn=[], delay_days=[], diversion_pct=0, **kw)
        t = base["ets"].get("chargeable_t") or 0.0
        fuel_t = base["fuel"]["t"]
        days = base["voyage"]["days"]
        fe = base["fueleu"]
        for bp in bunker_prices:
            for ap in allowance_prices:
                tot = fuel_t * bp + t * ap + (days * charter_per_day if charter_per_day is not None else 0.0)
                out.append({"year": y, "bunker_per_t": bp, "allowance_per_t": ap, "ets_chargeable_t": t, "ets_cost": t * ap, "fuel_cost": fuel_t * bp, "total_cost": tot,
                            "ets_share_of_total": (t * ap / tot) if tot else None, "fueleu_limit": fe.get("limit_gco2e_per_mj"), "fueleu_voyage_balance_tco2e": fe.get("compliance_balance_tco2e")})
    return {"currency": currency, "rows": out, "note": "Prices are your scenario values in one currency. Total = fuel + ETS" + (" + time" if charter_per_day is not None else "") + ". FuelEU penalty not included (no rate supplied)."}
