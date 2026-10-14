"""
Cold extremes for any place: how cold the coldest day of a year gets, how often, and how often it passes a plant's design minimum.

Why. Winter Storm Uri (Texas, February 2021) cost Dow Chemical alone 600 to 650 million USD of insured downstream loss and took about two thirds of the Texas grid's
generation offline; Gulf Coast plants are built for winters that almost never freeze hard. Eastern Canada's ice storm of 1998 cost about 1.5 to 2.2 billion CAD insured. The engine
had cold only as a thermal-comfort category. A freeze loss needs the extreme-value question: how far below the plant's design minimum does the 1-in-50 cold day go?

Method. ERA5 daily minimum temperature (Open-Meteo archive, 1950 onward; NASA POWER MERRA-2 from 1981 when ERA5 cannot be read, smoother so extreme cold reads milder). Annual minima fitted
with a Gumbel distribution by L-moments on the negated series. Return levels at 5 to 250 years; the return period of any design minimum the user gives; counts of hard-freeze days; the longest
run of ice days (maximum below 0 C) per winter.

Validated on one documented event (scripts/validate_scenario_events.py): Uri, with the event's year left out of the fit, falls at about 1-in-29 years at Houston and 1-in-35 at Dallas
on the smoother fallback data. One event is a sanity check, not a validation of the tail.

Not modelled and not claimed: freezing rain and ice accretion (no hazard), snow load, wind chill, how a plant fails when it freezes (instrument lines, pumps, power loss), or any loss.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Callable, Optional

import numpy as np

EULER = 0.5772156649015329
RPS = (5, 10, 20, 50, 100, 250)


def _gumbel(x) -> tuple:
    x = np.sort(np.asarray(x, float)); n = len(x)
    b0 = float(x.mean()); b1 = float(np.sum(np.arange(n) / (n - 1) * x) / n)
    beta = (2 * b1 - b0) / math.log(2)
    return b0 - EULER * beta, beta


def _rl(mu: float, beta: float, T: float) -> float:
    return mu - beta * math.log(-math.log(1.0 - 1.0 / T))


def _rp(v: float, mu: float, beta: float) -> float:
    p = 1 - math.exp(-math.exp(-(v - mu) / beta))
    return float("inf") if p <= 0 else 1.0 / p


def assess(lat: float, lon: float, design_min_c: Optional[float] = None, today: Optional[dt.date] = None, fetch: Optional[Callable] = None, years: int = 45) -> dict:
    from . import offshore_wind as OW
    today = today or dt.date.today()
    y1 = today.year - 1
    y0 = y1 - years + 1
    try:
        d = OW._cold_daily(lat, lon, y0, y1, fetch or OW._get)
    except Exception as e:                                         # noqa: BLE001
        return {"status": "null", "reason": f"temperature record unreadable ({type(e).__name__}: {str(e)[:120]}); cold extremes not assessed (not zero)"}
    tn = np.array([np.nan if x is None else x for x in d["tmin"]], float)
    tx = np.array([np.nan if x is None else x for x in d["tmax"]], float)
    yr = np.array([int(t[:4]) for t in d["time"]])
    mins, runs, hard, years_ok = {}, {}, {k: [] for k in (0, -5, -10, -20)}, []
    for y in sorted(set(yr)):
        m = yr == y
        if m.sum() < 330 or np.isnan(tn[m]).mean() > 0.05:
            continue
        a = tn[m]; b = tx[m]
        mins[y] = float(np.nanmin(a)); years_ok.append(y)
        for k in hard:
            hard[k].append(int(np.nansum(a < k)))
        best = cur = 0
        for v in b:
            cur = cur + 1 if (not np.isnan(v) and v < 0) else 0
            best = max(best, cur)
        runs[y] = best
    if len(mins) < 25:
        return {"status": "null", "reason": f"too few complete years ({len(mins)})"}
    neg = [-v for v in mins.values()]
    mu, beta = _gumbel(neg)
    n = len(mins)
    if not (beta > 0.05):
        return {"status": "null", "reason": "annual minima barely vary in this record, so an extreme-value fit is not meaningful; cold extremes not assessed (not zero)"}
    rl = {T: round(-(mu - beta * math.log(-math.log(1.0 - 1.0 / T))), 1) for T in RPS}
    med = float(np.median(list(mins.values())))
    ys = sorted(mins, key=mins.get)
    surprise = round(med - rl[50], 1)
    sev = "NEGLIGIBLE" if rl[50] >= 0 else ("HIGH" if surprise >= 8 and med > -12 else "MODERATE" if surprise >= 5 else "LOW")
    out = {"status": "constructed" if "POWER" in d["source"] else "modelled", "weather_source": d["source"], "years": [int(years_ok[0]), int(years_ok[-1])], "n_years": n,
           "annual_minimum_c": {"median": round(med, 1), "coldest": round(min(mins.values()), 1), "coldest_year": int(ys[0]), "second_coldest": [int(ys[1]), round(mins[ys[1]], 1)]},
           "return_levels_c": rl, "extrapolated_return_periods": [T for T in RPS if T > 1.3 * n],
           "hard_freeze_days_per_year": {f"below_{k}c": round(float(np.mean(v)), 1) for k, v in hard.items()},
           "ice_day_run_days": {"median": round(float(np.median(list(runs.values()))), 1), "max": int(max(runs.values()))},
           "surprise_c": surprise, "severity": sev,
           "severity_basis": "how much colder the 1-in-50 day is than a typical year's coldest day (8 C or more where hard freezes are uncommon is graded HIGH: plants there are rarely built for it); a place that freezes hard every winter is graded by that gap alone and is normally designed for it",
           "limits": ["A reanalysis grid cell is not the plant: cold air pooling, urban warmth and local microclimate are not resolved.",
                      "Annual minima over 45 years support the 1-in-10 to 1-in-50 levels; 100 and 250-year levels are extrapolations.",
                      "NASA POWER fallback (MERRA-2) reads extreme cold milder than ERA5; the status is then 'constructed'.",
                      "Freezing rain, ice accretion, snow load and the failure of plant (instrument lines, pumps, power supply) are not modelled; no loss is computed."]}
    if design_min_c is not None:
        rp = _rp(-float(design_min_c), mu, beta)
        out["design_check"] = {"design_minimum_c": float(design_min_c), "return_period_years": (round(rp, 0) if rp < 1000 else ">1000"), "annual_exceedance_probability": round(1 / rp, 4) if rp < 1e6 else 0.0,
                               "years_in_record_below_design": int(sum(1 for v in mins.values() if v < design_min_c))}
    return out
