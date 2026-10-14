"""
Thermal stress (UTCI) for any place on Earth: heat AND cold, hourly, from 1981 to now, with a sensitivity to CMIP6 warming.

Why built here: published UTCI archives cover a limited period and are not available as a live service for any coordinate. The engine constructs the
same quantity itself: ERA5 hourly air temperature, humidity, pressure, 10 m wind and sunshine (Open-Meteo archive, 1940 to within days of today) go through
a mean radiant temperature of a standing person (solar geometry and direct/diffuse split after Liljegren 2008; six-direction radiation budget after Thorsson 2007), then the UTCI polynomial (Broede et al. 2012).

Past: hours per year in each stress category, annual extremes, and a trend (Theil-Sen with Kendall significance).
Future: a sensitivity, not a forecast. The recent climate's hourly weather is warmed by the CMIP6 ensemble change in the annual maximum daily temperature
(for heat categories) or the annual mean temperature (for cold categories), the radiant excess over air temperature is kept, and humidity is held in two ways
(relative humidity constant; vapour pressure constant): the two bracket the answer. It has not been backtested.
Validation of the reanalysis-driven values against 51 stations: scripts/validate_utci.py, docs/VALIDATION_DOSSIER.md.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import urllib.parse
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..modellab import data as D, utci as U

HOURLY = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,shortwave_radiation"
FIRST_YEAR = 1981
HEAT_T = (("moderate_heat_ge_32", 32.0), ("strong_heat_ge_38", 38.0), ("very_strong_heat_ge_46", 46.0))
COLD_T = (("moderate_cold_le_0", 0.0), ("strong_cold_le_minus13", -13.0), ("very_strong_cold_le_minus27", -27.0))
# Mean reanalysis-driven minus station-driven UTCI on the hottest 5 % of hours, from the held-out station test (51 sites, 2018-2022), by region group.
HOT_BIAS_C = {"Europe": -0.26, "Canada": -0.37, "US": -0.84, "elsewhere": -1.73}
HOT_BIAS_SITES = {"Europe": 16, "Canada": 8, "US": 11, "elsewhere": 16}
VALIDATION = {"test": "reanalysis-driven UTCI against station-driven UTCI at 51 airports, 2018-2022; scripts/validate_utci.py",
              "mean_bias_c": {"Europe": -0.13, "Canada": -0.05, "US": -0.79, "elsewhere": -1.02}, "rmse_c": {"Europe": 1.47, "Canada": 2.02, "US": 1.87, "elsewhere": 2.19},
              "note": "Cold-end bias is not consistent in sign and a fitted correction did not help held-out sites, so none is applied to cold stress."}


def region_group(lat: float, lon: float) -> str:
    if 35.0 <= lat <= 72.0 and -12.0 <= lon <= 45.0:
        return "Europe"
    if 24.0 <= lat <= 72.0 and -170.0 <= lon <= -50.0:
        return "Canada" if lat >= 49.5 else "US"
    return "elsewhere"


def _cache_dir() -> Path:
    d = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "thermal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _chunk(lat: float, lon: float, y0: int, y1: int, end: Optional[dt.date], fetch: Callable) -> dict:
    """One block of hourly ERA5 (complete calendar years, or the current year up to `end`). Complete blocks are cached for good; the open one is keyed by month."""
    e = end or dt.date(y1, 12, 31)
    tag = f"{lat:.2f}_{lon:.2f}_{y0}_{e:%Y%m}" if end else f"{lat:.2f}_{lon:.2f}_{y0}_{y1}"
    f = _cache_dir() / f"{tag}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:                                          # noqa: BLE001
            pass
    q = {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": f"{y0}-01-01", "end_date": e.isoformat(), "hourly": HOURLY, "timezone": "UTC", "wind_speed_unit": "ms"}
    d = fetch("https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(q))
    if "error" in d:
        raise RuntimeError(str(d.get("reason", d["error"]))[:160])
    h = d["hourly"]
    out = {"time": h["time"], **{k: h[k] for k in HOURLY.split(",")}}
    try:
        f.write_text(json.dumps(out))
    except Exception:                                              # noqa: BLE001
        pass
    return out


def _hourly(lat: float, lon: float, today: dt.date, fetch: Callable) -> dict:
    last_full = today.year - 1
    blocks = [(y, min(y + 10, last_full)) for y in range(FIRST_YEAR, last_full + 1, 11)]
    parts = [_chunk(lat, lon, a, b, None, fetch) for a, b in blocks]
    cur_end = today - dt.timedelta(days=6)                         # ERA5 lags real time by several days
    if cur_end.year == today.year and cur_end.month >= 2:
        parts.append(_chunk(lat, lon, today.year, today.year, cur_end, fetch))
    out = {k: [] for k in ["time"] + HOURLY.split(",")}
    for p in parts:
        for k in out:
            out[k].extend(p[k])
    return out


def _arr(v) -> np.ndarray:
    return np.array([np.nan if x is None else float(x) for x in v], float)


def _series(h: dict, lat: float, lon: float) -> dict:
    t = _arr(h["temperature_2m"]); rh = np.clip(_arr(h["relative_humidity_2m"]), 1, 100); p = _arr(h["surface_pressure"]); ws = _arr(h["wind_speed_10m"]); sw = np.nan_to_num(_arr(h["shortwave_radiation"]), nan=0.0)
    ok = np.isfinite(t) & np.isfinite(rh) & np.isfinite(ws)
    p = np.where(np.isfinite(p), p, 1000.0)
    tm = np.array(h["time"], dtype="datetime64[s]") - np.timedelta64(1800, "s")
    u = np.full(len(t), np.nan); tmrt = np.full(len(t), np.nan)
    r = U.hourly_utci(t[ok], rh[ok], p[ok], ws[ok], sw[ok], tm[ok], lat, lon)
    u[ok], tmrt[ok] = r["utci"], r["tmrt"]
    yrs = np.array([int(x[:4]) for x in h["time"]])
    return {"utci": u, "tmrt": tmrt, "t": t, "rh": rh, "ws": ws, "yrs": yrs, "time": h["time"]}


def _annual(s: dict) -> dict:
    out = {}
    for y in sorted(set(s["yrs"].tolist())):
        m = s["yrs"] == y
        u = s["utci"][m]; n = int(np.isfinite(u).sum())
        if n < 0.9 * m.sum() * (1.0 if y < s["yrs"].max() else 0.0) and y < s["yrs"].max():
            continue
        row = {"hours": n, "partial": bool(y == s["yrs"].max() and n < 8000)}
        for k, th in HEAT_T:
            row[k] = int((u >= th).sum())
        for k, th in COLD_T:
            row[k] = int((u <= th).sum())
        row["utci_max"] = round(float(np.nanmax(u)), 1) if n else None
        row["utci_min"] = round(float(np.nanmin(u)), 1) if n else None
        out[int(y)] = row
    return out


def _trend(years: list, vals: list) -> dict:
    from scipy import stats
    if len(years) < 12:
        return {"per_decade": None, "p": None}
    sl = stats.theilslopes(vals, years)[0]
    tau, p = stats.kendalltau(years, vals)
    return {"per_decade": round(float(sl) * 10, 2), "p": round(float(p), 3)}


def _future(s: dict, lat: float, lon: float, fetch_factors: Optional[Callable]) -> dict:
    from ..modellab import scenarios as SC
    try:
        fh = SC.change_factors("heat", lat, lon, ssps=("ssp126", "ssp245", "ssp585"))
        fc = SC.change_factors("tas", lat, lon, ssps=("ssp126", "ssp245", "ssp585"))
    except Exception as e:                                         # noqa: BLE001
        return {"status": "null", "reason": f"climate-model change factors unavailable ({type(e).__name__}: {str(e)[:100]})"}
    base_yrs = [y for y in sorted(set(s["yrs"].tolist())) if y >= 2011 and y < max(s["yrs"])]
    m = np.isin(s["yrs"], base_yrs)
    t, rh, ws, tm = s["t"][m], s["rh"][m], s["ws"][m], s["tmrt"][m]
    ok = np.isfinite(t) & np.isfinite(rh) & np.isfinite(ws) & np.isfinite(tm)
    t, rh, ws, tm = t[ok], rh[ok], ws[ok], tm[ok]
    pa0 = U.sat_vapour_pa(t) * rh / 100.0 / 1000.0
    n_years = max(len(base_yrs), 1) * (ok.sum() / max(m.sum(), 1))

    def run(dt_c: float, hold: str) -> np.ndarray:
        t2, tm2 = t + dt_c, tm + dt_c
        pa = pa0 if hold == "vapour" else U.sat_vapour_pa(t2) * rh / 100.0 / 1000.0
        return U.utci_pa(t2, pa, ws, tm2)

    base = {k: round(float((run(0.0, "rh") >= th).sum() / n_years), 0) for k, th in HEAT_T[:2]}
    base.update({k: round(float((run(0.0, "rh") <= th).sum() / n_years), 0) for k, th in COLD_T[:2]})
    rows = []
    for ssp in ("ssp126", "ssp245", "ssp585"):
        for per in fh["periods"]:
            a, b = fh["ssp"].get(ssp, {}).get(per), fc["ssp"].get(ssp, {}).get(per)
            if not a or not b:
                continue
            row = {"scenario": ssp, "period": per, "txx_shift_c": round(a["median"], 1), "tas_shift_c": round(b["median"], 1), "heat_hours_per_year": {}, "cold_hours_per_year": {}}
            for k, th in HEAT_T[:2]:
                vals = [float((run(a[q], hold) >= th).sum() / n_years) for q in ("p10", "median", "p90") for hold in ("rh", "vapour")]
                row["heat_hours_per_year"][k] = {"low": round(min(vals), 0), "central_rh_held": round(float((run(a["median"], "rh") >= th).sum() / n_years), 0), "high": round(max(vals), 0)}
            for k, th in COLD_T[:2]:
                vals = [float((run(b[q], "rh") <= th).sum() / n_years) for q in ("p10", "median", "p90")]
                row["cold_hours_per_year"][k] = {"low": round(min(vals), 0), "central": round(float((run(b["median"], "rh") <= th).sum() / n_years), 0), "high": round(max(vals), 0)}
            rows.append(row)
    return {"status": "ok", "baseline_period": f"{base_yrs[0]}-{base_yrs[-1]}", "baseline_hours_per_year": base, "rows": rows, "source": SC.NOTE,
            "note": "A sensitivity of the recent hourly weather to the ensemble's warming, not a forecast; radiant excess kept, wind and sunshine unchanged, humidity bracketed. Not backtested."}


def assess(lat: float, lon: float, today: Optional[dt.date] = None, fetch: Callable = D._get, future: bool = True) -> dict:
    today = today or dt.date.today()
    try:
        h = _hourly(lat, lon, today, fetch)
    except Exception as e:                                         # noqa: BLE001
        return {"status": "null", "reason": f"hourly reanalysis could not be read ({type(e).__name__}: {str(e)[:140]})"}
    if sum(1 for x in h["temperature_2m"] if x is not None) < 24 * 365 * 5:
        return {"status": "null", "reason": "too little reanalysis data at this location (open sea or polar cell)"}
    s = _series(h, lat, lon)
    ann = _annual(s)
    full = [y for y, r in ann.items() if not r["partial"]]
    if len(full) < 10:
        return {"status": "null", "reason": "fewer than ten complete years of thermal-stress record"}
    grp = region_group(lat, lon)
    recent = full[-10:]
    mean = lambda k, ys: round(float(np.mean([ann[y][k] for y in ys])), 0)      # noqa: E731
    first = [y for y in full if y <= 2000]
    summary = {"years": [full[0], full[-1]], "last10": [recent[0], recent[-1]],
               "heat_hours_per_year_last10": {k: mean(k, recent) for k, _ in HEAT_T}, "cold_hours_per_year_last10": {k: mean(k, recent) for k, _ in COLD_T},
               "heat_hours_per_year_1981_2000": {k: mean(k, first) for k, _ in HEAT_T} if first else None, "cold_hours_per_year_1981_2000": {k: mean(k, first) for k, _ in COLD_T} if first else None,
               "utci_max_last10": round(float(np.mean([ann[y]["utci_max"] for y in recent])), 1), "utci_min_last10": round(float(np.mean([ann[y]["utci_min"] for y in recent])), 1),
               "trend_moderate_heat_hours": _trend(full, [ann[y]["moderate_heat_ge_32"] for y in full]), "trend_strong_cold_hours": _trend(full, [ann[y]["strong_cold_le_minus13"] for y in full])}
    bias = HOT_BIAS_C[grp]
    adj = {k: round(float(np.nansum(s["utci"][np.isin(s["yrs"], recent)] - bias >= th) / len(recent)), 0) for k, th in HEAT_T[:2]}
    fut = _future(s, lat, lon, None) if future else {"status": "skipped"}
    top = summary["heat_hours_per_year_last10"]["moderate_heat_ge_32"]
    return {"status": "modelled", "headline_value": top, "region_group": grp, "summary": summary, "annual": ann,
            "bias": {"hot_tail_bias_c": bias, "sites_in_group": HOT_BIAS_SITES[grp], "heat_hours_per_year_last10_bias_adjusted": adj,
                     "meaning": "Reanalysis-driven UTCI ran this many degrees below station-driven UTCI on the hottest hours in this region group; the adjusted counts add it back. The unadjusted counts are the lower bound."},
            "future": fut, "validation": VALIDATION,
            "method": "ERA5 hourly weather -> standing-person mean radiant temperature (Thorsson 2007 budget) -> UTCI (Broede 2012 polynomial)",
            "limits": ["Open ground in the sun for a walking person; indoors, shade and street canyons differ.", "ERA5 is a 25 km cell; sites in cities or industrial yards run hotter than the cell.",
                       "Wind and sunshine errors are not tested (station comparison shares them).", "Future values are a sensitivity of recent weather to ensemble warming, not a backtested projection."]}
