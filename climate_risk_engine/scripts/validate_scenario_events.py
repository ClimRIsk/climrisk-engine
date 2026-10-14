"""
Do the engine's hazard layers recognise real, documented extreme events at the places the scenarios use?

For each event (public sources in docs/SCENARIO_RESEARCH.md) the hazard series is built the way the engine builds it, WITHOUT the event year being used to fit (the extreme-value
fit leaves the event's year out), and the event is placed on that fit: its return period, or its percentile. A layer that does not call a record-breaking event rare is blind to it.

  event                        place                 variable (engine layer)                         expectation
  Hurricane Harvey, Aug 2017   Houston               5-day rain (lab:rain)                           return period of 50 years or more
  UAE floods, 15-16 Apr 2024   Dubai                 2-day rain (lab:rain)                           return period of 100 years or more
  Thailand 2011, Ahr 2021, Harvey 2017: the river flood MAP is checked at the flooded places instead (local rain is the wrong variable for a basin-scale flood: Ayutthaya's own
  Jul-Oct rain in 2011 ranked 9th of 45 on the fallback data, because the water came down the Chao Phraya from upstream)
  Fort McMurray fire, 3 May 2016   Fort McMurray     fire weather index (lab:fire)                   above the local 99th percentile of May-September days
  Palisades fire, 7 Jan 2025   Los Angeles           fire weather index (lab:fire)                   above the local 99th percentile of November-March days
  Winter Storm Uri, Feb 2021   Houston, Dallas       annual minimum daily low (cold study)           return period of 20 years or more

  python scripts/validate_scenario_events.py     (resumable; run until it says complete)

Data: ERA5 through Open-Meteo when its allowance lets us, else NASA POWER (MERRA-2: smoother, so rainfall extremes read LOW and the test is harder for the engine to pass).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "data" / "backtest" / "scenario_events_validation.json"
EULER = 0.5772156649015329


def _once(url, timeout=150):
    import urllib.error
    import urllib.request
    from cri.modellab import data as D
    try:
        return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "climrisk-research/0.1"}), timeout=timeout).read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise D.QuotaExceeded("rate limited") from e
        raise


def daily(lat, lon, era5_vars: str, power_params: str):
    """-> (times, {name: array}, source). Names are the ERA5 variable names; POWER values are mapped onto them."""
    from cri.modellab import data as D
    try:
        d = D.archive(lat, lon, 1981, 2025, daily=era5_vars, fetch=_once)["daily"]
        return d["time"], {k: np.array([np.nan if x is None else x for x in d[k]], float) for k in era5_vars.split(",")}, "ERA5 (Open-Meteo)"
    except D.QuotaExceeded:
        pass
    p = D.power_daily(lat, lon, 1981, 2025, power_params, fetch=_once)
    m = {"precipitation_sum": "PRECTOTCORR", "temperature_2m_min": "T2M_MIN", "temperature_2m_max": "T2M_MAX"}
    return p["time"], {k: np.array([np.nan if x is None else x for x in p[m[k]]], float) for k in era5_vars.split(",") if k in m}, "NASA POWER (MERRA-2)"


def gumbel_fit(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    b0 = x.mean(); b1 = float(np.sum((np.arange(n)) / (n - 1) * x) / n)
    beta = (2 * b1 - b0) / math.log(2); mu = b0 - EULER * beta
    return mu, beta


def return_period(v, mu, beta):
    p = 1 - math.exp(-math.exp(-(v - mu) / beta))
    return float("inf") if p <= 0 else 1.0 / p


def _rolling_sum(a, n):
    c = np.nancumsum(np.where(np.isnan(a), 0, a))
    s = c.copy(); s[n:] = c[n:] - c[:-n]
    return s


def _annual_max_excl(times, series, exclude_year):
    yr = np.array([int(t[:4]) for t in times]); out = {}
    for y in sorted(set(yr)):
        m = yr == y
        if m.sum() > 300 and not np.isnan(series[m]).all():
            out[y] = float(np.nanmax(series[m]))
    return out


def rain_event(name, lat, lon, ndays, start, end, need_rp):
    t, v, src = daily(lat, lon, "precipitation_sum", "PRECTOTCORR")
    r = _rolling_sum(v["precipitation_sum"], ndays)
    ev_year = int(start[:4]); idx = [i for i, x in enumerate(t) if start <= x <= end]
    ev = float(max(r[i] for i in idx))
    am = _annual_max_excl(t, r, ev_year); fit = [x for y, x in am.items() if y != ev_year]
    mu, b = gumbel_fit(fit); rp = return_period(ev, mu, b)
    return {"event": name, "place": [lat, lon], "source": src, "variable": f"{ndays}-day rain (mm)", "event_value": round(ev, 0), "fit_years": len(fit), "median_annual_max": round(float(np.median(fit)), 0),
            "largest_other_year": round(max(fit), 0), "return_period_years": (round(rp, 0) if rp < 1000 else ">1000 (beyond what a 40-year record can say)"), "expectation": f"return period of {need_rp} years or more", "passed": rp >= need_rp}


def season_total_event(name, lat, lon, year, months, expect_rank):
    t, v, src = daily(lat, lon, "precipitation_sum", "PRECTOTCORR")
    yr = np.array([int(x[:4]) for x in t]); mo = np.array([int(x[5:7]) for x in t])
    tot = {y: float(np.nansum(v["precipitation_sum"][(yr == y) & np.isin(mo, months)])) for y in sorted(set(yr)) if ((yr == y) & np.isin(mo, months)).sum() > 100}
    rank = 1 + sum(1 for y, x in tot.items() if x > tot[year])
    return {"event": name, "place": [lat, lon], "source": src, "variable": f"rain total months {months[0]}-{months[-1]} (mm)", "event_value": round(tot[year], 0), "rank_in_record": rank, "record_years": len(tot),
            "median": round(float(np.median(list(tot.values()))), 0), "expectation": f"rank {expect_rank} or better", "passed": rank <= expect_rank}


def fwi_event(name, lat, lon, date, months, pct):
    sys.path.insert(0, str(ROOT / "scripts"))
    import validate_wildfire as VW
    from cri.modellab import fire
    t, T, RH, W, P, src = VW._series(lat, lon, 2025)
    mo = [int(x[5:7]) for x in t]
    fwi = np.array(fire.run_series(T, RH, W, P, mo, tropical=abs(lat) < 25)["fwi"])
    yr = np.array([int(x[:4]) for x in t]); month = np.array(mo); ev_year = int(date[:4]) if int(date[5:7]) >= months[0] or months[0] <= months[-1] else int(date[:4])
    sel = np.isin(month, months) & (yr != ev_year)
    i = t.index(date); ev = float(fwi[i]); ref = fwi[sel]
    p = float((ref < ev).mean() * 100)
    return {"event": name, "place": [lat, lon], "source": src, "variable": "Canadian FWI on the event day", "event_value": round(ev, 1), "local_p99": round(float(np.percentile(ref, 99)), 1), "percentile_of_event_day": round(p, 1),
            "expectation": f"above the local {pct}th percentile of the season's days", "passed": p >= pct}


def cold_event(name, lat, lon, start, end, need_rp):
    t, v, src = daily(lat, lon, "temperature_2m_min", "T2M_MIN")
    tn = v["temperature_2m_min"]
    ev_year = int(start[:4]); idx = [i for i, x in enumerate(t) if start <= x <= end]
    ev = float(min(tn[i] for i in idx))
    yr = np.array([int(x[:4]) for x in t]); am = {}
    for y in sorted(set(yr)):
        m = yr == y
        if y != ev_year and m.sum() > 300:
            am[y] = float(np.nanmin(tn[m]))
    neg = [-x for x in am.values()]                              # minima as maxima of the negative
    mu, b = gumbel_fit(neg); rp = return_period(-ev, mu, b)
    return {"event": name, "place": [lat, lon], "source": src, "variable": "annual minimum of the daily low (C)", "event_value": round(ev, 1), "median_annual_min": round(float(np.median(list(am.values()))), 1),
            "coldest_other_year": round(min(am.values()), 1), "return_period_years": (round(rp, 0) if rp < 1000 else ">1000 (beyond what a 40-year record can say)"), "expectation": f"return period of {need_rp} years or more", "passed": rp >= need_rp}


def flood_map_event(name, lat, lon, expect_wet, note):
    """Does the engine's river flood map (JRC, 1-in-100-year, undefended) show water where a documented flood happened?"""
    from cri.finance.flood_hazard import site_flood
    f = site_flood(lat, lon, 250.0, None, None)
    r = f.get("river") or {}
    d = (r.get("depths") or {}).get("100") or []
    mean = round(sum(d) / len(d), 2) if d else None
    return {"event": name, "place": [lat, lon], "source": r.get("source"), "variable": "river map, 1-in-100-year mean depth over a 250 m footprint (m)", "map_available": r.get("available"), "event_value": mean,
            "expectation": ("map shows more than 0.3 m" if expect_wet else "no documented expectation; shows how the map behaves"), "note": note, "passed": (None if not expect_wet else bool(mean and mean > 0.3))}


def main() -> None:
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    jobs = [
        ("harvey_houston", lambda: rain_event("Hurricane Harvey", 29.76, -95.37, 5, "2017-08-25", "2017-08-30", 50)),
        ("uae_dubai", lambda: rain_event("UAE floods April 2024", 25.20, 55.27, 2, "2024-04-15", "2024-04-17", 100)),
        ("mcmurray_fire", lambda: fwi_event("Fort McMurray wildfire", 56.73, -111.38, "2016-05-03", [5, 6, 7, 8, 9], 99)),
        ("palisades_fire", lambda: fwi_event("Palisades fire", 34.05, -118.55, "2025-01-07", [11, 12, 1, 2, 3], 99)),
        ("map_ayutthaya_2011", lambda: flood_map_event("Thailand floods 2011 (Ayutthaya)", 14.35, 100.57, True, "large river basin (Chao Phraya)")),
        ("map_ahrweiler_2021", lambda: flood_map_event("Ahr valley flood July 2021 (Bad Neuenahr-Ahrweiler)", 50.5385, 7.1127, True, "small catchment; 134 deaths in the Ahr valley, about 8.2 billion EUR insured in Germany; discharge far above statistical 100-year values")),
        ("map_buffalo_bayou_2017", lambda: flood_map_event("Hurricane Harvey (Buffalo Bayou, Houston)", 29.76, -95.37, True, "Buffalo Bayou crested 2 ft above its 1992 record")),
        ("map_houston_ship_channel", lambda: flood_map_event("Houston Ship Channel refinery area (no documented expectation)", 29.72, -95.00, False, "industrial area near Galveston Bay")),
        ("map_dubai_marina", lambda: flood_map_event("Dubai Marina (no documented expectation)", 25.08, 55.14, False, "desert city: a river-flood depth here would be a map artefact or wadi flow")),
        ("uri_houston", lambda: cold_event("Winter Storm Uri", 29.76, -95.37, "2021-02-12", "2021-02-20", 20)),
        ("uri_dallas", lambda: cold_event("Winter Storm Uri", 32.78, -96.80, "2021-02-12", "2021-02-20", 20)),
    ]
    for k, fn in jobs:
        if k in res and "error" not in res[k]:
            continue
        if time.time() - t0 > 140:
            print("budget reached; run again"); OUT.write_text(json.dumps(res, indent=1)); return
        try:
            res[k] = fn()
        except Exception as e:                                       # noqa: BLE001
            res[k] = {"error": f"{type(e).__name__}: {str(e)[:140]}"}
        OUT.write_text(json.dumps(res, indent=1))
    for k, r in res.items():
        print(k, json.dumps(r))
    print("complete" if all("error" not in r for r in res.values()) else "some events could not be fetched; run again")


if __name__ == "__main__":
    main()
