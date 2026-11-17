"""Design rainfall for a place from TWO gridded records, for 1-day and 3-day totals, with the disagreement and the largest recorded event stated.  (11 Nov 2026)

Why: the surface-water model used one source (ERA5 daily rain) and one duration (a 1 h to 6 h share of the daily total). At Chennai, ERA5 puts the largest daily
total of 2015 at 101 mm (the 1-in-100 design at 203 mm) while 290 to 345 mm fell at city stations on 1 December 2015 and 494 mm at Tambaram; NASA POWER (IMERG-
calibrated) gives 133 mm for the day and 238 mm over three days. A long, humid-tropics event is a multi-day volume problem, not a one-hour burst, and one smooth
reanalysis cannot be the only input.

Sources (all free to use, no key):
  GAUGE      nearest long-record rain gauge, NOAA GHCN-Daily (public domain), up to 60 km; the only POINT measurement. At Chennai: 344.9 mm on 2 Dec 2015, 3.5 km from Velachery.
  ERA5       Open-Meteo archive, daily precipitation_sum, 1991-2020
  NASA POWER daily PRECTOTCORR (satellite-corrected precipitation, IMERG-calibrated from 2000), 1991-2020
For each source and duration (1 and 3 days): annual maxima of the running total -> Gumbel fit -> return level. The DESIGN value is the larger of the sources
(stated, never averaged away) and `uplift_if_sources_disagree` reports the ratio. Gridded rain is an area average: point extremes in a storm can be higher,
so every value is a floor, and `largest_in_record` shows what the record already contains.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

Y0, Y1 = 1991, 2020
_UA = {"User-Agent": "ClimRisk-raindesign/1.0"}


def _cache(key: str) -> Path:
    import hashlib
    d = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "raindesign"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.md5(key.encode()).hexdigest() + ".json")


def _get(url: str, timeout: float = 60.0) -> dict:
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:                                         # noqa: BLE001
            last = e
            time.sleep(1.5)
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:120]}")


def daily_era5(lat: float, lon: float, y0=Y0, y1=Y1, fetch: Callable = _get) -> dict:
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(
        {"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31", "daily": "precipitation_sum", "timezone": "UTC"})
    p = _cache(url)
    if p.exists():
        return json.loads(p.read_text())
    d = fetch(url)
    out = {"source": "ERA5 reanalysis via Open-Meteo archive", "t": d["daily"]["time"], "mm": d["daily"]["precipitation_sum"], "url": url}
    p.write_text(json.dumps(out))
    return out


def daily_power(lat: float, lon: float, y0=Y0, y1=Y1, fetch: Callable = _get) -> dict:
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(
        {"parameters": "PRECTOTCORR", "community": "AG", "longitude": round(lon, 3), "latitude": round(lat, 3), "start": f"{y0}0101", "end": f"{y1}1231", "format": "json"}))
    p = _cache(url)
    if p.exists():
        return json.loads(p.read_text())
    d = fetch(url, 120)
    v = d["properties"]["parameter"]["PRECTOTCORR"]
    ts = sorted(v)
    out = {"source": "NASA POWER daily PRECTOTCORR (satellite-corrected, IMERG-calibrated)", "t": [f"{t[:4]}-{t[4:6]}-{t[6:]}" for t in ts], "mm": [None if v[t] is None or v[t] < 0 else v[t] for t in ts], "url": url}
    p.write_text(json.dumps(out))
    return out


def daily_gauge(lat: float, lon: float, y0: int = 1950, y1: int = 2025, fetch: Optional[Callable] = None, max_km: float = 60.0) -> dict:
    """Daily rain from the nearest long-record rain gauge (NOAA GHCN-Daily, public domain). A gauge is a measurement at a point; the grids above are area averages.
    Missing days stay None, so a running total over a gap is dropped by `annual_maxima` rather than undercounted."""
    import csv
    import io
    from ..modellab import stationcheck as SC
    cands = SC.nearest(lat, lon, "PRCP", k=3, max_km=max_km, min_years=25)
    if not cands:
        raise RuntimeError(f"no long-record rain gauge within {max_km:.0f} km")
    last = None
    for st in cands:
        url = SC.daily_url(st["id"], "PRCP", y0, y1)
        try:
            txt = SC._text(url, fetch)
        except Exception as e:                                       # noqa: BLE001
            last = e
            continue
        rows, dropped = SC.clean_rows(txt)                           # values NOAA flagged as failing a quality check are removed, and reported below
        day: dict = {d: v for _, d, v in rows}
        if len(day) < 25 * 300:
            last = RuntimeError(f"{st['name']}: only {len(day)} daily values")
            continue
        import datetime as dt
        a, b = dt.date.fromisoformat(min(day)), dt.date.fromisoformat(max(day))
        t = [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]
        return {"source": f"rain gauge {st['name']} (GHCN-Daily {st['id']}, {st['km']:.0f} km away)", "t": t, "mm": [day.get(d) for d in t], "km": st["km"], "station": st["name"], "id": st["id"], "url": url,
                "qc_dropped": {"n": len(dropped), "largest": sorted(dropped, key=lambda x: -x[1])[:3], "rule": "NOAA GHCN-Daily quality flag non-blank"}}
    raise RuntimeError(f"gauge records unavailable ({last})")


def annual_maxima(series: dict, days: int) -> dict:
    """{year: (max running `days`-day total in mm, end date)} from a daily series."""
    mm = np.array([np.nan if x is None else float(x) for x in series["mm"]])
    t = series["t"]
    if days > 1:
        k = np.ones(days)
        ok = np.convolve(~np.isnan(mm), k, "valid") == days
        run = np.convolve(np.nan_to_num(mm), k, "valid")
        run = np.where(ok, run, np.nan)
        ends = t[days - 1:]
    else:
        run, ends = mm, t
    best: dict = {}
    for v, d in zip(run, ends):
        if np.isnan(v):
            continue
        y = int(d[:4])
        if y not in best or v > best[y][0]:
            best[y] = (float(v), d)
    return best


def design(lat: float, lon: float, rp: int = 100, fetch: Optional[Callable] = None, sources: tuple = ("era5", "power", "gauge")) -> dict:
    """{available, rp, d1_mm, d3_mm, by_source{...}, source_used{1,3}, largest_in_record{...}, uplift_if_sources_disagree, notes}. Never invents: a source that fails is listed with its reason."""
    from ..modellab import hydrology
    getters = {"era5": daily_era5, "power": daily_power, "gauge": daily_gauge}
    by, fails = {}, {}
    for name in sources:
        try:
            ser = getters[name](lat, lon, **({"fetch": fetch} if fetch else {}))
            by[name] = {"source": ser["source"]}
            if "km" in ser:
                by[name]["km"] = round(ser["km"], 1)
            if ser.get("qc_dropped"):
                by[name]["qc_dropped"] = ser["qc_dropped"]
            for days in (1, 3):
                am = annual_maxima(ser, days)
                if len(am) < (25 if name == "gauge" else 15):
                    raise RuntimeError(f"only {len(am)} years of data")
                vals = [v for v, _ in am.values()]
                mu, beta = hydrology.gumbel_fit(vals)
                top = max(am.items(), key=lambda kv: kv[1][0])
                by[name][f"d{days}_mm"] = round(float(hydrology.gumbel_quantile(mu, beta, rp)), 1)
                by[name][f"max{days}_mm"] = round(top[1][0], 1)
                by[name][f"max{days}_date"] = top[1][1]
                by[name][f"d{days}_rp10_mm"] = round(float(hydrology.gumbel_quantile(mu, beta, 10)), 1)
        except Exception as e:                                       # noqa: BLE001
            fails[name] = f"{type(e).__name__}: {str(e)[:120]}"
    if not by:
        return {"available": False, "reason": "no rainfall record could be read (" + "; ".join(f"{k}: {v}" for k, v in fails.items()) + ")"}
    out = {"available": True, "rp": rp, "by_source": by, "failed_sources": fails}
    for days in (1, 3):
        c = {n: s[f"d{days}_mm"] for n, s in by.items() if f"d{days}_mm" in s}
        best = max(c, key=c.get)
        out[f"d{days}_mm"] = c[best]
        out.setdefault("source_used", {})[f"{days}d"] = by[best]["source"]
        out.setdefault("largest_in_record", {})[f"{days}d"] = max(({"mm": s[f"max{days}_mm"], "date": s[f"max{days}_date"], "source": s["source"]} for s in by.values()), key=lambda r: r["mm"])
        if len(c) > 1:
            out.setdefault("uplift_if_sources_disagree", {})[f"{days}d"] = round(max(c.values()) / max(min(c.values()), 1e-6), 2)
    out["notes"] = ["Gridded rain is an area average: a storm's point maximum can be higher, so grid-only values are floors." if "gauge" not in by else
                    "A rain gauge is a point measurement; it is within %.0f km of the site, so it can differ from rain over the site itself." % by["gauge"].get("km", 0),
                    "The design value is the larger of the sources; the others are in by_source."]
    return out


def event_cross_check(d: dict, lat: float, lon: float, reader: Optional[Callable] = None) -> dict:
    """Check the largest 1-day total in the records against CHIRPS (satellite plus gauge blend) around the same date. Returns {available, record, chirps_days, best_day_mm,
    two_day_mm, ratio_best_day, ratio_two_day, reading}. A gauge's observation day (for India, 08:30 local to 08:30 the next morning) straddles two UTC days, which CHIRPS uses, so
    the record is compared with the best single CHIRPS day AND the sum of that date and the day before. One storm is a check on plausibility, not a design rainfall."""
    import datetime as dt
    from ..connectors import chirps
    big = (((d or {}).get("largest_in_record") or {}).get("1d"))
    if not big:
        return {"available": False, "reason": "no largest-day record to check (design rain not available)"}
    d1 = dt.date.fromisoformat(big["date"][:10])
    days = [chirps.day_total(lat, lon, (d1 - dt.timedelta(days=k)).isoformat(), reader=reader) for k in (1, 0)]
    if not all(x.get("available") for x in days):
        return {"available": False, "record": big, "reason": next(x.get("reason") for x in days if not x.get("available"))}
    best, two = max(x["cell_mm"] for x in days), round(sum(x["cell_mm"] for x in days), 1)
    rb, r2 = (round(best / big["mm"], 2), round(two / big["mm"], 2)) if big["mm"] else (None, None)
    agree = bool(r2 and 0.75 <= r2 <= 1.33)
    return {"available": True, "record": big, "chirps_days": [{"date": x["date"], "cell_mm": x["cell_mm"], "max_within_mm": x["max_within_mm"]} for x in days], "best_day_mm": best, "two_day_mm": two,
            "ratio_best_day": rb, "ratio_two_day": r2, "agree_within_25pct_two_day": agree, "source": chirps.ATTRIBUTION,
            "reading": (f"The largest day in the records is {big['mm']:.0f} mm on {big['date']} ({big['source']}). CHIRPS (satellite plus gauge blend, 5 km cell) reads "
                        + ", ".join(f"{x['cell_mm']:.0f} mm on {x['date']}" for x in days) + f": best single day {best:.0f} mm, two days {two:.0f} mm. "
                        + ("The two-day total is within about 25 % of the record." if agree else "They differ by more than 25 %, so the sources do not agree on this storm; treat the larger as a floor, not a settled value."))}
