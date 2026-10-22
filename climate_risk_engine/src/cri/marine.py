"""
marine.py - marine heatwaves at a coastal or offshore site from NOAA OISST sea-surface temperature.

Method: the Hobday et al. (2016) definition. For every day of the year the climatological mean and the 90th percentile of sea-surface
temperature are taken from 1982-2011 (an 11-day window around the day, then smoothed over 31 days). A marine heatwave is a run of at least
5 days above the 90th percentile; two events separated by 2 days or fewer are joined. Category (Hobday et al. 2018) is the largest ratio of the anomaly
to the gap between the 90th percentile and the mean: 1 moderate (under 2), 2 strong, 3 severe, 4 extreme (4 or more).

Data: NOAA OISST v2.1 daily, 0.25 degree grid, 1982 to within days of today, read from the NOAA Open Data Dissemination Zarr copy on AWS
(noaa-cdr-sea-surface-temp-optimum-interpolation-pds). Each day is one small request, so the FIRST call for a new site reads about 16,000 daily files
(minutes, and it needs the `blosc` package); the series is then cached on disk and later calls fetch only the new days. The Zarr copy stops in 2025, so newer
days are read one at a time from NCEI's THREDDS point service. A site on land uses the
nearest ocean cell within 0.5 degree and says how far away it is.

Tested in docs/CONVECTIVE_MARINE_BENCHMARK.md: the implementation against the author's reference code, and detection of documented events.
What it does not do: it says nothing about the harm (that depends on species, depth and the asset), about water below the surface, about
coastal water finer than 25 km, or about future events.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

S3 = "https://noaa-cdr-sea-surface-temp-optimum-interpolation-pds.s3.amazonaws.com/data/v2.1/avhrr-zarr/"
BASE_Y0, BASE_Y1 = 1982, 2011
FIRST_DAY = dt.date(1982, 1, 1)
LAG_DAYS = 4                           # the newest days are still being produced
CATEGORIES = {1: "Moderate", 2: "Strong", 3: "Severe", 4: "Extreme"}
THREADS = int(os.environ.get("CRI_MARINE_THREADS", "128"))


def _cache_dir() -> Path:
    d = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "marine"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cell_index(lat: float, lon: float) -> tuple:
    """Row and column of the OISST 0.25 degree cell containing the point (rows from 90 S, columns from 0 E)."""
    return int(math.floor((lat + 90.0) / 0.25)), int(math.floor((lon % 360.0) / 0.25)) % 1440


def cell_centre(i: int, j: int) -> tuple:
    return -89.875 + 0.25 * i, ((0.125 + 0.25 * j + 180.0) % 360.0) - 180.0


def _blosc():
    try:
        import blosc
        return blosc
    except ImportError as e:                                    # noqa: BLE001
        raise RuntimeError("the marine heatwave model needs the 'blosc' package to read NOAA's compressed OISST chunks (pip install blosc)") from e


_POOL = None


def _http_get(url: str, timeout: float = 40.0) -> tuple:
    """(status, body). Connections are pooled and reused: a thousand cold connections a second overwhelm name resolution on some networks."""
    global _POOL
    try:
        import urllib3
    except ImportError:                                         # noqa: BLE001
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "climrisk-engine/1.0"}), timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, b""
    if _POOL is None:
        _POOL = urllib3.PoolManager(num_pools=8, maxsize=max(THREADS, 16), block=False, headers={"User-Agent": "climrisk-engine/1.0"})
    r = _POOL.request("GET", url, timeout=urllib3.Timeout(connect=15.0, read=timeout), retries=False)
    return r.status, r.data


def _chunk(day: dt.date, ci: int, cj: int) -> Optional[np.ndarray]:
    """One 180 x 180 block of that day's SST in hundredths of a degree (int16, -999 = no data); None if the day is not published."""
    bl = _blosc()
    u = f"{S3}{day:%Y/%m}/oisst-avhrr-v02r01.{day:%Y%m%d}.zarr/sst/0.0.{ci}.{cj}"
    last: Optional[Exception] = None
    for k in range(7):
        try:
            status, body = _http_get(u)
            if status == 200:
                return np.frombuffer(bl.decompress(body), dtype="<i2").reshape(180, 180)
            if status in (403, 404):
                return None
            last = RuntimeError(f"HTTP {status}")
        except Exception as e:                                  # noqa: BLE001
            last = e
        time.sleep(0.5 * (k + 1))
    raise RuntimeError(f"could not read NOAA OISST ({last})")


NCSS = "https://www.ncei.noaa.gov/thredds/ncss/grid/OisstBase/NetCDF/V2.1/AVHRR/"


def _ncss_value(day: dt.date, lat: float, lon: float) -> Optional[float]:
    """SST for one cell and day from NCEI's THREDDS point service. Used only for days newer than the Zarr copy. None if the day is not published."""
    lo = lon % 360.0
    for suffix in ("", "_preliminary"):
        u = f"{NCSS}{day:%Y%m}/oisst-avhrr-v02r01.{day:%Y%m%d}{suffix}.nc?var=sst&latitude={lat:.3f}&longitude={lo:.3f}&accept=csv"
        for k in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "climrisk-engine/1.0"}), timeout=40) as r:
                    v = r.read().decode().strip().splitlines()[-1].split(",")[-1]
                x = float(v)
                return None if not math.isfinite(x) or x < -3.0 else round(x, 2)
            except urllib.error.HTTPError as e:
                if e.code in (400, 404):
                    break
            except Exception:                                   # noqa: BLE001
                pass
            time.sleep(0.8 * (k + 1))
    return None


def ocean_cell(lat: float, lon: float, max_km: float = 55.0) -> dict:
    """The nearest OISST cell with sea-surface temperature (a recent day is used as the land and ice mask)."""
    i0, j0 = cell_index(lat, lon)
    day = dt.date(2020, 6, 15)                                  # any day inside the Zarr copy serves as the land and ice mask (the mask does not move)
    best = None
    blocks: dict = {}
    for di in range(-2, 3):
        for dj in range(-2, 3):
            i, j = i0 + di, (j0 + dj) % 1440
            if not (0 <= i < 720):
                continue
            if (i // 180, j // 180) not in blocks:
                blocks[(i // 180, j // 180)] = _chunk(day, i // 180, j // 180)
            ch = blocks[(i // 180, j // 180)]
            if ch is None or ch[i % 180, j % 180] == -999:
                continue
            la, lo = cell_centre(i, j)
            km = 111.2 * math.hypot(la - lat, ((lo - lon + 180.0) % 360.0 - 180.0) * math.cos(math.radians(lat)))
            if best is None or km < best["distance_km"]:
                best = {"i": i, "j": j, "lat": la, "lon": lo, "distance_km": round(km, 1)}
    if best is None or best["distance_km"] > max_km:
        raise RuntimeError("no open-ocean cell within 0.5 degree of this point: not a marine site for this model")
    return best


def _cache_file(i: int, j: int) -> Path:
    return _cache_dir() / f"oisst_{i}_{j}.json"


def fetch_cells(cells: list, budget_s: Optional[float] = None, progress: Optional[Callable] = None, end: Optional[dt.date] = None) -> dict:
    """Make sure the daily SST series 1982 to now is cached for every (i, j) cell. Cells that share a 45 x 45 degree block share each day's download.
    budget_s stops early (call again to continue; nothing already read is repeated). Returns {(i, j): number of days now cached}."""
    t0 = time.time()
    state = {}
    for c in cells:
        f = _cache_file(*c)
        if f.exists():
            try:
                state[c] = json.loads(f.read_text())
                continue
            except Exception:                                   # noqa: BLE001
                pass
        state[c] = {"start": FIRST_DAY.isoformat(), "sst": []}
    end = end or (dt.date.today() - dt.timedelta(days=LAG_DAYS))
    next_day = {c: dt.date.fromisoformat(v["start"]) + dt.timedelta(days=len(v["sst"])) for c, v in state.items()}
    start = min(next_day.values())
    batch = 240
    d0 = start
    with cf.ThreadPoolExecutor(THREADS) as ex:
        while d0 <= end and (budget_s is None or time.time() - t0 < budget_s):
            days = [d0 + dt.timedelta(n) for n in range(batch) if d0 + dt.timedelta(n) <= end]
            blocks = sorted({(c[0] // 180, c[1] // 180) for c in cells})
            futs = {(d, b): ex.submit(_chunk, d, *b) for d in days for b in blocks}
            data = {k: f.result() for k, f in futs.items()}
            tail = [(c, d) for c in cells for d in days if d >= next_day[c] and data[(d, (c[0] // 180, c[1] // 180))] is None]
            tailv = {}
            if tail:                                            # days newer than the Zarr copy: one point request each, a few at a time
                with cf.ThreadPoolExecutor(6) as ex2:
                    for (c, d), v in zip(tail, ex2.map(lambda t: _ncss_value(t[1], *cell_centre(*t[0])), tail)):
                        tailv[(c, d)] = v
            for c in cells:
                for d in days:
                    if d < next_day[c]:
                        continue
                    ch = data[(d, (c[0] // 180, c[1] // 180))]
                    if ch is None:
                        state[c]["sst"].append(tailv.get((c, d)))
                    else:
                        v = int(ch[c[0] % 180, c[1] % 180])
                        state[c]["sst"].append(None if v == -999 else round(v * 0.01, 2))
                    next_day[c] = d + dt.timedelta(days=1)
                _cache_file(*c).write_text(json.dumps(state[c]))
            d0 = days[-1] + dt.timedelta(days=1)
            if progress:
                progress(d0, time.time() - t0)
    return {c: len(v["sst"]) for c, v in state.items()}


def sst_series(lat: float, lon: float, budget_s: Optional[float] = None, end: Optional[dt.date] = None) -> dict:
    """Daily SST at the ocean cell nearest the point. Raises if no ocean cell lies within ~0.5 degree, or if the series is still being downloaded."""
    oc = ocean_cell(lat, lon)
    c = (oc["i"], oc["j"])
    end = end or (dt.date.today() - dt.timedelta(days=LAG_DAYS))
    n = fetch_cells([c], budget_s=budget_s, end=end)[c]
    st = json.loads(_cache_file(*c).read_text())
    t0 = dt.date.fromisoformat(st["start"])
    if t0 + dt.timedelta(days=n - 1) < end - dt.timedelta(days=1):
        raise RuntimeError(f"the sea-temperature record for this site is still downloading ({n} days so far); call again")
    v = st["sst"]
    while v and v[-1] is None:
        v.pop()
    t = [(t0 + dt.timedelta(days=k)).isoformat() for k in range(len(v))]
    return {"time": t, "sst": v, "cell_lat": oc["lat"], "cell_lon": oc["lon"], "distance_km": oc["distance_km"]}


# ------------------------------------------------------------------ detection

def _doy(dates) -> np.ndarray:
    d = np.array(dates, dtype="datetime64[D]")
    y = d.astype("datetime64[Y]")
    doy = (d - y).astype(int) + 1
    leap = ((y.astype(int) + 1970) % 4 == 0) & (((y.astype(int) + 1970) % 100 != 0) | ((y.astype(int) + 1970) % 400 == 0))
    # map every year onto a 366-day calendar: after Feb 28 non-leap days shift by one so 29 Feb is its own day
    return np.where(~leap & (doy > 59), doy + 1, doy)


def _circ_smooth(x: np.ndarray, width: int) -> np.ndarray:
    k = np.ones(width) / width
    h = width // 2
    ext = np.concatenate([x[-h:], x, x[:h]])
    return np.convolve(ext, k, mode="valid")


def climatology(dates, sst, pctile: float = 90.0, window: int = 5, smooth: int = 31):
    """Mean and percentile threshold by day of year (366), baseline 1982-2011, 11-day pooling then 31-day smoothing (Hobday et al. 2016)."""
    doy = _doy(dates)
    yr = np.array(dates, dtype="datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    base = (yr >= BASE_Y0) & (yr <= BASE_Y1) & np.isfinite(sst)
    mean, thr = np.full(366, np.nan), np.full(366, np.nan)
    for d in range(1, 367):
        idx = ((doy - d + 183) % 366 - 183)                         # circular distance in days
        m = base & (np.abs(idx) <= window)
        if m.sum() >= 20:
            mean[d - 1] = sst[m].mean()
            thr[d - 1] = np.percentile(sst[m], pctile)
    for a in (mean, thr):
        ok = np.isfinite(a)
        if not ok.all() and ok.any():
            a[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), a[ok], period=366)
    return _circ_smooth(mean, smooth), _circ_smooth(thr, smooth)


def detect(dates, sst, min_duration: int = 5, max_gap: int = 2) -> dict:
    """Marine heatwave events in a daily series. Returns events (start, end, duration, peak anomaly, mean anomaly, category) and the climatology."""
    sst = np.asarray(sst, dtype=float)
    mean, thr = climatology(dates, sst)
    doy = _doy(dates) - 1
    anom = sst - mean[doy]
    hot = np.isfinite(sst) & (sst > thr[doy])
    runs = []
    i, n = 0, len(sst)
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            runs.append([i, j])
            i = j + 1
        else:
            i += 1
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_duration]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    events = []
    for a, b in merged:
        seg = slice(a, b + 1)
        ratio = np.nanmax((sst[seg] - mean[doy[seg]]) / np.maximum(thr[doy[seg]] - mean[doy[seg]], 1e-6))
        events.append({"start": str(np.array(dates)[a]), "end": str(np.array(dates)[b]), "duration_days": int(b - a + 1),
                       "peak_anomaly_c": round(float(np.nanmax(anom[seg])), 2), "mean_anomaly_c": round(float(np.nanmean(anom[seg])), 2),
                       "category": int(min(4, max(1, math.floor(ratio)))), "i0": int(a), "i1": int(b)})
    return {"events": events, "mean": mean, "thr": thr, "anom": anom}


def marine_heatwave(lat: float, lon: float, budget_s: Optional[float] = None, end: Optional[dt.date] = None) -> dict:
    """Marine heatwave record at a site. Raises RuntimeError if the point is not near open ocean or the data cannot be read."""
    s = sst_series(lat, lon, budget_s, end)
    t = s["time"]
    v = np.array([np.nan if x is None else x for x in s["sst"]], dtype=float)
    d = detect(t, v)
    ev = d["events"]
    years = np.array(t, dtype="datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    ys = sorted(set(years.tolist()))
    full = [y for y in ys if (years == y).sum() >= 350]
    per_year = {y: sum(e["duration_days"] for e in ev if int(e["start"][:4]) == y) for y in full}
    recent = [y for y in full if y >= max(full) - 9]
    base_days = float(np.mean([per_year[y] for y in full if BASE_Y0 <= y <= BASE_Y1]))
    slope = float(np.polyfit(full, [per_year[y] for y in full], 1)[0]) if len(full) >= 10 else None
    big = max(ev, key=lambda e: (e["category"], e["peak_anomaly_c"])) if ev else None
    last_ev = ev[-1] if ev else None
    return {"covered": True, "cell": [s["cell_lat"], s["cell_lon"]], "distance_to_ocean_cell_km": s["distance_km"],
            "series": [t[0], t[-1]],
            "heatwave_days_per_year_baseline_1982_2011": round(base_days, 1),
            "heatwave_days_per_year_last_10_years": round(float(np.mean([per_year[y] for y in recent])), 1),
            "heatwave_days_trend_per_decade": None if slope is None else round(slope * 10.0, 1),
            "events_per_year_last_10_years": round(sum(1 for e in ev if int(e["start"][:4]) in recent) / max(len(recent), 1), 2),
            "strongest_event": None if big is None else {k: big[k] for k in ("start", "end", "duration_days", "peak_anomaly_c", "category")},
            "latest_event": None if last_ev is None else {k: last_ev[k] for k in ("start", "end", "duration_days", "peak_anomaly_c", "category")},
            "events_total": len(ev), "category_names": CATEGORIES, "source": "NOAA OISST v2.1 daily 0.25 degree (NOAA Open Data Dissemination, AWS), Hobday et al. 2016 definition, 1982-2011 baseline",
            "basis": ("Surface sea temperature on a 25 km cell, not water at depth or in a harbour or estuary. It says how often and how strongly the surface "
                      "sea has been unusually warm, not what that does to a given species, farm or asset.")}
