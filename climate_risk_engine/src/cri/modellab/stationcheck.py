"""Independent check against weather stations: NOAA GHCN-Daily (US Government data, public domain; global, ~19,000 long-record stations bundled as an index).

The studies run on gridded products (reanalysis, satellite-corrected). A station is a measurement. This module finds the nearest long-record station, builds its annual maxima, and asks
whether the gridded series and the station agree on (a) which years were extreme (rank correlation of annual maxima) and (b) how large the extremes are (ratio of design levels for rain,
difference of annual-maximum temperature for heat). A grid cell averages over tens of kilometres, so point rainfall extremes are expected to exceed grid values: the agreement band says so.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from . import coastal as C, data as D, evt

NCEI = "https://www.ncei.noaa.gov/access/services/data/v1"
UA = {"User-Agent": "ClimRisk-ModelLab/1.0 (climate risk research prototype)"}
ELEMENT = {"rain": "PRCP", "heat": "TMAX"}
ATTRIBUTION = "Station data: NOAA National Centers for Environmental Information, Global Historical Climatology Network daily (GHCN-Daily), public domain."


def _index():
    from functools import lru_cache
    return _load()


def _load():
    z = np.load(Path(__file__).resolve().parent.parent / "data" / "ghcn_long.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def nearest(lat: float, lon: float, element: str, k: int = 3, max_km: float = 100.0, min_years: int = 25) -> list:
    d = _load()
    a, b = (d["p0"], d["p1"]) if element == "PRCP" else (d["t0"], d["t1"])
    dist = C.haversine_km(lat, lon, d["lat"].astype(float), d["lon"].astype(float))
    ok = (b >= 2005) & ((b - a + 1) >= min_years) & (dist <= max_km)
    idx = np.where(ok)[0]
    idx = idx[np.argsort(dist[idx])][:k]
    return [{"id": str(d["ids"][i]), "name": str(d["names"][i]).title(), "km": float(dist[i]), "first": int(a[i]), "last": int(b[i])} for i in idx]


def _text(url: str, fetch=None) -> str:
    if fetch is not None:
        return fetch(url)
    p = D._cache_path(url).with_suffix(".csv")
    if p.exists():
        return p.read_text()
    last = None
    for i in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                t = r.read().decode("utf-8", "ignore")
            try:
                p.write_text(t)
            except Exception:                                          # noqa: BLE001
                pass
            return t
        except Exception as e:                                         # noqa: BLE001
            last = e
            __import__("time").sleep(3 * (i + 1))
    raise RuntimeError(f"weather-station service unavailable ({last})")


def annual_max(station: str, element: str, y0: int = 1950, y1: int = 2025, fetch=None, min_days: int = 270) -> tuple:
    """(years, annual maxima, name) for years with at least min_days daily values. PRCP in mm, TMAX in °C."""
    url = f"{NCEI}?dataset=daily-summaries&stations={station}&startDate={y0}-01-01&endDate={y1}-12-31&dataTypes={element}&format=csv&units=metric&includeStationName=true"
    txt = _text(url, fetch)
    by: dict = {}
    n: dict = {}
    nm = ""
    import csv
    for p in list(csv.reader(io.StringIO(txt)))[1:]:
        if len(p) < 4:
            continue
        try:
            y = int(p[2][:4]); v = float(p[3])
        except ValueError:
            continue
        nm = nm or p[1]
        by[y] = max(by.get(y, -1e9), v); n[y] = n.get(y, 0) + 1
    ys = [y for y in sorted(by) if n[y] >= min_days]
    return ys, [by[y] for y in ys], nm


def _one(hazard, st, years, x, fetch, grid_levels):
    el = ELEMENT[hazard]
    sy, sx, nm = annual_max(st["id"], el, fetch=fetch)
    gy = dict(zip(years, x)); common = [y for y in sy if y in gy]
    if len(common) < 10:
        raise ValueError(f"{st['name']}: only {len(common)} overlapping complete years")
    s = np.array([sx[sy.index(y)] for y in common], float); g = np.array([gy[y] for y in common], float)
    from scipy import stats
    rho, p = stats.spearmanr(s, g)
    out = {"station": st["name"], "id": st["id"], "km": st["km"], "n": len(common), "rho": float(rho), "p": float(p)}
    if hazard == "rain":
        lv_s = {T: evt.level(evt.best_model(sx)["params"], T) for T in (10, 50)}
        lv_g = grid_levels or {T: evt.level(evt.best_model(g)["params"], T) for T in (10, 50)}
        out.update({"station_levels": lv_s, "grid_levels": {T: lv_g[T] for T in (10, 50)}, "ratio50": float(lv_s[50] / lv_g[50]), "mean_ratio": float(s.mean() / g.mean())})
        out["detail"] = f"{st['name']} ({st['km']:.0f} km, {len(common)} years): 50-year daily rain {lv_s[50]:.0f} mm at the station against {lv_g[50]:.0f} mm gridded (ratio {out['ratio50']:.2f}); rank correlation of annual maxima {rho:.2f}"
    else:
        out.update({"mean_diff": float(s.mean() - g.mean())})
        out["detail"] = f"{st['name']} ({st['km']:.0f} km, {len(common)} years): typical hottest day {s.mean():.1f} °C at the station against {g.mean():.1f} °C gridded (difference {out['mean_diff']:+.1f} °C); rank correlation {rho:.2f}"
    return out


def compare(hazard: str, lat: float, lon: float, years, x, fetch=None, grid_levels: Optional[dict] = None, ci: Optional[dict] = None, k: int = 3) -> dict:
    """Compare the gridded annual maxima (years, x; mm/day for rain, air temperature °C for heat) with up to k nearest long-record stations; the verdict uses their medians."""
    el = ELEMENT[hazard]
    cands = nearest(lat, lon, el, k=k)
    if not cands:
        return {"ok": False, "reason": "no long-record weather station within 100 km (GHCN-Daily)"}
    res, why = [], []
    for st in cands:
        try:
            res.append(_one(hazard, st, years, x, fetch, grid_levels))
        except Exception as e:                                         # noqa: BLE001
            why.append(str(e)[:100])
    if not res:
        return {"ok": False, "reason": "; ".join(why) or "no usable station record"}
    rho = float(np.median([r["rho"] for r in res]))
    out = {"ok": True, "stations": res, "n_stations": len(res), "rho": rho, "unit": "mm/day" if hazard == "rain" else "°C", "attribution": ATTRIBUTION}
    if hazard == "rain":
        ratio = float(np.median([r["ratio50"] for r in res])); out["ratio50"] = ratio
        out["station_levels"] = {T: float(np.median([r["station_levels"][T] for r in res])) for T in (10, 50)}
        out["grid_levels"] = res[0]["grid_levels"]
        out["level_ok"] = 0.6 <= ratio <= 1.7
    else:
        diff = float(np.median([r["mean_diff"] for r in res])); out["mean_diff"] = diff
        out["level_ok"] = abs(diff) <= 2.5
    out["rank_ok"] = rho >= 0.3
    both = out["level_ok"] and out["rank_ok"]; none = (not out["level_ok"]) and (not out["rank_ok"])
    out["agree"] = True if both else False if none else None
    out["verdict"] = "consistent" if both else "inconsistent" if none else "partly consistent"
    if max(r["n"] for r in res) < 15:
        out["agree"] = None; out["verdict"] = "inconclusive (fewer than 15 overlapping complete years)"
    lead = ("Median over %d stations: " % len(res)) if len(res) > 1 else ""
    out["note"] = lead + (f"design-level ratio {out['ratio50']:.2f} (agreement band 0.6–1.7), " if hazard == "rain" else f"hottest-day difference {out['mean_diff']:+.1f} °C (band ±2.5), ") + f"rank correlation {rho:.2f} (needs 0.3): {out['verdict']}."
    out["detail"] = " ".join(r["detail"] + "." for r in res)
    return out
