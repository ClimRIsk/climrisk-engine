"""
convective.py - severe hail and tornado climatology at a site, from NOAA Storm Prediction Center reports (United States only).

What it is: the number of days a year on which a report of hail of 1 inch (2.5 cm) or more, of hail of 2 inches or more, or of a tornado
fell within 40 km of the site, 1996-2024. "Days with a report within 25 miles" is the standard measure in the severe-weather climatology
literature (Brooks et al. 2003; Gensini and Ashley 2011), and it avoids counting one storm's many reports as many events.

What it is not: a physical model, a damage or loss estimate, or a hail-size or tornado-intensity forecast. Reports come from people and
spotters, so thinly populated land is under-counted (the data show reports, not hail). Tested by holding out 2010-2024
(docs/CONVECTIVE_MARINE_BENCHMARK.md). Outside the United States it says it is not covered and never returns zero.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DATA = Path(__file__).resolve().parent / "data"
EARTH_KM = 6371.0088
# Screening bands in days a year with a report within 40 km. Judgement set from the spread over random US land points; not calibrated to loss.
HAIL_BANDS = ((0.3, "Low"), (1.0, "Moderate"), (2.5, "High"), (1e9, "Very high"))
TORNADO_BANDS = ((0.1, "Low"), (0.3, "Moderate"), (0.6, "High"), (1e9, "Very high"))   # on days with an EF1 or stronger tornado: EF0 reports track how many people are watching


def _xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, dtype=float)), np.radians(np.asarray(lon, dtype=float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


@lru_cache(maxsize=1)
def _store() -> dict:
    p = _DATA / "spc_reports_1996_2024.npz"
    if not p.exists():
        raise RuntimeError("SPC report table is not installed (data/spc_reports_1996_2024.npz)")
    from scipy.spatial import cKDTree
    z = np.load(p)
    out = {"y0": int(z["y0"]), "y1": int(z["y1"])}
    for k in ("hail", "tor"):
        out[k] = {"day": z[f"{k}_day"], "tree": cKDTree(_xyz(z[f"{k}_lat"], z[f"{k}_lon"])),
                  "val": z["hail_in"] if k == "hail" else z["tor_mag"]}
    return out


def in_united_states(lat: float, lon: float, margin_deg: float = 0.3) -> bool:
    """True only if the site is in the United States and no other country lies within about 33 km (0.3 degree) of it. Inside that margin a 40 km search
    circle would take in land the source does not cover (Canada, Mexico), which would read as fewer storms than there are."""
    try:
        from .finance.geo import admin_at
        a = admin_at(lat, lon)
        if not (a.get("found") and a.get("iso3") == "USA"):
            return False
        for dla, dlo in ((margin_deg, 0), (-margin_deg, 0), (0, margin_deg), (0, -margin_deg)):
            b = admin_at(lat + dla, lon + dlo)
            if b.get("found") and b.get("iso3") != "USA":
                return False
        return True
    except Exception:                                   # noqa: BLE001
        return False


def days_with_report(kind: str, lat: float, lon: float, radius_km: float = 40.0, y0: Optional[int] = None, y1: Optional[int] = None,
                     min_val: float = -1e9) -> dict:
    """Days per year with at least one report within radius_km. y0/y1 restrict the years (used by the hold-out test)."""
    s = _store()
    d = s[kind]
    a, b = y0 or s["y0"], y1 or s["y1"]
    lo = np.datetime64(f"{a}-01-01").astype("datetime64[D]").astype(int)
    hi = np.datetime64(f"{b}-12-31").astype("datetime64[D]").astype(int)
    idx = np.asarray(d["tree"].query_ball_point(_xyz(lat, lon), 2.0 * np.sin(radius_km / EARTH_KM / 2.0)), dtype=int)
    if idx.size:
        keep = (d["day"][idx] >= lo) & (d["day"][idx] <= hi) & (d["val"][idx] >= min_val)
        idx = idx[keep]
    days = np.unique(d["day"][idx]).size if idx.size else 0
    return {"days_per_year": days / float(b - a + 1), "days": int(days), "reports": int(idx.size), "years": [a, b],
            "max_value": float(d["val"][idx].max()) if idx.size else None}


def _band(x: float, bands) -> str:
    for edge, name in bands:
        if x < edge:
            return name
    return bands[-1][1]


NOT_COVERED = {"covered": False, "reason": "outside the United States, or within about 35 km of its border: the source (NOAA Storm Prediction Center) records US reports only, so absence of reports there says nothing about the hazard"}
SOURCE = "NOAA Storm Prediction Center public storm reports, 1996-2024 (public domain)"


def severe_hail(lat: float, lon: float, radius_km: float = 40.0) -> dict:
    if not in_united_states(lat, lon):
        return dict(NOT_COVERED)
    h1 = days_with_report("hail", lat, lon, radius_km, min_val=1.0)
    h2 = days_with_report("hail", lat, lon, radius_km, min_val=2.0)
    return {"covered": True, "hail_days_per_year": round(h1["days_per_year"], 2), "hail_2in_days_per_year": round(h2["days_per_year"], 2),
            "largest_reported_in": h1["max_value"], "reports": h1["reports"], "radius_km": radius_km, "years": h1["years"],
            "band": _band(h1["days_per_year"], HAIL_BANDS), "source": SOURCE,
            "basis": (f"Days a year with a report of hail of 1 inch or more within {radius_km:.0f} km. A report climatology, not a physical model: "
                      "thinly populated land is under-counted, reported sizes are estimates, and nothing is said about damage or about hail in future climates.")}


def tornado(lat: float, lon: float, radius_km: float = 40.0) -> dict:
    if not in_united_states(lat, lon):
        return dict(NOT_COVERED)
    t = days_with_report("tor", lat, lon, radius_km)
    t1 = days_with_report("tor", lat, lon, radius_km, min_val=1)
    t2 = days_with_report("tor", lat, lon, radius_km, min_val=2)
    return {"covered": True, "tornado_days_per_year": round(t["days_per_year"], 3), "ef1_plus_days_per_year": round(t1["days_per_year"], 3),
            "ef2_plus_days_per_year": round(t2["days_per_year"], 3), "strongest_rating": None if t["max_value"] is None else int(t["max_value"]),
            "tornadoes": t["reports"], "radius_km": radius_km, "years": t["years"], "band": _band(t1["days_per_year"], TORNADO_BANDS), "source": SOURCE,
            "basis": (f"Days a year with a tornado touching down within {radius_km:.0f} km (start point), 1996-2024, from the final F/EF-rated tornado list. "
                      "The band follows days with an EF1 or stronger tornado; weak (EF0) tornadoes and waterspouts moving ashore depend heavily on how many people are watching. A report climatology: it gives no probability of a tornado hitting a given building.")}
