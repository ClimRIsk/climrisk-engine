"""
climate/tropical_cyclone.py — Global tropical-cyclone wind hazard from observed tracks.

Model
─────
1. Tracks: NOAA IBTrACS v04r01 since 1980, all basins (≈4,650 storms, compact
   copy in cri/data/ibtracs_since1980.npz; rebuild with
   scripts/build_ibtracs_compact.py).
2. Wind field: Holland (1980) radial profile driven by best-track maximum
   sustained wind, with
     • shape parameter B from Willoughby & Rahn (2004)
     • radius of maximum wind from IBTrACS USA_RMW, else Willoughby et al.
       (2006) Rmax(Vmax, latitude)
     • optional forward-motion asymmetry (ASYM; off — it did not improve the
       validation against station measurements)
   Tracks are interpolated to 30-minute steps so the closest approach is not
   missed between 3/6-hourly fixes.
3. Hazard at a site: modelled peak wind from every storm since 1980 →
   event rate above thresholds, return-period winds (exponential tail fitted
   to peaks ≥ 17.5 m/s, i.e. tropical-storm force), worst historical storm.

Units: winds are 1-minute sustained at 10 m over open terrain (m/s), matching
best-track convention; multiply by 0.88 for 10-minute mean (WMO guidance).

Validation against measured station winds (scripts/validate_tc_wind.py →
cri/data/tc_validation.json): 26 storms in 5 basins, 148 station peaks.
Pattern: r = 0.53 overall (0.69 North Atlantic). Level: raw open-terrain winds
run ~1.75× measured station peaks; a single factor k = 0.57 — the product
expected from land roughness (~0.75), station exposure (~0.85) and hourly
sampling (~0.9) — removes the bias (held-out median ratio 1.01, 63 % within
±30 %). Outputs are reported as open-terrain 1-min winds (engineering
convention) plus the "typical weather-station" equivalent (× k).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_DATA = Path(__file__).resolve().parent.parent / "data"
KT = 0.514444                     # m/s per knot
NMI = 1.852                       # km per nautical mile
R_EARTH = 6371.0
YEAR0, YEAR1 = 1980, 2025
ASYM = 0.0   # forward-motion asymmetry: validation showed no gain (r 0.55 vs 0.54), so off by default
SEARCH_KM = 600.0

SAFFIR = [(17.5, "tropical storm"), (33.0, "Cat 1"), (43.0, "Cat 2"), (50.0, "Cat 3"),
          (58.0, "Cat 4"), (70.0, "Cat 5")]


@lru_cache(maxsize=1)
def tracks() -> dict:
    z = np.load(_DATA / "ibtracs_since1980.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["years"] = int(d["season"].max()) - YEAR0 + 1
    return d


@lru_cache(maxsize=1)
def validation() -> dict:
    try:
        return json.loads((_DATA / "tc_validation.json").read_text())
    except Exception:                 # noqa: BLE001
        return {}


# ── geometry ────────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Bearing from point 1 to point 2 (deg clockwise from north)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


# ── wind profile ────────────────────────────────────────────────────────────

def rmax_km(vmax_ms, lat):
    """Willoughby et al. (2006) eq. 7a."""
    return 46.4 * np.exp(-0.0155 * vmax_ms + 0.0169 * np.abs(lat))


def holland_b(vmax_ms, rmax, lat):
    """Willoughby & Rahn (2004) regression, bounded to the physical range."""
    b = 1.0036 + 0.0173 * vmax_ms - 0.0313 * np.log(np.maximum(rmax, 1.0)) + 0.0087 * np.abs(lat)
    return np.clip(b, 1.0, 2.5)


def holland_wind(r_km, vmax_ms, rmax, b):
    """Symmetric surface wind at distance r (Holland 1980, cyclostrophic form)."""
    x = (rmax / np.maximum(r_km, 0.5)) ** b
    return vmax_ms * np.sqrt(x * np.exp(1.0 - x))


def _interp_storm(ix: np.ndarray, T: dict, step_h: float = 0.5) -> dict:
    t = T["t"][ix].astype(np.float64)
    order = np.argsort(t)
    t = t[order]
    if len(t) < 2:
        return {k: T[k][ix][order].astype(np.float64) for k in ("lat", "lon", "vmax", "rmw", "spd", "dir")} | {"t": t}
    tt = np.arange(t[0], t[-1] + 1e-6, step_h)
    out = {"t": tt}
    lon = T["lon"][ix][order].astype(np.float64)
    lon = np.degrees(np.unwrap(np.radians(lon)))          # dateline-safe
    for k, v in (("lat", T["lat"][ix][order]), ("lon", lon), ("vmax", T["vmax"][ix][order]),
                 ("rmw", T["rmw"][ix][order]), ("spd", T["spd"][ix][order]), ("dir", T["dir"][ix][order])):
        v = np.asarray(v, np.float64)
        ok = np.isfinite(v)
        out[k] = np.interp(tt, t[ok], v[ok]) if ok.sum() >= 2 else np.full(len(tt), np.nan)
    out["lon"] = (out["lon"] + 180) % 360 - 180
    return out


def storm_peak_at(lat: float, lon: float, storm_idx: np.ndarray, T: dict, asym: float = ASYM) -> tuple[float, float, float]:
    """Peak modelled 1-min wind (m/s) at the site from one storm, closest distance (km), time (h)."""
    s = _interp_storm(storm_idx, T)
    r = _haversine_km(s["lat"], s["lon"], lat, lon)
    if not np.isfinite(r).any():
        return 0.0, float("nan"), float("nan")
    v = s["vmax"] * KT
    rm = np.where(np.isfinite(s["rmw"]) & (s["rmw"] > 0), s["rmw"] * NMI, rmax_km(v, s["lat"]))
    b = holland_b(v, rm, s["lat"])
    w = holland_wind(r, v, rm, b)
    # forward-motion asymmetry
    if asym and np.isfinite(s["spd"]).any():
        vt = np.nan_to_num(s["spd"]) * KT
        brg = _bearing_deg(s["lat"], s["lon"], lat, lon)            # centre → site
        right = (s["dir"] + (90.0 if lat >= 0 else -90.0)) % 360    # strong side
        w = w + asym * vt * np.cos(np.radians(brg - np.nan_to_num(right)))
    w = np.clip(w, 0, None)
    i = int(np.nanargmax(w))
    return float(w[i]), float(np.nanmin(r)), float(s["t"][i])


# ── site hazard ─────────────────────────────────────────────────────────────

@dataclass
class TCHazard:
    lat: float
    lon: float
    years: int
    storms_within_200km: int = 0
    events_ge_ts: int = 0                   # peak ≥ 17.5 m/s (tropical-storm force)
    events_ge_cat1: int = 0                 # ≥ 33 m/s
    events_ge_cat3: int = 0                 # ≥ 50 m/s
    annual_rate_ge_ts: float = 0.0
    annual_rate_ge_cat1: float = 0.0
    rp_wind_ms: dict = field(default_factory=dict)       # {10: .., 50: .., 100: .., 250: ..}
    rp_category: dict = field(default_factory=dict)
    worst: Optional[dict] = None
    top_events: list = field(default_factory=list)
    station_equiv_factor: Optional[float] = None       # × open-terrain wind → typical station wind
    validation: dict = field(default_factory=dict)
    source: str = ""
    method: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def category(v: float) -> str:
    lab = "below tropical-storm force"
    for thr, name in SAFFIR:
        if v >= thr:
            lab = name
    return lab


def site_hazard(lat: float, lon: float, search_km: float = SEARCH_KM) -> TCHazard:
    T = tracks()
    years = T["years"]
    d = _haversine_km(T["lat"], T["lon"], lat, lon)
    near_storms = np.unique(T["storm"][d <= search_km])
    peaks = []
    for si in near_storms:
        ix = np.where(T["storm"] == si)[0]
        v, rmin, tpk = storm_peak_at(lat, lon, ix, T)
        if v >= 10.0:
            peaks.append((v, int(si), rmin))
    k = (validation().get("calibration") or {}).get("k_land_exposure")
    hz = TCHazard(lat=lat, lon=lon, years=years,
                  source=str(T["source"]),
                  method="Holland (1980) wind profile on IBTrACS best tracks, Willoughby B & Rmax; "
                         "1-min sustained 10 m open-terrain winds")
    hz.station_equiv_factor = k
    v = validation()
    if v:
        hz.validation = {"storms": v.get("n_storms"), "stations": (v.get("model_calibrated_heldout") or {}).get("n"),
                         "pattern_r": (v.get("model_symmetric") or {}).get("pearson_r"),
                         "heldout_within_30pct": (v.get("model_calibrated_heldout") or {}).get("within_30pct"),
                         "heldout_median_ratio": (v.get("model_calibrated_heldout") or {}).get("median_ratio_model_over_obs")}
    hz.storms_within_200km = int(len(np.unique(T["storm"][d <= 200.0])))
    v_all = np.array([p[0] for p in peaks]) if peaks else np.array([])
    hz.events_ge_ts = int((v_all >= 17.5).sum())
    hz.events_ge_cat1 = int((v_all >= 33.0).sum())
    hz.events_ge_cat3 = int((v_all >= 50.0).sum())
    hz.annual_rate_ge_ts = round(hz.events_ge_ts / years, 3)
    hz.annual_rate_ge_cat1 = round(hz.events_ge_cat1 / years, 3)

    # Return-period winds: exponential (GPD shape 0) tail above 17.5 m/s
    exc = v_all[v_all >= 17.5]
    if len(exc) >= 3:
        lam = len(exc) / years
        sigma = max(float(np.mean(exc - 17.5)), 1.0)
        for T_rp in (10, 50, 100, 250):
            v_T = 17.5 + sigma * math.log(lam * T_rp) if lam * T_rp > 1 else 0.0
            v_T = float(np.clip(v_T, 0, 95.0))
            hz.rp_wind_ms[T_rp] = round(v_T, 1)
            hz.rp_category[T_rp] = category(v_T)
    elif len(exc):
        # too few events for a tail fit: report empirical only
        for T_rp in (10, 50, 100, 250):
            v_T = float(np.max(exc)) if T_rp >= years else 0.0
            hz.rp_wind_ms[T_rp] = round(v_T, 1)
            hz.rp_category[T_rp] = category(v_T)
    else:
        for T_rp in (10, 50, 100, 250):
            hz.rp_wind_ms[T_rp] = 0.0
            hz.rp_category[T_rp] = category(0.0)

    if peaks:
        peaks.sort(reverse=True)
        def ev(p):
            si = p[1]
            return {"name": str(T["name"][si]).title(), "season": int(T["season"][si]),
                    "basin": str(T["basins"][T["basin"][si]]), "peak_wind_ms": round(p[0], 1),
                    "category_at_site": category(p[0]), "closest_km": round(p[2], 0)}
        hz.worst = ev(peaks[0])
        hz.top_events = [ev(p) for p in peaks[:5]]
    return hz
