"""Earthquake hazard: Gutenberg–Richter occurrence from the USGS catalogue, declustering, a ground-motion relation, and a hindcast test."""
from __future__ import annotations

import csv
import io
import math
import urllib.parse
import urllib.request
from typing import Callable, Optional

import numpy as np

from . import data as D

LOG10E = math.log10(math.e)


def fetch_catalog(lat: float, lon: float, radius_km: float = 300.0, y0: int = 1973, y1: int = 2025, minmag: float = 4.0, fetch: Optional[Callable] = None) -> dict:
    """USGS ComCat events (FDSN web service), fetched decade by decade to stay under the 20 000-event cap."""
    rows = []
    for a in range(y0, y1 + 1, 10):
        b = min(a + 9, y1)
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode({"format": "csv", "starttime": f"{a}-01-01", "endtime": f"{b + 1}-01-01", "latitude": round(lat, 3), "longitude": round(lon, 3),
                                                                                           "maxradiuskm": radius_km, "minmagnitude": minmag, "orderby": "time-asc", "limit": 20000})
        p = D._cache_path(url)
        txt = None
        if p.exists():
            try:
                txt = p.read_text()
            except Exception:                                          # noqa: BLE001
                txt = None
        if txt is None:
            if fetch:
                txt = fetch(url)
            else:
                txt = urllib.request.urlopen(urllib.request.Request(url, headers=D._UA), timeout=90).read().decode()
            try:
                p.write_text(txt)
            except Exception:                                          # noqa: BLE001
                pass
        for r in csv.DictReader(io.StringIO(txt)):
            try:
                rows.append((r["time"], float(r["mag"]), float(r["latitude"]), float(r["longitude"]), float(r["depth"] or 10.0)))
            except (ValueError, KeyError, TypeError):
                continue
    import datetime as dt
    yr = np.array([int(x[0][:4]) + (dt.datetime.fromisoformat(x[0][:19]).timetuple().tm_yday - 1) / 365.25 for x in rows])
    return {"t": yr, "mag": np.array([x[1] for x in rows]), "lat": np.array([x[2] for x in rows]), "lon": np.array([x[3] for x in rows]), "depth": np.array([x[4] for x in rows]),
            "source": "USGS ComCat earthquake catalogue (FDSN web service)"}


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(np.asarray(lon2) - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def mc_max_curvature(mag, dm: float = 0.1) -> float:
    """Magnitude of completeness by maximum curvature (+0.2 correction, Woessner & Wiemer 2005)."""
    bins = np.arange(np.floor(mag.min() * 10) / 10, mag.max() + dm, dm)
    h, edges = np.histogram(mag, bins=bins)
    return float(edges[int(np.argmax(h))] + 0.2)


def b_value(mag, mc: float, dm: float = 0.1) -> tuple:
    """Aki–Utsu maximum likelihood with binning correction; returns (b, standard error)."""
    m = mag[mag >= mc]
    n = len(m)
    b = LOG10E / (np.mean(m) - (mc - dm / 2))
    return float(b), float(b / math.sqrt(max(n, 1)))


def decluster_gk(t, mag, lat, lon):
    """Gardner & Knopoff (1974) windows (Uhrhammer's fit): remove events inside the time-distance window of a larger one."""
    order = np.argsort(-mag)
    keep = np.ones(len(t), bool)
    for i in order:
        if not keep[i]:
            continue
        dist_km = 10 ** (0.1238 * mag[i] + 0.983)
        days = 10 ** (0.032 * mag[i] + 2.7389) if mag[i] >= 6.5 else 10 ** (0.5409 * mag[i] - 0.547)
        near = (np.abs(t - t[i]) * 365.25 <= days) & (haversine(lat[i], lon[i], lat, lon) <= dist_km)
        near[i] = False
        keep &= ~(near & (mag <= mag[i]))
    return keep


def jb81_pga_g(M, d_km, sigma: bool = False):
    """Joyner & Boore (1981) peak ground acceleration (g), r = √(d² + 7.3²). Median unless sigma=True (log10 σ = 0.26)."""
    r = np.sqrt(np.asarray(d_km, float) ** 2 + 7.3 ** 2)
    lg = -1.02 + 0.249 * np.asarray(M, float) - np.log10(r) - 0.00255 * r
    return (10 ** lg, 0.26) if sigma else 10 ** lg


def exceed_prob(M, d_km, x_g):
    """P(PGA > x | M, d) with lognormal scatter (σ_log10 = 0.26)."""
    from scipy.stats import norm
    med, sg = jb81_pga_g(M, d_km, sigma=True)
    return norm.sf((np.log10(x_g) - np.log10(med)) / sg)


def hazard_curve(t_years: float, mag, dist_km, xs):
    """Empirical annual rate of PGA > x at the site: every catalogued event contributes its exceedance probability."""
    return np.array([float(np.sum(exceed_prob(mag, dist_km, x))) / t_years for x in xs])


def pga_return_level(rate_fn_xs, rates, T: float) -> float:
    """PGA whose annual exceedance rate is 1/T (log-log interpolation)."""
    xs, r = np.asarray(rate_fn_xs), np.asarray(rates)
    ok = r > 0
    if ok.sum() < 2 or 1.0 / T > r[ok].max() or 1.0 / T < r[ok].min():
        return float("nan")
    return float(np.exp(np.interp(math.log(1.0 / T), np.log(r[ok])[::-1], np.log(xs[ok])[::-1])))


# ── verification ─────────────────────────────────────────────────────────────

def _synthetic_gr(n: int, b: float, mc: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = mc - 0.05 + rng.exponential(1.0 / (b * math.log(10)), n)
    return np.round(m * 10) / 10                                         # catalogue magnitudes come in 0.1 bins


def bench_b_value_recovery() -> dict:
    m = _synthetic_gr(8000, 1.0, 4.5)
    b, se = b_value(m, 4.5)
    err = abs(b - 1.0)
    return {"name": "b-value recovered from a synthetic Gutenberg–Richter catalogue", "metric": f"b = {b:.3f} ± {se:.3f} (true 1.000)", "value": err, "limit": 0.06, "ok": err < 0.06, "note": "Aki–Utsu maximum likelihood on 8 000 binned magnitudes."}


def bench_completeness() -> dict:
    rng = np.random.default_rng(4)
    full = _synthetic_gr(20000, 1.0, 3.5, 4)
    # detection ramps from 0 at M4.0 to 1 at M4.5: the catalogue is complete from 4.5 up
    p = np.clip((full - 4.0) / 0.5, 0, 1)
    kept = full[rng.random(len(full)) < p]
    mc = mc_max_curvature(kept)
    ok = 4.4 <= mc <= 4.8
    return {"name": "Magnitude of completeness found in an incomplete catalogue", "metric": f"Mc = {mc:.1f} (detection is complete from 4.5; accepted 4.4–4.8)", "value": abs(mc - 4.6), "limit": 0.2, "ok": bool(ok),
            "note": "Maximum curvature + 0.2 (Woessner & Wiemer 2005) lands just above the completeness magnitude, as designed."}


def bench_ground_motion() -> dict:
    g = float(jb81_pga_g(6.5, 10.0))
    m_up = float(jb81_pga_g(7.5, 20.0) / jb81_pga_g(6.5, 20.0))
    far = float(jb81_pga_g(6.5, 100.0)) < float(jb81_pga_g(6.5, 10.0)) * 0.3
    ok = abs(g - 0.298) < 0.01 and abs(m_up - 10 ** 0.249) < 1e-9 and far
    return {"name": "Ground-motion relation: hand value and scaling", "metric": f"PGA(M 6.5, 10 km) = {g:.3f} g (hand 0.298); one magnitude unit multiplies by {m_up:.2f} (10^0.249 = 1.77)", "value": abs(g - 0.298), "limit": 0.01, "ok": bool(ok),
            "note": "Joyner–Boore (1981) evaluated by hand; stronger events and nearer sites shake harder."}


def bench_poisson_test_calibrated() -> dict:
    from scipy import stats
    ok = 0
    n = 200
    for s in range(n):
        r = np.random.default_rng(900 + s)
        tr, te = r.poisson(6.0, 30).sum(), r.poisson(6.0, 15).sum()
        lam = tr / 30 * 15
        p = float(stats.poisson.sf(te - 1, lam)) if te > lam else float(stats.poisson.cdf(te, lam))
        ok += int(2 * min(p, 0.5) > 0.05)
    rate = ok / n
    return {"name": "Poisson count test is calibrated", "metric": "fraction of true-Poisson records passing (expect ≈ 0.95)", "value": rate, "limit": 0.88, "ok": rate >= 0.88, "note": "The same test is used for the hindcast."}


def run_benchmarks() -> list:
    return [bench_b_value_recovery(), bench_completeness(), bench_ground_motion(), bench_poisson_test_calibrated()]
