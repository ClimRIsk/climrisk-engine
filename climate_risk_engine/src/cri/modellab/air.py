"""
Air-quality physics: Gaussian plume, a well-mixed urban box model, and the AQI.

Gaussian plume (steady, ground reflection, no deposition):
    C(x,y,z) = Q / (2π u σy σz) · exp(−y²/2σy²) · [exp(−(z−H)²/2σz²) + exp(−(z+H)²/2σz²)]
  σy, σz: Briggs (1973) urban dispersion curves by Pasquill–Gifford stability class.
Box model (city of length L, area A, mixing height H, wind u, background Cb):
    dC/dt = E/H − (u/L)(C − Cb) − (vd/H) C − k C
  Solved exactly for piecewise-constant hourly forcing, so there is no numerical error to argue about.
AQI: US EPA PM2.5 breakpoints (2024 revision).
"""
from __future__ import annotations

import math

import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

# ── dispersion ───────────────────────────────────────────────────────────────

_BRIGGS_URBAN = {            # σy = a x (1+b x)^-0.5 ; σz = c x (1+d x)^e
    "A": (0.32, 0.0004, 0.24, 0.001, 0.5), "B": (0.32, 0.0004, 0.24, 0.001, 0.5),
    "C": (0.22, 0.0004, 0.20, 0.0, 0.0), "D": (0.16, 0.0004, 0.14, 0.0003, -0.5),
    "E": (0.11, 0.0004, 0.08, 0.0015, -0.5), "F": (0.11, 0.0004, 0.08, 0.0015, -0.5),
}


def stability_class(wind_ms: float, daytime: bool, insolation: str = "moderate", cloud_frac: float = 0.5) -> str:
    """Pasquill stability class from wind speed and sun/cloud (Turner 1970 table, simplified to the standard lookups)."""
    u = wind_ms
    if daytime:
        row = {"strong": ["A", "A-B", "B", "C", "C"], "moderate": ["A-B", "B", "B-C", "C", "D"], "slight": ["B", "C", "C", "D", "D"]}[insolation]
        c = row[0 if u < 2 else 1 if u < 3 else 2 if u < 5 else 3 if u < 6 else 4]
    else:
        if cloud_frac >= 0.5:
            c = "E" if u < 3 else "D"
            c = "E" if u < 2 else c
        else:
            c = "F" if u < 2 else "E" if u < 3 else "D"
    return {"A-B": "B", "B-C": "C"}.get(c, c)


def sigmas(x: np.ndarray, cls: str) -> tuple:
    a, b, c, d, e = _BRIGGS_URBAN[cls]
    x = np.maximum(np.asarray(x, float), 1.0)
    return a * x * (1 + b * x) ** -0.5, c * x * (1 + d * x) ** e


def plume(x, y, z, Q_g_s: float, u_ms: float, H_m: float, cls: str = "D"):
    """Concentration in g/m³ at downwind distance x, crosswind y, height z."""
    sy, sz = sigmas(x, cls)
    x = np.asarray(x, float)
    c = Q_g_s / (2 * math.pi * u_ms * sy * sz) * np.exp(-np.asarray(y, float) ** 2 / (2 * sy ** 2)) * (
        np.exp(-(z - H_m) ** 2 / (2 * sz ** 2)) + np.exp(-(z + H_m) ** 2 / (2 * sz ** 2)))
    return np.where(x > 0, c, 0.0)


# ── box model ────────────────────────────────────────────────────────────────

def box_steady_state(E: float, H: float, u: float, L: float, Cb: float, vd: float = 0.0, k: float = 0.0) -> float:
    """C* = (E/H + (u/L) Cb) / (u/L + vd/H + k). E in µg m⁻² s⁻¹, H m, u m s⁻¹, L m, C in µg m⁻³."""
    lam = u / L + vd / H + k
    return (E / H + (u / L) * Cb) / lam


def box_series(E, H, u, L: float, Cb, vd: float = 0.002, k: float = 0.0, c0=None, dt_s: float = 3600.0) -> np.ndarray:
    """Exact piecewise-constant solution. E, H, u, Cb are arrays (one value per step); returns C at the END of each step."""
    E, H, u = (np.asarray(v, float) for v in (E, H, u))
    Cb = np.broadcast_to(np.asarray(Cb, float), E.shape)
    n = E.size
    c = np.empty(n)
    cur = (float(Cb[0]) if c0 is None else float(c0))
    for i in range(n):
        Hi = max(H[i], 50.0)
        lam = max(u[i], 0.3) / L + vd / Hi + k
        cs = (E[i] / Hi + max(u[i], 0.3) / L * Cb[i]) / lam
        cur = cs + (cur - cs) * math.exp(-lam * dt_s)
        c[i] = cur
    return c


# ── AQI ──────────────────────────────────────────────────────────────────────

_PM25_BP = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150), (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)]
_CAT = [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for sensitive groups"), (200, "Unhealthy"), (300, "Very unhealthy"), (500, "Hazardous")]


def aqi_pm25(c: float) -> int:
    """US EPA AQI for a 24-h PM2.5 concentration (µg m⁻³, truncated to 0.1)."""
    c = math.floor(max(c, 0.0) * 10) / 10
    for lo, hi, ilo, ihi in _PM25_BP:
        if c <= hi:
            return int(round((ihi - ilo) / (hi - lo) * (c - lo) + ilo))
    return 500


def aqi_category(aqi: float) -> str:
    return next((n for lim, n in _CAT if aqi <= lim), "Hazardous")


# ── verification benchmarks ──────────────────────────────────────────────────

def bench_plume_mass_flux() -> dict:
    """With no deposition the plume carries all the emitted mass: ∬ C u dy dz at any distance equals Q."""
    Q, u, H = 100.0, 4.0, 30.0
    worst = 0.0
    for x in (500.0, 2000.0, 8000.0):
        sy, sz = sigmas(np.array([x]), "D")
        y = np.linspace(-8 * sy[0], 8 * sy[0], 801)
        z = np.linspace(0.0, 12 * sz[0] + H, 1601)
        Y, Z = np.meshgrid(y, z)
        C = plume(np.full_like(Y, x), Y, Z, Q, u, H, "D")
        flux = _trapz(_trapz(C * u, y, axis=1), z)
        worst = max(worst, abs(flux - Q) / Q)
    return {"name": "Gaussian plume conserves emitted mass", "metric": "max relative flux error at 0.5, 2, 8 km", "value": worst, "limit": 0.02, "ok": worst < 0.02,
            "note": "Integrating concentration × wind across the plume returns the emission rate."}


def bench_plume_centreline() -> dict:
    """Ground-level centreline of a ground source (H=0): C = Q/(π u σy σz)."""
    Q, u, x = 50.0, 3.0, 1000.0
    sy, sz = sigmas(np.array([x]), "C")
    exact = Q / (math.pi * u * sy[0] * sz[0])
    got = float(plume(np.array([x]), np.array([0.0]), 0.0, Q, u, 0.0, "C")[0])
    rel = abs(got - exact) / exact
    return {"name": "Plume centreline matches the closed form", "metric": "relative error", "value": rel, "limit": 1e-9, "ok": rel < 1e-9, "note": "Q / (π u σy σz) for a ground-level source."}


def bench_box_steady_and_decay() -> dict:
    """Constant forcing converges to C*; with the source switched off the excess decays as exp(−λt)."""
    E, H, u, L, Cb, vd = 0.05, 800.0, 2.0, 20000.0, 10.0, 0.002
    cs = box_steady_state(E, H, u, L, Cb, vd)
    n = 400
    c = box_series(np.full(n, E), np.full(n, H), np.full(n, u), L, Cb, vd, c0=Cb)
    e1 = abs(c[-1] - cs) / cs
    lam = u / L + vd / H
    c2 = box_series(np.zeros(60), np.full(60, H), np.full(60, u), L, Cb, vd, c0=cs)
    pred = Cb + (cs - Cb) * math.exp(-lam * 3600 * 60) * 1.0
    # with E=0 the new steady state is Cb·(u/L)/(u/L+vd/H); compare decay to that
    cs0 = box_steady_state(0.0, H, u, L, Cb, vd)
    pred = cs0 + (cs - cs0) * math.exp(-lam * 3600 * 60)
    e2 = abs(c2[-1] - pred) / pred
    worst = max(e1, e2)
    return {"name": "Box model: steady state and exponential decay", "metric": "worst relative error vs analytic", "value": worst, "limit": 1e-6, "ok": worst < 1e-6, "note": "C* = (E/H + u Cb/L)/(u/L + vd/H); the excess decays at the ventilation rate."}


def bench_aqi_breakpoints() -> dict:
    cases = {0.0: 0, 9.0: 50, 12.0: 56, 35.4: 100, 55.4: 150, 125.4: 200, 225.4: 300}
    bad = {k: (aqi_pm25(k), v) for k, v in cases.items() if aqi_pm25(k) != v}
    return {"name": "AQI reproduces the EPA breakpoints", "metric": "mismatching breakpoints", "value": float(len(bad)), "limit": 0.0, "ok": not bad, "note": "PM2.5 0, 9.0, 12.0, 35.4, 55.4, 125.4, 225.4 µg/m³ → 0, 50, 56, 100, 150, 200, 300."}


def run_benchmarks() -> list:
    return [bench_plume_centreline(), bench_plume_mass_flux(), bench_box_steady_and_decay(), bench_aqi_breakpoints()]
