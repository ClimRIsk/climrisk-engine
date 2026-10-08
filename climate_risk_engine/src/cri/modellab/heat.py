"""Heat-stress physics: vapour pressure, wet-bulb temperature (exact and Stull 2011), simplified WBGT."""
from __future__ import annotations

import math

import numpy as np


def sat_vapour_pressure_hpa(t_c):
    """Magnus form (Alduchov & Eskridge 1996): e_s = 6.1094 exp(17.625 T / (T + 243.04)) hPa."""
    t = np.asarray(t_c, float)
    return 6.1094 * np.exp(17.625 * t / (t + 243.04))


def vapour_pressure_hpa(t_c, rh_pct):
    return np.asarray(rh_pct, float) / 100.0 * sat_vapour_pressure_hpa(t_c)


def wet_bulb_exact(t_c: float, rh_pct: float, p_hpa: float = 1013.25) -> float:
    """Solve the psychrometric equation e = e_s(Tw) − γ P (T − Tw), γ = 6.6e-4 (1 + 1.15e-3 Tw), by bisection."""
    e = float(vapour_pressure_hpa(t_c, rh_pct))
    lo, hi = t_c - 40.0, t_c
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = float(sat_vapour_pressure_hpa(mid)) - 6.6e-4 * (1 + 1.15e-3 * mid) * p_hpa * (t_c - mid) - e
        if f > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def wet_bulb_stull(t_c, rh_pct):
    """Stull (2011) closed-form wet-bulb; stated error < 1 °C for RH 5–99 % and T −20…50 °C."""
    t, rh = np.asarray(t_c, float), np.asarray(rh_pct, float)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(t + rh) - np.arctan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def wbgt_shade(t_c, rh_pct):
    """Australian Bureau of Meteorology simplified WBGT (shade, light wind): 0.567 T + 0.393 e + 3.94, e in hPa."""
    t = np.asarray(t_c, float)
    return 0.567 * t + 0.393 * vapour_pressure_hpa(t, rh_pct) + 3.94


def bench_saturation_and_wbgt() -> dict:
    e20 = float(sat_vapour_pressure_hpa(20.0))                      # 23.37 hPa, textbook
    w = float(wbgt_shade(35.0, 50.0))
    ok = abs(e20 - 23.37) < 0.05 and abs(w - 34.8) < 0.3
    return {"name": "Saturation vapour pressure and WBGT hand values", "metric": f"e_s(20 °C) = {e20:.2f} hPa (23.37), WBGT(35 °C, 50 %) = {w:.1f} (34.8)", "value": abs(e20 - 23.37) + abs(w - 34.8), "limit": 0.35, "ok": bool(ok),
            "note": "Textbook values reproduced."}


def bench_wet_bulb_limits() -> dict:
    """Physical limits: at 100 % RH the wet-bulb equals the air temperature; Stull's formula stays within 1 °C of the exact solution over its stated range."""
    sat = abs(wet_bulb_exact(30.0, 100.0) - 30.0)
    worst = 0.0
    for t in range(5, 46, 5):
        for rh in (10, 30, 50, 70, 90):
            worst = max(worst, abs(float(wet_bulb_stull(t, rh)) - wet_bulb_exact(float(t), float(rh))))
    ok = sat < 1e-6 and worst < 1.0
    return {"name": "Wet-bulb: equals air temperature at saturation; Stull within 1 °C of exact", "metric": f"saturation error {sat:.1e} °C, worst Stull error {worst:.2f} °C", "value": max(sat, worst), "limit": 1.0, "ok": bool(ok),
            "note": "The closed-form approximation is checked against the full psychrometric equation."}


def run_benchmarks() -> list:
    return [bench_saturation_and_wbgt(), bench_wet_bulb_limits()]
