"""Surface-fire behaviour (rate of spread, flame length, intensity) from fuel, terrain slope and weather: Rothermel (1972) with Anderson (1982) fuel models.

This is the standard US fire-behaviour calculation (the engine of BEHAVE / BehavePlus / FARSITE), implemented in a form that needs only what we can read anywhere:
  fuel model  from ESA WorldCover (grass -> FM1, shrub -> FM5, tree -> FM10, cropland -> FM1), a coarse proxy for real fuel mapping (LANDFIRE-class data are US-only)
  slope       from the terrain grid
  weather     ERA5 daily T max, RH min, wind max -> equilibrium moisture of the fine dead fuel (Simard 1968) and midflame wind
Not modelled: live-fuel moisture (fixed 120 %), crown fire, spotting, fuel breaks and buildings, suppression, ignition. Wind is ERA5 10 m daily maximum scaled to midflame
height by a fuel-type factor. Daily ERA5 extremes overstate the afternoon average, so ROS in dry windy climates is on the high side.
Checked against the Canadian National Fire Database in `scripts/fire_behaviour_benchmark.py` / `docs/FIRE_BEHAVIOUR_BENCHMARK.md`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

RHO_P, H_BTU, ST, SE = 32.0, 8000.0, 0.0555, 0.010


@dataclass(frozen=True)
class FuelModel:
    name: str
    loads: tuple          # (1 h, 10 h, 100 h, live herb/woody) oven-dry load, lb/ft2
    sigmas: tuple         # surface-area-to-volume, 1/ft
    depth_ft: float
    mx: float             # dead-fuel moisture of extinction, fraction
    wind_factor: float    # midflame wind / 10 m wind


MODELS = {
    "FM1": FuelModel("short grass", (0.034, 0, 0, 0), (3500, 109, 30, 1500), 1.0, 0.12, 0.40),
    "FM2": FuelModel("timber grass and understory", (0.092, 0.046, 0.023, 0.023), (3000, 109, 30, 1500), 1.0, 0.15, 0.30),
    "FM5": FuelModel("brush", (0.046, 0.023, 0, 0.092), (2000, 109, 30, 1500), 2.0, 0.20, 0.40),
    "FM6": FuelModel("dormant brush", (0.069, 0.115, 0.092, 0), (1750, 109, 30, 1500), 2.5, 0.25, 0.40),
    "FM8": FuelModel("closed timber litter", (0.069, 0.046, 0.115, 0), (2000, 109, 30, 1500), 0.2, 0.30, 0.20),
    "FM10": FuelModel("timber with heavy litter and understory", (0.138, 0.092, 0.230, 0.092), (2000, 109, 30, 1500), 1.0, 0.25, 0.20),
}
WORLDCOVER_FUEL = {10: "FM10", 20: "FM5", 30: "FM1", 40: "FM1", 60: "FM1", 90: "FM1", 95: "FM1", 100: "FM1"}      # tree, shrub, grass, crop, bare, wetland, mangrove, moss


def emc(rh_pct: float, t_c: float) -> float:
    """Equilibrium moisture content of fine dead fuel (fraction), Simard (1968) as used in the Fosberg fire-weather index."""
    h, tf = float(np.clip(rh_pct, 1, 100)), t_c * 9 / 5 + 32
    if h < 10:
        e = 0.03229 + 0.281073 * h - 0.000578 * h * tf
    elif h < 50:
        e = 2.22749 + 0.160107 * h - 0.01478 * tf
    else:
        e = 21.0606 + 0.005565 * h * h - 0.00035 * h * tf - 0.483199 * h
    return float(max(0.02, e / 100.0))


def spread(model: str, m1: float, wind_midflame_kmh: float, slope_deg: float, live_moisture: float = 1.2) -> dict:
    """Head-fire rate of spread (m/min), flame length (m), Byram intensity (kW/m) for fuel model, dead 1-h moisture (fraction), midflame wind and slope."""
    fm = MODELS[model]
    m = {0: m1, 1: m1 + 0.01, 2: m1 + 0.02, 3: live_moisture}              # 10 h and 100 h fuels lag the 1 h fuel; offsets are a stated heuristic
    cats = {"dead": [0, 1, 2], "live": [3]}
    a = {i: fm.loads[i] * fm.sigmas[i] / RHO_P for i in range(4)}
    a_cat = {c: sum(a[i] for i in idx) for c, idx in cats.items()}
    a_tot = sum(a_cat.values())
    f_cat = {c: a_cat[c] / a_tot for c in cats}
    sigma = sum(f_cat[c] * sum(a[i] / a_cat[c] * fm.sigmas[i] for i in cats[c]) for c in cats if a_cat[c] > 0)
    w_tot = sum(fm.loads)
    beta = (w_tot / fm.depth_ft) / RHO_P
    beta_op = 3.348 * sigma ** -0.8189
    ratio = beta / beta_op
    gmax = sigma ** 1.5 / (495 + 0.0594 * sigma ** 1.5)
    A = 133 * sigma ** -0.7913
    gamma = gmax * ratio ** A * math.exp(A * (1 - ratio))
    eta_s = min(1.0, 0.174 * SE ** -0.19)
    # dead-fuel weighted moisture and the live moisture of extinction (Rothermel 1972, eq. 88)
    fine = lambda i: fm.loads[i] * math.exp(-138.0 / fm.sigmas[i])                                  # noqa: E731
    w_fine_dead = sum(fine(i) for i in cats["dead"])
    m_fine_dead = sum(m[i] * fine(i) for i in cats["dead"]) / w_fine_dead if w_fine_dead > 0 else m1
    live_term = sum(fm.loads[3] * math.exp(-500.0 / fm.sigmas[3]) for _ in [0])
    mx_live = fm.mx
    if live_term > 0 and w_fine_dead > 0:
        mx_live = max(fm.mx, 2.9 * (w_fine_dead / live_term) * (1 - m_fine_dead / fm.mx) - 0.226)
    ir = 0.0
    for c in cats:
        if a_cat[c] <= 0:
            continue
        wn = sum(a[i] / a_cat[c] * fm.loads[i] * (1 - ST) for i in cats[c])
        mf = sum(a[i] / a_cat[c] * m[i] for i in cats[c])
        mx = fm.mx if c == "dead" else mx_live
        rm = min(1.0, mf / mx)
        eta_m = max(0.0, 1 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3)
        ir += gamma * wn * H_BTU * eta_m * eta_s                                # area-weighted net load per category; categories add
    xi = math.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / (192 + 0.2595 * sigma)
    u = max(0.0, wind_midflame_kmh) * 1000 / 60 * 3.2808                         # km/h -> ft/min
    C, B, E = 7.47 * math.exp(-0.133 * sigma ** 0.55), 0.02526 * sigma ** 0.54, 0.715 * math.exp(-3.59e-4 * sigma)
    phi_w = C * u ** B * ratio ** -E if u > 0 else 0.0
    phi_s = 5.275 * beta ** -0.3 * math.tan(math.radians(min(max(slope_deg, 0), 60))) ** 2
    ue_cap = 0.9 * ir
    phi_cap = C * ue_cap ** B * ratio ** -E
    phi = min(phi_w + phi_s, phi_cap)
    rho_b = w_tot / fm.depth_ft
    sink = 0.0
    for c in cats:
        if a_cat[c] <= 0:
            continue
        sink += f_cat[c] * sum(a[i] / a_cat[c] * math.exp(-138.0 / fm.sigmas[i]) * (250 + 1116 * m[i]) for i in cats[c])
    sink *= rho_b
    ros_ft_min = ir * xi * (1 + phi) / sink if sink > 0 else 0.0
    tau = 384.0 / sigma
    ib = ir * tau * ros_ft_min / 60.0                                              # Btu/ft/s
    flame_ft = 0.45 * ib ** 0.46 if ib > 0 else 0.0
    return {"ros_m_min": round(float(ros_ft_min * 0.3048), 3), "flame_length_m": round(float(flame_ft * 0.3048), 2), "intensity_kw_m": round(float(ib * 3.4614), 1)}


def flame_rating(flame_m: float) -> str:
    """US suppression-interpretation classes: <1.2 m hand crews hold the line; 1.2-2.4 m engines/dozers; 2.4-3.4 m control problems; >3.4 m crowning, spotting, uncontrollable."""
    return "NEGLIGIBLE" if flame_m < 0.3 else "LOW" if flame_m < 1.2 else "MODERATE" if flame_m < 2.4 else "HIGH" if flame_m < 3.4 else "CRITICAL"


def daily_spread(model: str, tmax_c, rh_min, wind_max_kmh, slope_deg: float = 0.0) -> np.ndarray:
    """Vector of daily head-fire rate of spread (m/min) from ERA5-type daily extremes."""
    fm = MODELS[model]
    return np.array([spread(model, emc(h, t), w * fm.wind_factor, slope_deg)["ros_m_min"] for t, h, w in zip(tmax_c, rh_min, wind_max_kmh)])


def moisture_from_ffmc(ffmc_value: float) -> float:
    """Fine dead fuel moisture (fraction) from the Canadian Fine Fuel Moisture Code (Van Wagner 1987: m = 147.2 (101 - F) / (59.5 + F) per cent)."""
    return float(max(0.02, 147.2 * (101.0 - min(ffmc_value, 100.0)) / (59.5 + min(ffmc_value, 100.0)) / 100.0))


def daily_spread_ffmc(model: str, ffmc_series, wind_max_kmh, slope_deg: float = 0.0) -> np.ndarray:
    """Daily rate of spread (m/min) with 1-h dead moisture taken from the FFMC, which carries the memory of recent rain and drying that a single day's weather lacks."""
    fm = MODELS[model]
    return np.array([spread(model, moisture_from_ffmc(f), w * fm.wind_factor, slope_deg)["ros_m_min"] for f, w in zip(ffmc_series, wind_max_kmh)])


def _dominant_fuel(lat: float, lon: float, radius_km: float = 0.5):
    from ..connectors.worldcover import lc_grid
    g, _ = lc_grid(lat, lon, 1.0)
    g = np.asarray(g)
    n = g.shape[0]
    c = n // 2
    k = max(1, int(radius_km * 1000 / 10))
    win = g[max(0, c - k): c + k + 1, max(0, c - k): c + k + 1]
    win = win[np.isfinite(win)]
    if win.size == 0:
        return None, {}
    cls, cnt = np.unique(win, return_counts=True)
    share = {int(a): float(b) / win.size for a, b in zip(cls, cnt)}
    fuel_classes = {k_: v for k_, v in share.items() if k_ in WORLDCOVER_FUEL}
    if not fuel_classes or sum(fuel_classes.values()) < 0.25:
        return None, share
    top = max(fuel_classes, key=fuel_classes.get)
    return WORLDCOVER_FUEL[top], share


def site_fire_behaviour(lat: float, lon: float, y0: int = 1991, y1: int = 2023, fuel: Optional[str] = None, slope_deg: Optional[float] = None, series=None) -> dict:
    """{available, fuel_model, ros_p95_m_min, flame_length_p95_m, intensity_p95_kw_m, rating, ...}: head-fire behaviour at the 95th-percentile warm-season day of ERA5 weather."""
    try:
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return {"available": False, "reason": "non-finite coordinates"}
        share = {}
        if fuel is None:
            fuel, share = _dominant_fuel(lat, lon)
        if fuel is None:
            return {"available": True, "applicable": False, "reason": "no flammable vegetation dominates within 500 m of the pin"}
        if slope_deg is None:
            from ..connectors.terrain_tiles import get_elevation_grid
            g = get_elevation_grid(lat, lon, radius_km=0.4)
            z = np.asarray(g.elevations, float)
            gy, gx = np.gradient(z, g.cell_size_m)
            slope_deg = float(np.degrees(np.arctan(np.nanmean(np.hypot(gy, gx)))))
        if series is None:
            from ..modellab import data as D
            v = "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum"
            dd = D.archive(lat, lon, y0, y1, daily=v)["daily"]
            ok = [i for i in range(len(dd["time"])) if None not in (dd[k][i] for k in v.split(","))]
            series = ([dd["time"][i] for i in ok], np.array([dd["temperature_2m_max"][i] for i in ok], float), np.array([dd["relative_humidity_2m_min"][i] for i in ok], float),
                      np.array([dd["wind_speed_10m_max"][i] for i in ok], float), np.array([dd["precipitation_sum"][i] for i in ok], float))
        t, T, RH, W, P = series
        if len(T) < 365 * 10:
            return {"available": False, "reason": "weather record too short"}
        from ..modellab import fire
        mo = [int(x[5:7]) for x in t]
        ff = np.array(fire.run_series(T, RH, W, P, mo, tropical=abs(lat) < 25)["ffmc"])
        warm = T >= 15.0
        if warm.sum() < 200:
            return {"available": True, "applicable": False, "reason": "too cool for a fire season (daily maxima rarely reach 15 C)"}
        fm = MODELS[fuel]
        idx = np.nonzero(warm)[0]
        ros = np.array([spread(fuel, moisture_from_ffmc(ff[i]), W[i] * fm.wind_factor, slope_deg)["ros_m_min"] for i in idx])
        j = idx[int(np.argsort(ros)[int(0.95 * (len(ros) - 1))])]
        b = spread(fuel, moisture_from_ffmc(ff[j]), W[j] * fm.wind_factor, slope_deg)
        return {"available": True, "applicable": True, "fuel_model": fuel, "fuel_name": fm.name, "slope_deg": round(slope_deg, 1), "ros_p95_m_min": float(b["ros_m_min"]),
                "flame_length_p95_m": float(b["flame_length_m"]), "intensity_p95_kw_m": float(b["intensity_kw_m"]), "rating": flame_rating(b["flame_length_m"]),
                "p95_day_weather": {"tmax_c": round(float(T[j]), 1), "rh_min_pct": round(float(RH[j]), 0), "wind_max_kmh": round(float(W[j]), 0), "ffmc": round(float(ff[j]), 1)},
                "warm_days_used": int(warm.sum()), "weather": "ERA5 daily via Open-Meteo", "land_cover_shares": {str(k): round(v, 2) for k, v in share.items()}}
    except Exception as e:                                                  # noqa: BLE001
        return {"available": False, "reason": f"fire behaviour could not run ({type(e).__name__}: {str(e)[:100]})"}
