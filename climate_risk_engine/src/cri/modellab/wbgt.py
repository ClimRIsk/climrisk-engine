"""
cri/modellab/wbgt.py — outdoor Wet Bulb Globe Temperature by the Liljegren method (vectorised NumPy port).

    WBGT = 0.1 * Tair + 0.2 * Tglobe + 0.7 * Tnwb

Tglobe (black globe) and Tnwb (natural wet bulb) come from the heat balance of a 15 cm globe / a wetted wick in sun, wind
and humidity, solved iteratively. This replaces the shade shortcut sWBGT = 0.567 T + 0.393 e + 3.94, which Kong & Huber (2022,
Earth's Future) showed overstates heat stress in hot, humid places and which the engine's own cross-check found 3.5 to 5.2 °C
above a physical bracket (scripts/heat_index_crosscheck.py).

Source and licence
  Port of "WBGT, Version 1.1", J. C. Liljegren, Argonne National Laboratory, Copyright (c) 2008 UChicago Argonne, LLC (open source
  licence: redistribution with notice allowed; derivative works must be noted). This file is a derivative work: it re-implements the
  C routines Tglobe(), Twb(), h_sphere_in_air(), h_cylinder_in_air(), esat(), emis_atm(), diffusivity(), viscosity(),
  thermal_cond(), evap(), dew_point(), stab_srdt(), est_wind_speed() and calc_solar_parameters() in NumPy, by ClimRisk.
  Reference: Liljegren, Carhart, Lawday, Tschopp, Sharp (2008), J. Occup. Environ. Hyg. 5:10, 645-655.
  This port is checked against the compiled reference (pywbgt 3.0.7, which wraps the same C code) in tests/test_wbgt.py and
  scripts/validate_wbgt.py; differences come only from the reference using 32-bit floats.

Inputs: air temperature (°C), relative humidity (%), pressure (hPa), 10 m wind (m/s), downward shortwave irradiance (W/m2),
and UTC time + latitude/longitude (for the solar zenith angle). Non-converged points return NaN, as the C code returns -9999.
"""
from __future__ import annotations

import numpy as np

SOLAR_CONST, STEFANB, CP, M_AIR, M_H2O, R_GAS = 1367.0, 5.6696e-8, 1003.5, 28.97, 18.015, 8314.34
RATIO = CP * M_AIR / M_H2O
R_AIR = R_GAS / M_AIR
PR = CP / (CP + 1.25 * R_AIR)
EMIS_WICK, ALB_WICK, D_WICK, L_WICK = 0.95, 0.4, 0.007, 0.0254
EMIS_GLOBE, ALB_GLOBE, D_GLOBE = 0.95, 0.05, 0.0508
EMIS_SFC, ALB_SFC = 0.999, 0.45
CZA_MIN, NORMSOLAR_MAX, REF_HEIGHT, MIN_SPEED, CONVERGENCE, MAX_ITER = 0.00873, 0.85, 2.0, 0.13, 0.02, 50
DEG_RAD = np.pi / 180.0
_LSRDT = np.array([[1, 1, 2, 4, 0, 5, 6, 0], [1, 2, 3, 4, 0, 5, 6, 0], [2, 2, 3, 4, 0, 4, 4, 0], [3, 3, 4, 4, 0, 0, 0, 0], [3, 4, 4, 4, 0, 0, 0, 0], [0] * 8])
_URBAN_EXP = np.array([0.15, 0.15, 0.20, 0.25, 0.30, 0.30])
_RURAL_EXP = np.array([0.07, 0.07, 0.10, 0.15, 0.35, 0.55])


def esat(tk):
    """Saturation vapour pressure over water, hPa (Buck 1981 form, x1.004 moist-air correction), as in the reference."""
    tk = np.asarray(tk, float)
    return 1.004 * 6.1121 * np.exp(17.502 * (tk - 273.15) / (tk - 32.18))


def dew_point(e):
    z = np.log(e / (6.1121 * 1.004))
    return 273.15 + 240.97 * z / (17.502 - z)


def emis_atm(tk, rh):
    return 0.575 * (rh * esat(tk)) ** 0.143


def evap(tk):
    return (313.15 - tk) / 30.0 * (-71100.0) + 2.4073e6


def viscosity(tk):
    tr = tk / 97.0
    omega = (tr - 2.9) / 0.4 * (-0.034) + 1.048
    return 2.6693e-6 * np.sqrt(M_AIR * tk) / (3.617 * 3.617 * omega)


def thermal_cond(tk):
    return (CP + 1.25 * R_AIR) * viscosity(tk)


def diffusivity(tk, p_hpa):
    pc13 = (36.4 * 218.0) ** (1.0 / 3.0)
    tc512 = (132.0 * 647.3) ** (5.0 / 12.0)
    tc12 = np.sqrt(132.0 * 647.3)
    mmix = np.sqrt(1.0 / M_AIR + 1.0 / M_H2O)
    return 3.640e-4 * (tk / tc12) ** 2.334 * pc13 * tc512 * mmix / (p_hpa / 1013.25) * 1e-4


def h_sphere(d, tk, p_hpa, speed):
    dens = p_hpa * 100.0 / (R_AIR * tk)
    re = speed * dens * d / viscosity(tk)
    return (2.0 + 0.6 * np.sqrt(re) * PR ** 0.3333) * thermal_cond(tk) / d


def h_cylinder(d, length, tk, p_hpa, speed):
    dens = p_hpa * 100.0 / (R_AIR * tk)
    re = speed * dens * d / viscosity(tk)
    return 0.281 * re ** (1.0 - 0.4) * PR ** (1.0 - 0.56) * thermal_cond(tk) / d


# ── solar geometry (Astronomical Almanac low-precision formulae, as in the reference) ──

def solar_cza_and_distance(t_utc: np.ndarray, lat: float, lon: float) -> tuple:
    """Cosine of the solar zenith angle (refraction added to altitude, as in the reference) and Earth-Sun distance (AU). t_utc: datetime64."""
    t = np.asarray(t_utc, dtype="datetime64[s]")
    days_j2000 = (t - np.datetime64("2000-01-01T12:00:00")).astype("timedelta64[s]").astype(float) / 86400.0
    cent = np.floor(days_j2000 + 0.5) / 36525.0                                   # Julian centuries at 0 h UT of the date
    ut = ((t - t.astype("datetime64[D]")).astype("timedelta64[s]").astype(float)) / 3600.0
    ma = np.mod((357.528 + 0.9856003 * days_j2000) / 360.0, 1.0) * 2 * np.pi
    ml = np.mod((280.460 + 0.9856474 * days_j2000) / 360.0, 1.0) * 2 * np.pi
    obl = (23.439 - 4.0e-7 * days_j2000) * DEG_RAD
    ecl = (1.915 * np.sin(ma) + 0.020 * np.sin(2 * ma)) * DEG_RAD + ml
    dist = 1.00014 - 0.01671 * np.cos(ma) - 0.00014 * np.cos(2 * ma)
    ra = np.mod(np.arctan2(np.cos(obl) * np.sin(ecl), np.cos(ecl)), 2 * np.pi) / (2 * np.pi) * 24.0
    dec = np.arcsin(np.sin(obl) * np.sin(ecl))
    gmst0 = 24110.54841 + cent * (8640184.812866 + cent * (0.093104 - cent * 6.2e-6))
    gmst0 = np.mod(gmst0 / 3600.0 / 24.0, 1.0) * 24.0
    lmst = np.mod((gmst0 + ut * 1.00273790934 + lon / 15.0) / 24.0, 1.0) * 24.0
    ha = lmst - ra
    ha = np.where(ha < -12.0, ha + 24.0, np.where(ha > 12.0, ha - 24.0, ha)) / 24.0 * 2 * np.pi
    la = lat * DEG_RAD
    alt = np.arcsin(np.sin(dec) * np.sin(la) + np.cos(dec) * np.cos(ha) * np.cos(la))
    alt_d = alt / DEG_RAD
    tan_alt = np.tan(np.where(np.abs(alt) < 1.57079615, alt, 1.0))
    refr = np.where(alt_d < -1.0, 0.0, np.where(alt_d < 19.225,
                    (0.1594 + alt_d * (0.0196 + 0.00002 * alt_d)) * 1013.25 / ((1.0 + alt_d * (0.505 + 0.0845 * alt_d)) * 288.0),
                    0.00452 * (1013.25 / 288.0) / tan_alt))
    return np.cos((90.0 - (alt_d + refr)) * DEG_RAD), dist


def solar_parameters(solar, cza, dist) -> tuple:
    """Adjusted irradiance and direct-beam fraction (Liljegren): clip the clearness at 0.85, direct fraction exp(3 - 1.34 k - 1.65 / k)."""
    solar = np.asarray(solar, float)
    toa = np.where(cza < CZA_MIN, 0.0, SOLAR_CONST * np.maximum(0.0, cza) / (dist * dist))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(toa > 0, np.minimum(solar / np.where(toa > 0, toa, 1.0), NORMSOLAR_MAX), 0.0)
        fdir = np.where(k > 0, np.clip(np.exp(3.0 - 1.34 * k - 1.65 / np.where(k > 0, k, 1.0)), 0.0, 0.9), 0.0)
    return np.where(toa > 0, k * toa, 0.0), fdir


def adjust_wind(speed10, solar, cza, urban: bool = False, dT: float = -1.0, min_speed: float = MIN_SPEED) -> np.ndarray:
    """10 m wind to 2 m using the Pasquill stability class from sun and wind (EPA-454/5-99-005 s6.2.5), as in the reference."""
    s = np.asarray(speed10, float)
    day = cza > 0
    jd = np.where(solar >= 925.0, 0, np.where(solar >= 675.0, 1, np.where(solar >= 175.0, 2, 3)))
    idd = np.where(s >= 6.0, 4, np.where(s >= 5.0, 3, np.where(s >= 3.0, 2, np.where(s >= 2.0, 1, 0))))
    jn = 6 if dT >= 0 else 5
    inn = np.where(s >= 2.5, 2, np.where(s >= 2.0, 1, 0))
    cls = _LSRDT[np.where(day, idd, inn), np.where(day, jd, jn)]
    exp = (_URBAN_EXP if urban else _RURAL_EXP)[np.clip(cls - 1, 0, 5)]
    return np.maximum(s * (REF_HEIGHT / 10.0) ** exp, max(min_speed, MIN_SPEED))


# ── heat balances ───────────────────────────────────────────────────────────────

def _iterate(step, x0):
    """The reference's relaxation loop: x_prev <- 0.9 x_prev + 0.1 x_new until |x_new - x_prev| < 0.02 K, at most 50 passes; returns x_new at convergence, else NaN."""
    prev = x0.copy()
    out = np.full_like(x0, np.nan)
    live = np.ones(x0.shape, bool)
    for _ in range(MAX_ITER):
        new = step(prev)
        done = live & (np.abs(new - prev) < CONVERGENCE)
        out[done] = new[done]
        live &= ~done
        prev = np.where(live, 0.9 * prev + 0.1 * new, prev)
        if not live.any():
            break
    return out


def globe_temp_c(tair_c, rh_pct, p_hpa, speed2, solar, fdir, cza, d_globe: float = D_GLOBE) -> np.ndarray:
    tk = np.asarray(tair_c, float) + 273.15
    rh = np.asarray(rh_pct, float) * 0.01
    tk, rh, p, sp, so, fd, cz = np.broadcast_arrays(tk, rh, np.asarray(p_hpa, float), np.asarray(speed2, float), np.asarray(solar, float), np.asarray(fdir, float), np.asarray(cza, float))
    base = 0.5 * (emis_atm(tk, rh) * tk ** 4 + EMIS_SFC * tk ** 4)

    def step(prev):
        h = h_sphere(d_globe, 0.5 * (prev + tk), p, sp)
        with np.errstate(divide="ignore", invalid="ignore"):
            rad = so / (2.0 * STEFANB * EMIS_GLOBE) * (1.0 - ALB_GLOBE) * (fd * (1.0 / (2.0 * np.maximum(cz, 1e-9)) - 1.0) + 1.0 + ALB_SFC)
        return np.maximum(base - h / (STEFANB * EMIS_GLOBE) * (prev - tk) + rad, 0.0) ** 0.25
    return _iterate(step, tk.copy()) - 273.15


def natural_wetbulb_c(tair_c, rh_pct, p_hpa, speed2, solar, fdir, cza, radiation: bool = True) -> np.ndarray:
    tk = np.asarray(tair_c, float) + 273.15
    rh = np.asarray(rh_pct, float) * 0.01
    tk, rh, p, sp, so, fd, cz = np.broadcast_arrays(tk, rh, np.asarray(p_hpa, float), np.asarray(speed2, float), np.asarray(solar, float), np.asarray(fdir, float), np.asarray(cza, float))
    eair = rh * esat(tk)
    sza = np.arccos(np.clip(cz, -1.0, 1.0))
    fatm_const = STEFANB * EMIS_WICK * 0.5 * (emis_atm(tk, rh) * tk ** 4 + EMIS_SFC * tk ** 4)

    def step(prev):
        tref = 0.5 * (prev + tk)
        heat = 0.0
        if radiation:
            with np.errstate(invalid="ignore"):
                fatm = (fatm_const - STEFANB * EMIS_WICK * prev ** 4 + (1.0 - ALB_WICK) * so * ((1.0 - fd) * (1.0 + 0.25 * D_WICK / L_WICK)
                        + fd * (np.tan(sza) / np.pi + 0.25 * D_WICK / L_WICK) + ALB_SFC))
            heat = fatm / h_cylinder(D_WICK, L_WICK, tref, p, sp)
        ew = esat(prev)
        dens = p * 100.0 / (R_AIR * tref)
        sc = viscosity(tref) / (dens * diffusivity(tref, p))
        return tk - evap(tref) / RATIO * (ew - eair) / (p - ew) * (PR / sc) ** 0.56 + heat
    return _iterate(step, dew_point(eair)) - 273.15


def wbgt(tair_c, rh_pct, p_hpa, speed10, solar, t_utc, lat: float, lon: float, urban: bool = False, fdir_override=None, wind_height: float = 10.0) -> dict:
    """Outdoor WBGT and its parts for hourly series. `t_utc` marks the MIDDLE of the averaging period of the radiation (hourly means: subtract 30 min)."""
    tair_c, rh_pct, p_hpa, speed10, solar = (np.asarray(a, float) for a in (tair_c, rh_pct, p_hpa, speed10, solar))
    cza, dist = solar_cza_and_distance(t_utc, lat, lon)
    sol, fdir = solar_parameters(solar, cza, dist)
    if fdir_override is not None:
        fdir = np.where(sol > 0, np.clip(np.asarray(fdir_override, float), 0.0, 1.0), 0.0)
    sp2 = adjust_wind(speed10, sol, cza, urban) if wind_height != REF_HEIGHT else np.maximum(speed10, MIN_SPEED)
    tg = globe_temp_c(tair_c, rh_pct, p_hpa, sp2, sol, fdir, cza)
    tn = natural_wetbulb_c(tair_c, rh_pct, p_hpa, sp2, sol, fdir, cza, True)
    tp = natural_wetbulb_c(tair_c, rh_pct, p_hpa, sp2, sol, fdir, cza, False)
    return {"wbgt": 0.1 * tair_c + 0.2 * tg + 0.7 * tn, "globe": tg, "nat_wetbulb": tn, "psy_wetbulb": tp, "cza": cza, "fdir": fdir, "solar": sol, "speed2": sp2}
