"""
Release plume for the 3D twin: a steady Gaussian plume from a release the USER declares (rate, height, wind, stability), on the engine's verified
dispersion code (`modellab.air`: Briggs urban sigmas, ground reflection). Screening-grade, and honest about what it leaves out.

What it is:     passive, neutrally buoyant gas or fine particles, steady wind, flat ground, Briggs (1973) urban dispersion, ~10-60 minute averaging.
What it is not: dense gases (chlorine, LNG, LPG, ammonia cold releases), jet or plume rise, building wake, terrain channelling, deposition, chemistry, odour,
                fire plumes, time-varying releases. Do not use it for emergency planning or toxic-release consequence assessment.
Inputs the user must supply (nothing is guessed): emission rate Q (g/s), effective release height H (m), 10 m wind speed and the direction it blows FROM,
                and a Pasquill stability class (or day / night and cloud, from which a class is read with `modellab.air.stability_class`).
Output:         a 3D concentration field in the plume's own frame (x downwind, y crosswind, z up) in ug/m3, a ground map in site coordinates, and a few
                numbers (peak ground value and where, distance to each decade level, structures inside each level). The first 30 m downwind are masked
                (null, not zero): the Gaussian form is not valid that close to a source.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .modellab import air

NEAR_FIELD_M = 30.0
URBAN_WIND_EXPONENT = {"A": 0.15, "B": 0.15, "C": 0.20, "D": 0.25, "E": 0.30, "F": 0.30}      # EPA ISC urban profile exponents
LEVELS_UG_M3 = (1.0, 10.0, 100.0, 1000.0, 10000.0)                                               # decades: not health thresholds
LIMITS = [
    "Gaussian plume: steady wind, flat ground, passive neutrally buoyant gas or fine particles. Dense gases, jets, plume rise, building wake, terrain channelling, deposition and chemistry are not modelled.",
    "Briggs urban dispersion (1973) by Pasquill class; real concentrations at one place and time are patchy and can differ from this mean by a factor of two or more.",
    "The release rate, height and wind are the values you declared; this tool does not know what your plant can release.",
    f"The first {NEAR_FIELD_M:.0f} m downwind are masked (null): the formula is not valid that close to a source.",
    "Decade levels (1 to 10,000 ug/m3) are for reading the picture, not health or regulatory thresholds. Enter a threshold from the substance's own criteria (for example an AEGL or ERPG value) to see where it is exceeded.",
    "Screening only. Not for emergency planning or toxic-release consequence assessment.",
]


def _u_at(u10: float, H: float, cls: str) -> float:
    return max(0.5, u10) * (max(H, 10.0) / 10.0) ** URBAN_WIND_EXPONENT[cls]


def field(Q_g_s: float, H_m: float, u10_ms: float, wind_from_deg: float, cls: Optional[str] = None, daytime: bool = True, cloud_frac: float = 0.5,
          src_xy=(0.0, 0.0), half_m: float = 400.0, components: Optional[list] = None, threshold_ug_m3: Optional[float] = None,
          nx: int = 56, ny: int = 36, nz: int = 14, ground_n: int = 64) -> dict:
    """Concentration field for one declared release. Site coordinates: x east, y north, metres, origin at the twin's pin."""
    if not (Q_g_s > 0 and math.isfinite(Q_g_s)):
        raise ValueError("release rate must be a positive number of grams per second")
    if not (0 <= H_m <= 300):
        raise ValueError("release height must be between 0 and 300 m")
    if not (0 <= u10_ms <= 40):
        raise ValueError("10 m wind speed must be between 0 and 40 m/s")
    declared = cls is not None
    cls = cls or air.stability_class(max(u10_ms, 0.1), daytime, "moderate", cloud_frac)
    if cls not in air._BRIGGS_URBAN:
        raise ValueError("stability class must be A to F")
    u = _u_at(u10_ms, H_m, cls)
    to_rad = math.radians((wind_from_deg + 180.0) % 360.0)                       # direction the wind blows TOWARD, compass degrees
    ex, ey = math.sin(to_rad), math.cos(to_rad)                                  # unit vector downwind (east, north)
    L = 2.0 * half_m
    # --- 3D field in the plume frame ---
    xs = np.linspace(0.0, L, nx)
    sy_L, sz_L = air.sigmas(np.array([L]), cls)
    W = float(max(60.0, min(half_m, 3.0 * sy_L[0])))                             # crosswind half width shown
    Z = float(min(250.0, max(H_m * 2.0 + 30.0, 4.0 * sz_L[0] * 0.5, 60.0)))
    ys = np.linspace(-W, W, ny); zs = np.linspace(0.0, Z, nz)
    X, Y, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    c3 = air.plume(X, Y, ZZ, Q_g_s, u, H_m, cls) * 1e6                           # g/m3 -> ug/m3
    c3 = np.where(X < NEAR_FIELD_M, 0.0, c3)
    # --- ground map in site coordinates (what the viewer drapes on the terrain) ---
    g = np.linspace(-half_m, half_m, ground_n)
    GX, GY = np.meshgrid(g, g)                                                   # GY increases north
    dx, dy = GX - src_xy[0], GY - src_xy[1]
    xd = dx * ex + dy * ey                                                       # downwind distance
    yc = -dx * ey + dy * ex                                                      # crosswind (left of downwind positive)
    cg = air.plume(np.maximum(xd, 1.0), yc, 0.0, Q_g_s, u, H_m, cls) * 1e6
    ground_null = (xd < NEAR_FIELD_M)
    cg = np.where(ground_null, np.nan, cg)
    peak = None
    if np.isfinite(cg).any():
        i = int(np.nanargmax(cg)); r, c = divmod(i, ground_n)
        peak = {"ug_m3": float(cg[r, c]), "x_m": float(GX[r, c]), "y_m": float(GY[r, c]), "downwind_m": float(xd[r, c])}
    # distance along the centreline to each level (ground)
    xc = np.linspace(NEAR_FIELD_M, 5000.0, 1500)
    cc = air.plume(xc, np.zeros_like(xc), 0.0, Q_g_s, u, H_m, cls) * 1e6
    reach = {}
    levels = list(LEVELS_UG_M3) + ([float(threshold_ug_m3)] if threshold_ug_m3 else [])
    for lv in levels:
        idx = np.nonzero(cc >= lv)[0]
        reach[f"{lv:g}"] = (float(xc[idx[-1]]) if idx.size else None)
    reach_note = "furthest downwind distance on the centreline where the ground value is at or above the level; null = never reached beyond 30 m; capped at 5 km"
    # structures inside each level on the ground map (centroid test on the coarse grid)
    inside = {}
    if components:
        for lv in levels:
            hit = []
            for comp in components:
                px, py = comp.get("x", 0.0), comp.get("y", 0.0)
                d_x, d_y = px - src_xy[0], py - src_xy[1]
                xd1, yc1 = d_x * ex + d_y * ey, -d_x * ey + d_y * ex
                if xd1 < NEAR_FIELD_M:
                    continue
                v = float(air.plume(np.array([xd1]), np.array([yc1]), 0.0, Q_g_s, u, H_m, cls)[0] * 1e6)
                if v >= lv:
                    hit.append({"id": comp.get("id"), "name": comp.get("name"), "ug_m3": round(v, 3 if v < 10 else 1)})
            inside[f"{lv:g}"] = hit
    def r3(a):
        return [[round(float(v), 4) if v == v else None for v in row] for row in a]
    return {
        "inputs": {"Q_g_s": Q_g_s, "H_m": H_m, "u10_ms": u10_ms, "u_at_release_ms": round(u, 2), "wind_from_deg": wind_from_deg % 360, "toward_deg": round(math.degrees(to_rad), 1),
                   "stability": cls, "stability_source": "declared by the user" if declared else "read from wind speed and day/night (Turner table)", "src_xy_m": list(src_xy),
                   "wind_exponent": URBAN_WIND_EXPONENT[cls]},
        "frame": {"ex": ex, "ey": ey, "nx": nx, "ny": ny, "nz": nz, "L_m": L, "W_m": W, "Z_m": Z, "near_field_m": NEAR_FIELD_M},
        "conc_ug_m3_xyz": [[[float(f"{v:.4g}") for v in row] for row in plane] for plane in c3.tolist()],
        "ground": {"half_m": half_m, "n": ground_n, "ug_m3": r3(cg), "masked_note": f"null within {NEAR_FIELD_M:.0f} m downwind of the source and upwind of it"},
        "levels_ug_m3": levels,
        "peak_ground": peak,
        "centreline_reach_m": reach, "centreline_reach_note": reach_note,
        "structures_at_or_above": inside,
        "status": "modelled",
        "method": "Gaussian plume with ground reflection; Briggs (1973) urban sigmas; EPA ISC urban wind-profile exponents; modellab.air (verified: mass flux and closed-form centreline benchmarks)",
        "limits": LIMITS,
    }
