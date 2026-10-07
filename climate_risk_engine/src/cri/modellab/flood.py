"""
2-D flood inundation by the local-inertial (reduced shallow-water) equations — Bates, Horritt & Fewtrell (2010).

Governing equations (depth h, bed z, free surface η = z + h, unit discharge q, Manning n, rain r):
    ∂h/∂t + ∂qx/∂x + ∂qy/∂y = r
    ∂q/∂t + g h ∂η/∂x + g n² |q| q / h^(7/3) = 0        (convective acceleration neglected)
Discretised on a staggered grid, semi-implicit in friction:
    q' + g Δt n² q'|q'| / h_f^(7/3) = q − g h_f Δt ∂η/∂x   (solved exactly for q'),   h_f = max(η_i, η_j) − max(z_i, z_j)
Mass is conserved exactly: a donor-cell limiter stops any cell giving away more water than it holds.
Time step is adaptive: Δt = α Δx / sqrt(g h_max) (CFL), capped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

G = 9.80665
H_MIN = 1e-4
MIN_EDGE_SLOPE = 5e-4          # a free outflow edge never drains slower than this (flat floodplain edge)


@dataclass
class FloodResult:
    h_max: np.ndarray                      # maximum depth over the run (m)
    h_end: np.ndarray                      # depth at the end (m)
    t_wet: np.ndarray                      # seconds each cell was wet (> 5 cm)
    dx: float
    steps: int
    t_end: float
    vol_in: float                          # m³ added by rain/inflow
    vol_out: float                         # m³ left through open boundaries
    vol_stored: float                      # m³ on the domain at the end
    mass_error: float                      # |in − out − stored| / max(in, 1)
    max_courant: float
    series: dict = field(default_factory=dict)   # t (s), probe depth (m), total_in/out/stored


def _implicit_friction(m, c):
    """Solve q' + c q'|q'| = m exactly (fully implicit Manning friction): q' = 2m / (1 + sqrt(1 + 4 c |m|)).
    Unlike the lagged-|q| form of Bates et al. it has no time-step threshold in thin, fast sheet flow."""
    return 2.0 * m / (1.0 + np.sqrt(1.0 + 4.0 * c * np.abs(m)))


def _face_depth(eta_a, eta_b, z_a, z_b):
    return np.maximum(eta_a, eta_b) - np.maximum(z_a, z_b)


def simulate(z: np.ndarray, dx: float, duration_s: float, rain_m_s=0.0, manning_n=0.035, inflow: Optional[dict] = None,
             open_boundary=True, h0: Optional[np.ndarray] = None, probe: Optional[tuple] = None, alpha: float = 0.7, dt_max: float = 30.0,
             record_every_s: float = 300.0, rain_mask: Optional[np.ndarray] = None) -> FloodResult:
    """
    z: bed elevation (rows × cols, m). rain_m_s: scalar, or callable t→m/s, or (times_s, rates) arrays.
    inflow: {"cells": [(i, j), ...], "q_m3_s": scalar|callable} spread equally over the cells.
    open_boundary: True (all four edges), False (closed), or a string such as "S" / "NS" naming the edges that drain by
                   Manning normal-depth flux (a free outflow); the others are walls.
    """
    z = np.asarray(z, float)
    ny, nx = z.shape
    h = np.zeros_like(z) if h0 is None else np.asarray(h0, float).copy()
    qx = np.zeros((ny, nx - 1))
    qy = np.zeros((ny - 1, nx))
    n2 = np.broadcast_to(np.asarray(manning_n, float) ** 2, z.shape)
    n2x = 0.5 * (n2[:, :-1] + n2[:, 1:])
    n2y = 0.5 * (n2[:-1, :] + n2[1:, :])
    area = dx * dx

    def rate(t):
        if callable(rain_m_s):
            return float(rain_m_s(t))
        if isinstance(rain_m_s, tuple):
            ts, rs = rain_m_s
            return float(np.interp(t, ts, rs, right=0.0))
        return float(rain_m_s)

    edges = set("NSEW") if open_boundary is True else set(open_boundary or "")
    open_boundary = bool(edges)
    inflow_cells = np.array(inflow["cells"], int) if inflow else None
    t, steps, vol_in, vol_out = 0.0, 0, 0.0, 0.0
    h_max = h.copy()
    t_wet = np.zeros_like(z)
    max_c = 0.0
    rec = {"t": [], "probe": [], "in": [], "out": [], "stored": []}
    next_rec = 0.0
    rm = None if rain_mask is None else np.asarray(rain_mask, float)

    # outward-slope Manning drainage factor on the four edges (free outflow), computed once from the bed
    def _edge_slope(a, b):
        return np.maximum((a - b) / dx, 0.0)

    while t < duration_s - 1e-9:
        hm = float(h.max())
        dt = min(dt_max, alpha * dx / math.sqrt(G * max(hm, 0.02)), duration_s - t)
        max_c = max(max_c, dt * math.sqrt(G * max(hm, 0.0)) / dx)
        eta = z + h

        hf = _face_depth(eta[:, :-1], eta[:, 1:], z[:, :-1], z[:, 1:])
        wet = hf > H_MIN
        hfs = np.where(wet, hf, 1.0)
        qx = np.where(wet, _implicit_friction(qx - G * hfs * dt * (eta[:, 1:] - eta[:, :-1]) / dx, G * dt * n2x / hfs ** (7.0 / 3.0)), 0.0)
        hf = _face_depth(eta[:-1, :], eta[1:, :], z[:-1, :], z[1:, :])
        wet = hf > H_MIN
        hfs = np.where(wet, hf, 1.0)
        qy = np.where(wet, _implicit_friction(qy - G * hfs * dt * (eta[1:, :] - eta[:-1, :]) / dx, G * dt * n2y / hfs ** (7.0 / 3.0)), 0.0)

        # donor-cell limiter: no cell may give away more than it holds
        out = np.zeros_like(z)
        out[:, :-1] += np.maximum(qx, 0.0)
        out[:, 1:] += np.maximum(-qx, 0.0)
        out[:-1, :] += np.maximum(qy, 0.0)
        out[1:, :] += np.maximum(-qy, 0.0)

        # open boundaries: Manning normal-depth outflow where the bed falls outward
        qb = {}
        if open_boundary:
            def _norm_q(hh, s, nn):
                return np.where((hh > H_MIN) & (s > 0), hh ** (5.0 / 3.0) * np.sqrt(np.maximum(s, 1e-12)) / nn, 0.0)
            nsq = np.sqrt(n2)
            qb["N"] = _norm_q(h[0, :], np.maximum(_edge_slope(z[1, :], z[0, :]), MIN_EDGE_SLOPE), nsq[0, :])
            qb["S"] = _norm_q(h[-1, :], np.maximum(_edge_slope(z[-2, :], z[-1, :]), MIN_EDGE_SLOPE), nsq[-1, :])
            qb["W"] = _norm_q(h[:, 0], np.maximum(_edge_slope(z[:, 1], z[:, 0]), MIN_EDGE_SLOPE), nsq[:, 0])
            qb["E"] = _norm_q(h[:, -1], np.maximum(_edge_slope(z[:, -2], z[:, -1]), MIN_EDGE_SLOPE), nsq[:, -1])
            for k in "NSEW":
                if k not in edges:
                    qb[k] = 0.0 * qb[k]
            out[0, :] += qb["N"]; out[-1, :] += qb["S"]; out[:, 0] += qb["W"]; out[:, -1] += qb["E"]

        lost = out * dt / dx                                           # depth a cell would lose
        fac = np.where(lost > 1e-15, np.minimum(1.0, h / np.maximum(lost, 1e-15)), 1.0)
        qx = np.where(qx > 0, qx * fac[:, :-1], qx * fac[:, 1:])
        qy = np.where(qy > 0, qy * fac[:-1, :], qy * fac[1:, :])
        if open_boundary:
            qb["N"] = qb["N"] * fac[0, :]; qb["S"] = qb["S"] * fac[-1, :]; qb["W"] = qb["W"] * fac[:, 0]; qb["E"] = qb["E"] * fac[:, -1]

        dh = np.zeros_like(z)
        dh[:, :-1] -= qx; dh[:, 1:] += qx
        dh[:-1, :] -= qy; dh[1:, :] += qy
        dh *= dt / dx
        if open_boundary:
            dh[0, :] -= qb["N"] * dt / dx; dh[-1, :] -= qb["S"] * dt / dx; dh[:, 0] -= qb["W"] * dt / dx; dh[:, -1] -= qb["E"] * dt / dx
            vol_out += float((qb["N"].sum() + qb["S"].sum() + qb["W"].sum() + qb["E"].sum()) * dt * dx)

        r = rate(t)
        if r:
            add = r * dt
            dh += add if rm is None else add * rm
            vol_in += float(add * area * (z.size if rm is None else rm.sum()))
        if inflow_cells is not None:
            qq = inflow["q_m3_s"](t) if callable(inflow["q_m3_s"]) else float(inflow["q_m3_s"])
            if qq:
                per = qq * dt / len(inflow_cells) / area
                for (ii, jj) in inflow_cells:
                    dh[ii, jj] += per
                vol_in += float(qq * dt)

        h = np.maximum(h + dh, 0.0)
        t += dt
        steps += 1
        np.maximum(h_max, h, out=h_max)
        t_wet += np.where(h > 0.05, dt, 0.0)
        if t >= next_rec:
            rec["t"].append(t); rec["probe"].append(float(h[probe]) if probe else float(h.max()))
            rec["in"].append(vol_in); rec["out"].append(vol_out); rec["stored"].append(float(h.sum() * area))
            next_rec = t + record_every_s

    stored = float(h.sum() * area) - (0.0 if h0 is None else float(np.asarray(h0).sum() * area))
    err = abs(vol_in - vol_out - stored) / max(vol_in, 1.0)
    return FloodResult(h_max=h_max, h_end=h, t_wet=t_wet, dx=dx, steps=steps, t_end=t, vol_in=vol_in, vol_out=vol_out, vol_stored=stored,
                       mass_error=err, max_courant=max_c, series=rec)


# ── verification benchmarks (also run live by every study) ───────────────────

def bench_lake_at_rest(n: int = 40, dx: float = 20.0) -> dict:
    """A lake over a rough bed must stay still: well-balancedness. Spurious velocity should be ~machine zero."""
    rng = np.random.default_rng(3)
    z = rng.uniform(0.0, 0.6, (n, n))
    eta0 = 1.0
    h0 = np.maximum(eta0 - z, 0.0)
    r = simulate(z, dx, 600.0, open_boundary=False, h0=h0)
    drift = float(np.abs(r.h_end - h0).max())
    return {"name": "Lake at rest over a rough bed", "metric": "max depth change (m)", "value": drift, "limit": 1e-6, "ok": drift < 1e-6,
            "note": "Still water must stay still: the discretisation is well-balanced."}


def bench_mass_conservation(n: int = 36, dx: float = 25.0) -> dict:
    rng = np.random.default_rng(5)
    z = np.cumsum(rng.normal(0, 0.05, (n, n)), axis=0)
    r = simulate(z, dx, 1800.0, rain_m_s=30e-3 / 3600, open_boundary=False)
    return {"name": "Mass conservation (closed basin, rain)", "metric": "relative volume error", "value": r.mass_error, "limit": 1e-9, "ok": r.mass_error < 1e-9,
            "note": "Every cubic metre of rain is still on the grid."}


def bench_normal_depth(dx: float = 10.0, rows: int = 60, cols: int = 5, s0: float = 0.001, n: float = 0.03, q0: float = 0.5) -> dict:
    """Steady uniform flow down a plane must reach the Manning normal depth h = (q n / √S)^(3/5)."""
    zz = (rows - 1 - np.arange(rows))[:, None] * dx * s0 * np.ones((1, cols))
    cells = [(0, j) for j in range(cols)]
    r = simulate(zz, dx, 5400.0, manning_n=n, inflow={"cells": cells, "q_m3_s": q0 * cols * dx}, open_boundary="S", probe=(rows // 2, cols // 2), dt_max=5.0)
    h_mid = float(r.h_end[rows // 2, cols // 2])
    h_n = (q0 * n / math.sqrt(s0)) ** 0.6
    rel = abs(h_mid - h_n) / h_n
    return {"name": "Steady uniform flow vs Manning normal depth", "metric": f"relative error (analytic {h_n:.3f} m, model {h_mid:.3f} m)", "value": rel, "limit": 0.05, "ok": rel < 0.05,
            "note": "The friction law reproduces the textbook normal depth."}


def _rain_on_plane(dx: float) -> float:
    """Peak depth halfway down a 600 m × 200 m plane (S = 0.01, n = 0.05) under 60 mm/h for 20 min, resolved at cell size dx."""
    rows, cols = int(600 / dx), int(200 / dx)
    z = (rows - 1 - np.arange(rows))[:, None] * dx * 0.01 * np.ones((1, cols))
    r = simulate(z, dx, 2400.0, rain_m_s=lambda t: 60e-3 / 3600 if t < 1200 else 0.0, manning_n=0.05, open_boundary="S", dt_max=3.0)
    return float(r.h_max[rows // 2, cols // 2])


def bench_grid_convergence() -> dict:
    """A transient rain-on-slope flood must give the same peak depth when the cell size is halved (convergence in dx)."""
    coarse, fine = _rain_on_plane(20.0), _rain_on_plane(10.0)
    rel = abs(coarse - fine) / fine
    return {"name": "Grid convergence (transient, cell size halved)", "metric": f"relative change in peak depth ({coarse:.3f} m → {fine:.3f} m)", "value": rel, "limit": 0.05, "ok": rel < 0.05,
            "note": "The result does not depend on the resolution chosen."}


def bench_rain_equilibrium_and_dt() -> dict:
    """Steady rain on a steep plane: unit discharge at distance L is r·L, so h = (r L n / √S)^(3/5). Must hold at a 30 s step and a 3 s step alike."""
    n_c, dx, nm, r, s0 = 24, 25.0, 0.04, 20e-3 / 3600, 0.02
    z = (n_c - 1 - np.arange(n_c))[:, None] * dx * s0 * np.ones((1, 6))
    worst = 0.0
    for dtm in (30.0, 3.0):
        res = simulate(z, dx, 4 * 3600, rain_m_s=r, manning_n=nm, open_boundary="S", dt_max=dtm)
        for row in (6, 12, 20):
            exp = (r * (row + 1) * dx * nm / math.sqrt(s0)) ** 0.6
            worst = max(worst, abs(float(res.h_end[row, 3]) - exp) / exp)
    return {"name": "Rain on a slope reaches the analytic depth at any time step", "metric": "max relative error at 3 locations, 30 s and 3 s steps", "value": worst, "limit": 0.03, "ok": worst < 0.03,
            "note": "h = (r L n / √S)^(3/5): the answer must not depend on the time step chosen."}


def run_benchmarks() -> list:
    return [bench_lake_at_rest(), bench_mass_conservation(), bench_normal_depth(), bench_rain_equilibrium_and_dt(), bench_grid_convergence()]
