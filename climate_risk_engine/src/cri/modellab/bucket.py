"""
Conceptual water-balance physics: a soil store with evapotranspiration, drainage and saturation-excess runoff, plus linear-reservoir routing.

Soil store S (mm), capacity Sc, relative wetness s = S/Sc, potential evapotranspiration E0 and rain P per day:
    ET = kc · E0 · min(1, s / fc)            evaporation is stressed once wetness falls below fc
    D  = kd · Sc · s^m                       drainage out of the root zone
    Q  = max(0, S + P − ET − D − Sc)         saturation excess runs off
    S' = min(Sc, S + P − ET − D)             ← water balance closes exactly: P = ET + D + Q + ΔS
Routing: R' = R + (Q + D·g) − k R,  flow = A · k R   (one linear reservoir; g = share of drainage that reaches the river).
"""
from __future__ import annotations

import math

import numpy as np


def soil(P, E0, Sc, kc, kd, m=2.0, fc=0.6, s0=0.5):
    """Run the soil store. Returns S, ET, D, Q arrays (mm)."""
    n = len(P)
    S = np.empty(n); ET = np.empty(n); D = np.empty(n); Q = np.empty(n)
    cur = s0 * Sc
    for i in range(n):
        s = min(max(cur / Sc, 0.0), 1.0)
        et = min(kc * E0[i] * min(1.0, s / fc), cur + P[i])
        d = min(kd * Sc * s ** m, max(cur + P[i] - et, 0.0))
        pre = cur + P[i] - et - d
        q = max(0.0, pre - Sc)
        cur = min(max(pre, 0.0), Sc)
        S[i], ET[i], D[i], Q[i] = cur, et, d, q
    return S, ET, D, Q


def _lin(inp, k, r0=0.0):
    """R_t = (1 − k) R_{t−1} + inp_t, vectorised (exactly the reservoir recursion)."""
    from scipy.signal import lfilter
    y, _ = lfilter([1.0], [1.0, -(1.0 - k)], inp, zi=[(1.0 - k) * r0])
    return y


def route(Q, D, g, k, A, r0=0.0):
    """Linear reservoir: returns flow (same unit as A·mm)."""
    return A * k * _lin(np.asarray(Q) + g * np.asarray(D), k, r0)


def route2(Q, D, g, kf, ks, wf, A, r0=0.0):
    """Two parallel reservoirs (fast storm-flow and slow base-flow); wf is the share of the input sent to the fast one."""
    inp = np.asarray(Q) + g * np.asarray(D)
    return A * (kf * _lin(wf * inp, kf, r0) + ks * _lin((1 - wf) * inp, ks, r0))


# ── verification benchmarks ──────────────────────────────────────────────────

def bench_water_balance_closes() -> dict:
    rng = np.random.default_rng(2)
    n = 800
    P = np.where(rng.random(n) < 0.3, rng.gamma(1.5, 9.0, n), 0.0)
    E0 = 3 + 2 * np.sin(np.arange(n) / 58.0)
    S, ET, D, Q = soil(P, E0, 250.0, 0.9, 0.05, 2.5, 0.5, s0=0.4)
    dS = np.diff(np.concatenate([[0.4 * 250.0], S]))
    err = float(np.max(np.abs(P - ET - D - Q - dS)))
    return {"name": "Water balance closes: P = ET + drainage + runoff + ΔS", "metric": "max daily imbalance (mm)", "value": err, "limit": 1e-9, "ok": err < 1e-9, "note": "No water is created or lost."}


def bench_bounds_and_drydown() -> dict:
    n = 120
    S, ET, D, Q = soil(np.zeros(n), np.full(n, 4.0), 200.0, 1.0, 0.04, 2.0, 0.6, s0=0.9)
    mono = bool(np.all(np.diff(S) <= 1e-12) and S.min() >= 0)
    S2, _, _, Q2 = soil(np.full(30, 400.0), np.full(30, 3.0), 200.0, 1.0, 0.04, 2.0, 0.6, s0=0.2)
    sat = bool(abs(S2[-1] - 200.0) < 1e-9 and Q2[-1] > 300)
    return {"name": "Dry spell only dries the soil; a flood fills it to capacity and the rest runs off", "metric": "violations", "value": 0.0 if (mono and sat) else 1.0, "limit": 0.5, "ok": mono and sat, "note": "No negative storage, no storage above capacity."}


def bench_equilibrium() -> dict:
    """Constant rain and evaporative demand: the store must settle at the analytic equilibrium P = ET(s) + D(s)."""
    Sc, kc, kd, m, fc, P0, E00 = 300.0, 1.0, 0.03, 2.0, 0.5, 3.0, 4.0
    S, *_ = soil(np.full(3000, P0), np.full(3000, E00), Sc, kc, kd, m, fc, s0=0.1)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f = kc * E00 * min(1, mid / fc) + kd * Sc * mid ** m - P0
        lo, hi = (mid, hi) if f < 0 else (lo, mid)
    exact = 0.5 * (lo + hi) * Sc
    rel = abs(S[-1] - exact) / exact
    return {"name": "Steady rain and demand settle at the analytic equilibrium", "metric": f"relative error (analytic {exact:.1f} mm, model {S[-1]:.1f} mm)", "value": rel, "limit": 1e-4, "ok": rel < 1e-4, "note": "Solve P = ET(s) + D(s) for s and compare."}


def bench_linear_reservoir_recession() -> dict:
    k = 0.08
    Q = np.zeros(200); Q[0] = 100.0
    f = route(Q, np.zeros(200), 0.0, k, 1.0)
    # after the impulse, flow decays exactly as (1-k)^t
    ratio = f[60] / f[10]
    exact = (1 - k) ** 50
    rel = abs(ratio - exact) / exact
    return {"name": "Reservoir recession is exactly geometric", "metric": "relative error", "value": rel, "limit": 1e-9, "ok": rel < 1e-9, "note": "Flow after an impulse falls as (1 − k)^t."}


def run_benchmarks() -> list:
    return [bench_water_balance_closes(), bench_bounds_and_drydown(), bench_equilibrium(), bench_linear_reservoir_recession()]
