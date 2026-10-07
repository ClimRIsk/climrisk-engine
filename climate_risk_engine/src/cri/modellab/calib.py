"""Calibration, split-sample validation and sensitivity — generic, model-agnostic."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def least_squares_fit(residual: Callable, x0, bounds, restarts: int = 4, seed: int = 11) -> dict:
    """Bounded non-linear least squares with random restarts (SciPy). Returns best parameters and the fit RMS."""
    from scipy.optimize import least_squares
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds[0], float), np.array(bounds[1], float)
    starts = [np.clip(np.array(x0, float), lo, hi)] + [lo + rng.random(len(lo)) * (hi - lo) for _ in range(restarts - 1)]
    best = None
    for s in starts:
        try:
            r = least_squares(residual, s, bounds=(lo, hi), x_scale=np.maximum(np.abs(s), 1e-6))
        except Exception:                                          # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("calibration failed to converge from any start")
    n = max(len(best.fun), 1)
    return {"x": best.x.tolist(), "rms": float(math.sqrt(2 * best.cost / n)), "n_obs": n, "at_bound": [bool(np.isclose(v, l) or np.isclose(v, h)) for v, l, h in zip(best.x, lo, hi)]}


def grid_search(run: Callable[[float], float], values, score: Callable[[float], float] = lambda e: e):
    """1-D search for the value with the lowest score(error). run(value) must return an error (lower = better)."""
    errs = [(float(v), float(run(v))) for v in values]
    best = min(errs, key=lambda t: score(t[1]))
    return {"best": best[0], "error": best[1], "trials": errs}


def oat_sensitivity(run: Callable[[dict], float], base: dict, ranges: dict) -> list:
    """
    One-at-a-time sensitivity. run(params)→scalar output. ranges: {name: (low, high)}.
    Returns, per parameter: the output at low / base / high, the swing, and a normalised elasticity
    (% change in output per % change in the parameter, over the tested range). Sorted by swing.
    """
    y0 = float(run(dict(base)))
    rows = []
    for k, (lo, hi) in ranges.items():
        p = dict(base); p[k] = lo; ylo = float(run(p))
        p = dict(base); p[k] = hi; yhi = float(run(p))
        b = float(base[k])
        d_in = (hi - lo) / b if b else float("nan")
        el = ((yhi - ylo) / y0) / d_in if y0 and d_in else float("nan")
        rows.append({"param": k, "low": lo, "high": hi, "y_low": ylo, "y_base": y0, "y_high": yhi, "swing": abs(yhi - ylo), "elasticity": el})
    return sorted(rows, key=lambda r: -r["swing"])


def aic(n: int, rss: float, k: int) -> float:
    """Akaike information criterion for least-squares fits (small-sample corrected)."""
    n = max(n, k + 2)
    a = n * math.log(max(rss, 1e-12) / n) + 2 * k
    return a + (2 * k * (k + 1)) / (n - k - 1)
