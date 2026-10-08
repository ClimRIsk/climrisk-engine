"""Shared event-rate tools: Poisson hindcast tests used by the cyclone and earthquake studies."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def poisson_two_sided(k_obs: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k_obs == 0 else 0.0
    lo = float(stats.poisson.cdf(k_obs, lam))
    hi = float(stats.poisson.sf(k_obs - 1, lam))
    return float(min(1.0, 2 * min(lo, hi)))


def rate_hindcast(counts_by_year: np.ndarray, frac: float = 0.67, labels=("level",)) -> dict:
    """Estimate the annual rate from the first `frac` of the years and predict the count in the rest."""
    n = len(counts_by_year)
    cut = int(round(n * frac))
    tr, te = counts_by_year[:cut], counts_by_year[cut:]
    lam = float(tr.mean()) * len(te)
    k = int(te.sum())
    return {"n_train": cut, "n_test": len(te), "rate_train": float(tr.mean()), "rate_test": float(te.mean()), "expected": lam, "observed": k, "p": poisson_two_sided(k, lam)}


def dispersion(counts) -> float:
    """Variance-to-mean ratio of annual counts (1 for a Poisson process; clustering gives > 1)."""
    m = float(np.mean(counts))
    return float(np.var(counts, ddof=1) / m) if m > 0 and len(counts) > 2 else float("nan")


def pot_fit(peaks, u: float):
    """Peaks-over-threshold exponential tail: returns (scale, n_exceed)."""
    ex = np.asarray([p - u for p in peaks if p > u], float)
    return (float(ex.mean()) if len(ex) else float("nan")), int(len(ex))


def pot_level(u: float, scale: float, rate_per_year: float, T: float) -> float:
    return float(u + scale * math.log(max(rate_per_year * T, 1.0)))
