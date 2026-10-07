"""Rainfall-runoff and extreme-value statistics used to drive the flood model."""
from __future__ import annotations

import math

import numpy as np


def scs_cn_runoff(p_mm: float, cn: float, ia_ratio: float = 0.2) -> float:
    """SCS Curve Number direct runoff depth (mm) for a storm total p_mm. Q = (P-Ia)^2/(P-Ia+S), S=25400/CN-254."""
    if p_mm <= 0 or cn <= 0:
        return 0.0
    s = 25400.0 / cn - 254.0
    ia = ia_ratio * s
    return 0.0 if p_mm <= ia else (p_mm - ia) ** 2 / (p_mm - ia + s)


def effective_rain_series(rain_mm_per_step, cn: float, ia_ratio: float = 0.2) -> np.ndarray:
    """Cumulative SCS-CN applied to a hyetograph: effective rain per step (mm); sums to scs_cn_runoff(total)."""
    r = np.asarray(rain_mm_per_step, float)
    cum = np.cumsum(r)
    q = np.array([scs_cn_runoff(c, cn, ia_ratio) for c in cum])
    return np.diff(np.concatenate([[0.0], q]))


def gumbel_fit(annual_maxima) -> tuple:
    """Method-of-moments Gumbel (EV1): returns (location mu, scale beta)."""
    x = np.asarray(annual_maxima, float)
    x = x[np.isfinite(x)]
    beta = float(np.std(x, ddof=1) * math.sqrt(6) / math.pi)
    mu = float(np.mean(x) - 0.5772156649 * beta)
    return mu, beta


def gumbel_quantile(mu: float, beta: float, return_period: float) -> float:
    return float(mu - beta * math.log(-math.log(1 - 1.0 / return_period)))


def block_hyetograph(total_mm: float, hours: float, step_s: float, peak_frac: float = 0.4) -> np.ndarray:
    """Triangular design hyetograph; returns mm per step, summing exactly to total_mm."""
    n = max(2, int(round(hours * 3600 / step_s)))
    t = (np.arange(n) + 0.5) / n
    w = np.where(t < peak_frac, t / peak_frac, (1 - t) / (1 - peak_frac))
    w = np.maximum(w, 0.02)
    return total_mm * w / w.sum()
