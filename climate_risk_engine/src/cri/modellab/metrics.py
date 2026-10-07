"""Skill metrics used by every study. Plain NumPy; each is tested against hand-computed values."""
from __future__ import annotations

import numpy as np


def _clean(obs, sim):
    o, s = np.asarray(obs, float).ravel(), np.asarray(sim, float).ravel()
    m = np.isfinite(o) & np.isfinite(s)
    return o[m], s[m]


def rmse(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(np.sqrt(np.mean((s - o) ** 2))) if o.size else float("nan")


def mae(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(np.mean(np.abs(s - o))) if o.size else float("nan")


def bias(obs, sim) -> float:
    o, s = _clean(obs, sim)
    return float(np.mean(s - o)) if o.size else float("nan")


def nmb(obs, sim) -> float:
    """Normalised mean bias (fraction): sum(sim-obs)/sum(obs)."""
    o, s = _clean(obs, sim)
    return float(np.sum(s - o) / np.sum(o)) if o.size and np.sum(o) != 0 else float("nan")


def pearson_r(obs, sim) -> float:
    o, s = _clean(obs, sim)
    if o.size < 3 or np.std(o) == 0 or np.std(s) == 0:
        return float("nan")
    return float(np.corrcoef(o, s)[0, 1])


def nse(obs, sim) -> float:
    """Nash–Sutcliffe efficiency: 1 is perfect, 0 is no better than the mean of the observations."""
    o, s = _clean(obs, sim)
    d = np.sum((o - o.mean()) ** 2) if o.size else 0
    return float(1 - np.sum((s - o) ** 2) / d) if d > 0 else float("nan")


def kge(obs, sim) -> float:
    """Kling–Gupta efficiency (2009)."""
    o, s = _clean(obs, sim)
    if o.size < 3 or np.std(o) == 0 or np.mean(o) == 0:
        return float("nan")
    r = pearson_r(o, s)
    if not np.isfinite(r):
        return float("nan")
    a, b = np.std(s) / np.std(o), np.mean(s) / np.mean(o)
    return float(1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2))


def fac2(obs, sim) -> float:
    """Fraction of predictions within a factor of two of the observations (air-quality standard)."""
    o, s = _clean(obs, sim)
    m = o > 0
    if not m.any():
        return float("nan")
    ratio = s[m] / o[m]
    return float(np.mean((ratio >= 0.5) & (ratio <= 2.0)))


def skill_vs(obs, sim, baseline) -> float:
    """1 - RMSE(sim)/RMSE(baseline). Positive = the model beats the baseline."""
    b = rmse(obs, baseline)
    return float(1 - rmse(obs, sim) / b) if b and np.isfinite(b) else float("nan")


def contingency(obs_mask, sim_mask) -> dict:
    o, s = np.asarray(obs_mask, bool).ravel(), np.asarray(sim_mask, bool).ravel()
    hits, miss, fa = int(np.sum(o & s)), int(np.sum(o & ~s)), int(np.sum(~o & s))
    csi = hits / (hits + miss + fa) if (hits + miss + fa) else float("nan")
    f1 = 2 * hits / (2 * hits + miss + fa) if (2 * hits + miss + fa) else float("nan")
    return {"hits": hits, "misses": miss, "false_alarms": fa, "csi": csi, "f1": f1,
            "pod": hits / (hits + miss) if (hits + miss) else float("nan"), "far": fa / (hits + fa) if (hits + fa) else float("nan")}
