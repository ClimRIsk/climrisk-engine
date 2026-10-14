"""
cri/analysis/flood_event_model.py — terrain flood susceptibility fitted on OBSERVED flood extents.

Why this exists
---------------
The first event benchmark (scripts/flood_event_benchmark.py: Copernicus EMS observed flood polygons, 125 windows,
72 real events 2023–2026) showed that the old terrain rule (built-up land below 3 m / 6 m) has no skill inland
(mean per-window AUC 0.54), while height above the nearest drainage (HAND) and local relative elevation do
(AUC ≈ 0.70 and ≈ 0.77). This module is the small, interpretable model fitted on those events.

It ranks places WITHIN a flood-affected landscape (which cells are the ones that go under); it is not a
return-period or depth model and says nothing about whether a random place floods at all.

Four inputs, all available for any point from `terrain.hand.hand_metrics`:
  hand_0.1      height above nearest channel draining ≥ 0.1 km²  (m)
  hand_1.0      same for ≥ 1 km²                                   (m)
  elev_pct      share of the 8 km window that lies below the site (0–1)
  rel_p10       site elevation minus the window's 10th percentile  (m)

Fitted by L2-regularised logistic regression with every window carrying equal total weight (so large windows do not
dominate). The weights and their held-out skill are stored in `src/cri/data/flood_event_model.json` (written by
`scripts/flood_event_benchmark.py --stage fit`) and read back here; nothing is fitted at run time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

FEATURES = ("log_hand_0.1", "log_hand_1.0", "elev_pct", "log_rel_p10")
_PARAMS = Path(__file__).resolve().parents[1] / "data" / "flood_event_model.json"


def design(hand01, hand1, elev_pct, rel_p10) -> np.ndarray:
    """Feature matrix (n × 4) from raw arrays or scalars. HAND missing → large (undrained cells are high ground)."""
    h01 = np.nan_to_num(np.asarray(hand01, float), nan=99.0)
    h1 = np.nan_to_num(np.asarray(hand1, float), nan=99.0)
    ep = np.nan_to_num(np.asarray(elev_pct, float), nan=0.5)
    rp = np.clip(np.nan_to_num(np.asarray(rel_p10, float), nan=0.0), 0.0, 200.0)
    return np.column_stack([np.log1p(np.clip(h01, 0, 99)), np.log1p(np.clip(h1, 0, 99)), ep, np.log1p(rp)])


def cells_design(a: dict) -> np.ndarray:
    """Design matrix for a stored benchmark window (dict of equal-length arrays)."""
    e = np.asarray(a["elev"], float)
    p10 = np.percentile(e, 10)
    return design(a["hand_0.1"], a["hand_1.0"], a["elev_pct"], e - p10)


def fit_logistic(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None, l2: float = 1.0, iters: int = 50) -> dict:
    """Weighted L2 logistic regression by Newton–Raphson. Features are standardised; returns raw-scale coefficients."""
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    w = w / w.sum() * n
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(n), (X - mu) / sd])
    beta = np.zeros(k + 1)
    pen = np.diag([0.0] + [l2] * k)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -30, 30)))
        g = Z.T @ (w * (p - y)) + pen @ beta
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + pen + 1e-8 * np.eye(k + 1)
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    coef = beta[1:] / sd
    return {"features": list(FEATURES), "coef": coef.tolist(), "intercept": float(beta[0] - np.sum(beta[1:] * mu / sd)), "l2": l2}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    z = X @ np.asarray(model["coef"]) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def load() -> Optional[dict]:
    """The shipped fitted model, or None if the benchmark has not been fitted yet."""
    try:
        return json.loads(_PARAMS.read_text())
    except Exception:                                               # noqa: BLE001
        return None


def site_score(hm: Optional[dict]) -> Optional[dict]:
    """
    Terrain flood susceptibility of one site from `terrain.hand.hand_metrics(lat, lon, radius_km=4.0)`, or None if
    no fitted model is shipped or the metrics are incomplete.

    `rank` (0–1) = share of benchmark cells (half flooded, half not, from 72 real events) that look LESS flood-like
    than this site. It is a relative terrain position, not a probability of flooding or a depth.
    """
    m = load()
    if not m or not hm:
        return None
    h01, h1, ep = hm.get("hand_m_at_0.1km2"), hm.get("hand_m_at_1.0km2"), hm.get("elev_percentile_in_window")
    if ep is None or (h01 is None and h1 is None):
        return None
    h01 = 99.0 if h01 is None else h01
    h1 = 99.0 if h1 is None else h1
    p = float(predict(m["model"], design(h01, h1, ep, hm.get("elev_minus_window_p10_m") or 0.0))[0])
    q = np.asarray(m.get("reference_quantiles") or [])
    rank = float(np.searchsorted(q, p) / len(q)) if q.size else None
    return {"probability": round(p, 4), "rank": None if rank is None else round(rank, 3),
            "heldout_auc": (m.get("heldout") or {}).get("mean_auc"), "n_events": m.get("n_events"),
            "hand_m_0.1": hm.get("hand_m_at_0.1km2"), "hand_m_1.0": hm.get("hand_m_at_1.0km2"), "elev_percentile": ep}
