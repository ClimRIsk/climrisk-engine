"""Event-trained terrain flood model: shipped parameters, ranking behaviour, and the benchmark's own metric."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from cri.analysis import flood_event_model as fem

ROOT = Path(__file__).resolve().parents[1]


def _auc():
    spec = importlib.util.spec_from_file_location("feb", ROOT / "scripts" / "flood_event_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.auc


def test_auc_metric_known_values():
    auc = _auc()
    y = np.array([0, 0, 1, 1])
    assert auc(np.array([1, 2, 3, 4.0]), y) == 1.0
    assert auc(np.array([4, 3, 2, 1.0]), y) == 0.0
    assert abs(auc(np.array([1, 1, 1, 1.0]), y) - 0.5) < 1e-9


def test_logistic_recovers_signal_and_is_weighted():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 4))
    y = (x[:, 0] * -1.5 + rng.normal(size=4000) > 0).astype(float)
    m = fem.fit_logistic(x, y, np.ones(4000), l2=1.0)
    assert m["coef"][0] < -0.5 and abs(m["coef"][1]) < 0.3


def test_shipped_model_exists_with_heldout_skill():
    m = fem.load()
    assert m and m["trained_on_all_events"] and m["n_events"] >= 50
    assert m["heldout"]["mean_auc"] >= 0.70                       # held-out events, never fitted on
    assert len(m["reference_quantiles"]) == 199


def test_flood_prone_position_outranks_high_ground():
    low = {"hand_m_at_0.1km2": 0.4, "hand_m_at_1.0km2": 0.8, "elev_percentile_in_window": 0.05, "elev_minus_window_p10_m": 1.0}
    high = {"hand_m_at_0.1km2": 30.0, "hand_m_at_1.0km2": 40.0, "elev_percentile_in_window": 0.95, "elev_minus_window_p10_m": 60.0}
    a, b = fem.site_score(low), fem.site_score(high)
    assert a["rank"] > 0.8 > 0.3 > b["rank"]


def test_incomplete_metrics_give_no_score():
    assert fem.site_score(None) is None
    assert fem.site_score({"hand_m_at_0.1km2": None, "hand_m_at_1.0km2": None, "elev_percentile_in_window": None}) is None


def test_benchmark_reports_exist_and_show_gain_over_old_rule():
    b = json.loads((ROOT / "data" / "backtest" / "flood_event_benchmark.json").read_text())
    g = b["heldout"]["mean_auc_[mean, lo95, hi95]"]["gain_vs_current_rule"]
    assert g[1] > 0.1                                             # lower 95% bound of the AUC gain
    h = json.loads((ROOT / "data" / "backtest" / "flood_history_benchmark.json").read_text())
    assert h["n_windows"] >= 200 and h["overall"]["event_model"][1] > 0.7
    assert abs(h["event_model_auc_vs_year_spearman"][0]) < 0.3     # no strong decay with event age


def test_flood_finding_adds_points_for_flood_like_terrain_and_stays_capped():
    from cri.analysis import asset_intelligence as ai
    hm = {"hand_m_at_0.1km2": 0.3, "hand_m_at_1.0km2": 0.6, "elev_percentile_in_window": 0.03, "elev_minus_window_p10_m": 0.8}
    f = ai._flood({}, {}, {}, None, None, hm)
    assert f["score"] == 2 and f["severity"] == "MODERATE"          # relative terrain evidence alone cannot exceed MODERATE
    assert any("flooded cells in real events" in e["metric"] for e in f["evidence"])
    dry = ai._flood({}, {}, {}, None, None, {"hand_m_at_0.1km2": 40, "hand_m_at_1.0km2": 50, "elev_percentile_in_window": 0.97, "elev_minus_window_p10_m": 80})
    assert dry["score"] == 0
