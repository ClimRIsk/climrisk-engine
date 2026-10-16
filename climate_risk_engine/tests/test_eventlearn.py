"""Event-learning models: the tested claims must stay true, and the training rows must score 1.0 against their own reference answers."""
import json
import collections

from cri import eventlearn as EL
from cri import eventlearn_tc as TC
from cri.llmtrain import eventlearn_data as ED
from cri.llmtrain import evaluate as EV


def test_few_events_gives_no_analogs():
    rows = EL.load()
    assert EL.analogs("drought", rows=rows) is None            # fewer than MIN_EVENTS recorded events: "not enough", never a guess
    a = EL.analogs("tropical cyclone", rows=rows)
    assert a and a["n_events"] >= EL.MIN_EVENTS and a["p25_pct"] <= a["median_pct"] <= a["p75_pct"]


def test_company_revenue_has_no_hazard_skill():
    r = EL.loeo_eval()
    d, (lo, hi) = r["hazard_vs_all_events_mdae_diff"]
    assert not (hi < 0), "hazard type now improves the revenue prediction: update the dossier and the training rows"


def test_storm_model_beats_guessing_on_unseen_storms():
    r = TC.loo(True)
    assert r["median_factor_error"] < r["baseline_median_factor_error"]
    assert r["within_factor_3_pct"] > r["baseline_within_factor_3_pct"]
    assert r["rank_correlation_spearman"] > 0.5


def test_storm_prediction_rises_with_wind_and_range_brackets_it():
    a, b = TC.predict(70), TC.predict(130)
    assert b["expected_musd"] > a["expected_musd"]
    assert a["p10_musd"] < a["expected_musd"] < a["p90_musd"]


def test_training_rows_score_one_on_their_own_answers():
    rows = ED.build(8, 40)
    agg = collections.defaultdict(list)
    for r in rows:
        r["meta"].setdefault("nulls", [])
        for k, v in EV.score({"kind": r["kind"], "meta": r["meta"], "messages": r["messages"]}, r["messages"][2]["content"]).items():
            agg[k].append(v)
    assert all(abs(sum(v) / len(v) - 1.0) < 1e-9 for v in agg.values()), {k: sum(v) / len(v) for k, v in agg.items() if sum(v) / len(v) < 1}
