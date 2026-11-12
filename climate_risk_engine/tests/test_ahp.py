import numpy as np
import pytest

from cri import ahp


def test_saaty_textbook_example():
    A = np.array([[1, 3, 5], [1 / 3, 1, 3], [1 / 5, 1 / 3, 1]])
    r = ahp.weights(A)
    assert r["weights"] == pytest.approx([0.637, 0.258, 0.105], abs=0.002)
    assert r["lambda_max"] == pytest.approx(3.039, abs=0.002)
    assert r["cr"] == pytest.approx(0.033, abs=0.003) and r["consistent"]


def test_perfectly_consistent_matrix_has_zero_ratio():
    w = np.array([0.5, 0.3, 0.2])
    A = np.outer(w, 1 / w)
    r = ahp.weights(A)
    assert r["cr"] == pytest.approx(0.0, abs=1e-9) and r["weights"] == pytest.approx(w, abs=1e-9)


def test_no_judgments_is_not_assessed_not_a_default():
    r = ahp.criteria_weights(["flood", "heat", "wind"], None)
    assert r["status"] == "not_assessed" and "weights" not in r


def test_missing_or_bad_judgments_are_rejected():
    assert ahp.criteria_weights(["a", "b", "c"], {"a|b": 3})["status"] == "invalid"            # a vs c and b vs c missing
    assert ahp.criteria_weights(["a", "b"], {"a|b": 12})["status"] == "invalid"                # outside the scale
    assert ahp.criteria_weights(["a", "b"], {"a|b": 3, "b|a": 1 / 3})["status"] == "invalid"   # pair twice
    assert ahp.criteria_weights(["a", "a"], {"a|a": 1})["status"] == "invalid"


def test_inconsistent_judgments_are_flagged_and_the_culprit_named():
    j = {"a|b": 9, "b|c": 9, "a|c": 1 / 9}                                                     # a >> b >> c but c >> a
    r = ahp.criteria_weights(["a", "b", "c"], j)
    assert r["status"] == "inconsistent" and r["consistency_ratio"] > 0.10 and "Revisit" in r["advice"]


def test_two_items_have_no_consistency_ratio_and_reciprocal_weights():
    r = ahp.criteria_weights(["x", "y"], {"x|y": 3})
    assert r["status"] == "ok" and r["consistency_ratio"] is None
    assert r["weights"]["x"] == pytest.approx(0.75, abs=1e-3)


def test_group_geometric_mean():
    g = ahp.group_judgments([{"a|b": 2}, {"a|b": 8}])
    assert g[("a", "b")] == pytest.approx(4.0)
    g2 = ahp.group_judgments([{"a|b": 2}, {"b|a": 0.125}])                                     # second expert wrote the reverse pair: a is 8 times b
    assert g2[("a", "b")] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        ahp.group_judgments([{"a|b": 2}, {"a|c": 3}])


def test_rank_orders_by_weighted_score_and_reports_stability():
    alts = {"site A": {"flood": 0.9, "heat": 0.2}, "site B": {"flood": 0.1, "heat": 0.8}, "site C": {"flood": 0.5, "heat": 0.5}}
    r = ahp.rank(alts, {"flood": 0.8, "heat": 0.2})
    assert [x["name"] for x in r["ranking"]][0] == "site A"
    assert r["ranking"][0]["is_first_share"] > 0.9 and "robust" in r["stability_note"]
    close = ahp.rank(alts, {"flood": 0.5, "heat": 0.5})
    assert close["ranking"][0]["is_first_share"] < 0.9 and "depends on the weights" in close["stability_note"]


def test_rank_never_treats_a_missing_score_as_zero_and_respects_cost_criteria():
    r = ahp.rank({"a": {"x": 1.0, "y": None}, "b": {"x": 2.0, "y": 1.0}}, {"x": 0.5, "y": 0.5})
    assert r["status"] == "not_assessed" and "not treated as zero" in r["reason"]
    c = ahp.rank({"cheap": {"cost": 1.0}, "dear": {"cost": 9.0}}, {"cost": 1.0}, benefit={"cost": False})
    assert c["ranking"][0]["name"] == "cheap"


def test_endpoint_roundtrip():
    from fastapi.testclient import TestClient
    from cri.api.main import app
    c = TestClient(app)
    r = c.post("/ahp", json={"criteria": ["flood", "heat"], "judgments": {"flood|heat": 3}, "alternatives": {"A": {"flood": 1, "heat": 0}, "B": {"flood": 0, "heat": 1}}})
    if r.status_code in (401, 403):                       # a deployment that requires a key
        pytest.skip("endpoint needs an API key in this environment")
    assert r.status_code == 200
    body = r.json()
    assert body["weights"]["status"] == "ok" and body["ranking"]["ranking"][0]["name"] == "A"
    n = c.post("/ahp", json={"criteria": ["flood", "heat"]})
    assert n.json()["weights"]["status"] == "not_assessed" and "ranking" not in n.json()
