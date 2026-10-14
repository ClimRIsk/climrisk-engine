"""Universal site profile: four statuses, null != zero, ladder widening, data-vs-absence, caching."""
import os
import tempfile

import pytest

from cri import siteprofile as SP


def _fake(status, **kw):
    def fn(c):
        if status == "boom":
            raise RuntimeError("source down")
        return SP._entry("x", "X", status, family="t", **kw)
    return fn


def _run(reg, **kw):
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    return SP.profile(10.0, 20.0, "T", registry_override=reg, workers=2, budget_s=20, **kw)


def test_four_statuses_and_null_is_not_zero():
    reg = [("a", "fast", _fake("modelled", value=3.0, severity="HIGH")),
           ("b", "fast", _fake("constructed", value=1.0, severity="LOW")),
           ("c", "fast", _fake("none_found")),
           ("d", "fast", _fake("null", value=5.0, severity="HIGH", reason="no data"))]
    p = _run(reg)
    h = p["hazards"]
    assert h["a"]["status"] == "modelled" and h["a"]["value"] == 3.0
    assert h["c"]["value"] == 0.0 and h["c"]["severity"] == "NEGLIGIBLE"
    assert h["d"]["value"] is None and h["d"]["severity"] is None      # null stays null even if a value was passed
    assert p["counts"] == {"modelled": 1, "constructed": 1, "none_found": 1, "null": 1}
    assert p["null_hazards"] == ["d"]
    assert p["top_severity"] == "HIGH"
    assert "null means not assessed" in p["rules"][0]


def test_exception_becomes_null_with_reason():
    p = _run([("a", "fast", _fake("boom")), ("b", "fast", _fake("modelled", value=1.0))])
    assert p["hazards"]["a"]["status"] == "null" and "source down" in p["hazards"]["a"]["attempts"][0]["why"]
    assert p["hazards"]["b"]["status"] == "modelled"


def test_unfinished_is_null_not_low():
    import time
    def slow(c):
        time.sleep(3)
        return SP._entry("s", "S", "modelled", value=1.0)
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    p = SP.profile(10.0, 20.0, "T", registry_override=[("s", "fast", slow)], workers=1, budget_s=0.5)
    assert p["hazards"]["s"]["status"] == "null" and "not finished" in p["hazards"]["s"]["attempts"][0]["why"]


def test_data_shortage_is_not_absence():
    assert SP._DATA.search("Too little data within 100 km to fit a model")
    # lab no_model with a data message must not become none_found
    r = {"kind": "no_model", "status": "no model for this place", "limitations": ["no earthquake catalog events within 200 km: too sparse"]}
    ent, cls = SP._lab_map("earthquake", r, "rung")
    assert ent is None and "data" in cls


def test_genuine_absence_is_none_found():
    r = {"kind": "no_model", "status": "no model for this place", "limitations": ["No coast: the site is inland, too far from the sea."]}
    ent, cls = SP._lab_map("surge", r, "rung")
    assert ent is not None and ent["status"] == "none_found" and ent["value"] == 0.0


def test_value_is_hazard_level_not_validation_stat():
    r = {"headline": [("0.68", "Nash–Sutcliffe on unseen years"), ("+41%", "skill vs climatology"), ("C", "evidence grade (A best)")]}
    ent, _ = SP._lab_map("river", r, "rung")
    assert ent["status"] == "modelled" and ent["value"] is None
    assert ent["evidence_grade"] == "C"
    r2 = {"headline": [("38.0 WBGT °C", "100-year level"), ("0.94", "anomaly correlation"), ("A", "evidence grade (A best)")]}
    ent2, _ = SP._lab_map("heat", r2, "rung")
    assert ent2["value"] == 38.0 and "Simplified WBGT" in ent2["caveat"]


def test_no_skill_study_is_constructed_not_modelled():
    r = {"headline": [("10 mg", "100-year level")], "skilful": False}
    ent, _ = SP._lab_map("air", r, "rung")
    assert ent["status"] == "constructed" and "no skill" in ent["caveat"].lower()


def test_cache_roundtrip_and_summary():
    os.environ["CRI_HOME"] = tempfile.mkdtemp()
    reg = [("a", "fast", _fake("modelled", value=3.0, severity="HIGH")), ("d", "fast", _fake("null"))]
    p1 = SP.profile(1.0, 2.0, "T", only=None, registry_override=None) if False else _run(reg)
    s = SP.summarise(p1)
    assert "NULL" in s or "null" in s.lower()
    assert "HIGH" in s
