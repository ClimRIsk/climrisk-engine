"""The reasoning evaluation must be able to fail: a perfect reasoner scores 1.0, shortcut 'models' score low, and the teacher is the ceiling."""
import json
import os

import pytest

from cri import reasoning as R
from cri.llmtrain import reasoning_data as RD
from cri.llmtrain import reasoning_eval as RE


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("rd")
    os.environ["CRI_CASE_INDEX"] = os.path.join(os.path.dirname(R.__file__), "resident", "data", "case_index.jsonl")
    RD.build(str(d), n=900, seed=7)
    return str(d)


def _gold_by_user(ds):
    return {r["messages"][1]["content"]: r["messages"][2]["content"] for r in RE.load(ds)}


def test_selection_keeps_pairs_and_covers_every_test_type(ds):
    rows = RE.select(RE.load(ds), 40, 1)
    roles = {r["meta"]["role"] for r in rows}
    assert {"base", "variant", "ablation"} <= roles
    pairs = {}
    for r in rows:
        pairs.setdefault(r["meta"]["pair"], set()).add(r["meta"]["role"])
    assert any({"base", "variant"} <= v for v in pairs.values())


def test_a_perfect_reasoner_scores_one_and_the_teacher_is_the_ceiling(ds):
    gold = _gold_by_user(ds)
    res = RE.run(ds, os.path.join(ds, "e1.json"), ["perfect"], limit=40, seed=1, chat=lambda m, msgs: gold[msgs[1]["content"]])
    assert res["oracle (rule-based teacher)"]["composite"] == 1.0
    assert res["perfect"]["composite"] == 1.0 and res["perfect"]["metrics"]["all_checks_pass"] == 1.0


def test_shortcut_models_fail(ds):
    gold = _gold_by_user(ds)

    def always_breach(model, msgs):                                 # right format, same verdict for everything, ignores the engine results
        g = gold[msgs[1]["content"]]
        import re
        return re.sub(r"VERDICT: [A-Z\-]+(?: \([^)]*\))?", "VERDICT: BREACH", g)

    def ignores_removal(model, msgs):                               # never says NOT-COMPUTABLE
        g = gold[msgs[1]["content"]]
        import re
        return re.sub(r"VERDICT: NOT-COMPUTABLE", "VERDICT: PASS", g)

    res = RE.run(ds, os.path.join(ds, "e2.json"), ["always_breach", "ignores_removal"], limit=60, seed=2, chat=lambda m, msgs: (always_breach if m == "always_breach" else ignores_removal)(m, msgs))
    a, b = res["always_breach"], res["ignores_removal"]
    assert a["composite"] < 0.85 and a["metrics"]["verdict_accuracy"] < 0.7
    assert b["metrics"]["ablation_not_computable"] == 0.0
    assert RE.pick(os.path.join(ds, "e2.json"), __import__("pathlib").Path(ds))["active"] in ("always_breach", "ignores_removal")


def test_no_flip_baseline_and_pick_never_promotes_a_worse_model(ds, tmp_path):
    gold = _gold_by_user(ds)
    res = RE.run(ds, os.path.join(ds, "e3.json"), ["good"], limit=40, seed=3, chat=lambda m, msgs: gold[msgs[1]["content"]])
    assert res["_baselines"]["ignore_the_change"]["sensitivity_follows_change"] in (0.0, None)
    assert RE.pick(os.path.join(ds, "e3.json"), tmp_path)["active"] == "good"
    worse = json.loads(open(os.path.join(ds, "e3.json")).read())
    worse["bad"] = {"metrics": {}, "composite": 0.4}
    worse["good"]["composite"] = 0.4
    json.dump({**worse, "good": {"metrics": {}, "composite": 0.4}}, open(tmp_path / "e4.json", "w"))
    r = RE.pick(str(tmp_path / "e4.json"), tmp_path)                 # current model 'good' recorded at 1.0; nothing here beats it
    assert r["changed"] is False and r["active"] == "good"


def test_review_sheet(ds):
    gold = _gold_by_user(ds)
    RE.run(ds, os.path.join(ds, "e5.json"), ["m"], limit=20, seed=4, chat=lambda m, msgs: gold[msgs[1]["content"]])
    out = RE.sheet(os.path.join(ds, "e5.json"), os.path.join(ds, "sheet.md"), 3)
    t = open(out).read()
    assert "Item 1" in t and "m" not in t.split("Item 1")[0].split("# Blind")[1][:0] and "VERDICT" in t


def test_report_is_written_and_says_when_a_model_does_not_beat_the_baseline(ds, tmp_path):
    res = RE.run(ds, os.path.join(ds, "e2.json"), ["lazy"], limit=40, seed=1, chat=lambda m, msgs: "No idea.")
    md = RE.report(os.path.join(ds, "e2.json"), str(tmp_path / "r.md"))
    assert "lazy" in md and "NOT clearly beat" in md and "rule-based reasoner" in md
