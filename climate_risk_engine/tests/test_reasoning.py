"""The reasoning layer: engine arithmetic, rule-based chain, and a verifier that fails wrong answers."""
import json
import random

import pytest

from cri import reasoning as R
from cri.llmtrain import reasoning_data as RD


def _scn(**over):
    s = {"name": "Testco plc", "hazard": "flood", "assets": "a chemical plant on a river",
         "fin": {"revenue": 1000.0, "ebitda": 200.0, "net_debt": 500.0, "interest_expense": 40.0, "equity": 600.0, "total_assets": 1500.0},
         "loss": {"asset_damage": 150.0, "insured_recovery": 30.0, "bi_revenue_lost": 50.0},
         "cov": {"max_net_debt_to_ebitda": 3.0, "min_interest_cover": 4.0}, "analogs": [{"text": "Event X (case summary): the driver was a river flood; the exposed assets were chemical plants."}]}
    for k, v in over.items():
        s[k] = v
    return s


def _run(scn, engine=True, distractors=None):
    led = R.ledger(scn)
    S, ids = R.sources(scn, led, engine, distractors)
    ans = R.compose(scn, led, ids, ["lost revenue while the site is closed"], engine, "Event X")
    return led, S, ids, ans


def test_oracle_chain_passes_every_check_and_matches_the_engine():
    led, S, ids, ans = _run(_scn())
    v = R.verify(ans, S, led["verdict"], ids, [], {"new": [R.COV_NAMES[k] for k in led["breach_new"]], "pre": [R.COV_NAMES[k] for k in led["breach_pre"]]})
    assert v["ok"], v
    assert led["verdict"] in R.VERDICTS and f"VERDICT: {led['verdict']}" in ans
    assert "STEP 1 [FACT]" in ans and "[PRECEDENT]" in ans
    # the arithmetic is the engine's: uninsured 120, EBITDA hit 50 x 20% = 10, pre-tax loss 130
    assert "uninsured asset loss 120.0" in S[4] and "EBITDA hit 10.0" in S[4] and "pre-tax loss 130.0" in S[4]


def test_the_verifier_fails_each_kind_of_wrong_answer():
    led, S, ids, ans = _run(_scn())
    ev = led["verdict"]
    assert not R.verify(ans.replace(f"VERDICT: {ev}", "VERDICT: PASS" if ev != "PASS" else "VERDICT: BREACH"), S, ev)["checks"]["verdict_correct"]
    assert not R.verify(ans.replace("120.0", "999.0", 1), S, ev)["checks"]["numbers_in_sources"]
    assert not R.verify(ans.replace("[5]", "[99]", 1), S, ev)["checks"]["citations_exist"]
    bad = ans.replace("INFERENCE from 2", "INFERENCE", 1)
    assert not R.verify(bad, S, ev)["checks"]["inferences_grounded"]
    assert not R.verify(ans.replace(" [5].", ".", 1), S, ev)["checks"]["facts_cited"]
    assert not R.verify("The loss is probably fine.", S, ev)["checks"]["format"]


def test_missing_engine_result_must_give_not_computable():
    scn = _scn()
    led, S, ids, ans = _run(scn, engine=False)
    assert R.verify(ans, S, R.expected_verdict(led, False), ids)["ok"] and "VERDICT: NOT-COMPUTABLE" in ans
    led2, S2, ids2, ans2 = _run(scn)
    wrong = R.verify(ans2, S, R.expected_verdict(led, False), ids)        # a full analysis offered where the results were withheld
    assert not wrong["ok"]


def test_an_irrelevant_passage_must_not_be_cited():
    scn = _scn()
    led = R.ledger(scn)
    d = [{"text": "Another event (case summary): a wildfire destroyed a town."}]
    S, ids = R.sources(scn, led, True, d)
    ans = R.compose(scn, led, ids, None, True, "Event X")
    assert R.verify(ans, S, led["verdict"], ids, ids["distractor"])["ok"]
    cheat = ans.replace("[8].", f"[8][{ids['distractor'][0]}].", 1)
    assert not R.verify(cheat, S, led["verdict"], ids, ids["distractor"])["checks"]["ignores_irrelevant"]


def test_attribution_new_versus_pre_existing_breach():
    scn = _scn(cov={"max_net_debt_to_ebitda": 2.0})                     # leverage was 2.5x before the loss: already in breach
    led = R.ledger(scn)
    assert led["breach_pre"] == ["net_debt_to_ebitda"] and not led["breach_new"]
    S, ids = R.sources(scn, led)
    ans = R.compose(scn, led, ids, None, True, "Event X")
    assert "already in breach before the loss" in ans
    attr = {"new": [], "pre": ["net debt/EBITDA"]}
    assert R.verify(ans, S, "BREACH", ids, [], attr)["ok"]
    swapped = ans.replace("was already in breach before the loss", "is pushed over")
    assert not R.verify(swapped, S, "BREACH", ids, [], attr)["checks"]["attribution_correct"]
    scn2 = _scn(cov={"max_net_debt_to_ebitda": 3.0})                    # 2.5x before, ~3.1x after: the loss causes the breach
    led2 = R.ledger(scn2)
    assert led2["verdict"] == "BREACH" and led2["breach_new"] == ["net_debt_to_ebitda"]


def test_counterfactual_changes_the_engine_verdict_in_the_right_direction():
    base = _scn(cov={"max_net_debt_to_ebitda": 3.5})
    ins = _scn(cov={"max_net_debt_to_ebitda": 3.5}, loss={"asset_damage": 150.0, "insured_recovery": 0.0, "bi_revenue_lost": 50.0})
    assert R.ledger(ins)["tr"]["after"]["net_debt"] > R.ledger(base)["tr"]["after"]["net_debt"]       # less insurance -> more debt
    tight = _scn(cov={"max_net_debt_to_ebitda": 2.2})
    assert R.ledger(tight)["verdict"] == "BREACH" and R.ledger(base)["verdict"] != "BREACH"


def test_generated_dataset_is_balanced_and_every_gold_chain_verifies(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("CRI_CASE_INDEX", os.path.join(os.path.dirname(R.__file__), "resident", "data", "case_index.jsonl"))
    st = RD.build(str(tmp_path), n=150, seed=2)
    assert st["train"]["rows"] > 100 and {"BREACH", "PASS", "PASS-TIGHT", "NO-COVENANT"} <= set(st["train"]["verdicts"])
    bad = 0
    for sp in ("train", "valid", "test"):
        for line in open(tmp_path / f"{sp}_meta.jsonl"):
            r = json.loads(line)
            m = r["meta"]
            v = R.verify(r["messages"][2]["content"], m["sources"], m["expected"], None, m["irrelevant"], m["attribution"] if m["engine"] else None)
            bad += 0 if v["ok"] else 1
    assert bad == 0
    assert any(json.loads(l)["kind"] == "reason_ablation" for l in open(tmp_path / "train_meta.jsonl"))


def test_split_is_by_precedent_event(tmp_path, monkeypatch):
    import os
    monkeypatch.setenv("CRI_CASE_INDEX", os.path.join(os.path.dirname(R.__file__), "resident", "data", "case_index.jsonl"))
    RD.build(str(tmp_path), n=300, seed=4)
    ev = {sp: {json.loads(l)["meta"]["analog_event"] for l in open(tmp_path / f"{sp}_meta.jsonl")} - {None} for sp in ("train", "valid", "test")}
    assert not (ev["train"] & ev["test"]) and not (ev["train"] & ev["valid"])
