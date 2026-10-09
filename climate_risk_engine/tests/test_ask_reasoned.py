import json, os
import cri.resident.answer as AN
import cri.reasoning as R

SC = {"name": "Harbor Chemicals", "hazard": "flood", "assets": "a chemical plant on a river",
      "fin": {"revenue": 1200, "ebitda": 240, "net_debt": 600, "interest_expense": 40, "equity": 500, "total_assets": 1500},
      "loss": {"asset_damage": 300, "insured_recovery": 120, "bi_revenue_lost": 200, "extra_costs": 10}, "cov": {"max_net_debt_to_ebitda": 3.5, "min_interest_cover": 4.0}}


def _idx(monkeypatch):
    monkeypatch.setenv("CRI_CASE_INDEX", os.path.join(os.path.dirname(AN.__file__), "data", "case_index.jsonl"))
    monkeypatch.setenv("CRI_LLM", "0")
    monkeypatch.setenv("CRI_HOME", "/tmp/cri_home_test")


def test_rule_based_chain_passes_every_check(monkeypatch):
    _idx(monkeypatch)
    r = AN.ask_reasoned(SC, use_llm=False)
    assert r["ok"] and r["checks_ok"] and r["verdict"] == R.ledger({**SC})["verdict"]
    assert "Step 1" in r["reply"] and "VERDICT" in r["reply"]


def test_model_chain_with_wrong_verdict_is_discarded(monkeypatch):
    _idx(monkeypatch)
    good = AN.ask_reasoned(SC, use_llm=False)["reply"].replace("Step ", "STEP ")
    bad = good.replace("VERDICT: " + good.split("VERDICT: ")[1].split()[0], "VERDICT: PASS")
    monkeypatch.setattr(AN, "_ollama_post", lambda *a, **k: {"message": {"content": bad}})
    r = AN.ask_reasoned(SC, model="x")
    assert r["checks_ok"] and "rule-based" in r["mode"] and "failed the checks" in (r["llm_note"] or "")


def test_model_chain_that_passes_is_used(monkeypatch):
    _idx(monkeypatch)
    good = AN.ask_reasoned(SC, use_llm=False)["reply"].replace("Step ", "STEP ")
    monkeypatch.setattr(AN, "_ollama_post", lambda *a, **k: {"message": {"content": good}})
    r = AN.ask_reasoned(SC, model="x")
    assert "language model" in r["mode"] and r["checks_ok"]


def test_insurer_result_and_missing_inputs(monkeypatch):
    _idx(monkeypatch)
    sc = {k: v for k, v in SC.items() if k != "loss"}
    r = AN.ask_reasoned({**sc, "insurer": {"gross_loss": 300, "insured_loss": 120}}, use_llm=False)
    assert r["ok"] and r["summary"]["pre_tax_loss"] > 0
    assert not AN.ask_reasoned({"name": "x"}, use_llm=False)["ok"]
    assert not AN.ask_reasoned({**SC, "fin": {"revenue": 5}}, use_llm=False)["ok"]
