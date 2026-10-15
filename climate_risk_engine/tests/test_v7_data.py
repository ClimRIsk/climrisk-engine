import json
import random

from cri.llmtrain import v7_data as V7
from cri.llmtrain import v7_report as R


def test_v7_data_shape_and_no_leak(tmp_path):
    rows = V7.build(seed=3, n_stress=40)
    rep = V7.write(rows, tmp_path, 3)
    assert rep["test_prompts_also_in_train"] == 0
    for k in ("stress", "siteprofile", "chat", "scenario", "gold"):
        assert rep["train"].get(k, 0) > 0, k
    tr = [json.loads(l) for l in open(tmp_path / "train_meta.jsonl")]
    te = [json.loads(l) for l in open(tmp_path / "test_meta.jsonl")]
    assert te and all(r["messages"][-1]["role"] == "assistant" for r in tr + te)
    # held-out scenario combinations are not training combinations
    combos = lambda rs: {tuple(r["meta"]["combo"]) for r in rs if r["kind"] == "scenario"}      # noqa: E731
    assert not (combos(tr) & combos(te))


def test_gate_blocks_regression(tmp_path):
    def w(n, d):
        p = tmp_path / n
        p.write_text(json.dumps(d))
        return str(p)
    new = w("n.json", {"a": {"metrics": {"stress.numbers_ok": 0.9}}, "b": {"metrics": {"stress.numbers_ok": 0.5}}})
    ok_r = w("r.json", {"a": {"composite": 0.9}, "b": {"composite": 0.9}})
    bad_r = w("r2.json", {"a": {"composite": 0.5}, "b": {"composite": 0.9}})
    g = w("g.json", {"a": {"metrics": {"case.numbers_ok": 0.9}}, "b": {"metrics": {"case.numbers_ok": 0.9}}})
    assert R.gate(new, ok_r, g, "a", "b")["promote"] is True
    res = R.gate(new, bad_r, g, "a", "b")
    assert res["promote"] is False and "reasoning" in res["why"]
