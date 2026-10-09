import json
from cri.llmtrain import mix as MX


def test_mix_ratio_and_unchanged_rows(tmp_path):
    for d, n, tag in (("a", 70, "N"), ("b", 100, "R")):
        (tmp_path / d).mkdir()
        for sp in ("train", "valid", "test"):
            (tmp_path / d / f"{sp}.jsonl").write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"{tag}{i}"}]}) + "\n" for i in range(n if sp == "train" else 10)))
    r = MX.mix(str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "o"), 0.3)
    assert r["train"] == {"new": 70, "replay": 30, "total": 100}
    rows = [json.loads(l)["messages"][0]["content"] for l in open(tmp_path / "o" / "train.jsonl")]
    assert sum(x.startswith("R") for x in rows) == 30 and sum(x.startswith("N") for x in rows) == 70
    assert all(json.loads(l)["messages"][0]["content"].startswith("N") for l in open(tmp_path / "o" / "test.jsonl"))
