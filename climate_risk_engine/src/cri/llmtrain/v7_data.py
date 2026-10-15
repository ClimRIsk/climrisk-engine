"""
cri/llmtrain/v7_data.py — training data for the v7 round: only the NEW task families, every answer composed from the engine (no text copied from outside).

    python -m cri.llmtrain.v7_data OUT_DIR [--seed 7] [--stress 900] [--no-scenarios]

Families: stress (linked hazard/plant/supplier/freight/tariff/cash runs of the real stresstest module), scenario (scenario list -> engine warming facts, split by distinct scenario),
plan_review (supplied plan -> review that states only what the supplied facts show), profile (site hazard profile with null semantics), chat (routing), gold (hand-written reference answers).
Writes train/valid/test.jsonl, *_meta.jsonl (same layout as dataset.main, so evaluate.py scores them) and v7_report.json.
Replay of the earlier tasks is added afterwards by cri.llmtrain.mix, so v7 does not forget them.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from . import dataset as D
from . import gold as G


def build(seed: int = 7, n_stress: int = 900, scenarios: bool = True, repo: Path | None = None):
    repo = repo or Path(__file__).resolve().parents[3]
    rows = []
    rows += D.build_stress(random.Random(seed + 90), n_stress, "train") + D.build_stress(random.Random(seed + 91), max(24, n_stress // 8), "valid") + D.build_stress(random.Random(seed + 92), max(40, n_stress // 6), "test")
    rows += D.build_profile(random.Random(seed + 70), 300, "train") + D.build_profile(random.Random(seed + 71), 30, "valid") + D.build_profile(random.Random(seed + 72), 40, "test")
    rows += D.build_chat(random.Random(seed + 80), 300, "train") + D.build_chat(random.Random(seed + 81), 30, "valid") + D.build_chat(random.Random(seed + 82), 40, "test")
    scn = repo / "data" / "scenarios" / "climate_risk_scenarios_1000.csv"
    if scenarios and scn.exists():
        rows += D.build_scenarios_csv(scn, seed + 100)
        ans = scn.with_name("climate_risk_answers_1000.csv")
        if ans.exists():
            rows += D.build_plan_review(scn, ans, seed + 100)
    gold = D.build_gold()
    rows += gold + [dict(r) for r in gold if r["split"] == "train"] * 2
    for r in rows:
        r["messages"][-1]["content"] = re.sub(r"\s+", " ", r["messages"][-1]["content"]).strip()
    return rows


def write(rows, out: Path, seed: int = 7):
    out.mkdir(parents=True, exist_ok=True)
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
    seen = {r["messages"][1]["content"] for r in splits["train"]}                      # a held-out prompt that also appears in training would flatter the score: drop it from valid/test
    for s in ("valid", "test"):
        splits[s] = [r for r in splits[s] if r["messages"][1]["content"] not in seen]
    for s, rs in splits.items():
        random.Random(seed).shuffle(rs)
        with open(out / f"{s}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        with open(out / f"{s}_meta.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps({"kind": r["kind"], "meta": r["meta"], "messages": r["messages"]}, ensure_ascii=False) + "\n")
    rep = {s: dict(Counter(r["kind"] for r in rs)) for s, rs in splits.items()}
    # leakage guard: no held-out prompt may also be a training prompt
    tr = {r["messages"][1]["content"] for r in splits["train"]}
    rep["test_prompts_also_in_train"] = sum(r["messages"][1]["content"] in tr for r in splits["test"])
    (out / "v7_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main():
    a = sys.argv[1:]
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 7
    n = int(a[a.index("--stress") + 1]) if "--stress" in a else 900
    rows = build(seed, n, "--no-scenarios" not in a)
    print(json.dumps(write(rows, Path(a[0]), seed), indent=1))


if __name__ == "__main__":
    main()
