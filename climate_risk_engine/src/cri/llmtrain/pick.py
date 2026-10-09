"""
cri/llmtrain/pick.py — promote a fine-tuned model to be the one the app uses, only if it did not get worse.

Reads a held-out evaluation (evaluate.py output), scores every climrisk-llm* model on the honesty metrics that matter for the demo
(no invented numbers, valid and sufficient citations, says so when the sources do not cover the question, labels inference), and writes the
winner's name to ~/.climrisk/active_model. A new model replaces the current one only if its score is at least the current model's score
minus a small tolerance, so a retrain can never silently make the app worse.

  python -m cri.llmtrain.pick EVAL.json [--home DIR] [--tolerance 0.01]
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

GOOD = re.compile(r"(numbers_ok|no_invented_numbers|cites_valid|cites_needed|cites_all_needed|sent_cited|says_missing|flags_inference|hedges_inference|refuses|declines)")
BAD = re.compile(r"invented_numbers$")


def composite(metrics: dict) -> tuple:
    good = [v for k, v in metrics.items() if GOOD.search(k) and isinstance(v, (int, float))]
    bad = [v for k, v in metrics.items() if BAD.search(k) and "no_invented" not in k and isinstance(v, (int, float))]
    score = (sum(good) / len(good) if good else 0.0) - 0.1 * (sum(bad) / len(bad) if bad else 0.0)
    return round(score, 4), len(good)


def pick(eval_json: Path, home: Path, tolerance: float = 0.01) -> dict:
    d = json.loads(Path(eval_json).read_text())
    scores = {m: composite(v["metrics"]) for m, v in d.items() if m.startswith("climrisk-llm")}
    if not scores:
        return {"changed": False, "why": "no climrisk-llm model in the evaluation"}
    best = max(scores, key=lambda m: scores[m][0])
    f = Path(home) / "active_model"
    hist = Path(home) / "active_model.json"
    prev = json.loads(hist.read_text()) if hist.exists() else {}
    cur = f.read_text().strip() if f.exists() else ""
    cur_score = scores[cur][0] if cur in scores else prev.get("score")
    if cur and cur_score is not None and scores[best][0] < cur_score - tolerance and cur != best:
        return {"changed": False, "active": cur, "why": f"{best} scored {scores[best][0]} against {cur_score} for the current model; kept {cur}", "scores": scores}
    Path(home).mkdir(parents=True, exist_ok=True)
    f.write_text(best + "\n")
    hist.write_text(json.dumps({"model": best, "score": scores[best][0], "scores": scores, "eval": str(eval_json)}, indent=1))
    return {"changed": best != cur, "active": best, "scores": scores}


def main():
    a = sys.argv[1:]
    home = Path(a[a.index("--home") + 1]) if "--home" in a else Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))
    tol = float(a[a.index("--tolerance") + 1]) if "--tolerance" in a else 0.01
    print(json.dumps(pick(Path(a[0]), home, tol), indent=1))


if __name__ == "__main__":
    main()
