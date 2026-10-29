"""Measure the resident intent model on REAL out-of-scope queries it was never trained on.

Source: CLINC150 (Larson et al., EMNLP 2019, CC BY 3.0) test split, out-of-scope rows, 185 kept (tests/data/clinc_oos_heldout.txt).
Rows whose subject overlaps ClimRisk's own domain (earthquake maps, global warming, local temperature trends, energy bills, outages) were
left out of this file by hand because a "correct" answer is arguable; they are listed in docs/RESIDENT_TRAINING.md.

  scope reply   = intent unknown AND offtopic flag  -> the agent says it is outside what it does   (best)
  asks          = intent unknown, no flag            -> the agent asks what is meant              (acceptable)
  wrong tool    = any other intent                   -> the agent would run a climate tool on an unrelated question (bad)
"""
import json
import pathlib
import sys
sys.path[:0] = ["src"]
from cri.resident import nlu as N                                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    qs = [l.strip() for l in (ROOT / "tests/data/clinc_oos_heldout.txt").read_text().splitlines() if l.strip()]
    m = N.get_model()
    scope = asks = 0
    wrong = []
    for q in qs:
        p = m.predict(q)
        if p["intent"] == "unknown":
            if p.get("offtopic"):
                scope += 1
            else:
                asks += 1
        else:
            wrong.append((p["intent"], round(p["confidence"], 2), q))
    n = len(qs)
    out = {"n": n, "scope_reply": scope, "asks": asks, "wrong_tool": len(wrong), "scope_rate": round(scope / n, 3), "wrong_rate": round(len(wrong) / n, 3)}
    print(json.dumps(out))
    for w in wrong[:40]:
        print("  wrong", w)


if __name__ == "__main__":
    main()
