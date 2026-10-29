"""Retrain the resident agent's intent model and report before/after on the independent batteries.

  python tools/retrain_resident.py            # corpus + logged user corrections, writes src/cri/data/resident_nlu.json
  python tools/retrain_resident.py --dry      # measure only

Batteries: tests/nlu_battery.py (A, guided template writing) and tests/nlu_battery_b.py (B, clean). Add a real failing utterance to the right list in a battery only AFTER
recording it in docs/RESIDENT_TRAINING.md, and add a training template for the pattern (not the sentence) to src/cri/resident/corpus_more.py.
Logged corrections (the user picked an intent after the agent asked) are read from resident memory and added to the training rows.
"""
import json
import sys
sys.path[:0] = ["src", "tests"]
from cri.resident import nlu as N                                   # noqa: E402
import nlu_battery as A                                             # noqa: E402
import nlu_battery_b as B                                           # noqa: E402


def score(m, bat):
    ok = ask = wrong = 0
    miss = []
    for intent, qs in bat.items():
        for q in qs:
            g = m.predict(q)["intent"]
            if g == intent:
                ok += 1
            else:
                (ask := ask + 1) if g == "unknown" else (wrong := wrong + 1)
                miss.append((intent, g, q))
    n = ok + ask + wrong
    return {"n": n, "correct": ok, "accuracy": round(ok / n, 4), "asked": ask, "wrong": wrong}, miss


def main(dry=False):
    before = {k: score(N.get_model(), b)[0] for k, b in (("A", A.BATTERY), ("B", B.BATTERY))}
    print("before:", json.dumps(before))
    if dry:
        return
    extra = []
    try:
        from cri.resident import memory as M
        extra = [(r["text"], r["intent"]) for r in M.feedback_rows()] if hasattr(M, "feedback_rows") else []
    except Exception as e:                                          # noqa: BLE001
        print("no feedback rows read:", e)
    rep = N.train(extra=extra, save=True)
    print("corpus held-out templates:", rep["heldout_templates"], "grouped CV mean:", rep["grouped_cv"]["mean"], "feedback rows:", len(extra))
    after = {}
    for k, b in (("A", A.BATTERY), ("B", B.BATTERY)):
        s, miss = score(N.get_model(), b)
        after[k] = s
        for m in miss:
            print("  miss", k, m)
    print("after:", json.dumps(after))


if __name__ == "__main__":
    main("--dry" in sys.argv)
