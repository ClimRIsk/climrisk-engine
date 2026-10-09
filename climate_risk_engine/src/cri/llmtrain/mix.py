"""
cri/llmtrain/mix.py — mix a new task's training rows with replay rows from the previous model's data, so learning the new task does not erase the old one.

    python -m cri.llmtrain.mix NEW_DS REPLAY_DS OUT_DS [--replay 0.3] [--seed 1] [--test-max 300]

OUT_DS/train.jsonl = all NEW train rows + enough REPLAY train rows that replay is `--replay` of the total; valid likewise; test = the NEW task's test rows (what we measure).
Rows are copied unchanged. Writes OUT_DS/mix_report.json.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def _rows(p: Path) -> list:
    return [l for l in open(p) if l.strip()] if p.exists() else []


def mix(new_ds: str, replay_ds: str, out_ds: str, replay: float = 0.3, seed: int = 1, test_max: int = 300) -> dict:
    rng = random.Random(seed)
    out = Path(out_ds)
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for sp in ("train", "valid"):
        a, b = _rows(Path(new_ds) / f"{sp}.jsonl"), _rows(Path(replay_ds) / f"{sp}.jsonl")
        want = int(round(len(a) * replay / (1 - replay))) if replay < 1 else len(b)
        b = rng.sample(b, min(want, len(b)))
        rows = a + b
        rng.shuffle(rows)
        (out / f"{sp}.jsonl").write_text("".join(r if r.endswith("\n") else r + "\n" for r in rows))
        rep[sp] = {"new": len(a), "replay": len(b), "total": len(rows)}
    t = _rows(Path(new_ds) / "test.jsonl")[:test_max]
    (out / "test.jsonl").write_text("".join(r if r.endswith("\n") else r + "\n" for r in t))
    rep["test"] = {"new": len(t)}
    (out / "mix_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main() -> None:
    a = sys.argv[1:]
    r = mix(a[0], a[1], a[2], float(a[a.index("--replay") + 1]) if "--replay" in a else 0.3, int(a[a.index("--seed") + 1]) if "--seed" in a else 1, int(a[a.index("--test-max") + 1]) if "--test-max" in a else 300)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
