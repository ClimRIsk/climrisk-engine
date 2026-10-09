"""
cri/resident/demo_check.py — a scripted, mechanical check of the investor-demo path, run on the machine that will give the demo.

It asks the real answer path a fixed mix of questions built from the case index (event summaries, company filings, responses, precedents for an
exposure, a company asked about an event it never reported on, and questions outside the library) and records, for each:
  - which path answered (case library with the model, case library assembled, general knowledge, or declined)
  - seconds to answer
  - the mechanical honesty check against the evidence shown (every number in the sources, every [n] real, factual sentences cited)
Writes demo_report.json and prints a one-screen summary: pass rate, median and 95th percentile latency, anything that failed.

  python -m cri.resident.demo_check [OUT.json] [--n 24] [--seed 7]
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

from . import answer as AN
from . import casebase as CB

MISSING = ("do not cover", "does not cover", "not covered", "do not report", "no information", "would need", "could not find", "will not guess", "not in the sources", "do not hold")


def _questions(n: int, seed: int) -> list:
    ix = CB._load()
    rng = random.Random(seed)
    qs = []
    cards = list(ix["cards"].values())
    for fr in rng.sample(cards, min(len(cards), max(3, n // 6))):
        nm = fr["event_name"]
        qs.append(("event_summary", f"Summarise what happened to the finances in {nm}.", True))
        qs.append(("event_response", f"How did the companies and institutions respond in {nm}?", True))
    pairs = {}
    for r in ix["items"]:
        if r["kind"] == "filing" and r.get("company"):
            pairs.setdefault((r["event"], r["company"]), r)
    keys = list(pairs)
    for ev, co in rng.sample(keys, min(len(keys), max(4, n // 4))):
        qs.append(("company", f"What did {co} report about {pairs[(ev, co)]['event_name']}?", True))
    ev_names = {r["event"]: r["event_name"] for r in ix["items"]}
    for ev, co in rng.sample(keys, min(len(keys), max(3, n // 8))):
        other = rng.choice([e for e in ev_names if e != ev and (e, co) not in pairs])
        qs.append(("company_unknown", f"What did {co} report about {ev_names[other]}?", False))
    for q in ("Which past cases are closest to a flood at a chemical plant on a river, and what worked?",
              "What precedents exist for a wildfire hitting a utility's network, and how did the company cope?",
              "Our asset is a coastal refinery exposed to a hurricane. What do the cases suggest about how the loss reaches the financials?"):
        qs.append(("analog", q, True))
    for q in ("What did the 1998 North American ice storm cost Hydro-Quebec?", "How did the 2013 Colorado floods affect Anadarko's production guidance?"):
        qs.append(("outside_library", q, False))
    return qs[:max(n, 12)]


def _evaluate(kind: str, q: str, r: dict) -> dict:
    ev = r.get("evidence") or []
    ps = [{"id": e["n"], "text": e["text"]} for e in ev]
    chk = CB.check(r["reply"], ps) if ps else {"ok": None, "numbers_not_in_sources": [], "bad_citations": [], "uncited_sentences": [], "cited": []}
    says_missing = any(m in r["reply"].lower() for m in MISSING)
    if kind in ("company_unknown", "outside_library"):
        good = says_missing or not r.get("ok")
        # a general-knowledge answer that carries its own citations is acceptable for a question outside the case library
        good = good or (kind == "outside_library" and r.get("ok") and "[1]" in r["reply"])
    else:
        good = bool(r.get("ok")) and (chk["ok"] is not False)
    return {"kind": kind, "question": q, "ok": bool(good), "path": ("case/model" if r.get("case_mode") and "language model" in r.get("mode", "") else "case/assembled" if r.get("case_mode") else "general" if r.get("ok") else "declined"),
            "seconds": r.get("elapsed_s"), "numbers_not_in_sources": chk["numbers_not_in_sources"], "bad_citations": chk["bad_citations"], "says_missing": says_missing, "reply": r["reply"][:600]}


def run(out: Path = Path("demo_report.json"), n: int = 24, seed: int = 7, allow_learn: bool = False) -> dict:
    if not CB.available():
        raise SystemExit("no case index found: build it first (python -m cri.resident.casebase build VERIFIED.json EDGAR_CASES.jsonl)")
    warm = AN.warm(force=True)
    rows = []
    for kind, q, _expect in _questions(n, seed):
        t0 = time.time()
        try:
            if kind == "analog":
                r = AN.ask_analogs(q, assets=q)
            else:
                r = AN.ask_cases(q) or AN.ask(q, allow_learn=allow_learn)
        except Exception as e:                                       # noqa: BLE001
            r = {"ok": False, "reply": f"EXCEPTION {type(e).__name__}: {e}", "elapsed_s": round(time.time() - t0, 1)}
        row = _evaluate(kind, q, r)
        row["wall_s"] = round(time.time() - t0, 2)
        row["llm_timing"] = r.get("llm_timing")                      # where Ollama says the time went (load / prompt / generation)
        row["llm_note"] = r.get("llm_note")
        rows.append(row)
        print(f"{'PASS' if row['ok'] else 'FAIL'}  {row['wall_s']:6.2f}s  {row['path']:15s} {kind:16s} {q[:70]}", flush=True)
    secs = sorted(r["wall_s"] for r in rows)
    p95 = secs[min(len(secs) - 1, int(0.95 * len(secs)))]
    summary = {"questions": len(rows), "passed": sum(r["ok"] for r in rows), "pass_rate": round(sum(r["ok"] for r in rows) / max(1, len(rows)), 3),
               "median_s": round(statistics.median(secs), 2), "p95_s": p95, "max_s": secs[-1], "warm": warm, "index": CB.stats(),
               "by_kind": {k: f"{sum(r['ok'] for r in rows if r['kind'] == k)}/{sum(1 for r in rows if r['kind'] == k)}" for k in sorted({r['kind'] for r in rows})},
               "by_path": {k: sum(1 for r in rows if r["path"] == k) for k in sorted({r["path"] for r in rows})},
               "failures": [{"kind": r["kind"], "q": r["question"], "path": r["path"], "numbers_not_in_sources": r["numbers_not_in_sources"], "reply": r["reply"][:200]} for r in rows if not r["ok"]]}
    summary["slowest"] = [{"s": r["wall_s"], "q": r["question"][:80], "timing": r.get("llm_timing")} for r in sorted(rows, key=lambda r: -r["wall_s"])[:3]]
    Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))
    return summary


def main():
    a = sys.argv[1:]
    out = Path(a[0]) if a and not a[0].startswith("--") else Path("demo_report.json")
    n = int(a[a.index("--n") + 1]) if "--n" in a else 24
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 7
    run(out, n, seed)


if __name__ == "__main__":
    main()
