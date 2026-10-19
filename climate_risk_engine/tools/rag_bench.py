"""Retrieval benchmark for the case library.

  python tools/rag_bench.py [--json OUT.json]

Set A (paraphrase): hand-written questions that avoid the event names; gold = the event(s) whose case card answers them. Split p1 (dev) / p2 (test).
Set B (entity): "What did <company> report after <event>?" for 60 company-event pairs drawn with a fixed seed; gold = a record of that company AND event.
Metrics at event level (A) or company+event level (B): hit@1, hit@3, hit@6, MRR. Also: routing recall = share of A questions that the chat router would send to the case library.
Reports are descriptive: A has 25 events x 2 phrasings, B is template-generated (optimistic, because the question names the entity).
"""
import json
import random
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from cri.resident import casebase as CB  # noqa: E402


def _metrics(ranks, n):
    def hit(k): return sum(1 for r in ranks if r is not None and r <= k) / n
    return {"n": n, "hit@1": round(hit(1), 3), "hit@3": round(hit(3), 3), "hit@6": round(hit(6), 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3)}


def _rank(passages, ok):
    for i, p in enumerate(passages, 1):
        if ok(p):
            return i
    return None


def run_a(retrieve=None):
    retrieve = retrieve or CB.retrieve
    qs = json.loads((ROOT / "rag_bench_questions.json").read_text())["paraphrase"]
    out = {}
    for split in ("p1", "p2"):
        ranks, route = [], 0
        for q in qs:
            ps = retrieve(q[split], 6)
            # event-level rank: the first passage whose event is gold (dedupe so repeated events count once)
            seen, r = [], None
            for p in ps:
                if p["event"] not in seen:
                    seen.append(p["event"])
                if p["event"] in q["gold"]:
                    r = len(seen)
                    break
            ranks.append(r)
            route += 1 if CB.is_case_question(q[split]) else 0
        m = _metrics(ranks, len(qs))
        m["routed_to_case_library"] = round(route / len(qs), 3)
        out[split] = m
    qj = json.loads((ROOT / "rag_bench_questions.json").read_text())
    neg = qj.get("not_case_questions", [])
    out["false_routes"] = {"n": len(neg), "routed": sum(1 for x in neg if CB.is_case_question(x))}
    return out


def run_b(retrieve=None, n=60, seed=7):
    retrieve = retrieve or CB.retrieve
    ix = CB._load()
    pairs = sorted({(r["event"], r["company"]) for r in ix["items"] if r["kind"] == "filing" and r.get("company") and r["event"] != "climate-reports"})
    random.Random(seed).shuffle(pairs)
    ranks = []
    for ev, co in pairs[:n]:
        name = ix["events"][ev]["name"]
        q = f"What did {co} report about {name}?"
        ps = retrieve(q, 6)
        ranks.append(_rank(ps, lambda p: p["event"] == ev and CB._co_norm(p.get("company") or "") == CB._co_norm(co)))
    return _metrics(ranks, len(ranks))


def main():
    res = {"A_paraphrase": run_a(), "B_entity": run_b()}
    print(json.dumps(res, indent=1))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
