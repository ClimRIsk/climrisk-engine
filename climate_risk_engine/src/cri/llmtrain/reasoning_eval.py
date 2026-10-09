"""
cri/llmtrain/reasoning_eval.py — an evaluation of reasoning that can fail.

Rule-following is easy to score; thinking is not. This scores what a model does when the right answer has to FOLLOW the inputs:
    base          the verdict (BREACH / PASS-TIGHT / PASS / NO-COVENANT) must equal what the engine's numbers imply, with no invented figures and every step grounded
    near_limit    the same, restricted to scenarios whose tightest covenant is within 8% of its limit (the comparison, not the topic, decides)
    sensitivity   the same scenario with ONE input changed (insurance, loss size, covenant limit): when the true verdict changes, the model's verdict must change with it
    ablation      the engine results are removed from the sources: the only right verdict is NOT-COMPUTABLE
    distractor    an unrelated case passage is added: the verdict must not move and the passage must not be cited
    ood           sizes, leverage and loss multiples outside the training ranges
    attribution   breaches caused by the loss versus breaches that already existed before it
Held-out rows come from precedent events never seen in training. Baselines are printed beside every model: always guessing the commonest verdict, and "ignore the change" (answer the base
scenario's verdict for its variant). A model that thinks must beat both. The rule-based reasoner (the teacher) is scored the same way as a ceiling, and must reach 1.0 or the verifier is wrong.

  python -m cri.llmtrain.reasoning_eval DS OUT.json MODEL [MODEL ...] [--limit 160] [--seed 3]
  python -m cri.llmtrain.reasoning_eval pick OUT.json [--home DIR] [--tolerance 0.02]
  python -m cri.llmtrain.reasoning_eval sheet OUT.json SHEET.md [--n 12]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .. import reasoning as R


def _chat(model: str, messages: list, deadline: float = 180.0) -> str:
    from ..resident import answer as AN
    body = json.dumps({"model": model, "stream": False, "keep_alive": "20m", "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 760}, "messages": messages}).encode()
    try:
        return AN._ollama_post("/api/chat", body, deadline).get("message", {}).get("content", "")
    except Exception as e:                                           # noqa: BLE001
        return f"ERROR {type(e).__name__}: {e}"


def load(ds: str) -> list:
    return [json.loads(l) for l in open(Path(ds) / "test_meta.jsonl")]


def select(rows: list, limit: int, seed: int) -> list:
    """A stratified sample that keeps each counterfactual pair together."""
    rng = random.Random(seed)
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[r["meta"]["pair"]].setdefault(r["meta"]["role"], r)
    pairs = [p for p in by_pair.values() if "base" in p and "variant" in p]
    abl = [p["ablation"] for p in by_pair.values() if "ablation" in p]
    dis = [p["distractor"] for p in by_pair.values() if "distractor" in p]
    ood = [r for r in rows if r["meta"].get("ood")]
    rng.shuffle(ood)
    near = [p["base"] for p in by_pair.values() if "base" in p and p["base"]["meta"]["min_headroom"] is not None and abs(p["base"]["meta"]["min_headroom"]) <= 8 and "variant" not in p]
    rng.shuffle(pairs), rng.shuffle(abl), rng.shuffle(dis), rng.shuffle(near)
    q = max(1, limit // 5)
    out = []
    for p in pairs[:q]:
        out += [p["base"], p["variant"]]
    out += abl[:q] + dis[:q] + near[:q] + ood[:q]
    seen, uniq = set(), []
    for r in out:
        k = (r["meta"]["pair"], r["meta"]["role"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def score_row(r: dict, answer: str) -> dict:
    m = r["meta"]
    v = R.verify(answer, m["sources"], m["expected"], None, m["irrelevant"], m["attribution"] if m["engine"] else None)
    return {"pair": m["pair"], "role": m["role"], "ood": bool(m.get("ood")), "expected": m["expected"], "got": v["verdict"], "ok": v["ok"], "checks": v["checks"], "problems": v["problems"], "min_headroom": m["min_headroom"],
            "perturbation": m["perturbation"], "answer": answer}


def aggregate(scored: list) -> dict:
    def rate(rows, f):
        return round(sum(1 for r in rows if f(r)) / len(rows), 3) if rows else None
    base = [r for r in scored if r["role"] in ("base", "variant")]
    out = {"n": len(scored)}
    out["verdict_accuracy"] = rate(base, lambda r: r["got"] == r["expected"])
    out["ood_verdict_accuracy"] = rate([r for r in base if r.get("ood")], lambda r: r["got"] == r["expected"])
    out["near_limit_verdict_accuracy"] = rate([r for r in base if r["min_headroom"] is not None and abs(r["min_headroom"]) <= 8], lambda r: r["got"] == r["expected"])
    out["ablation_not_computable"] = rate([r for r in scored if r["role"] == "ablation"], lambda r: r["got"] == "NOT-COMPUTABLE")
    dist = [r for r in scored if r["role"] == "distractor"]
    out["distractor_verdict_ok"] = rate(dist, lambda r: r["checks"].get("verdict_correct"))
    out["distractor_ignored"] = rate(dist, lambda r: r["checks"].get("ignores_irrelevant"))
    pairs = defaultdict(dict)
    for r in scored:
        if r["role"] in ("base", "variant"):
            pairs[r["pair"]][r["role"]] = r
    both = [p for p in pairs.values() if len(p) == 2]
    flips = [p for p in both if p["base"]["expected"] != p["variant"]["expected"]]
    out["sensitivity_pairs"] = len(both)
    out["sensitivity_flip_pairs"] = len(flips)
    out["sensitivity_follows_change"] = rate(flips, lambda p: p["base"]["got"] == p["base"]["expected"] and p["variant"]["got"] == p["variant"]["expected"])
    out["sensitivity_consistent_when_no_flip"] = rate([p for p in both if p not in flips], lambda p: p["base"]["got"] == p["base"]["expected"] and p["variant"]["got"] == p["variant"]["expected"])
    for name in ("format", "numbers_in_sources", "citations_exist", "facts_cited", "inferences_grounded", "attribution_correct"):
        rows = [r for r in scored if name in r["checks"]]
        out[name] = rate(rows, lambda r: r["checks"][name])
    out["all_checks_pass"] = rate(scored, lambda r: r["ok"])
    return out


def baselines(scored: list) -> dict:
    base = [r for r in scored if r["role"] in ("base", "variant")]
    c = Counter(r["expected"] for r in base)
    maj = c.most_common(1)[0][0] if c else None
    pairs = defaultdict(dict)
    for r in base:
        pairs[r["pair"]][r["role"]] = r
    var = [p for p in pairs.values() if len(p) == 2]
    flips = [p for p in var if p["base"]["expected"] != p["variant"]["expected"]]
    return {"always_the_commonest_verdict": {"verdict": maj, "verdict_accuracy": round(c[maj] / len(base), 3) if base else None},
            "ignore_the_change": {"sensitivity_follows_change": 0.0 if flips else None, "note": "answering the base verdict for the variant is wrong on every pair whose true verdict changes"},
            "verdict_counts": dict(c)}


def composite(a: dict) -> float:
    keys = ("verdict_accuracy", "near_limit_verdict_accuracy", "ood_verdict_accuracy", "ablation_not_computable", "distractor_ignored", "sensitivity_follows_change", "numbers_in_sources", "format", "attribution_correct")
    vals = [a[k] for k in keys if a.get(k) is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def run(ds: str, out: str, models: list, limit: int = 160, seed: int = 3, chat=_chat) -> dict:
    rows = select(load(ds), limit, seed)
    res = {"_meta": {"rows": len(rows), "seed": seed, "limit": limit}}
    oracle = [score_row(r, r["messages"][2]["content"]) for r in rows]
    res["oracle (rule-based teacher)"] = {"metrics": aggregate(oracle), "composite": composite(aggregate(oracle))}
    res["_baselines"] = baselines(oracle)
    for mdl in models:
        t0, scored = time.time(), []
        if chat is _chat:                                            # 16 GB: never keep two models resident while timing one (this caused the 15-minute stalls)
            try:
                from ..resident import answer as AN
                AN.unload_others(mdl)
            except Exception:                                        # noqa: BLE001
                pass
        for i, r in enumerate(rows):
            ans = chat(mdl, r["messages"][:2])
            scored.append(score_row(r, ans))
            print(f"{mdl} {i + 1}/{len(rows)}", flush=True)
        agg = aggregate(scored)
        res[mdl] = {"metrics": agg, "composite": composite(agg), "seconds": round(time.time() - t0), "failures": [{k: s[k] for k in ("pair", "role", "expected", "got", "problems")} for s in scored if not s["ok"]][:40],
                    "samples": [{"expected": s["expected"], "got": s["got"], "answer": s["answer"]} for s in scored[:6]]}
        Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def pick(eval_json: str, home: Path, tolerance: float = 0.02) -> dict:
    d = json.loads(Path(eval_json).read_text())
    sc = {k: v["composite"] for k, v in d.items() if not k.startswith("_") and not k.startswith("oracle") and isinstance(v, dict) and "composite" in v}
    if not sc:
        return {"changed": False, "why": "no model in the evaluation"}
    best = max(sc, key=sc.get)
    f, hist = Path(home) / "active_reasoning_model", Path(home) / "active_reasoning_model.json"
    cur = f.read_text().strip() if f.exists() else ""
    prev = json.loads(hist.read_text()) if hist.exists() else {}
    cur_score = sc.get(cur, prev.get("score"))
    if cur and cur != best and cur_score is not None and sc[best] < cur_score - tolerance:
        return {"changed": False, "active": cur, "why": f"{best} scored {sc[best]} against {cur_score}; kept {cur}", "scores": sc}
    Path(home).mkdir(parents=True, exist_ok=True)
    f.write_text(best + "\n")
    hist.write_text(json.dumps({"model": best, "score": sc[best], "scores": sc, "eval": str(eval_json)}, indent=1))
    return {"changed": best != cur, "active": best, "scores": sc}


def sheet(eval_json: str, out: str, n: int = 12) -> str:
    """A blind review sheet for risk professionals: scenario sources and an answer, no model name, with the questions a reviewer should answer."""
    d = json.loads(Path(eval_json).read_text())
    rng = random.Random(1)
    items = []
    for k, v in d.items():
        if k.startswith("_") or k.startswith("oracle") or not isinstance(v, dict):
            continue
        items += [(i, s) for i, s in enumerate(v.get("samples", []))]
    rng.shuffle(items)
    lines = ["# Blind review sheet: reasoning answers", "", "For each item: (1) does the VERDICT follow from the engine results in the sources? (2) is any statement misleading or unsupported? (3) would you act on it?  Score 1 (no) to 5 (yes) and note anything wrong.", ""]
    for j, (_, s) in enumerate(items[:n], 1):
        lines += [f"## Item {j}", "", "```", s["answer"][:2500], "```", "", "Verdict follows from the numbers (1-5): ___   Misleading or unsupported content (describe): ___   Would act on it (1-5): ___", ""]
    Path(out).write_text("\n".join(lines))
    return out


def report(eval_json: str, out: str) -> str:
    """Plain-language results: every model against the baselines and the teacher, with what the numbers do and do not show."""
    d = json.loads(Path(eval_json).read_text())
    models = {k: v for k, v in d.items() if not k.startswith("_") and not k.startswith("oracle") and isinstance(v, dict) and "metrics" in v}
    cols = [("verdict_accuracy", "verdict"), ("near_limit_verdict_accuracy", "near limit"), ("ood_verdict_accuracy", "out of range"), ("sensitivity_follows_change", "follows a change"),
            ("ablation_not_computable", "says not computable"), ("distractor_ignored", "ignores irrelevant"), ("numbers_in_sources", "no invented numbers"), ("all_checks_pass", "all checks")]
    f = lambda v: "n/a" if v is None else f"{100 * v:.0f}%"
    L = ["# Reasoning evaluation", "", f"{d['_meta']['rows']} held-out rows from precedent events never used in training, scored by code (no model grades another model).", "",
         "| model | composite | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for k, v in sorted(models.items(), key=lambda kv: -kv[1]["composite"]):
        L.append(f"| {k} | {v['composite']:.3f} | " + " | ".join(f(v["metrics"].get(c[0])) for c in cols) + " |")
    o = d.get("oracle (rule-based teacher)")
    if o:
        L.append(f"| rule-based teacher (ceiling) | {o['composite']:.3f} | " + " | ".join(f(o["metrics"].get(c[0])) for c in cols) + " |")
    b = d["_baselines"]
    L += ["", f"Baselines: always answering the commonest verdict ({b['always_the_commonest_verdict']['verdict']}) scores {f(b['always_the_commonest_verdict']['verdict_accuracy'])} on verdict accuracy; ignoring the change in a counterfactual pair scores 0% on following a change.", ""]
    if models:
        best = max(models, key=lambda k: models[k]["composite"])
        bm = models[best]["metrics"]
        maj = b["always_the_commonest_verdict"]["verdict_accuracy"] or 0
        L += [f"Best model: **{best}**. " + ("It beats the commonest-verdict baseline on verdict accuracy." if (bm.get("verdict_accuracy") or 0) > maj + 0.05 else "It does NOT clearly beat the commonest-verdict baseline, so it is not reasoning from the numbers.")
              + (" It follows a changed input in " + f(bm.get("sensitivity_follows_change")) + " of pairs whose true verdict changes." if bm.get("sensitivity_follows_change") is not None else ""), ""]
    L += ["What this does and does not show", "",
          "- The engine computes every figure. A model is scored on whether it reads those figures correctly, reaches the verdict they imply, refuses when the engine result is absent, and ignores a passage that does not bear on the question.",
          "- The reference answers come from a rule-based reasoner, so a model that matches them has learned the analysis procedure and its phrasing; this is not evidence of general intelligence or of new knowledge.",
          "- Scenarios are synthetic inputs paired with real precedent passages from only a few dozen curated events. Real covenant packages are more varied. A lender's or insurer's own calculation remains the authority.",
          "- At run time every chain is re-checked by the same code; a failing chain is replaced by the rule-based chain and labelled as such.",
          "- A blind review by risk professionals is still needed before anyone relies on the narrative: see the review sheet."]
    Path(out).write_text("\n".join(L) + "\n")
    return "\n".join(L)


def main() -> None:
    a = sys.argv[1:]
    if a and a[0] == "report":
        print(report(a[1], a[2]))
        return
    if a and a[0] == "pick":
        home = Path(a[a.index("--home") + 1]) if "--home" in a else Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))
        tol = float(a[a.index("--tolerance") + 1]) if "--tolerance" in a else 0.02
        print(json.dumps(pick(a[1], home, tol), indent=1))
        return
    if a and a[0] == "sheet":
        print(sheet(a[1], a[2], int(a[a.index("--n") + 1]) if "--n" in a else 12))
        return
    ds, out = a[0], a[1]
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else 160
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 3
    models = [x for i, x in enumerate(a[2:], 2) if not x.startswith("--") and a[i - 1] not in ("--limit", "--seed")]
    r = run(ds, out, models, limit, seed)
    print(json.dumps({k: (v["metrics"] if isinstance(v, dict) and "metrics" in v else v) for k, v in r.items() if not k.startswith("_")}, indent=1)[:4000])
    print(json.dumps(r["_baselines"], indent=1))


if __name__ == "__main__":
    main()
