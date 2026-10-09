"""Build train/valid/test chat JSONL.  python -m cri.llmtrain.dataset CORPUS_DIR OUT_DIR

Kinds (all in the exact prompt formats the engine uses at run time):
  qa       numbered open-source passages (public-domain / permissive only) -> cited answer from the right passage, ignoring distractors
  refuse   passages that do not cover the question -> says what is missing, cites what the sources do cover
  insurer  engine facts string -> faithful narration / Q&A that copies numbers exactly, and declines what the facts do not contain
  gold     hand-written definitions and honest-behaviour answers (original text)
Split is by SOURCE for qa/refuse (a test source is never seen in training) and by seed for insurer.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

from . import facts as F
from . import gold as G

QA_SYS = ("You are ClimRisk's climate-risk analyst. Answer the question using ONLY the numbered sources. "
          "Cite every factual sentence with its source number in square brackets, like [2]. "
          "If the sources do not contain the answer, say exactly what is missing instead of guessing. "
          "Write 120 to 220 words in plain prose: lead with the answer, then the mechanism, then what it means for an asset owner or lender. No bullet lists.")
INS_SYS = ("You write for an insurance or risk manager. Using ONLY the facts given, answer in 70 to 140 words of plain prose. "
           "Copy every number exactly as written. Never introduce a number that is not in the facts. If the facts do not contain what is asked, say so and name what would be needed. "
           "State once that this is a screening model. No lists, no headings.")

STOP = set("""about above after again against also because been before being below between both cannot could does doing down during each from further have having here into just more most much only other over same should some such than that their them then there these they this those through under until very were what when where which while with would your will can may might must shall one two three four five six seven eight nine ten first second many several often usually typically however therefore including include includes used using use made make makes areas area region regions can't""".split())
TOK = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")


def _terms(t):
    return [w.lower() for w in TOK.findall(t) if w.lower() not in STOP]


def _sents(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])", t) if 7 <= len(s.split()) <= 60]


_DANGLING = re.compile(r"^(this|these|those|it|they|its|their|he|she|however|also|thus|therefore|moreover|additionally|furthermore|in addition|as a result|consequently|alternatively|but|and|so)\b", re.I)


def _h(s: str) -> float:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _split_of(key: str) -> str:
    x = _h(key)
    return "test" if x < 0.08 else "valid" if x < 0.16 else "train"


_BAD = re.compile(r"\||\bTable\b|\bFigure\b|\bClick\b|\bDownload\b|\bhttp|©|\*|\bPDF\b|\bhires\b|\bGIS\b", re.I)


def _clean_sents(t):
    out = []
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t):
        s = s.strip()
        w = s.split()
        if not (9 <= len(w) <= 45) or not s[0].isupper() or not s.endswith((".", "!", "?")) or _BAD.search(s):
            continue
        if sum(c.isdigit() for c in s) > 0.1 * len(s):
            continue
        out.append(s)
    return out


def load_passages(corpus_dir: Path):
    out = []
    for line in open(corpus_dir / "corpus.jsonl"):
        r = json.loads(line)
        if not r.get("train"):
            continue
        for p in r["passages"]:
            cs = _clean_sents(p)
            if len(cs) >= 3:
                out.append({"url": r["url"], "title": r["title"], "topic": r["topic"], "text": p, "sents": cs})
    return out


def _cands(t):
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{2,}|[.,;:!?()]", t.lower())
    out, prev = [], None
    for w in toks:
        if len(w) == 1 or w in STOP or len(w) < 4:
            prev = None
            continue
        out.append(w)
        if prev:
            out.append(prev + " " + w)
        prev = w
    return out


def _keyterms(passages):
    df = Counter()
    for p in passages:
        df.update(set(_cands(" ".join(p["sents"]))))
    N = len(passages)
    for p in passages:
        tf = Counter(_cands(" ".join(p["sents"])))
        sc = {}
        for t, c in tf.items():
            if df[t] > N * 0.12:
                continue
            sc[t] = (c + (1.5 if " " in t else 0)) * math.log(N / df[t])
        picked = []
        for t, _ in sorted(sc.items(), key=lambda x: -x[1]):
            if any(t in q or q in t for q in picked):
                continue
            picked.append(t)
            if len(picked) == 3:
                break
        p["keys"] = picked


Q_TEMPL = ["What do the sources say about {a} and {b}?", "Explain {a} and {b} for a risk manager.", "How do {a} and {b} matter for an asset owner?",
           "What should I know about {a} and {b}?", "What is known about {a} and {b}?"]
Q1_TEMPL = ["What do the sources say about {a}?", "What should I know about {a}?", "Explain {a} for a risk manager.", "Why does {a} matter for exposed assets?"]


def _question(rng, keys):
    keys = keys[:2] if rng.random() < 0.3 else keys[:1]
    if len(keys) >= 2:
        return rng.choice(Q_TEMPL).format(a=keys[0], b=keys[1]), keys
    return rng.choice(Q1_TEMPL).format(a=keys[0]), keys


def _compose(p, qterms, pos):
    sents = p["sents"]
    sc = []
    for i, s in enumerate(sents):
        low = s.lower()
        hit = sum(1 for t in qterms if t in low)
        sc.append((hit * 2 + (0.8 if i == 0 else 0) - 0.01 * len(s.split()), i, hit))
    best = sorted(sc, reverse=True)[:3]
    if not any(h for _, _, h in best):
        return ""
    pick = sorted(best, key=lambda x: x[1])
    return " ".join(f"{sents[i]} [{pos}]" for _, i, _ in pick)


def _sources_block(ps):
    return "\n\n".join(f"[{i + 1}] {p['title']}: {p['text']}" for i, p in enumerate(ps))


def build_qa(passages, rng, per=1):
    rows = []
    by_topic = {}
    for p in passages:
        by_topic.setdefault(p["topic"], []).append(p)
    for p in passages:
        if len(p["keys"]) < 1:
            continue
        for _ in range(per):
            q, qt = _question(rng, p["keys"] if rng.random() < 0.7 else rng.sample(p["keys"], min(2, len(p["keys"]))))
            pool = [x for x in (by_topic[p["topic"]] if rng.random() < 0.5 else passages) if x["url"] != p["url"]]
            ds = rng.sample(pool, min(len(pool), rng.randint(2, 4)))
            ctx = ds + [p]
            rng.shuffle(ctx)
            pos = ctx.index(p) + 1
            ans = _compose(p, qt, pos)
            if len(ans.split()) < 40:
                continue
            rows.append({"kind": "qa", "split": _split_of(p["url"]), "meta": {"target": pos, "n": len(ctx)},
                         "messages": [{"role": "system", "content": QA_SYS}, {"role": "user", "content": f"Question: {q}\n\nSources:\n{_sources_block(ctx)}"},
                                      {"role": "assistant", "content": ans}]})
    return rows


def build_refuse(passages, rng, n):
    rows = []
    for _ in range(n):
        p = rng.choice(passages)
        if not p["keys"]:
            continue
        q, qt = _question(rng, p["keys"][:2])
        pool = [x for x in passages if x["url"] != p["url"] and not (set(qt) & set(_terms(x["text"])))]
        if len(pool) < 3:
            continue
        ds = rng.sample(pool, rng.randint(2, 3))
        names = [d["title"].split(": ")[-1] for d in ds]
        a = " and ".join(f"{t} [{i + 1}]" for i, t in enumerate(names[:2]))
        what = " and ".join(qt)
        ans = rng.choice([
            f"The sources provided do not cover {what}. They deal with {a}, which is not enough to answer this without guessing. To answer it I would need a source that addresses {what} directly.",
            f"I cannot answer that from these sources: none of them discusses {what}. What they contain is {a}. A source on {what} is missing, so I will not guess.",
            f"These sources do not contain the answer. They are about {a}, not {what}. If you can point me to a document on {what}, I can answer from it."])
        rows.append({"kind": "refuse", "split": _split_of(p["url"]), "meta": {"n": len(ds)},
                     "messages": [{"role": "system", "content": QA_SYS}, {"role": "user", "content": f"Question: {q}\n\nSources:\n{_sources_block(ds)}"}, {"role": "assistant", "content": ans}]})
    return rows


# ── insurer narration ────────────────────────────────────────────────────────

def _g(res, rp, k="oep"):
    return F._row(res["stats"]["gross"]["rows"], rp).get(k, 0.0)


def _answer(res, qtype, rng):
    g = res["stats"]["gross"]
    tiv, aal = res["tiv"], g["aal"]
    pct = 100 * aal / tiv
    bp = res["stats"]["by_peril"]
    top = sorted([("wind", bp["wind"]), ("river flood", bp["river"]), ("coastal flood", bp["coast"])], key=lambda x: -x[1])
    s = max(res["sites"], key=lambda r: r["aal"])
    c = res.get("climate")
    ra = res.get("reinsurance") or {}
    dl = res["deductible_ladder"]
    SCR = "This is a screening model, not a vendor catastrophe model, and its damage curves are not calibrated to claims."
    if qtype == "summary":
        opening = rng.choice([f"Across {res['n_sites']} sites with {F.fm(tiv)} of insured value, the expected loss is {F.fm(aal)} a year, {pct:.2f}% of insured value.",
                              f"The schedule of {res['n_sites']} sites, insured for {F.fm(tiv)}, carries an average annual loss of {F.fm(aal)}, or {pct:.2f}% of value."])
        tail = rng.choice([f"A 1-in-100 year occurrence loss is {F.fm(_g(res, 100))}, and a 1-in-250 is {F.fm(_g(res, 250))}, so the tail is far heavier than the average suggests.",
                           f"In a 1-in-100 year the largest event costs {F.fm(_g(res, 100))}, rising to {F.fm(_g(res, 250))} at 1-in-250."])
        drv = f"{top[0][0].capitalize()} drives most of the average at {F.fm(top[0][1])}, and {s['name']} is the largest site at {F.fm(s['aal'])}."
        prem = f"The indicative technical premium is {F.fm(res['premium']['technical_total'])}, which is not a quote."
        extra = ""
        if c:
            extra = f" Under {c['ssp'].upper()} in {c['year']} the average loss moves {c['aal']['change_pct']:+.1f}% to {F.fm(c['aal']['scenario'])}."
        return f"{opening} {tail} {drv} {prem}{extra} {SCR}"
    if qtype == "tail":
        return (f"The tail is much heavier than the average: the 1-in-100 occurrence loss is {F.fm(_g(res, 100))} against an average annual loss of {F.fm(aal)}, "
                f"and tail value at risk at 99% is {F.fm(g['tvar99'])}. {s['name']} contributes the largest average loss at {F.fm(s['aal'])}, and {top[0][0]} is the peril behind most of the total at {F.fm(top[0][1])}. "
                f"Sites that share a storm or flood season tend to lose together, which widens the tail but not the average. {SCR}")
    if qtype == "deductible":
        return (f"Raising the deductible from {dl[0]['deductible_pct']:g}% to {dl[-1]['deductible_pct']:g}% of insured value lowers the average annual loss from {F.fm(dl[0]['aal'])} to {F.fm(dl[-1]['aal'])}, "
                f"but the 1-in-100 occurrence loss only moves from {F.fm(dl[0]['oep100'])} to {F.fm(dl[-1]['oep100'])}. A deductible removes many small claims, so it changes attritional cost far more than catastrophe capital. {SCR}")
    if qtype == "climate":
        if not c:
            return ("The facts do not include a climate-pathway run, so I cannot say how the losses change in a warmer climate. To answer it I would need the paired run for a chosen pathway, year and model range, "
                    "which the insurer view can produce with the same simulated years.")
        return (f"Under {c['ssp'].upper()} in {c['year']} (the {c['band']} model range) the average annual loss changes by {c['aal']['change_pct']:+.1f}% to {F.fm(c['aal']['scenario'])} from {F.fm(aal)}, "
                f"and the 1-in-100 occurrence loss changes by {c['tail']['oep100']['change_pct']:+.1f}% to {F.fm(c['tail']['oep100']['scenario'])}. The same random years are replayed with the pathway's hazard changes, so the difference is the climate effect. "
                f"A scenario is not a forecast, and no change in storm frequency is applied.")
    if qtype == "reinsurance":
        if not ra.get("ceded_aal"):
            return ("The facts do not include a reinsurance programme, so I cannot say what is ceded or what the net result is. To answer it I would need the quota share percentage and the excess-of-loss layers, "
                    "which can be entered in the insurer view.")
        return (f"The reinsurance programme cedes {F.fm(ra['ceded_aal'])} of average annual loss, leaving a net average annual loss of {F.fm(res['stats']['net']['aal'])} against {F.fm(aal)} gross. "
                f"Ceding lowers the tail as well as the average, but the cost of the cover is not in these facts, so the net benefit cannot be judged from them. {SCR}")
    if qtype == "premium":
        return (f"The indicative technical premium is {F.fm(res['premium']['technical_total'])}, or {res['premium']['rate_permille']:.1f} per mille of insured value. It is built from the {F.fm(aal)} average annual loss, "
                f"a load for model uncertainty, a cost of capital on the tail and an expense ratio, so it is a reference point and not a quote. A real price would also reflect reinsurance cost, competition and underwriting judgement. {SCR}")
    if qtype == "site":
        share = 100 * s["aal"] / max(aal, 1)
        return (f"The largest site by average annual loss is {s['name']} at {F.fm(s['aal'])}, about {share:.0f}% of the portfolio's {F.fm(aal)}. "
                f"That concentration matters for accumulation, because one event near that site can dominate a bad year. The facts do not break the other sites down, so I cannot rank them. {SCR}")
    raise ValueError(qtype)


Q = {"summary": ["Summarise this result.", "What does this schedule's risk look like?", "Give me the headline numbers and what they mean."],
     "tail": ["What drives the tail?", "How bad is a bad year compared with the average?", "Why is the 1-in-100 so much bigger than the average annual loss?"],
     "deductible": ["What does the deductible do here?", "How much does a higher deductible help?", "Would raising the deductible reduce our catastrophe exposure?"],
     "climate": ["How does climate change affect this?", "What happens to the losses under a warming scenario?", "How much worse does it get by mid-century?"],
     "reinsurance": ["What does reinsurance do to these numbers?", "What is our net position after the programme?"],
     "cost_unknown": [],
     "premium": ["What premium would we charge?", "Explain the technical premium.", "Is the technical premium a quote?"],
     "site": ["Which site matters most?", "Where is the concentration?"]}


UNKNOWN = [("What does the reinsurance cost us?", "the price of reinsurance", "a reinsurance quote for the layer in question"),
           ("What is the combined ratio on this book?", "the written premium, actual expenses or claims history", "the book's written premium, expenses and paid claims"),
           ("How much would claims cost inflation add?", "any claims-inflation or demand-surge assumption", "an inflation or demand-surge assumption to apply"),
           ("What is the earthquake exposure?", "any earthquake modelling, because the model covers cyclone wind, river flood and coastal flood only", "an earthquake model and exposure details"),
           ("What is the business interruption loss?", "a business-interruption loss, which is not modelled separately", "business-interruption values and indemnity periods"),
           ("Which site has the worst flood loss?", "a site-by-site flood breakdown beyond the largest site", "the per-site table from the insurer view"),
           ("What will the loss be next year?", "a forecast, since the model gives long-run averages and exceedance probabilities rather than predictions", "a forecast of next year's weather, which this model does not produce"),
           ("Is this portfolio profitable?", "the premiums actually charged or the insurer's costs", "the premiums, expenses and reinsurance cost for the book"),
           ("What are the hail and wildfire losses?", "any hail or wildfire loss, because those perils are not modelled", "a hail or wildfire model with exposure data")]


def build_insurer(rng, n, split):
    rows = []
    kinds = list(Q)
    for i in range(n):
        res = F.synth(rng)
        qt = kinds[i % len(kinds)]
        if qt == "cost_unknown" and rng.random() < 0.5:
            res.pop("reinsurance", None)
        fx = F.facts(res)
        if qt == "cost_unknown":
            question, missing, need = rng.choice(UNKNOWN)
            a = (f"The facts do not contain {missing}, so I cannot answer that from them. What they do show is a gross average annual loss of {F.fm(res['stats']['gross']['aal'])} "
                 f"and a 1-in-100 occurrence loss of {F.fm(_g(res, 100))}. To answer I would need {need}.")
        else:
            question = rng.choice(Q[qt])
            try:
                a = _answer(res, qt, rng)
            except Exception:
                continue
        rows.append({"kind": "insurer", "split": split, "meta": {"qtype": qt, "facts": fx},
                     "messages": [{"role": "system", "content": INS_SYS}, {"role": "user", "content": f"Facts: {fx}\n\nQuestion: {question}"}, {"role": "assistant", "content": a}]})
    return rows


def build_gold():
    rows = []
    for i, (topic, q, a) in enumerate(G.GOLD):
        split = "test" if i % 7 == 3 else "valid" if i % 7 == 5 else "train"
        rows.append({"kind": "gold", "split": split, "meta": {"topic": topic}, "messages": [{"role": "system", "content": G.GOLD_SYS}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    return rows


def main(corpus_dir: Path, out: Path, seed: int = 7):
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    ps = load_passages(corpus_dir)
    _keyterms(ps)
    rows = build_qa(ps, rng, per=3) + build_refuse(ps, rng, 360)
    rows += build_insurer(random.Random(seed + 1), 560, "train") + build_insurer(random.Random(seed + 2), 48, "valid") + build_insurer(random.Random(seed + 3), 64, "test")
    if "--v2" in sys.argv:
        from . import synth_v2
        v2 = synth_v2.build(seed=seed + 10)
        r2 = random.Random(seed + 11)
        for r in v2:
            x = r2.random()
            r["split"] = "test" if x < 0.08 else "valid" if x < 0.16 else "train"
        rows += v2
    if "--cases" in sys.argv:                                        # v3: real events, company financials and responses (verified facts only)
        from . import cases as CS
        vf = json.loads(Path(sys.argv[sys.argv.index("--cases") + 1]).read_text())
        rows += CS.build(vf, seed=seed + 20, reps=3)
    if "--edgar" in sys.argv:                                        # v3: real company disclosures from SEC filings (verbatim quotes)
        from . import edgar_cases as EC
        rows += EC.build_rows(Path(sys.argv[sys.argv.index("--edgar") + 1]), seed=seed + 30, per_case=2)
    gold = build_gold()
    rows += gold + [dict(r) for r in gold if r["split"] == "train"] * 2        # gold is scarce and high quality: upweight x3
    for r in rows:
        r["messages"][-1]["content"] = re.sub(r"\s+", " ", r["messages"][-1]["content"]).strip()
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
    for s, rs in splits.items():
        random.Random(seed).shuffle(rs)
        with open(out / f"{s}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        with open(out / f"{s}_meta.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps({"kind": r["kind"], "meta": r["meta"], "messages": r["messages"]}, ensure_ascii=False) + "\n")
    rep = {s: dict(Counter(r["kind"] for r in rs)) for s, rs in splits.items()}
    rep["passages"] = len(ps)
    rep["sources"] = len({p["url"] for p in ps})
    (out / "dataset_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
