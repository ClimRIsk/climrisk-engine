"""
cri/reasoning.py — structured, checkable reasoning from a loss scenario to the financial statements and covenants.

What this is, and is not
    The engine does the arithmetic (cri.riskmgmt.translate on the loss from the insurer run). The model's job is to READ those results and the cited case passages and write a short chain of steps:
        STEP n [FACT]            a statement taken from a numbered source, cited [k]
        STEP n [INFERENCE from i, j]   a conclusion that follows from earlier steps (no new numbers)
        STEP n [PRECEDENT]       what a past case shows, cited
        STEP n [CAVEAT]          a limit or assumption, cited
        VERDICT: BREACH | PASS-TIGHT | PASS | NO-COVENANT | NOT-COMPUTABLE
        CHECK NEXT: ...          NOT IN THE SOURCES: ...
    Every part of that is verified by code (verify()): numbers must be in the sources, citations must exist, inferences must point at earlier steps, the verdict must equal what the engine's
    numbers imply, a missing engine result must give NOT-COMPUTABLE, an irrelevant passage must not be cited. So the language model can phrase and select, but cannot change the truth.
    compose() writes the same chain by rules (the teacher for training data, and the fallback shown when a model answer fails the checks).

Scenario = {"name", "hazard", "assets", "fin": {...}, "loss": {...}, "cov": {...}, "analogs": [passage, ...]}   (money in millions of one currency)
"""
from __future__ import annotations

import re
from typing import Optional

from . import riskmgmt as RM

VERDICTS = ("BREACH", "PASS-TIGHT", "PASS", "NO-COVENANT", "NOT-COMPUTABLE")

SYS_REASON = ("You are ClimRisk's climate-risk analyst. Reason step by step using ONLY the numbered sources. "
              "Write numbered steps: 'STEP n [FACT]' states a fact from a source and cites it like [3]; 'STEP n [INFERENCE from i, j]' draws a conclusion from earlier steps and adds no new numbers; "
              "'STEP n [PRECEDENT]' says what a past case shows and cites it; 'STEP n [CAVEAT]' states a limit or assumption and cites it. "
              "Do not do arithmetic: use the engine results in the sources. Then write 'VERDICT:' followed by one of BREACH, PASS-TIGHT, PASS, NO-COVENANT or NOT-COMPUTABLE, "
              "then 'CHECK NEXT:' and 'NOT IN THE SOURCES:'. If the sources hold no engine result for the loss, the verdict is NOT-COMPUTABLE. Never add a number that is not in the sources, and never cite a source that does not bear on the step.")

COV_NAMES = {"net_debt_to_ebitda": "net debt/EBITDA", "interest_cover": "EBITDA/interest", "equity_ratio": "equity/total assets"}
_COV_LABEL = {"Net debt / EBITDA": "net_debt_to_ebitda", "EBITDA / interest": "interest_cover", "Equity / total assets": "equity_ratio"}

# ── formatting (one place, so the sources and the answer always show a number the same way) ────────────────────────────────────

def m(v) -> str:
    return f"{v:,.1f}"


def x(v) -> str:
    return f"{v:.2f}x"


def pc(v) -> str:
    return f"{v:.1f}%"


def jn(items: list) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _nums(t: str) -> set:
    t = re.sub(r"STEP \d+|from [\d, ]+(?=\])|\[[\d, ]+\]", " ", t)
    return {y.replace(",", "").rstrip(".").lstrip("0") or "0" for y in re.findall(r"\d[\d,]*\.?\d*", t)}


# ── the ledger: engine arithmetic + derived judgements ───────────────────────────────────────────────────────────────────────

def verdict_of(tests: list) -> tuple:
    """-> (verdict, [covenant names]) from translate()['covenants']."""
    if not tests:
        return "NO-COVENANT", []
    br = [t["covenant"] for t in tests if t["status"] == "breach"]
    if br:
        return "BREACH", br
    nc = [t["covenant"] for t in tests if t["status"] == "not computable"]
    if nc:
        return "NOT-COMPUTABLE", nc
    tight = [t["covenant"] for t in tests if t["status"].startswith("pass, under")]
    if tight:
        return "PASS-TIGHT", tight
    return "PASS", []


def ledger(scn: dict) -> dict:
    fin, loss, cov = scn["fin"], scn["loss"], scn.get("cov") or {}
    tr = RM.translate(fin, loss, cov or None)
    L = tr["loss"]
    pct = 100 * L["pre_tax_loss"] / fin["ebitda"] if fin["ebitda"] else None
    unins_share = next((k["value"] for k in tr["kris"] if k["kri"].startswith("Uninsured share")), None)
    dmg_u, bi_h = L["uninsured_asset_loss"], L["ebitda_hit"]
    driver = "uninsured damage" if dmg_u > 1.5 * bi_h else "business interruption and extra cost" if bi_h > 1.5 * dmg_u else "both uninsured damage and interruption"
    mat = None if pct is None else "small" if pct < 5 else "meaningful" if pct < 20 else "large" if pct < 50 else "very large"
    v, names = verdict_of(tr["covenants"])
    tests = tr["covenants"]
    bef = tr["before"]
    before_status = {}
    for t in tests:
        key = _COV_LABEL[t["covenant"]]
        b0 = bef.get(key)
        before_status[key] = None if b0 is None else ("pass" if (b0 <= t["threshold"] if t["rule"] == "<=" else b0 >= t["threshold"]) else "breach")
    breach_keys = [_COV_LABEL[t["covenant"]] for t in tests if t["status"] == "breach"]
    breach_new = [k for k in breach_keys if before_status.get(k) == "pass"]
    breach_pre = [k for k in breach_keys if before_status.get(k) == "breach"]
    binding = None
    for t in tests:
        if t.get("headroom_pct") is not None and (binding is None or t["headroom_pct"] < binding["headroom_pct"]):
            binding = t
    return {"tr": tr, "pct_of_ebitda": pct, "uninsured_share": unins_share, "driver": driver, "materiality": mat, "verdict": v, "verdict_names": names, "binding": binding, "before_status": before_status, "breach_new": breach_new, "breach_pre": breach_pre,
            "anomaly_recovery": any("insured_recovery exceeds" in a for a in tr["assumptions"])}


# ── the numbered sources the model sees ─────────────────────────────────────────────────────────────────────────────────────

def _first_sentence(t: str, limit: int = 34) -> str:
    body = t.split("): ", 1)[1] if "): " in t[:240] else t
    s = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])|;\s", body.strip())[0]
    w = s.split()
    return " ".join(w[:limit]).rstrip(".,;: ") if w else ""


def sources(scn: dict, led: dict, engine: bool = True, distractors: Optional[list] = None, distractors_first: bool = False) -> tuple:
    """-> (list of source strings in order, ids) where ids = {"inputs": [...], "engine": {"loss": k, "after": k, "cov": k, "assump": k} or {}, "analog": [...], "distractor": [...]}."""
    fin, loss, cov, tr = scn["fin"], scn["loss"], scn.get("cov") or {}, led["tr"]
    S, ids = [], {"inputs": [], "engine": {}, "analog": [], "distractor": []}

    def add(t, key=None, sub=None):
        S.append(t)
        k = len(S)
        if key == "inputs":
            ids["inputs"].append(k)
        elif key == "engine":
            ids["engine"][sub] = k
        elif key == "analog":
            ids["analog"].append(k)
        elif key == "distractor":
            ids["distractor"].append(k)
        return k

    add(f"Scenario: {scn['name']} owns {scn['assets']}. Hazard: {scn['hazard']}. All money is in millions of one currency.", "inputs")
    parts = [f"{lab} {m(fin[k])}" for k, lab in (("revenue", "revenue"), ("ebitda", "EBITDA"), ("net_debt", "net debt"), ("interest_expense", "interest expense"), ("equity", "equity"), ("total_assets", "total assets")) if fin.get(k) is not None]
    add("Financials before the event: " + "; ".join(parts) + ".", "inputs")
    lp = [f"asset damage {m(loss.get('asset_damage', 0.0))}", f"insured recovery {m(loss.get('insured_recovery', 0.0))}"]
    if loss.get("bi_revenue_lost"):
        mg = loss.get("bi_margin")
        lp.append(f"lost revenue from interruption {m(loss['bi_revenue_lost'])}" + (f" at an EBITDA flow-through of {pc(100 * mg)}" if mg is not None else " at the company's average EBITDA margin"))
    if loss.get("extra_costs"):
        lp.append(f"extra costs {m(loss['extra_costs'])}")
    add("Loss inputs: " + "; ".join(lp) + ".", "inputs")
    cp = []
    if cov.get("max_net_debt_to_ebitda") is not None:
        cp.append(f"net debt/EBITDA at most {x(cov['max_net_debt_to_ebitda'])}")
    if cov.get("min_interest_cover") is not None:
        cp.append(f"EBITDA/interest at least {x(cov['min_interest_cover'])}")
    if cov.get("min_equity_ratio") is not None:
        cp.append(f"equity/total assets at least {pc(100 * cov['min_equity_ratio'])}")
    add("Covenants: " + ("; ".join(cp) if cp else "none supplied") + ".", "inputs")
    if engine:
        L, b, a = tr["loss"], tr["before"], tr["after"]
        add(f"Engine result, loss: uninsured asset loss {m(L['uninsured_asset_loss'])}; EBITDA hit {m(L['ebitda_hit'])}; pre-tax loss {m(L['pre_tax_loss'])}, which is {pc(led['pct_of_ebitda'])} of EBITDA; "
            f"share of the loss that is uninsured {pc(led['uninsured_share'])}.", "engine", "loss")
        ap = [f"EBITDA {m(a['ebitda'])}", f"net debt {m(a['net_debt'])}"]
        for key, fmt in (("net_debt_to_ebitda", x), ("interest_cover", x)):
            if b.get(key) is not None:
                ap.append(f"{COV_NAMES[key]} {fmt(a[key]) if a.get(key) is not None else 'not computable because EBITDA is not positive'} (was {fmt(b[key])})")
        if b.get("equity_ratio") is not None:
            ap.append(f"equity/total assets {pc(100 * a['equity_ratio']) if a.get('equity_ratio') is not None else 'not computable'} (was {pc(100 * b['equity_ratio'])})")
        add("Engine result, after the loss: " + "; ".join(ap) + ".", "engine", "after")
        if tr["covenants"]:
            ts = []
            for t in tr["covenants"]:
                key = _COV_LABEL[t["covenant"]]
                fmt = pc if key == "equity_ratio" else x
                sc = 100 if key == "equity_ratio" else 1
                aft = fmt(sc * t["after"]) if t["after"] is not None else "not computable"
                rule = "at most" if t["rule"] == "<=" else "at least"
                hr = f", headroom {pc(t['headroom_pct'])}" if t["headroom_pct"] is not None else ""
                bs = led["before_status"].get(key)
                ts.append(f"{COV_NAMES[key]} {aft} against a limit of {rule} {fmt(sc * t['threshold'])}: {t['status']}{hr}" + (f"; before the loss it was a {bs}" if bs else ""))
            add("Engine result, covenant tests: " + "; ".join(ts) + ".", "engine", "cov")
        else:
            add("Engine result, covenant tests: there are no covenants to test.", "engine", "cov")
        add("Engine assumptions: the uninsured cash cost is funded by new debt; interest expense is unchanged; no covenant waiver, cure right, equity injection or insurer prepayment is assumed.", "engine", "assump")
    if distractors_first:
        for p in distractors or []:
            add(p["text"], "distractor")
    for p in scn.get("analogs") or []:
        add(p["text"], "analog")
    if not distractors_first:
        for p in distractors or []:
            add(p["text"], "distractor")
    return S, ids


QUESTIONS = (
    "Using only the sources, what does this loss do to {name}'s financials and covenants, and what should the risk manager check next?",
    "Assess the effect of this {hazard} loss on {name}: how large is it, does it threaten any covenant, and what is still unknown?",
    "Walk through how the {hazard} loss reaches {name}'s balance sheet and covenants. Show your steps and give a verdict.",
    "Does the {hazard} loss put {name}'s lending covenants at risk? Reason from the sources.",
)


def question(scn: dict, i: int = 0) -> str:
    return QUESTIONS[i % len(QUESTIONS)].format(name=scn["name"], hazard=scn["hazard"])


def user_message(scn: dict, S: list, i: int = 0) -> str:
    return f"Question: {question(scn, i)}\n\nSources:\n" + "\n\n".join(f"[{k + 1}] {t}" for k, t in enumerate(S))


# ── the rule-based reasoner (teacher and fallback) ───────────────────────────────────────────────────────────────────────────

def compose(scn: dict, led: dict, ids: dict, channels: Optional[list] = None, engine: bool = True, analog_label: Optional[str] = None) -> str:
    fin, loss, tr = scn["fin"], scn["loss"], led["tr"]
    I = ids["inputs"]
    nm = scn["name"]
    steps, n = [], 0

    def add(kind, text, frm=None):
        nonlocal n
        n += 1
        head = f"STEP {n} [{kind}" + (f" from {', '.join(str(i) for i in frm)}" if frm else "") + "]"
        steps.append(f"{head} {text}")
        return n

    bits = [f"asset damage of {m(loss.get('asset_damage', 0.0))}"]
    rec = loss.get("insured_recovery", 0.0)
    bits.append(f"insured recovery of {m(rec)}" if rec else "no insured recovery")
    if loss.get("bi_revenue_lost"):
        bits.append(f"{m(loss['bi_revenue_lost'])} of lost revenue from interruption")
    if loss.get("extra_costs"):
        bits.append(f"extra costs of {m(loss['extra_costs'])}")
    s1 = add("FACT", f"{nm} faces a {scn['hazard']} loss with " + ", ".join(bits[:-1]) + (" and " if len(bits) > 1 else "") + bits[-1] + f" [{I[2]}].")
    if not engine:
        s2 = add("FACT", f"The sources give the loss inputs [{I[2]}] but no engine result for the loss, for the financials after the loss or for the covenant tests.", None)
        add("INFERENCE", f"Without those results the effect on {nm}'s covenants cannot be stated from these sources.", [s2])
        return "\n".join(steps) + ("\nVERDICT: NOT-COMPUTABLE\nCHECK NEXT: run the engine on these loss inputs to get the after-loss EBITDA, net debt and covenant ratios, then repeat this analysis."
                                   "\nNOT IN THE SOURCES: the after-loss EBITDA, net debt and covenant ratios.")
    e = ids["engine"]
    L, b, a = tr["loss"], tr["before"], tr["after"]
    s2 = add("FACT", f"The engine puts the uninsured asset loss at {m(L['uninsured_asset_loss'])} and the EBITDA hit at {m(L['ebitda_hit'])}, so the pre-tax loss is {m(L['pre_tax_loss'])}, "
                     f"which is {pc(led['pct_of_ebitda'])} of EBITDA [{e['loss']}].")
    mat = {"small": "small next to annual earnings", "meaningful": "meaningful but within what a year of earnings can absorb", "large": "large relative to annual earnings",
           "very large": "very large: it exceeds half a year of EBITDA"}[led["materiality"]]
    ins = ("no recovery was supplied, so the whole damage falls on the balance sheet" if not rec else
           "insurance covers the damage in full, so what remains comes from interruption and extra cost" if led["anomaly_recovery"] or rec >= loss.get("asset_damage", 0.0) else
           "insurance covers only part of the damage")
    s3 = add("INFERENCE", f"The loss is {mat}; it is driven by {led['driver']}, and {ins}.", [s2])
    ap = f"After the loss EBITDA is {m(a['ebitda'])} and net debt is {m(a['net_debt'])}"
    if b.get("net_debt_to_ebitda") is not None and a.get("net_debt_to_ebitda") is not None:
        ap += f"; net debt/EBITDA moves from {x(b['net_debt_to_ebitda'])} to {x(a['net_debt_to_ebitda'])}"
    s4 = add("FACT", ap + f" [{e['after']}].")
    cstep = []
    for t in tr["covenants"]:
        key = _COV_LABEL[t["covenant"]]
        fmt, sc = (pc, 100) if key == "equity_ratio" else (x, 1)
        rule = "at most" if t["rule"] == "<=" else "at least"
        aft = fmt(sc * t["after"]) if t["after"] is not None else "not computable"
        hr = f" with headroom of {pc(t['headroom_pct'])}" if t["headroom_pct"] is not None else ""
        bs = led["before_status"].get(key)
        cstep.append(add("FACT", f"The {COV_NAMES[key]} test is {aft} against a limit of {rule} {fmt(sc * t['threshold'])}: {t['status']}{hr}" + (f"; before the loss it was a {bs}" if bs else "") + f" [{e['cov']}]."))
    if not tr["covenants"]:
        cstep.append(add("FACT", f"No covenants were supplied, so there is nothing to test [{e['cov']}]."))
    v, names = led["verdict"], led["verdict_names"]
    bnd = led["binding"]
    if v == "BREACH":
        parts = [f"the loss pushes {COV_NAMES[k]} into breach" for k in led["breach_new"]]
        parts += [f"{COV_NAMES[k]} was already in breach before the loss" for k in led["breach_pre"]]
        if not parts:
            parts.append(f"{jn([COV_NAMES[_COV_LABEL[c]] for c in names])} is in breach on these inputs")
        t = "On these inputs " + "; ".join(parts) + "." + (" The loss did not cause the pre-existing breaches and only widens the gap." if led["breach_pre"] else "")
    elif v == "PASS-TIGHT":
        t = f"No covenant is breached, but {jn([COV_NAMES[_COV_LABEL[c]] for c in names])} is within 15% of its limit, so a slightly larger loss or a slower insurance payment could tip it."
    elif v == "PASS":
        t = "Every covenant still passes with room to spare, so on these inputs the loss does not threaten the lending terms."
    elif v == "NOT-COMPUTABLE":
        t = "At least one covenant cannot be computed after the loss, so it cannot be cleared on these inputs."
    else:
        t = "With no covenants supplied the effect can be stated only as a change in the ratios, not as a breach or a pass."
    s5 = add("INFERENCE", t, cstep or [s4])
    last = s5
    if scn.get("analogs"):
        a0 = ids["analog"][0]
        lab = analog_label or "a past case"
        fs = _first_sentence(scn["analogs"][0]["text"])
        sp = add("PRECEDENT", f"In {lab}: {fs} [{a0}].")
        if channels:
            add("INFERENCE", "Analysis (not stated in the sources): past cases suggest checking " + jn(channels) + " as well as direct damage.", [sp, s3])
    cav = add("CAVEAT", f"The engine assumes the uninsured cost is debt-funded and that no waiver, cure right or equity injection is available [{e['assump']}].")
    if led["anomaly_recovery"]:
        add("CAVEAT", f"The insured recovery exceeds the modelled loss on these inputs [{I[2]}], so the policy limits and what is actually insured need checking.")
    nxt = {"BREACH": "whether a waiver, cure period or equity injection is available, and when the insurer would pay",
           "PASS-TIGHT": "how much of the remaining headroom a larger loss, a lower margin or a slower insurance payment would use",
           "PASS": "whether the loss inputs are complete, for example other sites or interruption cover",
           "NO-COVENANT": "which financial tests the lenders actually apply, then rerun this analysis with them",
           "NOT-COMPUTABLE": "the after-loss EBITDA and net debt, then rerun this analysis"}[v]
    extra = ("the policy limits and deductibles on the damaged assets" if "damage" in led["driver"] and "both" not in led["driver"] else
             "the business-interruption cover and its waiting period" if "interruption" in led["driver"] and "both" not in led["driver"] else "both the property and the interruption cover")
    vtxt = v + (f" ({', '.join(COV_NAMES[_COV_LABEL[c]] for c in names)})" if names else "")
    return "\n".join(steps) + f"\nVERDICT: {vtxt}\nCHECK NEXT: {nxt}; also {extra}.\nNOT IN THE SOURCES: the insurer's payment timing, any covenant waiver, and the company's actual response."


# ── the verifier ────────────────────────────────────────────────────────────────────────────────────────────────────────────

_STEP = re.compile(r"^STEP (\d+) \[(FACT|INFERENCE|PRECEDENT|CAVEAT)(?: from ([\d, ]+))?\]\s*(.*)$")
_VERD = re.compile(r"^VERDICT:\s*([A-Z\-]+)", re.M)


def parse(ans: str) -> dict:
    steps, bad = [], []
    for line in ans.splitlines():
        line = line.strip()
        if line.startswith("STEP"):
            mm = _STEP.match(line)
            if mm:
                steps.append({"n": int(mm.group(1)), "kind": mm.group(2), "from": [int(i) for i in re.findall(r"\d+", mm.group(3) or "")], "text": mm.group(4)})
            else:
                bad.append(line[:60])
    v = _VERD.search(ans)
    return {"steps": steps, "malformed": bad, "verdict": v.group(1) if v else None, "has_next": "CHECK NEXT:" in ans, "has_unknown": "NOT IN THE SOURCES:" in ans}


def verify(ans: str, S: list, expected_verdict: str, ids: Optional[dict] = None, irrelevant_ids: Optional[list] = None, attribution: Optional[dict] = None) -> dict:
    """Every check is mechanical. Returns {"ok": bool, "checks": {name: bool}, "problems": [..]}."""
    p = parse(ans)
    st = p["steps"]
    problems, c = [], {}
    c["format"] = bool(st) and not p["malformed"] and p["verdict"] is not None and p["has_next"] and p["has_unknown"] and [s["n"] for s in st] == list(range(1, len(st) + 1))
    if not c["format"]:
        problems.append("format: steps, VERDICT, CHECK NEXT and NOT IN THE SOURCES must all be present and numbered in order")
    src_nums = set().union(*[_nums(t) for t in S]) if S else set()
    bad_nums = sorted(_nums(ans) - src_nums - {str(i) for i in range(1, len(S) + 1)} - {str(i) for i in range(1, len(st) + 2)})
    c["numbers_in_sources"] = not bad_nums
    if bad_nums:
        problems.append(f"numbers not in the sources: {bad_nums[:5]}")
    cites = [int(i) for s in st for i in re.findall(r"\[(\d+)\]", s["text"])]
    c["citations_exist"] = all(1 <= i <= len(S) for i in cites)
    if not c["citations_exist"]:
        problems.append("a citation does not exist")
    c["facts_cited"] = all(re.search(r"\[\d+\]", s["text"]) for s in st if s["kind"] in ("FACT", "PRECEDENT", "CAVEAT"))
    if not c["facts_cited"]:
        problems.append("a FACT, PRECEDENT or CAVEAT step has no citation")
    inf_ok = True
    for s in st:
        if s["kind"] == "INFERENCE":
            if not s["from"] or any(f >= s["n"] or f < 1 for f in s["from"]):
                inf_ok = False
    c["inferences_grounded"] = inf_ok
    if not inf_ok:
        problems.append("an INFERENCE step does not point at earlier steps or introduces a number")
    c["verdict_correct"] = p["verdict"] == expected_verdict
    if not c["verdict_correct"]:
        problems.append(f"verdict {p['verdict']} but the engine numbers imply {expected_verdict}")
    if attribution is not None:                                      # which breaches the loss causes and which were already there: {"new": [names], "pre": [names]}
        inf = " ".join(s["text"] for s in st if s["kind"] == "INFERENCE")
        ok = all(f"pushes {n} into breach" in inf for n in attribution["new"]) and all(f"{n} was already in breach before the loss" in inf for n in attribution["pre"])
        ok = ok and not any(f"pushes {n} into breach" in inf for n in attribution["pre"]) and not any(f"{n} was already in breach" in inf for n in attribution["new"])
        c["attribution_correct"] = ok
        if not ok:
            problems.append("wrong attribution of breaches to the loss versus pre-existing")
    if irrelevant_ids:
        c["ignores_irrelevant"] = not (set(cites) & set(irrelevant_ids))
        if not c["ignores_irrelevant"]:
            problems.append("cited a source that does not bear on the question")
    return {"ok": all(c.values()), "checks": c, "problems": problems, "verdict": p["verdict"], "steps": len(st)}


def expected_verdict(led: dict, engine: bool = True) -> str:
    return led["verdict"] if engine else "NOT-COMPUTABLE"


def render_plain(ans: str) -> str:
    """Reader-friendly form: the structure is kept, the citation markers stay (they are the evidence trail)."""
    return ans.replace("STEP ", "Step ")
