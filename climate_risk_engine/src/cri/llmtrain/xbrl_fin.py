"""
cri/llmtrain/xbrl_fin.py — company financials around each event, from the SEC's structured (XBRL) annual data.

For every company that disclosed something about an event (edgar_cases.jsonl), fetch its annual 10-K figures from
https://data.sec.gov/api/xbrl/companyfacts and write ONE source sentence per company-event: revenue, operating income, net income,
total assets and long-term debt in the fiscal years either side of the event, plus the change in the event year. Every number in the
sentence is either copied from the SEC data or is a percentage computed by this code from two copied numbers; nothing is estimated.

Pre-XBRL years (before about 2009) have no structured data: those company-events simply get no financial sentence.

  collect(edgar_jsonl, out_dir)   writes out_dir/xbrl_items.jsonl and xbrl_report.json (resumable; raw responses cached in out_dir/raw)
  build_rows(items, seed, per_case)  training rows ('financials', 'financials_unknown'); the held-out events stay out of train
CLI:  python -m cri.llmtrain.xbrl_fin collect EDGAR_CASES.jsonl OUT_DIR
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

UA = {"User-Agent": "ClimRisk research (open-source climate-risk project; polite crawler) shrinivashdkannan@gmail.com", "Accept-Encoding": "identity"}
URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenuesNetOfInterestExpense", "RevenueFromContractWithCustomerIncludingAssessedTax", "PremiumsEarnedNet"],
    "operating income": ["OperatingIncomeLoss"],
    "net income": ["NetIncomeLoss", "ProfitLoss"],
    "total assets": ["Assets"],
    "long-term debt": ["LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
}
DURATION = {"revenue", "operating income", "net income"}


def _get(url: str, timeout: int = 60) -> dict:
    last = None
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:                                       # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def _d(s: str) -> date:
    return date.fromisoformat(s)


def annual(facts: dict, tags: list, duration: bool) -> dict:
    """{fiscal-year-end date: value} from 10-K annual facts; the latest filing wins when a year was restated."""
    best: dict = {}
    us = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        for x in us.get(tag, {}).get("units", {}).get("USD", []):
            if x.get("form") not in ("10-K", "10-K405", "10-K/A") or x.get("fp") != "FY":
                continue
            if duration:
                if "start" not in x or not (340 <= (_d(x["end"]) - _d(x["start"])).days <= 380):
                    continue
            cur = best.get(x["end"])
            if cur is None or x.get("filed", "") >= cur[1]:
                best[x["end"]] = (x["val"], x.get("filed", ""))
        if best:
            break                                                    # the first tag that has annual data for this company is the one it uses
    return {k: v[0] for k, v in best.items()}


def _money(v: float) -> str:
    a = abs(v)
    s = f"${a / 1e9:,.2f} billion" if a >= 1e9 else f"${a / 1e6:,.1f} million"
    return ("negative " + s) if v < 0 else s


def _pct(a: float, b: float):
    return None if not a or a <= 0 or b is None else round(100 * (b - a) / a, 1)


def event_year(ev: str, filed_years: list) -> int:
    m = re.search(r"(\d{4})", ev)
    return int(m.group(1)) if m else min(filed_years)


def sentence(company: str, ev_name: str, year: int, series: dict) -> tuple:
    """-> (text, amounts) or (None, []) when there is not enough data. Uses fiscal years ending in year-1, year and year+1."""
    parts, amounts = [], []
    for label, ser in series.items():
        pts = {}
        for end, v in ser.items():
            y = _d(end).year if _d(end).month >= 6 else _d(end).year - 1        # a fiscal year ending Jan-May belongs to the calendar year before
            pts[y] = (end, v)
        sel = [(y, pts[y]) for y in (year - 1, year, year + 1) if y in pts]
        if len(sel) < 2:
            continue
        seg = ", ".join(f"{_money(v)} in the fiscal year ended {end}" for _, (end, v) in sel)
        amounts += [_money(v) for _, (_, v) in sel]
        extra = ""
        if label in ("revenue", "net income") and year in pts and (year - 1) in pts:
            p = _pct(pts[year - 1][1], pts[year][1])
            if p is not None:
                extra = f" (a change of {p:+.1f}% in the year of the event)"
        parts.append(f"{label} was {seg}{extra}")
    if len(parts) < 2:
        return None, []
    return f"{company}'s annual financial data from its SEC Form 10-K filings around {ev_name}: " + "; ".join(parts) + ".", amounts


def _job(a):
    cik, company, pairs, raw = a
    f = raw / f"{cik}.json"
    try:
        facts = json.loads(f.read_text()) if f.exists() else None
        if facts is None:
            facts = _get(URL.format(cik=int(cik)))
            f.write_text(json.dumps({k: v for k, v in facts.items() if k in ("entityName", "cik", "facts")}))
            time.sleep(0.15)
        series = {lab: annual(facts, tags, lab in DURATION) for lab, tags in CONCEPTS.items()}
        series = {k: v for k, v in series.items() if v}
        out = []
        for ev, ev_name, yrs in pairs:
            text, amounts = sentence(company, ev_name, event_year(ev, yrs), series)
            if text:
                out.append({"event": ev, "event_name": ev_name, "company": company, "cik": cik, "kind": "xbrl", "text": text, "amounts": amounts,
                            "url": URL.format(cik=int(cik)), "form": "10-K (XBRL)"})
        return cik, out, None
    except Exception as e:                                           # noqa: BLE001
        return cik, [], f"{type(e).__name__}: {str(e)[:80]}"


def collect(edgar_jsonl: Path, out: Path, workers: int = 4) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    by: dict = {}
    for line in Path(edgar_jsonl).read_text().splitlines():
        r = json.loads(line)
        d = by.setdefault(r["cik"], {"company": r["company"], "ev": {}})
        yrs = d["ev"].setdefault((r["event"], r["event_name"]), [])
        yrs.append(int(r["filed"][:4]))
    jobs = [(cik, d["company"], [(ev, nm, yrs) for (ev, nm), yrs in d["ev"].items()], raw) for cik, d in by.items()]
    items, errs, done = [], {}, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for cik, got, err in pool.map(_job, jobs):
            done += 1
            items += got
            if err:
                errs[cik] = err
            if done % 50 == 0:
                print(f"xbrl {done}/{len(jobs)} companies, {len(items)} company-event financial sentences", flush=True)
    tmp = out / "xbrl_items.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(out / "xbrl_items.jsonl")
    rep = {"companies": len(jobs), "company_event_financials": len(items), "errors": len(errs), "sample_errors": dict(list(errs.items())[:5])}
    (out / "xbrl_report.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return rep


# ── training rows ────────────────────────────────────────────────────────────

SYS = ("You are ClimRisk's climate-risk analyst. Answer using ONLY the numbered sources, which are structured annual figures from company filings with the SEC. Cite every factual sentence with its source number in square brackets, like [2]. "
       "If the sources do not contain the answer, say exactly what is missing instead of guessing. Keep any inference clearly separate by starting it with 'Analysis (not stated in the sources):'. Never add a number that is not in the sources.")
CAVEAT = "Analysis (not stated in the sources): these are company-wide annual figures, so they show the movement around the event but do not isolate how much of it the event caused."
from .heldout import EVENTS as HELD_EVENTS


def load(path: Path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def _row(q, blk, ans, need, n, split, meta, kind="case"):
    return {"kind": kind, "split": split, "meta": {"need": need, "n": n, **meta},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {q}\n\nSources:\n{blk}"}, {"role": "assistant", "content": ans}]}


def build_rows(path: Path, seed: int = 9, per_case: int = 1) -> list:
    rng = random.Random(seed)
    items = load(path)
    rows = []
    for it in items:
        held = it["event"] in HELD_EVENTS
        split = "test" if held else ("valid" if rng.random() < 0.08 else "train")
        pool = [x for x in items if (x["event"] in HELD_EVENTS) == held and x is not it]
        if len(pool) < 2:
            continue
        for _ in range(per_case):
            others = rng.sample(pool, 2)
            srcs = [it] + others
            order = list(range(3))
            rng.shuffle(order)
            pos = {k: n for n, k in enumerate(order, 1)}
            blk = "\n\n".join(f"[{n}] {srcs[k]['text']}" for n, k in sorted(((pos[k], k) for k in range(3))))
            core = it["text"].rstrip(".")
            ans = f"{core} [{pos[0]}]. {CAVEAT}"
            q = rng.choice([f"How did {it['company']}'s revenue and profit move around {it['event_name']}, according to its SEC financial data?",
                            f"What do {it['company']}'s annual financials show around {it['event_name']}?",
                            f"Give me the before-and-after financial figures for {it['company']} around {it['event_name']}."])
            rows.append(_row(q, blk, ans, [pos[0]], 3, split, {"type": "financials", "case": f"{it['event']}:{it['company']}"}))
        # asked about an event this company has no financial sentence for: say it is not covered
        evs = {x["event"]: x["event_name"] for x in pool}
        mine = {x["event"] for x in items if x["company"] == it["company"]}
        free = [e for e in evs if e not in mine]
        if free and rng.random() < 0.4:
            e2 = rng.choice(free)
            others = rng.sample(pool, 2)
            srcs = [it] + others
            order = list(range(3))
            rng.shuffle(order)
            pos = {k: n for n, k in enumerate(order, 1)}
            blk = "\n\n".join(f"[{n}] {srcs[k]['text']}" for n, k in sorted(((pos[k], k) for k in range(3))))
            ans = (f"The sources provided do not cover {it['company']}'s financials around {evs[e2]}. They only show its figures around {it['event_name']}: {it['text'].split(': ', 1)[1].rstrip('.')} [{pos[0]}]. "
                   f"To answer I would need its annual filings for the relevant years.")
            rows.append(_row(f"How did {it['company']}'s revenue move around {evs[e2]}?", blk, ans, [pos[0]], 3, "test" if held else "train", {"type": "financials_unknown", "case": f"{e2}:{it['company']}"}, kind="case_gap"))
    return rows


def main():
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "collect":
        collect(Path(a[1]), Path(a[2]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
