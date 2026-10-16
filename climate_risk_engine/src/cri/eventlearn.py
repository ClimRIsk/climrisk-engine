"""
cri/eventlearn.py: what recorded events did to the companies that disclosed them, and how far that can be used to expect an outcome.

Data: one row per (company, named event) from the SEC's structured annual data (cri.llmtrain.xbrl_fin; every figure is copied from SEC
filings, change percentages are computed here from two copied numbers). Table in data/events/company_events.csv.

What this is NOT: a damage model. The revenue change of a company in the year of an event also moves with the economy, its own growth and
acquisitions, so an event-year change is a *descriptive analog*, not a loss. The leave-one-event-out test below says how much, if any,
information the hazard type and company size carry beyond the all-events average. The result is reported whatever it is.

    PYTHONPATH=src python -m cri.eventlearn build  XBRL_ITEMS.jsonl   data/events/company_events.csv
    PYTHONPATH=src python -m cri.eventlearn eval   data/events/company_events.csv  [OUT.json]
    from cri.eventlearn import analogs;  analogs("tropical cyclone")
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "data" / "events" / "company_events.csv"
MIN_EVENTS = 5                                   # fewer distinct events than this: no analog statistics, the answer is "not enough recorded events"
_MONEY = re.compile(r"(negative )?\$([\d,.]+)\s*(trillion|billion|million|thousand)?")
_SCALE = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3, None: 1.0}
_ITEM = re.compile(r"(revenue|net income|total assets) was (.*?)(?=; (?:revenue|operating income|net income|total assets|long-term debt) was |\.$)")


def _money(s: str):
    out = []
    for m in _MONEY.finditer(s):
        v = float(m.group(2).replace(",", "")) * _SCALE[m.group(3)]
        out.append(-v if m.group(1) else v)
    return out


def _hazards() -> dict:
    from .llmtrain import edgar_cases as E
    return {e[0]: (e[5], e[6], e[3][:4]) for e in E.EVENTS}


def build(items_path: str, out_path: str) -> dict:
    hz = _hazards()
    rows, skipped = [], 0
    for line in open(items_path):
        r = json.loads(line)
        fig = {m.group(1): _money(m.group(2)) for m in _ITEM.finditer(r["text"])}
        rev, ta, ni = fig.get("revenue", []), fig.get("total assets", []), fig.get("net income", [])
        if len(rev) < 2 or rev[0] <= 0 or r["event"] not in hz:
            skipped += 1
            continue
        h, region, year = hz[r["event"]]
        rows.append(dict(event=r["event"], event_name=r["event_name"], hazard=h, region=region, event_year=year, company=r["company"], cik=r["cik"],
                         revenue_prev=rev[0], revenue_event=rev[1], revenue_next=rev[2] if len(rev) > 2 else "",
                         rev_change_event_pct=round(100 * (rev[1] / rev[0] - 1), 2),
                         rev_change_next_pct=round(100 * (rev[2] / rev[1] - 1), 2) if len(rev) > 2 and rev[1] > 0 else "",
                         net_income_prev=ni[0] if len(ni) > 0 else "", net_income_event=ni[1] if len(ni) > 1 else "",
                         assets_event=ta[1] if len(ta) > 1 else "", source=r["url"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"rows": len(rows), "events": len({r["event"] for r in rows}), "skipped": skipped}


def load(path=None) -> list[dict]:
    p = Path(path) if path else DEFAULT
    out = []
    for r in csv.DictReader(open(p)):
        r["rev_change_event_pct"] = float(r["rev_change_event_pct"])
        r["assets_event"] = float(r["assets_event"]) if r["assets_event"] else None
        out.append(r)
    return out


def _q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def analogs(hazard: str, exclude_events=(), path=None, rows=None) -> dict | None:
    """Event-year revenue change of companies that disclosed events of this hazard type. None when fewer than MIN_EVENTS distinct events."""
    rows = rows if rows is not None else load(path)
    sel = [r for r in rows if r["hazard"] == hazard and r["event"] not in set(exclude_events)]
    ev = {r["event"] for r in sel}
    if len(ev) < MIN_EVENTS:
        return None
    ys = [r["rev_change_event_pct"] for r in sel]
    return dict(hazard=hazard, n_events=len(ev), n_company_events=len(ys), median_pct=round(st.median(ys), 1), p25_pct=round(_q(ys, .25), 1), p75_pct=round(_q(ys, .75), 1),
                p10_pct=round(_q(ys, .10), 1), p90_pct=round(_q(ys, .90), 1), share_decline_pct=round(100 * sum(y < 0 for y in ys) / len(ys)),
                share_decline_over_10_pct=round(100 * sum(y < -10 for y in ys) / len(ys)),
                examples=[dict(event=e, n=sum(1 for r in sel if r["event"] == e)) for e in sorted(ev)[:5]])


def _mdae(pairs):
    return st.median(abs(p - y) for p, y in pairs) if pairs else float("nan")


def loeo_eval(rows=None, path=None, seed=1, boot=400) -> dict:
    """Leave-one-event-out: predict every company's event-year revenue change from the OTHER events only."""
    rows = rows if rows is not None else load(path)
    events = sorted({r["event"] for r in rows})
    tercile = {}
    sz = sorted(r["assets_event"] for r in rows if r["assets_event"])
    cut = (_q(sz, 1 / 3), _q(sz, 2 / 3))
    def bucket(a):
        return None if a is None else (0 if a < cut[0] else 1 if a < cut[1] else 2)
    preds = {"zero": [], "all_events_median": [], "hazard_median": [], "hazard_size_median": []}
    bands = {"hazard_iqr_hits": [], "hazard_p10_p90_hits": [], "global_iqr_hits": []}
    per_event = {}
    for e in events:
        test = [r for r in rows if r["event"] == e]
        train = [r for r in rows if r["event"] != e]
        g = [r["rev_change_event_pct"] for r in train]
        gm = st.median(g)
        gq = (_q(g, .25), _q(g, .75))
        a = analogs(test[0]["hazard"], exclude_events=(e,), rows=rows)
        for r in test:
            y = r["rev_change_event_pct"]
            p_h = a["median_pct"] if a else gm
            same = [t["rev_change_event_pct"] for t in train if t["hazard"] == r["hazard"] and bucket(t["assets_event"]) == bucket(r["assets_event"])]
            p_hs = st.median(same) if len(same) >= 15 else p_h
            for k, p in (("zero", 0.0), ("all_events_median", gm), ("hazard_median", p_h), ("hazard_size_median", p_hs)):
                preds[k].append((p, y, e))
            bands["global_iqr_hits"].append(gq[0] <= y <= gq[1])
            if a:
                bands["hazard_iqr_hits"].append(a["p25_pct"] <= y <= a["p75_pct"])
                bands["hazard_p10_p90_hits"].append(a["p10_pct"] <= y <= a["p90_pct"])
        per_event[e] = len(test)
    rng = random.Random(seed)
    def boot_diff(a, b):
        by = {}
        for (pa, y, e), (pb, _, _) in zip(preds[a], preds[b]):
            by.setdefault(e, []).append((abs(pa - y), abs(pb - y)))
        ev = list(by)
        ds = []
        for _ in range(boot):
            samp = [x for e in (rng.choice(ev) for _ in ev) for x in by[e]]
            ds.append(st.median(x[0] for x in samp) - st.median(x[1] for x in samp))
        ds.sort()
        return round(st.median(x[0] for v in by.values() for x in v) - st.median(x[1] for v in by.values() for x in v), 3), (round(ds[int(.025 * boot)], 3), round(ds[int(.975 * boot)], 3))
    res = dict(n_company_events=len(rows), n_events=len(events),
               median_abs_error_pct_points={k: round(_mdae([(p, y) for p, y, _ in v]), 2) for k, v in preds.items()},
               band_coverage={"hazard_iqr (nominal 50%)": round(100 * sum(bands["hazard_iqr_hits"]) / max(1, len(bands["hazard_iqr_hits"]))),
                              "hazard_p10_p90 (nominal 80%)": round(100 * sum(bands["hazard_p10_p90_hits"]) / max(1, len(bands["hazard_p10_p90_hits"]))),
                              "all_events_iqr (nominal 50%)": round(100 * sum(bands["global_iqr_hits"]) / len(bands["global_iqr_hits"]))},
               hazard_vs_all_events_mdae_diff=boot_diff("hazard_median", "all_events_median"),
               hazard_size_vs_all_events_mdae_diff=boot_diff("hazard_size_median", "all_events_median"),
               note="Negative difference = hazard-type information improves the prediction. Interval = 95% bootstrap over events.")
    return res


def main():
    a = sys.argv[1:]
    if a and a[0] == "build":
        print(json.dumps(build(a[1], a[2])))
    elif a and a[0] == "eval":
        r = loeo_eval(path=a[1])
        r["hazards"] = {h: analogs(h, path=a[1]) for h in sorted({x["hazard"] for x in load(a[1])})}
        if len(a) > 2:
            Path(a[2]).write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
