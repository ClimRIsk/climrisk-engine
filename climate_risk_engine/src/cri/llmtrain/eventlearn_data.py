"""
cri/llmtrain/eventlearn_data.py: training rows that teach the language model to USE what the engine learned from recorded events, and to state its limits.

    python -m cri.llmtrain.eventlearn_data OUT_DIR [--seed 8]

The language model does not predict. The engine's event models (cri.eventlearn_tc: US storm wind -> total cost, tested leave-one-storm-out; cri.eventlearn: company revenue after
recorded events, tested leave-one-event-out) produce the numbers; the model reads them and explains them. Families:
  evtc_predict    "what loss for a storm of W knots?"   facts = the fitted model's expected value and 10-90% range, the backtest score, the storms it misses
  evtc_backtest   "how did the model do on Hurricane X?" facts = that storm's wind, observed cost, the leave-one-out prediction (storm held out of its own fit); split by storm
  evfin           "will revenue fall after a <hazard>?"  facts = analog statistics and the finding that hazard type adds no accuracy; or 'too few recorded events'
Every number in an answer is copied from the facts. Writes train/valid/test(+_meta).jsonl in the same layout as v7_data, plus eventlearn_report.json.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from .. import eventlearn as EL
from .. import eventlearn_tc as TC

SYS = ("You are ClimRisk's climate-risk analyst. The engine has learned from recorded events and tested itself on events it had not seen; use only the engine facts supplied. "
       "State the expected value and range with their meaning, the backtest result, and what the model gets wrong. Past events are not a forecast. Never give a number that is not in the facts. "
       "Where the facts say too few events or no skill, say so plainly. 50 to 130 words, plain prose, no lists.")

_Q_PRED = ["If a hurricane with {w} knot winds at the coast hit the US next year, what total loss should we expect?",
           "What does the event-trained model say about a {w} kt landfall?",
           "Based on past US storms, what is a {w}-knot hurricane likely to cost?",
           "How big a loss should I plan for from a storm that reaches {w} knots at the coast?"]
_Q_BACK = ["How well did the event-trained model do on {s}?", "What would the model have predicted for {s}, and what happened?", "Did the model get {s} right?"]
_Q_FIN = ["Will our revenue fall in the year after a major {h} event?", "What do recorded {h} events say about company revenue afterwards?",
          "Can you forecast a revenue hit from a {h} using past events?"]


def _fmt(x: float) -> str:
    return f"{x:,.0f}"


def _b(x: float) -> str:
    return f"{x / 1000:.1f}"


def _pred_row(rng, w: int, split: str, bt: dict):
    p = TC.predict(w, 2026)
    facts = dict(model="US tropical cyclones 1980-2024", n_storms=p["n_storms"], wind_kt=w, expected_musd=p["expected_musd"], expected_busd=float(_b(p["expected_musd"])),
                 p10_busd=float(_b(p["p10_musd"])), p90_busd=float(_b(p["p90_musd"])), backtest_median_factor_error=bt["median_factor_error"], baseline_median_factor_error=bt["baseline_median_factor_error"],
                 within_factor_3_pct=bt["within_factor_3_pct"], baseline_within_factor_3_pct=bt["baseline_within_factor_3_pct"], misses=["Katrina", "Harvey", "Allison"],
                 cost="total economic cost for the whole US event, not an insured loss, not a site loss")
    a = (f"For a storm reaching {w} knots at the US coast, the model fitted on {p['n_storms']} recorded US storms expects a total economic cost of about ${_b(p['expected_musd'])} billion, "
         f"with an 80% range of ${_b(p['p10_musd'])} to ${_b(p['p90_musd'])} billion. That is the cost of the whole event across the country, not an insured loss and not the loss at one site. "
         f"Tested on storms it had not seen, its typical error was a factor of {bt['median_factor_error']}, against {bt['baseline_median_factor_error']} for guessing the typical storm, and {bt['within_factor_3_pct']}% of storms were within a factor of 3 "
         f"(guessing: {bt['baseline_within_factor_3_pct']}%). It underestimates rain and surge storms such as Katrina, Harvey and Allison. Screening-grade, not a forecast.")
    q = rng.choice(_Q_PRED).format(w=w)
    return {"kind": "evtc", "split": split, "meta": {"qtype": "predict", "facts": json.dumps(facts), "nulls": []},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Engine results: {json.dumps(facts)}\n\nQuestion: {q}"}, {"role": "assistant", "content": a}]}


def _back_row(rng, d: dict, split: str):
    within = 1 / 3 <= d["factor"] <= 3
    facts = dict(storm=d["storm"], wind_kt=int(d["wind_kt"]), observed_busd=float(_b(d["observed_musd"])), predicted_busd=float(_b(d["predicted_musd"])), factor=d["factor"],
                 within_factor_3=within, note="prediction made by a model fitted without this storm")
    if d["factor"] < 1:
        verdict = f"under-predicted: the model's figure was {d['factor']} times the observed cost"
    else:
        verdict = f"over-predicted: the model's figure was {d['factor']} times the observed cost"
    a = (f"For {d['storm']}, with {int(d['wind_kt'])} knots at the coast, a model fitted without that storm would have predicted about ${_b(d['predicted_musd'])} billion; "
         f"the observed total economic cost was ${_b(d['observed_musd'])} billion. It {verdict}, {'within' if within else 'outside'} a factor of 3. "
         f"A wind-only model cannot see rain, surge or the value of what was in the way, so single storms can miss widely. Screening-grade, not a forecast.")
    q = rng.choice(_Q_BACK).format(s=d["storm"])
    return {"kind": "evtc", "split": split, "meta": {"qtype": "backtest", "facts": json.dumps(facts), "nulls": []},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Engine results: {json.dumps(facts)}\n\nQuestion: {q}"}, {"role": "assistant", "content": a}]}


def _fin_rows(rng, ev: dict, rows_per_hazard: int, split: str):
    out = []
    rows = EL.load()
    for h in sorted({r["hazard"] for r in rows}):
        a = EL.analogs(h, rows=rows)
        for _ in range(rows_per_hazard):
            if a is None:
                n = len({r["event"] for r in rows if r["hazard"] == h})
                facts = dict(hazard=h, recorded_events=n, minimum_events=EL.MIN_EVENTS, status="not enough recorded events")
                ans = (f"The engine has only {n} recorded {h} events with company financials, fewer than the {EL.MIN_EVENTS} it needs, so it gives no analog figures for {h}; this is not assessed, not zero. "
                       f"Past events are not a forecast.")
            else:
                facts = dict(hazard=h, n_events=a["n_events"], n_company_events=a["n_company_events"], median_pct=a["median_pct"], p25_pct=a["p25_pct"], p75_pct=a["p75_pct"],
                             share_decline_pct=a["share_decline_pct"], mdae_all_events=ev["median_abs_error_pct_points"]["all_events_median"], mdae_hazard=ev["median_abs_error_pct_points"]["hazard_median"],
                             finding="hazard type did not improve the prediction")
                ans = (f"Across {a['n_company_events']} company filings after {a['n_events']} recorded {h} events, the event-year revenue change had a median of {a['median_pct']}% "
                       f"(middle half {a['p25_pct']}% to {a['p75_pct']}%), and {a['share_decline_pct']}% of companies saw revenue fall. "
                       f"Tested on events it had not seen, knowing the hazard type did not improve the prediction (typical error {ev['median_abs_error_pct_points']['hazard_median']} points against {ev['median_abs_error_pct_points']['all_events_median']} "
                       f"for the all-event average), so these figures describe the past and are not a forecast of your revenue. Screening-grade.")
            q = rng.choice(_Q_FIN).format(h=h)
            out.append({"kind": "evfin", "split": split, "meta": {"qtype": "fin", "facts": json.dumps(facts), "nulls": ["revenue forecast"] if a is None else []},
                        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Engine results: {json.dumps(facts)}\n\nQuestion: {q}"}, {"role": "assistant", "content": ans}]})
    return out


def build(seed: int = 8, n_pred: int = 220):
    rng = random.Random(seed)
    loo = TC.loo(True)
    bt = {k: v for k, v in loo.items() if k != "detail"}
    ev = EL.loeo_eval()
    detail = loo["detail"]
    rows = []
    for _ in range(n_pred):
        rows.append(_pred_row(rng, rng.choice(range(40, 160, 1)), "train" if rng.random() < 0.85 else ("valid" if rng.random() < 0.4 else "test"), bt))
    names = sorted(d["storm"] for d in detail)
    held = set(rng.sample(names, max(8, len(names) // 6)))
    for d in detail:
        sp = "test" if d["storm"] in held else "train"
        for _ in range(3 if sp == "train" else 2):
            rows.append(_back_row(rng, d, sp))
    for sp, k in (("train", 5), ("valid", 1), ("test", 2)):
        rows += _fin_rows(rng, ev, k, sp)
    for r in rows:
        r["messages"][-1]["content"] = re.sub(r"\s+", " ", r["messages"][-1]["content"]).strip()
    return rows


def write(rows, out: Path, seed: int = 8):
    from .v7_data import write as W
    rep = W(rows, out, seed)
    (out / "eventlearn_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main():
    a = sys.argv[1:]
    seed = int(a[a.index("--seed") + 1]) if "--seed" in a else 8
    print(json.dumps(write(build(seed), Path(a[0]), seed), indent=1))


if __name__ == "__main__":
    main()
