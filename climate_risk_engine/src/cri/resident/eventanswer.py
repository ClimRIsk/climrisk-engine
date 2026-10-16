"""
cri/resident/eventanswer.py: the chat route for questions the engine can answer from RECORDED EVENTS.

  "what total loss from a 115 kt hurricane?"   -> cri.eventlearn_tc  (US storm wind -> total economic cost, fitted on 61 storms, tested leave-one-storm-out)
  "will revenue fall after a flood?"           -> cri.eventlearn     (company revenue after recorded events; no skill beyond the all-event average)

The numbers come from the engine. The local language model (v8, trained on exactly this fact layout) only words them; every figure in its reply is checked against the
facts, and if the model is absent, slow, or quotes a figure that is not in the facts, the reply is the code-built text instead. Nothing here forecasts a site or an insured loss.
"""
from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from typing import Optional

_STORM = re.compile(r"\b(hurricane|tropical storm|typhoon|cyclone|landfall|storm)\b", re.I)
_LOSS = re.compile(r"\b(loss(?:es)?|cost|costs|damage|damages|expect|how much|economic|billion|impact|estimate)\b", re.I)
_REV = re.compile(r"\b(revenue|revenues|sales|turnover)\b", re.I)
_REV_VERB = re.compile(r"\b(fall|drop|decline|hit|impact|forecast|predict|after|following|effect|expect)\b", re.I)
_NON_US = re.compile(r"\b(india|bangladesh|china|philippines|japan|vietnam|australia|europe|uk|united kingdom|mexico|caribbean|pacific|indian ocean|bay of bengal|taiwan|korea|pakistan|oman|new zealand)\b", re.I)
_HAZ = (("tropical cyclone", r"hurricane|tropical|typhoon|cyclone"), ("flood", r"flood"), ("wildfire", r"wild\s?fire|\bfire\b|bushfire"), ("severe convective storm", r"tornado|convective|thunderstorm|derecho"),
        ("hail", r"hail"), ("extreme cold", r"freeze|extreme cold|cold snap|winter storm"), ("drought", r"drought"))
_CAT_KT = {1: 73, 2: 89, 3: 104, 4: 125, 5: 145}                      # mid-points of the Saffir-Simpson 1-minute wind bands in knots


def _wind_kt(text: str) -> tuple:
    """(knots, note) from the text, or (None, '')."""
    t = text.lower()
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:kt|kts|knots?)\b", t)
    if m:
        return float(m.group(1)), ""
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*mph\b", t)
    if m:
        v = float(m.group(1)) / 1.15078
        return round(v), f"{m.group(1)} mph is {v:.0f} knots"
    m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:km/?h|kph|kmh)\b", t)
    if m:
        v = float(m.group(1)) / 1.852
        return round(v), f"{m.group(1)} km/h is {v:.0f} knots"
    m = re.search(r"\bcat(?:egory)?\s*([1-5])\b", t)
    if m:
        return float(_CAT_KT[int(m.group(1))]), f"category {m.group(1)} taken as {_CAT_KT[int(m.group(1))]} knots (mid-point of its wind band)"
    return None, ""


@lru_cache(maxsize=1)
def _backtest() -> dict:
    from .. import eventlearn_tc as TC
    lo = TC.loo(True)
    return {k: v for k, v in lo.items() if k != "detail"}


def detect(text: str) -> Optional[dict]:
    if _STORM.search(text) and _LOSS.search(text) and not _REV.search(text):
        w, note = _wind_kt(text)
        return {"kind": "tc", "wind_kt": w, "note": note, "non_us": bool(_NON_US.search(text))}
    if _REV.search(text) and _REV_VERB.search(text):
        for h, rx in _HAZ:
            if re.search(rx, text, re.I):
                return {"kind": "fin", "hazard": h}
    return None


def _b(x: float) -> str:
    return f"{x / 1000:.1f}"


def tc_facts(w: float) -> dict:
    from .. import eventlearn_tc as TC
    bt, p = _backtest(), TC.predict(w, date.today().year)
    return dict(model="US tropical cyclones 1980-2024", n_storms=p["n_storms"], wind_kt=int(round(w)), expected_musd=p["expected_musd"], expected_busd=float(_b(p["expected_musd"])),
                p10_busd=float(_b(p["p10_musd"])), p90_busd=float(_b(p["p90_musd"])), backtest_median_factor_error=bt["median_factor_error"],
                baseline_median_factor_error=bt["baseline_median_factor_error"], within_factor_3_pct=bt["within_factor_3_pct"], baseline_within_factor_3_pct=bt["baseline_within_factor_3_pct"],
                misses=["Katrina", "Harvey", "Allison"], cost="total economic cost for the whole US event, not an insured loss, not a site loss")


def tc_text(f: dict) -> str:
    return (f"For a storm reaching {f['wind_kt']} knots at the US coast, the model fitted on {f['n_storms']} recorded US storms expects a total economic cost of about ${f['expected_busd']} billion, "
            f"with an 80% range of ${f['p10_busd']} to ${f['p90_busd']} billion. That is the cost of the whole event across the country, not an insured loss and not the loss at one site. "
            f"Tested on storms it had not seen, its typical error was a factor of {f['backtest_median_factor_error']}, against {f['baseline_median_factor_error']} for guessing the typical storm, and "
            f"{f['within_factor_3_pct']}% of storms were within a factor of 3 (guessing: {f['baseline_within_factor_3_pct']}%). It underestimates rain and surge storms such as Katrina, Harvey and Allison. "
            "Screening-grade, not a forecast.")


def fin_facts(h: str) -> dict:
    from .. import eventlearn as EL
    rows = EL.load()
    a = EL.analogs(h, rows=rows)
    ev = EL.loeo_eval(rows=rows, boot=50)
    if a is None:
        n = len({r["event"] for r in rows if r["hazard"] == h})
        return dict(hazard=h, recorded_events=n, minimum_events=EL.MIN_EVENTS, status="not enough recorded events")
    return dict(hazard=h, n_events=a["n_events"], n_company_events=a["n_company_events"], median_pct=a["median_pct"], p25_pct=a["p25_pct"], p75_pct=a["p75_pct"], share_decline_pct=a["share_decline_pct"],
                mdae_all_events=ev["median_abs_error_pct_points"]["all_events_median"], mdae_hazard=ev["median_abs_error_pct_points"]["hazard_median"], finding="hazard type did not improve the prediction")


def fin_text(f: dict) -> str:
    from .. import eventlearn as EL
    h = f["hazard"]
    if f.get("status"):
        return (f"The engine has only {f['recorded_events']} recorded {h} events with company financials, fewer than the {EL.MIN_EVENTS} it needs, so it gives no analog figures for {h}; "
                "this is not assessed, not zero. Past events are not a forecast.")
    return (f"Across {f['n_company_events']} company filings after {f['n_events']} recorded {h} events, the event-year revenue change had a median of {f['median_pct']}% "
            f"(middle half {f['p25_pct']}% to {f['p75_pct']}%), and {f['share_decline_pct']}% of companies saw revenue fall. Tested on events it had not seen, knowing the hazard type did not improve "
            f"the prediction (typical error {f['mdae_hazard']} points against {f['mdae_all_events']} for the all-event average), so these figures describe the past and are not a forecast of your revenue. Screening-grade.")


def _word(facts: dict, question: str, fallback: str, total_s: float = 70.0) -> tuple:
    """The local model words the facts; any figure not in the facts, or no answer in time, and the code-built text is used instead."""
    from . import answer as AN, converse as CV
    from ..llmtrain.eventlearn_data import SYS
    try:
        ls = AN.llm_status()
        if not ls["available"]:
            return fallback, "engine text (the local model is not running)"
        r = AN._ollama_post("/api/chat", json.dumps({"model": ls["model"], "stream": False, "keep_alive": "30m", "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 260},
                                                     "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Engine results: {json.dumps(facts)}\n\nQuestion: {question}"}]}).encode(), total_s)
        out = ((r.get("message") or {}).get("content") or "").strip()
        if len(out) < 60 or CV.check_numbers(out, json.dumps(facts) + " " + question):
            return fallback, "engine text (the model's wording was rejected: it quoted a figure that is not in the engine results)"
        return out, f"local language model ({ls['model']}), figures checked against the engine"
    except Exception as e:                                                  # noqa: BLE001
        return fallback, f"engine text ({type(e).__name__})"


def answer(text: str, d: Optional[dict] = None, use_llm: bool = True) -> Optional[dict]:
    d = d or detect(text)
    if not d:
        return None
    if d["kind"] == "tc":
        if d.get("non_us"):
            return {"reply": "The storm-cost model was fitted on 61 storms that made US landfall, 1980 to 2024. It is not tested outside the US, so I will not extrapolate it to that place. "
                             "For a site anywhere, the engine's own cyclone run (IBTrACS tracks and a wind field at the coordinates) gives wind return periods and the loss curve; give me the coordinates.",
                    "blocks": [], "actions": [], "mode": "engine (scope refused)"}
        if d.get("wind_kt") is None:
            return {"reply": "I can give the recorded-storm estimate if you tell me the peak wind at the coast in knots (or mph, or the Saffir-Simpson category). "
                             "The model links US landfalling storm wind to total economic cost and has been tested on storms it had not seen.", "blocks": [], "actions": [], "mode": "engine"}
        f = tc_facts(d["wind_kt"])
        base = tc_text(f)
        body, mode = _word(f, text, base) if use_llm else (base, "engine text")
        pre = (d["note"] + ". ") if d.get("note") else ""
        rows = [["Peak wind at the coast", f"{f['wind_kt']} kt"], ["Expected total economic cost", f"${f['expected_busd']} bn"], ["80% range", f"${f['p10_busd']} to ${f['p90_busd']} bn"],
                ["Backtest: typical error (guessing: )".replace("(guessing: )", f"(guessing {f['baseline_median_factor_error']}x)"), f"{f['backtest_median_factor_error']}x"],
                ["Backtest: storms within 3x", f"{f['within_factor_3_pct']}% (guessing {f['baseline_within_factor_3_pct']}%)"], ["Not covered", "insured loss, one site, rain and surge storms"]]
        return {"reply": pre + body, "blocks": [{"type": "table", "head": ["Recorded-storm model (US, 1980-2024)", "Value"], "rows": rows}], "actions": [], "mode": mode, "facts": f}
    f = fin_facts(d["hazard"])
    base = fin_text(f)
    body, mode = _word(f, text, base) if use_llm else (base, "engine text")
    return {"reply": body, "blocks": [], "actions": [], "mode": mode, "facts": f}
