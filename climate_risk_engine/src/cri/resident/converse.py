"""
cri/resident/converse.py — talk to the resident agent like a colleague, with no key and no cloud.

The agent's tools (twin, hazards, reports, research) are chosen by the trained router. Everything else (greetings, follow-ups,
"why is that?", "what should I do about it?", thinking out loud about a site) goes through the local language model
(Ollama, on this computer) with the conversation so far and the facts the engine holds about the place under discussion:

  CONTEXT = the place, the last result, the cached all-hazards profile for the place (status of every hazard, with null
            kept as null), facts the user asked the agent to remember, and what the agent recently did.

Rules put to the model, and checked afterwards:
  * numbers about a place come only from CONTEXT; a figure in the reply that appears nowhere in CONTEXT, the question or the
    conversation is reported as general knowledge, not engine output;
  * null = not assessed (never low, never zero); none_found = established absent; constructed = built from a fallback;
  * sources the engine found itself or inferred are not facts about the plant; user statements are the user's;
  * no personal investment or insurance advice.

If no model is running the agent says so plainly and how to start it. It never asks for an API key to talk: there is none.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

from . import memory as M

SYSTEM = (
    "You are ClimRisk's resident climate-risk analyst, running on the user's own computer. No account or key is involved. "
    "Talk like a thoughtful colleague: answer the question asked, reason step by step in your head, then give a clear answer in plain prose, 60 to 160 words. "
    "Use ONLY the CONTEXT for any number about a place; if the number is not in CONTEXT, say the engine has not produced it and offer to run it. "
    "Never invent a site-specific figure. 'null' means NOT ASSESSED (no data, nothing could be built): never call it low or zero. "
    "'none_found' means established absent. 'constructed' means built from a fallback or inference, lower confidence. "
    "A source marked inferred or low confidence is a hypothesis, not a fact about a plant. "
    "You can run these for the user when they give a place: all hazards at a coordinate, 3D twin with flood and wind, scenarios, reports, "
    "local air quality with nearby sources, research scans, and recall of past events from filings. Suggest one next step when it helps. "
    "Say clearly what a screening model cannot do. No personal investment or insurance-purchase advice. Be honest when you do not know."
)


def _profile_lines(lat: float, lon: float) -> list:
    try:
        from .. import siteprofile as SP
        p = json.loads(SP._cache_path(lat, lon).read_text())
    except Exception:                                              # noqa: BLE001
        return []
    out = []
    for e in p.get("hazards", {}).values():
        if e["status"] == "null":
            out.append(f'- {e["label"]}: null (not assessed): {str(e.get("reason") or "")[:100]}')
        elif e["status"] == "none_found":
            out.append(f'- {e["label"]}: none found (absent here)')
        else:
            v = "" if e.get("value") is None else f' {e["value"]:g} {e.get("unit") or ""}'.rstrip()
            out.append(f'- {e["label"]}: {e["status"]}{v}' + (f' {e["severity"]}' if e.get("severity") else ""))
    al = p.get("hazards", {}).get("air_local") or {}
    for s in (al.get("metrics") or {}).get("sources", [])[:6]:
        out.append(f'- nearby source: {s["kind"]} {s.get("name") or ""} {s.get("distance_km")} km at {s.get("bearing_deg")} deg, {s.get("confidence")}: {str(s.get("provenance"))[:80]}')
    return out


def build_context(session: str) -> str:
    parts = []
    pl = M.get_state("last_place")
    if pl:
        parts.append(f'Place under discussion: {pl.get("name")} ({pl["lat"]:.4f}, {pl["lon"]:.4f}).')
        prof = _profile_lines(pl["lat"], pl["lon"])
        if prof:
            parts.append("All-hazards profile held by the engine:\n" + "\n".join(prof))
        else:
            parts.append("The engine has not yet run the all-hazards profile for this place.")
        try:
            fs = M.facts(pl.get("name"), 6)
            if fs:
                parts.append("Facts the user asked me to remember: " + "; ".join(f["text"] for f in fs))
        except Exception:                                          # noqa: BLE001
            pass
    lr = M.get_state("last_result")
    if lr:
        parts.append(f'Last result: {lr.get("summary")}. {str(lr.get("explain") or "")[:400]}')
    try:
        rs = M.runs(3)
        if rs:
            parts.append("Recently I ran: " + "; ".join(f'{r["intent"]}: {str(r["summary"])[:90]}' for r in rs))
    except Exception:                                              # noqa: BLE001
        pass
    return "\n".join(parts) or "No site is under discussion yet."


def _nums(t: str) -> set:
    return {x.rstrip(".") for x in re.findall(r"\d+(?:\.\d+)?", t)}


def check_numbers(reply: str, allowed_text: str) -> list:
    ok = _nums(allowed_text) | {str(y) for y in range(1950, 2101)} | {str(i) for i in range(0, 11)} | {"100", "50", "60", "30", "20", "15", "5"}
    return sorted(n for n in _nums(reply) if n not in ok)


def no_model_reply(why: str, smalltalk: bool = False) -> dict:
    head = "Hello. " if smalltalk else ""
    return {"reply": head + ("I can talk this through with you once my local language model is running. It runs on this computer, so no key or account is needed. "
                             f"It is not running right now ({why}). Start Ollama (or run the app's local-AI setup) and ask again. "
                             "Meanwhile I can still run everything else: give me a coordinate or place and I will run every hazard, build the 3D twin, "
                             "find what emits nearby and screen the air, or answer from the sources I hold, with citations."),
            "blocks": [], "actions": [{"type": "say", "label": "What can you do?", "text": "what can you do"}], "mode": "no_model"}


def converse(session: str, text: str, history_n: int = 8, total_s: float = 120.0) -> dict:
    from . import answer as AN
    ls = AN.llm_status()
    if not ls["available"]:
        return no_model_reply(ls["why"], smalltalk=len(text.split()) <= 4)
    ctx = build_context(session)
    hist = [m for m in M.recent_messages(session, history_n * 2 + 2)]
    msgs = [{"role": "system", "content": SYSTEM}]
    for m in hist[-history_n * 2:]:
        role = "user" if m["role"] == "user" else "assistant"
        if msgs[-1]["role"] == role and role == "user":
            continue
        msgs.append({"role": role, "content": str(m["text"])[:700]})
    if msgs[-1]["role"] == "user" and msgs[-1]["content"][:200] == text[:200]:
        msgs.pop()                                                  # the current message is added once, below, with the context
    msgs.append({"role": "user", "content": f"CONTEXT (from the engine, the only source of numbers):\n{ctx}\n\nUser: {text}"})
    t0 = time.time()
    try:
        r = AN._ollama_post("/api/chat", json.dumps({"model": ls["model"], "messages": msgs, "stream": False, "keep_alive": "30m",
                                                       "options": {"temperature": 0.4, "num_ctx": 6144, "num_predict": 420}}).encode(), total_s)
        reply = (r.get("message") or {}).get("content", "").strip()
    except Exception as e:                                          # noqa: BLE001
        return {"reply": f"The local model did not answer ({type(e).__name__}: {str(e)[:120]}). Try again in a moment; the engine tools still work.", "blocks": [], "actions": [], "mode": "model_error"}
    if not reply:
        return {"reply": "The local model returned an empty answer. Try rephrasing; the engine tools still work.", "blocks": [], "actions": [], "mode": "model_error"}
    reply = re.sub(r"\n{3,}", "\n\n", reply)
    bad = check_numbers(reply, ctx + " " + text + " " + " ".join(m["content"] for m in msgs[1:-1]))
    if bad:
        reply += (f"\n\n(Note: the figure{'s' if len(bad) > 1 else ''} {', '.join(bad[:4])} did not come from the engine's results for this place; "
                  "treat them as general background, or ask me to run the engine to get a site-specific value.)")
    return {"reply": reply, "blocks": [], "actions": [], "mode": f"local language model ({ls['model']})", "elapsed_s": round(time.time() - t0, 1), "unsupported_numbers": bad}
