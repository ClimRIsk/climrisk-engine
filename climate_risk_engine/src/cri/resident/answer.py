"""
cri/resident/answer.py — answers a free-form question from what the agent has learned, with citations.

  1. search the knowledge index (cri.resident.knowledge)
  2. if the index does not cover the question, learn the topic from Wikipedia right now, then search again
  3. compose the answer:
       - if a local language model is running (Ollama on this computer: no key, no cloud), it writes the answer
         from the numbered passages only;
       - otherwise the answer is assembled from the most relevant sentences of those passages, each marked [n].
  4. say plainly when nothing reliable was found. Never fill the gap with invented text.

Every answer carries its sources (title, address, date learned) and which mode wrote it.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.request
from typing import Optional

from . import knowledge as K

_OLLAMA = os.environ.get("CRI_OLLAMA_URL", "http://127.0.0.1:11434")
_PREFER = ["climrisk-llm", "qwen2.5:7b", "qwen2.5:3b", "llama3.2:3b", "llama3.1:8b", "llama3.2", "llama3.1", "mistral", "gemma2", "phi3"]
_LLM_CACHE: dict = {"t": 0.0, "v": None}


def llm_status(force: bool = False) -> dict:
    """Is a local model reachable? Cached for 60 s. Set CRI_LLM=0 to ignore it."""
    if os.environ.get("CRI_LLM", "1") == "0":
        return {"available": False, "model": None, "why": "disabled (CRI_LLM=0)"}
    if not force and time.time() - _LLM_CACHE["t"] < 60 and _LLM_CACHE["v"] is not None:
        return _LLM_CACHE["v"]
    v = {"available": False, "model": None, "why": "Ollama is not running on this computer"}
    try:
        with urllib.request.urlopen(urllib.request.Request(_OLLAMA + "/api/tags"), timeout=1.5) as r:
            tags = [m["name"] for m in json.loads(r.read()).get("models", [])]
        if tags:
            want = os.environ.get("CRI_LLM_MODEL")
            norm = {t.split(":latest")[0]: t for t in tags}              # "climrisk-llm:latest" is found as "climrisk-llm"
            pick = want if want in tags else next((norm[p] for p in _PREFER if p in norm), None) or next((t for t in tags if not re.search(r"embed|nomic", t)), tags[0])
            v = {"available": True, "model": pick, "why": "ok"}
        else:
            v["why"] = "Ollama is running but has no model; run: ollama pull llama3.2"
    except Exception:                                                # noqa: BLE001
        pass
    _LLM_CACHE.update(t=time.time(), v=v)
    return v


# ── composing ────────────────────────────────────────────────────────────────

_DANGLING = re.compile(r"^(this|these|those|it|they|its|their|he|she|however|also|thus|therefore|moreover|additionally|furthermore|in addition|as a result|consequently|alternatively|but|and|so)\b", re.I)


def _sentences(t: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])", t) if 7 <= len(s.split()) <= 70]


def _extractive(question: str, passages: list, max_sent: int = 5) -> str:
    ts = list(dict.fromkeys(K._terms(question)))
    if not ts:
        return ""
    n = len(passages)
    df = {t: sum(1 for p in passages if K.has(t, p["text"].lower())) for t in ts}
    w = {t: math.log(1 + n / max(1, df[t])) + (0.4 if len(t) > 6 else 0) for t in ts}
    top = max((p["score"] for p in passages), default=1.0) or 1.0
    need = min(2, len(ts))                                           # a sentence must carry at least two of the question's terms
    cands = []
    for pi, p in enumerate(passages[:4]):                            # only the best-ranked passages feed the answer
        rel = max(0.25, p["score"] / top)
        sents = _sentences(p["text"])
        for si, s in enumerate(sents):
            if _DANGLING.match(s):                                   # "However, it …" / "As a result, they …" cannot stand alone
                continue
            low = s.lower()
            hit = [t for t in ts if K.has(t, low)]
            if len(hit) < need:
                continue
            score = rel * sum(w[t] for t in hit) / (1 + 0.012 * len(s.split())) + (0.3 if si == 0 else 0) - 0.15 * pi
            cands.append((score, pi, si, s))
    cands.sort(key=lambda x: -x[0])
    chosen, seen = [], set()
    for score, pi, si, s in cands:
        key = re.sub(r"\W+", " ", s.lower())[:60]
        if key in seen:
            continue
        seen.add(key); chosen.append((pi, si, s))
        if len(chosen) >= max_sent:
            break
    chosen.sort(key=lambda x: (x[0], x[1]))                          # read in source order
    return " ".join(f"{s} [{pi + 1}]" for pi, si, s in chosen)


LAST_LLM: dict = {"why": "not called", "raw": ""}


def _normalise_cites(out: str, n: int) -> str:
    """Small models cite as (1), [Source 2], [1, 3] — fold those into plain [n] markers."""
    out = re.sub(r"\[(?:source|src|s)\s*(\d+)\]", r"[\1]", out, flags=re.I)
    out = re.sub(r"\((?:source|sources|src)?\s*(\d+)\)", lambda m: f"[{m.group(1)}]" if int(m.group(1)) <= n else m.group(0), out, flags=re.I)
    out = re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]", lambda m: "".join(f"[{x}]" for x in re.findall(r"\d+", m.group(1))), out)
    return out


def _llm(question: str, passages: list, model: str, context: Optional[str] = None) -> Optional[str]:
    for strict in (False, True):
        out = _llm_once(question, passages, model, context, strict)
        if out:
            return out
        if not LAST_LLM["why"].startswith("the model answered without citing"):
            return None                                              # only an uncited answer is worth a stricter retry
    return None


def _llm_once(question: str, passages: list, model: str, context: Optional[str], strict: bool) -> Optional[str]:
    src = "\n\n".join(f"[{i + 1}] {p['title']}: {p['text']}" for i, p in enumerate(passages))
    sysmsg = ("You are ClimRisk's climate-risk analyst. Answer the question using ONLY the numbered sources. "
              "Cite every factual sentence with its source number in square brackets, like [2]. "
              "If the sources do not contain the answer, say exactly what is missing instead of guessing. "
              "Write 120 to 220 words in plain prose: lead with the answer, then the mechanism, then what it means for an asset owner or lender. No bullet lists.")
    if strict:
        sysmsg += (" STRICT FORMAT: end EVERY sentence with its source marker, for example: "
                   "'Storm surge is wind-driven sea-level rise [2]. River floods come from rainfall upstream [3].' "
                   "An answer with no [n] markers is invalid.")
    user = (f"Context from the user's session: {context}\n\n" if context else "") + f"Question: {question}\n\nSources:\n{src}"
    body = json.dumps({"model": model, "stream": False, "keep_alive": "30m",
                       "options": {"temperature": 0.0 if strict else 0.2, "num_ctx": 6144, "num_predict": 520},
                       "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]}).encode()
    try:
        req = urllib.request.Request(_OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            out = json.loads(r.read()).get("message", {}).get("content", "").strip()
        out = _normalise_cites(out, len(passages))
        cited = {int(x) for x in re.findall(r"\[(\d+)\]", out)}
        LAST_LLM.update(raw=out[:400])
        if not out:
            LAST_LLM["why"] = "the model returned nothing"
            return None
        if not cited:
            LAST_LLM["why"] = "the model answered without citing a source, so the answer was rejected"
            return None
        if max(cited) > len(passages):
            LAST_LLM["why"] = "the model cited a source number that does not exist, so the answer was rejected"
            return None
        LAST_LLM["why"] = "ok"
        return out
    except Exception as e:                                           # noqa: BLE001
        LAST_LLM.update(why=f"call to the local model failed: {type(e).__name__}: {e}"[:300], raw="")
        return None


def _confident(ps: list) -> bool:
    return bool(ps) and ps[0]["coverage"] >= 0.5 and ps[0]["score"] >= 3.0


def ask(question: str, context: Optional[str] = None, allow_learn: bool = True, k: int = 6) -> dict:
    t0 = time.time()
    ps = K.search(question, k)
    learned: list = []
    if not _confident(ps) and allow_learn:
        r = K.learn_topic(" ".join(K._terms(question)[:8]) or question)
        learned = r.get("learned", [])
        if learned:
            ps = K.search(question, k)
    if not ps or not (ps[0]["coverage"] >= 0.34):
        return {"ok": False, "reply": ("I could not find reliable material on that, so I will not guess. "
                                       "Give me a page to read (say \"learn from <web address>\") or rephrase with the hazard, place or regulation you mean."),
                "sources": [], "mode": "none", "learned": learned, "elapsed_s": round(time.time() - t0, 1)}
    ls = llm_status()
    text, mode = None, "extractive"
    if ls["available"]:
        text = _llm(question, ps, ls["model"], context)
        if text:
            mode = f"language model ({ls['model']}, running on this computer)"
    if not text:
        text = _extractive(question, ps)
    if not text:
        return {"ok": False, "reply": "I found related pages but no sentence that answers the question directly; the sources are listed below.", "sources": _src(ps), "mode": "none", "learned": learned,
                "elapsed_s": round(time.time() - t0, 1)}
    used = sorted({int(x) for x in re.findall(r"\[(\d+)\]", text) if int(x) <= len(ps)})
    return {"ok": True, "reply": text, "sources": _src(ps, used), "mode": mode, "learned": learned, "elapsed_s": round(time.time() - t0, 1),
            "llm_note": (LAST_LLM["why"] if ls["available"] and mode == "extractive" else None)}


def _src(ps: list, used: Optional[list] = None) -> list:
    out, seen = [], set()
    for i, p in enumerate(ps, 1):
        if used and i not in used:
            continue
        key = p["url"]
        if key in seen:
            continue
        seen.add(key)
        out.append({"n": i, "title": p["title"], "url": p["url"], "learned": p["learned"]})
    return out
