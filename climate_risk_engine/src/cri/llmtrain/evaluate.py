"""Score models served by Ollama on the held-out set.  python -m cri.llmtrain.evaluate DS_DIR OUT.json model1 [model2 ...] [--limit N]

Metrics are mechanical, so a model cannot win by sounding good:
  qa       cites at least one source, every cited number exists, cites the source that holds the answer, share of sentences carrying a citation
  refuse   says what is missing instead of answering from nothing
  insurer  every number in the answer appears in the facts or the question (no invented figures); declines what the facts lack
  gold     word overlap with the reference answer; numbers that are in neither question nor reference (fabrication proxy)
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from collections import defaultdict

OLLAMA = "http://127.0.0.1:11434"
NUM = re.compile(r"\d[\d,]*\.?\d*")
MISSING = re.compile(r"do(es)? not (cover|contain|include|discuss|address)|cannot answer|can't answer|not enough|no source|is missing|won't guess|will not guess|without guessing|cannot say|can't say|need a source|would need", re.I)


def chat(model, messages, n_ctx=6144, n_pred=420):
    body = json.dumps({"model": model, "stream": False, "keep_alive": "20m", "options": {"temperature": 0.0, "num_ctx": n_ctx, "num_predict": n_pred}, "messages": messages}).encode()
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()).get("message", {}).get("content", "").strip()


def nums(t):
    return {x.replace(",", "").rstrip(".").lstrip("0") or "0" for x in NUM.findall(t)}


def words(t):
    return re.findall(r"[a-z]{3,}", t.lower())


def score(row, out):
    k = row["kind"]
    u = row["messages"][1]["content"]
    ref = row["messages"][2]["content"]
    s = {}
    if k == "chain":
        cites = {int(x) for x in re.findall(r"\[(\d+)\]", out)}
        need = set(row["meta"]["need"])
        s["cites_all_needed"] = float(need <= cites)
        s["cites_valid"] = float(bool(cites) and max(cites) <= row["meta"]["n"])
        s["no_invented_numbers"] = float(nums(out) <= nums(u) | nums(out) & set())
        s["hedges_inference"] = float(bool(re.search(r"do not (give|state|contain)|not stated|inference|no (carbon|volumes|damage)", out)))
    elif k in ("case", "case_gap"):
        cites = {int(x) for x in re.findall(r"\[(\d+)\]", out)}
        need = set(row["meta"].get("need", []))
        s["numbers_ok"] = float(nums(out) <= nums(u))
        s["cites_valid"] = float(not cites or max(cites) <= row["meta"]["n"])
        if k == "case":
            s["cites_needed"] = len(need & cites) / max(1, len(need))
            norm = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", r" \2\1", out)
            sents = [x for x in re.split(r"(?<=[.!?])\s+", norm) if len(x.split()) > 4 and not x.lower().startswith("analysis")]
            s["sent_cited"] = sum(bool(re.search(r"\[\d+\]", x)) for x in sents) / max(1, len(sents))
            s["flags_inference"] = float("analysis (not stated" in out.lower())
        else:
            s["says_missing"] = float(bool(MISSING.search(out)))
    elif k == "narr":
        s["numbers_ok"] = float(nums(out) <= nums(row["meta"]["facts"]) | nums(u))
        s["says_not_forecast"] = float("forecast" in out.lower())
        s["length_ok"] = float(40 <= len(out.split()) <= 150)
    elif k in ("qa", "refuse"):
        n = row["meta"].get("n", 0)
        cites = [int(x) for x in re.findall(r"\[(\d+)\]", out)]
        norm = re.sub(r"([.!?])\s*((?:\[\d+\]\s*)+)", r" \2\1", out)
        sents = [x for x in re.split(r"(?<=[.!?])\s+", norm) if len(x.split()) > 4]
        if k == "qa":
            s["cited"] = float(bool(cites))
            s["cites_valid"] = float(bool(cites) and max(cites) <= n)
            s["cites_target"] = float(row["meta"]["target"] in cites)
            s["sent_cited"] = sum(bool(re.search(r"\[\d+\]", x)) for x in sents) / max(1, len(sents))
        else:
            s["says_missing"] = float(bool(MISSING.search(out)))
            s["cites_valid"] = float(not cites or max(cites) <= n)
    elif k in ("siteprofile", "chat"):
        nul = row["meta"].get("nulls", [])
        low = out.lower()
        s["numbers_ok"] = float(nums(out) <= nums(row["meta"]["facts"]) | nums(u) | {"0"})
        if nul:
            s["names_nulls"] = sum(n.lower() in low for n in nul) / len(nul)
            s["says_not_assessed"] = float("not assessed" in low or "was not assessed" in low or "were not assessed" in low)
            tail = low.split("null")[0] if "null" in low else low
            s["null_not_called_low"] = float(not any(re.search(re.escape(n.lower()) + r"[^.;]{0,40}\b(low|zero|negligible|safe)\b", low) for n in nul))
        s["length_ok"] = float(20 <= len(out.split()) <= 140)
    elif k == "scenario":
        s["numbers_ok"] = float(nums(out) <= nums(row["meta"]["facts"]) | nums(u) | {"0", "1"})
        s["says_no_location"] = float(bool(re.search(r"no location|coordinates", out, re.I)))
        s["says_not_assessed"] = float(bool(re.search(r"not assessed|null", out, re.I)))
        if "user-stated exceedance" in row["meta"]["facts"]:
            s["flags_unverified_input"] = float(bool(re.search(r"your own input|not verified|has not verified", out, re.I)))
        s["length_ok"] = float(40 <= len(out.split()) <= 170)
    elif k == "stress":
        s["numbers_ok"] = float(nums(out) <= nums(row["meta"]["facts"]) | nums(u) | {"0", "1", "2", "3", "4", "5"})
        if row["meta"].get("nulls") or row["meta"].get("status") in ("null", "partial"):
            s["says_not_modelled"] = float(bool(re.search(r"not modelled|not known|not given|not assessed|no decision owner|is null|left null", out, re.I)))
        s["says_screening"] = float(bool(re.search(r"screening|not a forecast|does not forecast", out, re.I)))
        s["length_ok"] = float(25 <= len(out.split()) <= 170)
    elif k in ("evtc", "evfin"):
        low = out.lower()
        s["numbers_ok"] = float(nums(out) <= nums(row["meta"]["facts"]) | nums(u) | {"0", "1", "3", "80", "10", "90", "50"})
        s["says_not_forecast"] = float(bool(re.search(r"not a forecast|not forecast|does not forecast", low)))
        s["length_ok"] = float(25 <= len(out.split()) <= 160)
        if k == "evtc":
            s["says_not_insured_or_site"] = float(bool(re.search(r"not an insured|not insured|whole event|not the loss at one site|total economic", low)))
            if row["meta"]["qtype"] == "predict":
                s["gives_range"] = float(bool(re.search(r"range|between|to \$", low)))
                s["names_misses"] = float(any(n in low for n in ("katrina", "harvey", "allison", "rain", "surge")))
        else:
            if row["meta"].get("nulls"):
                s["says_not_enough"] = float(bool(re.search(r"not enough|fewer than|too few|not assessed|only \d+ recorded", low)))
            else:
                s["says_no_skill"] = float(bool(re.search(r"did not improve|no better|not improve|no skill", low)))
    elif k == "insurer":
        allowed = nums(row["meta"]["facts"]) | nums(u)
        used = nums(out)
        s["numbers_ok"] = float(used <= allowed)
        s["new_numbers"] = float(len(used - allowed))
        wc = len(out.split())
        s["length_ok"] = float(40 <= wc <= 190)
        if row["meta"]["qtype"] in ("cost_unknown",) or (row["meta"]["qtype"] == "climate" and "Climate SSP" not in row["meta"]["facts"]) or (row["meta"]["qtype"] == "reinsurance" and "Reinsurance ceded" not in row["meta"]["facts"]):
            s["declines_unknown"] = float(bool(MISSING.search(out) or re.search(r"cannot|can't|not (in|part of) the facts", out, re.I)))
    elif k == "gold":
        a, b = set(words(out)), set(words(ref))
        p = len(a & b) / max(1, len(a))
        r = len(a & b) / max(1, len(b))
        s["overlap_f1"] = 0.0 if not (p + r) else 2 * p * r / (p + r)
        s["invented_numbers"] = float(len(nums(out) - nums(ref) - nums(u)))
        s["length_ok"] = float(40 <= len(out.split()) <= 170)
    return s


def main():
    a = sys.argv[1:]
    limit = None
    if "--limit" in a:
        i = a.index("--limit"); limit = int(a[i + 1]); del a[i:i + 2]
    ds, outp, models = a[0], a[1], a[2:]
    rows = [json.loads(l) for l in open(f"{ds}/test_meta.jsonl")]
    if limit:
        by = defaultdict(list)
        for r in rows:
            by[r["kind"]].append(r)
        rows = [r for k in by for r in by[k][:limit]]
    res = {}
    for m in models:
        agg, samples, t0 = defaultdict(list), [], time.time()
        for i, r in enumerate(rows):
            try:
                out = chat(m, r["messages"][:2])
            except Exception as e:                                   # noqa: BLE001
                out = f"[error {type(e).__name__}]"
            sc = score(r, out)
            for k, v in sc.items():
                agg[(r["kind"], k)].append(v)
            if len(samples) < 12 and i % 5 == 0:
                samples.append({"kind": r["kind"], "q": r["messages"][1]["content"][-300:], "out": out[:600]})
            print(f"{m} {i + 1}/{len(rows)}", flush=True)
        res[m] = {"metrics": {f"{k}.{n}": round(sum(v) / len(v), 3) for (k, n), v in sorted(agg.items())}, "n_rows": len(rows), "seconds": round(time.time() - t0), "samples": samples}
        json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({m: v["metrics"] for m, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
