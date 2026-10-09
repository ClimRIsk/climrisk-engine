"""Case library tools: verify the facts against their sources, then build training rows that teach case-based reasoning.

    python -m cri.llmtrain.cases verify OUT.json          fetch every source page, check each fact's figures appear on it
    python -m cri.llmtrain.cases build  VERIFIED.json OUT_DIR   write cases_train/valid/test rows (jsonl)

Behaviours taught (every answer is composed from the same fields that built the source blocks, so it is correct by construction):
  summary      what happened to the assets and the financials, with the figures, each sentence cited
  response     what the company or institution did about it, by type
  gap          the sources do not report X  ->  say so, give what they do report, say what would be needed
  analog       an asset and hazard are described; pick the closest cases, name the channels, cite the evidence, and flag the inference
  uncertainty  several estimates for one event; cite each and explain why they differ
  unknown      an event that is not in the sources  ->  say the sources do not cover it
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

from .cases_data import CASES

SYS = ("You are ClimRisk's climate-risk analyst. Answer using ONLY the numbered sources. Cite every factual sentence with its source number in square brackets, like [2]. "
       "If the sources do not contain the answer, say exactly what is missing instead of guessing. Keep any inference clearly separate by starting it with 'Analysis (not stated in the sources):'. "
       "Never add a number that is not in the sources.")

HELD_OUT = ["basf-rhine-2018", "panama-2023", "tsmc-2021", "valencia-2024", "rs-brazil-2024", "kariba-2024"]

CH_DESC = {
    "asset_damage": "physical damage to buildings and equipment",
    "business_interruption": "lost revenue while the site or service is closed",
    "supply_chain": "disruption passed on through suppliers and customers",
    "input_shortage": "a shortage of an input such as power, water, transport or feedstock",
    "liability": "legal liability and compensation claims",
    "regulation_policy": "regulatory or policy changes",
    "market_price": "price moves in the markets the firm sells into or buys from",
    "financing_liquidity": "pressure on credit ratings, liquidity and capital raising",
    "insurance_availability": "the price and availability of insurance and reinsurance",
    "sovereign_backstop": "a state fund, public insurer or government support absorbing part of the loss",
    "demand_shift": "a shift in demand away from the firm's product",
}
UNC = {"jebi-2018": [0, 1, 2], "pakistan-2022": [0, 1, 2, 4, 5], "ian-2022": [0, 1], "valencia-2024": [1, 3, 4], "ahr-2021": [1, 4]}
SHORT = {c["id"]: c["title"].split(" (")[0] for c in CASES}
OUT_OF_SCOPE = [
    "What did Hurricane Katrina cost Entergy in 2005?", "How much did the Fukushima accident cost TEPCO?", "What was Con Edison's loss from Superstorm Sandy?",
    "How did the 2019 to 2020 Black Summer bushfires affect Australian insurers' results?", "What did Cyclone Gabrielle cost New Zealand insurers?",
    "How much did the 2021 Suez Canal blockage cost shipping lines?", "What did the 2012 US drought cost corn growers?", "How did the 2003 European heatwave affect French power prices?",
    "What did Hurricane Michael cost Tyndall Air Force Base?", "How did the 2022 Yangtze drought affect Chinese hydropower and factories?",
    "What did the 2017 Atlantic hurricanes cost Lloyd's of London?", "How much did the 2010 Russian wildfires cost the wheat sector?",
]
SYNTH_QS = ["Summarise what happened to the finances in {s}.", "What were the main financial effects in {s}, with the figures?", "What does the evidence say about the damage and costs in {s}?",
            "Walk me through the impact of {s} on the assets and the numbers."]
RESP_QS = ["How did the companies and institutions respond in {s}?", "What actions were taken in {s} to deal with the loss?", "What was done about it in {s}?"]
GAP_QS = ["Do the sources say anything about {g}?", "What do the sources report about {g}?"]
ANALOG_QS = ["A company owns {a}. A hazard of this type occurs: {d}. Using only the cases, which channels should it check first, and what has worked before?",
             "Our asset is {a}, exposed to {d}. What do the cases suggest about how the loss would reach the financials and how others responded?"]
UNC_QS = ["Why do the loss estimates for {s} differ?", "The numbers reported for {s} do not agree. What do the sources say and why might they differ?"]


# ── verification ─────────────────────────────────────────────────────────────

def _nums(t: str) -> list:
    return [x.replace(",", "") for x in re.findall(r"\d[\d,]*\.?\d*", t) if re.search(r"\d", x)]


_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
_SEC_UA = "ClimRisk research (open-source climate-risk project; polite crawler) shrinivashdkannan@gmail.com"


def _fetch(url: str) -> str:
    """Fetch a source page: SEC-compliant agent for sec.gov, a browser agent elsewhere, three tries, and the real error text kept."""
    import time
    import urllib.request
    ua = _SEC_UA if "sec.gov" in url else _BROWSER_UA
    last = None
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8", "Accept-Language": "en", "Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read(4_000_000)
                enc = r.headers.get_content_charset() or "utf-8"
            return raw.decode(enc, errors="replace")
        except Exception as e:                                       # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise last


def verify(out: Path) -> dict:
    from ..resident import knowledge as K
    res, pages = {}, {}
    for c in CASES:
        for si, (url, outlet, tier) in enumerate(c["sources"]):
            try:
                raw = _fetch(url)
                text = K._strip_html(raw)[1] if "<" in raw[:3000] else raw
                pages[(c["id"], si)] = re.sub(r"[ \t]+", " ", text).replace(",", "").lower()
            except Exception as e:                                   # noqa: BLE001
                pages[(c["id"], si)] = None
                res[f"{c['id']}#src{si}"] = {"fetch": f"{type(e).__name__}: {str(e)[:100]}"}
        for fi, (text, si) in enumerate(c["facts"]):
            page = pages.get((c["id"], si))
            tier = c["sources"][si][2]
            if page is None:
                st = "read" if tier == "read" else "unfetchable"
                res[f"{c['id']}#{fi}"] = {"status": st, "why": "source page could not be fetched"}
                continue
            ns = [n for n in _nums(text)]
            found = [n for n in ns if n in page]
            words = {w for w in re.findall(r"[a-zà-ÿ]{6,}", text.lower())}
            wf = sum(w in page for w in words) / max(1, len(words))
            ok = (len(found) >= max(1, round(0.8 * len(ns))) if ns else wf >= 0.6) and wf >= 0.35
            res[f"{c['id']}#{fi}"] = {"status": "verified" if ok else ("read" if tier == "read" else "unverified"), "numbers": f"{len(found)}/{len(ns)}", "words": round(wf, 2)}
    from collections import Counter
    n_src = sum(len(c["sources"]) for c in CASES)
    n_fail = sum(1 for k in res if "#src" in k)
    print(Counter(v.get("status") for k, v in res.items() if "#src" not in k), f"sources fetched {n_src - n_fail}/{n_src}")
    if n_fail > 0.5 * n_src:                                          # a network outage must not be recorded as 'unverifiable': keep the evidence, write nothing the pipeline would trust
        Path(str(out) + ".failed").write_text(json.dumps(res, indent=1))
        raise SystemExit(f"verification aborted: only {n_src - n_fail} of {n_src} source pages could be fetched (see {out}.failed). Re-run when the network is up.")
    out.write_text(json.dumps(res, indent=1))
    return res


def _status(vf, cid, fi):
    return (vf or {}).get(f"{cid}#{fi}", {}).get("status", "reported")


def usable(cards, vf, allow_unverified=False):
    """-> cards with only the facts that are verified or read (or all of them if allowed)."""
    out = []
    for c in cards:
        keep = [(i, f) for i, f in enumerate(c["facts"]) if allow_unverified or _status(vf, c["id"], i) in ("verified", "read")]
        if len(keep) >= 2:
            d = dict(c)
            d["facts"] = [f for _, f in keep]
            d["_idx"] = [i for i, _ in keep]
            out.append(d)
    return out


# ── rendering ────────────────────────────────────────────────────────────────

def _src(c, si):
    return c["sources"][si][1]


def _clean(t):
    return t.rstrip(".").rstrip()


def _blocks(items, rng):
    """items [(text, key)] -> numbered block string and key->position"""
    order = list(range(len(items)))
    rng.shuffle(order)
    pos, lines = {}, []
    for n, k in enumerate(order, 1):
        text, key = items[k]
        pos[key] = n
        lines.append(f"[{n}] {text}")
    return "\n\n".join(lines), pos


def _fact_item(c, fi):
    text, si = c["facts"][fi]
    return (f"{SHORT[c['id']]} ({_src(c, si)}): {text}", (c["id"], "f", fi))


def _frame_item(c):
    return (f"{SHORT[c['id']]} (case summary): the driver was {c['driver']}; the exposed assets were {c['assets']}; the event took place in {c['region']} in {c['year']}.", (c["id"], "frame"))


def _resp_item(c, ri):
    typ, text, si = c["responses"][ri]
    return (f"{SHORT[c['id']]} ({_src(c, si)}), response ({typ}): {text}", (c["id"], "r", ri))


def _ch_sentence(c):
    parts = [CH_DESC[x] for x in c["channels"]]
    return "Analysis (not stated in the sources): the loss reached the financials through " + ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] + "."


def _row(kind, q, blk, ans, need, n, split, meta=None):
    m = {"need": need, "n": n}
    if meta:
        m.update(meta)
    return {"kind": kind, "split": split, "meta": m,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {q}\n\nSources:\n{blk}"}, {"role": "assistant", "content": ans}]}


def _distract(cards, c, rng, k=2):
    others = [x for x in cards if x["id"] != c["id"]]
    out = []
    for o in rng.sample(others, min(k, len(others))):
        out.append(_fact_item(o, rng.randrange(len(o["facts"]))))
    return out


def _cite(text, p):
    return f"{_clean(text)} [{p}]."


def summary(c, cards, rng, split):
    k = min(len(c["facts"]), rng.choice([3, 4]))
    fis = sorted(rng.sample(range(len(c["facts"])), k))
    items = [_frame_item(c)] + [_fact_item(c, i) for i in fis] + _distract(cards, c, rng)
    blk, pos = _blocks(items, rng)
    s = f"{_cite('In ' + SHORT[c['id']] + ', the driver was ' + c['driver'] + ' and the exposed assets were ' + c['assets'], pos[(c['id'], 'frame')])} "
    s += " ".join(_cite(c["facts"][i][0], pos[(c["id"], "f", i)]) for i in fis) + " " + _ch_sentence(c)
    need = [pos[(c["id"], "frame")]] + [pos[(c["id"], "f", i)] for i in fis]
    return _row("case", rng.choice(SYNTH_QS).format(s=SHORT[c["id"]]), blk, s, need, len(items), split, {"type": "summary", "case": c["id"]})


def response(c, cards, rng, split):
    if not c["responses"]:
        return None
    ris = list(range(len(c["responses"])))
    items = [_resp_item(c, i) for i in ris] + [_fact_item(c, rng.randrange(len(c["facts"])))] + _distract(cards, c, rng)
    blk, pos = _blocks(items, rng)
    s = " ".join(f"{c['responses'][i][0].capitalize()} response: {_cite(c['responses'][i][1], pos[(c['id'], 'r', i)])}" for i in ris)
    g = rng.choice(c["gaps"])
    s += f" The sources do not report {g}."
    need = [pos[(c["id"], "r", i)] for i in ris]
    return _row("case", rng.choice(RESP_QS).format(s=SHORT[c["id"]]), blk, s, need, len(items), split, {"type": "response", "case": c["id"]})


def gap(c, cards, rng, split):
    g = rng.choice(c["gaps"])
    fis = rng.sample(range(len(c["facts"])), min(2, len(c["facts"])))
    items = [_frame_item(c)] + [_fact_item(c, i) for i in fis] + _distract(cards, c, rng)
    blk, pos = _blocks(items, rng)
    have = " ".join(_cite(c["facts"][i][0], pos[(c["id"], "f", i)]) for i in fis)
    s = (f"The sources provided do not cover {g}. What they do report is this. {have} To answer the question I would need the company's later filings or an official statement on it.")
    return _row("case_gap", rng.choice(GAP_QS).format(g=g), blk, s, [pos[(c["id"], "f", i)] for i in fis], len(items), split, {"type": "gap", "case": c["id"]})


def analog(c, cards, rng, split):
    partners = [o for o in cards if o["id"] != c["id"] and set(o["channels"]) & set(c["channels"])]
    if not partners:
        return None
    o = rng.choice(partners)
    shared = [x for x in c["channels"] if x in o["channels"]]
    fa = sorted(rng.sample(range(len(c["facts"])), min(2, len(c["facts"]))))
    fb = rng.randrange(len(o["facts"]))
    ra = rng.randrange(len(c["responses"])) if c["responses"] else None
    items = [_frame_item(c)] + [_fact_item(c, i) for i in fa] + ([_resp_item(c, ra)] if ra is not None else []) + [_fact_item(o, fb)]
    blk, pos = _blocks(items, rng)
    s = f"{_cite('The closest case is ' + SHORT[c['id']] + ', where the driver was ' + c['driver'], pos[(c['id'], 'frame')])} "
    s += " ".join(_cite(c["facts"][i][0], pos[(c["id"], "f", i)]) for i in fa) + " "
    if ra is not None:
        s += f"Response used: {_cite(c['responses'][ra][1], pos[(c['id'], 'r', ra)])} "
    s += f"A second case with the same channel is {SHORT[o['id']]}: {_cite(o['facts'][fb][0], pos[(o['id'], 'f', fb)])} "
    s += f"Analysis (not stated in the sources): the first things to check are {', '.join(CH_DESC[x] for x in shared)}. " + rng.choice(c["lessons"])
    need = [pos[(c["id"], "frame")]] + [pos[(c["id"], "f", i)] for i in fa] + [pos[(o["id"], "f", fb)]]
    q = rng.choice(ANALOG_QS).format(a=c["assets"], d=c["driver"])
    return _row("case", q, blk, s, need, len(items), split, {"type": "analog", "case": c["id"]})


def uncertainty(c, cards, rng, split):
    idx = UNC.get(c["id"])
    if not idx:
        return None
    ok = [i for i in idx if i < len(c["facts"])]
    if len(ok) < 2:
        return None
    items = [_fact_item(c, i) for i in ok] + _distract(cards, c, rng, 1)
    blk, pos = _blocks(items, rng)
    s = "The sources give different numbers. " + " ".join(_cite(c["facts"][i][0], pos[(c["id"], "f", i)]) for i in ok)
    s += " Analysis (not stated in the sources): estimates differ because they were made at different times, by different methods, or measure different things such as insured loss, economic loss or claims paid."
    return _row("case", rng.choice(UNC_QS).format(s=SHORT[c["id"]]), blk, s, [pos[(c["id"], "f", i)] for i in ok], len(items), split, {"type": "uncertainty", "case": c["id"]})


def unknown(cards, rng, split):
    q = rng.choice(OUT_OF_SCOPE)
    cs = rng.sample(cards, 3)
    items = [_fact_item(c, rng.randrange(len(c["facts"]))) for c in cs]
    blk, pos = _blocks(items, rng)
    have = "; ".join(f"{_clean(items[j][0].split('): ', 1)[1])} [{pos[items[j][1]]}]" for j in range(2))
    s = f"The sources provided do not cover that event. They only report other cases, for example: {have}. To answer I would need sources on that event itself."
    return _row("case_gap", q, blk, s, [], len(items), split, {"type": "unknown", "case": "none"})


def build(vf, seed=3, allow_unverified=False, reps=2):
    rng = random.Random(seed)
    cards = usable(CASES, vf, allow_unverified)
    train_cards = [c for c in cards if c["id"] not in HELD_OUT]
    rows = []
    for rep in range(reps):
        for c in cards:
            held = c["id"] in HELD_OUT
            pool = cards if held else train_cards                # train rows never show a held-out case, not even as a distractor
            for fn in (summary, summary, response, gap, gap, analog, analog, uncertainty):
                r = fn(c, pool, rng, "test" if held else ("valid" if rng.random() < 0.12 else "train"))
                if r:
                    rows.append(r)
        for _ in range(8):
            rows.append(unknown(train_cards, rng, "train"))
        rows.append(unknown(cards, rng, "test"))
    return rows


def passages(vf=None, allow_unverified=False):
    """The cards as retrieval passages for the agent's knowledge index: (title, url, text)."""
    out = []
    for c in usable(CASES, vf, allow_unverified):
        text = f"{c['title']}. Driver: {c['driver']}. Exposed: {c['assets']}. " + " ".join(f[0] for f in c["facts"]) + " Responses: " + " ".join(r[1] for r in c["responses"]) + " Not reported in the sources: " + "; ".join(c["gaps"]) + "."
        out.append((c["title"], c["sources"][0][0], text))
    return out


def main():
    cmd = sys.argv[1]
    if cmd == "verify":
        verify(Path(sys.argv[2]))
    elif cmd == "build":
        vf = json.loads(Path(sys.argv[2]).read_text())
        rows = build(vf)
        out = Path(sys.argv[3])
        out.mkdir(parents=True, exist_ok=True)
        from collections import Counter
        for s in ("train", "valid", "test"):
            rs = [r for r in rows if r["split"] == s]
            with open(out / f"cases_{s}.jsonl", "w") as f:
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print({s: dict(Counter(r["kind"] + ":" + r["meta"].get("type", "") for r in rows if r["split"] == s)) for s in ("train", "valid", "test")})


if __name__ == "__main__":
    main()
