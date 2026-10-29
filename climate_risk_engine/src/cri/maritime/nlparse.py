"""Sentence -> voyage fields. Order matters: numbers and prices are lifted out first (and replaced by a ';' so a name or port cannot run through them),
then the ship name, then the route. Anything guessed (a name/port split with no 'from') is flagged in ``flags`` so the answer can say so.

Fields returned (only those found): imo, name, origin, dest, origin_country, dest_country, distance_nm, speed_kn, speed_option_kn, fuel, daily_t, daily_speed_kn,
dwt, teu, gt, cbm, ship_type, year, delay_days, diversion_pct, market{bunker_usd_per_t, charter_usd_per_day, eua_price, currency}, flags[].
"""
from __future__ import annotations

import re

from . import classes as CL

SHIP_WORDS = r"(?:ship|vessel|tanker|bulker|bulk carrier|container ?ship|containership|boxship|feeder|ro-?ro|ro-?pax|ferry|cruise ship|cruise liner|car carrier|reefer|lng carrier|lpg carrier|gas carrier|carrier|vlcc|suezmax|aframax|panamax|capesize|handysize|supramax|MV|M/V|MT|M/T)"
STOP = {"is", "was", "has", "will", "would", "which", "that", "moves", "moving", "move", "sails", "sailing", "sail", "docked", "berthed", "moored", "currently", "now", "at", "in", "and", "from", "to", "with", "of", "the", "a", "an",
        "carrying", "loaded", "ballast", "heading", "bound", "going", "goes", "go", "if", "how", "what", "leaving", "departing", "leaves", "departs", "route", "voyage", "trip", "on", "for", "burning", "uses", "using", "needs", "must", "can", "could", "should",
        "travels", "travelling", "steams", "steaming", "proceeds", "transits", "makes", "called", "named", "burns", "runs", "operating", "emits", "emit", "sails", "via", "then"}
NOTNAME = {"imo", "vlsfo", "hsfo", "ifo", "hfo", "lfo", "mgo", "mdo", "lng", "lpg", "eua", "ets", "teu", "dwt", "gt", "usd", "eur", "cbm", "nm", "what", "how", "can", "emissions", "cost", "costs", "carbon", "the", "ship", "vessel", "if", "and", "then",
           "voyage", "which", "will", "our", "my", "a", "an", "is", "are", "we", "i", "does", "do", "should", "could", "would", "please", "fuel", "bunker", "hire", "charter"}
_CONNECT = {"of", "the", "de", "la", "del", "da", "do", "di", "el", "al", "von", "van"}
_CONN = r"(?:\s+(?:of|the|de|la|del|da|do|di|el|al|von|van)){0,2}"
_TOK = r"[A-Z][\w'’\-\.]*"
_PN = rf"{_TOK}(?:{_CONN}\s+{_TOK}|\s+{_TOK}){{0,2}}"
_PORTOF = r"(?:the\s+)?(?:[Pp]ort\s+of\s+)?"
_CTRY = r"(?:\s*,\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?))?"

FUEL_WORDS = [(r"\b(vlsfo|hsfo|ifo|hfo|heavy fuel|fuel oil|residual)\b", "hfo"), (r"\b(lfo|light fuel)\b", "lfo"), (r"\b(mgo|mdo|marine gas ?oil|marine diesel|diesel)\b", "mdo_mgo"),
              (r"\b(lng)\b(?!\s+carrier|\s+tanker)", "lng_otto_medium"), (r"\bmethanol\b", "methanol_ng"), (r"\bammonia\b", "nh3_ng_fuel_cell"), (r"\blpg\b(?!\s+carrier)", "lpg_propane")]


def _num(s):
    return float(s.replace(",", ""))


def _cut(t, m, fill=" ; "):
    return t[:m.start()] + fill + t[m.end():]


def normalise(text: str) -> str:
    t = text.replace("→", " to ").replace("->", " to ").replace("=>", " to ").replace("—", " to ").replace("–", " to ").replace("›", " to ")
    t = re.sub(r"(?<=[a-z]) - (?=[A-Z])", " to ", t)                        # "Genoa - Barcelona"
    t = re.sub(r"(?<=[a-z])-(?=[A-Z][a-z]{2,})(?<![A-Za-z]{2}-[A-Z])", " to ", t) if re.search(r"\b[A-Z][a-z]+-[A-Z][a-z]+\b", t) and not re.search(r"\b(?:Ro|Ro-Ro|Port-au)", t) else t
    t = re.sub(r"\bn\.?\s?m\.?\b", "nm", t, flags=re.I)
    return t


def parse(text: str) -> dict:
    out: dict = {"flags": []}
    t = re.sub(r"\b(?:the\s+)?Red\s+Sea\b|\bSuez(?:\s+Canal)?\b(?=\s+(?:route|diversion|closure|crisis))|\bCape\s+of\s+Good\s+Hope\b|\bvia\s+the\s+Cape\b", " ; ", normalise(text))

    m = re.search(r"\bIMO\s*(\d{7})\b", t, re.I)
    if m:
        out["imo"] = m.group(1)
        t = _cut(t, m)

    # --- money first, so its numbers are not read as anything else
    mk: dict = {}
    m = (re.search(r"\b(?:eua|ets|allowances?|carbon)\b[^.\d$€]{0,25}([€$])\s*(\d+(?:\.\d+)?)", t, re.I)
         or re.search(r"\b(?:eua|ets|allowances?|carbon)\b[^.\d$€]{0,25}()(\d+(?:\.\d+)?)\s*(?:eur|€|usd|\$|(?:/|per)\s*t(?:onne)?\b)", t, re.I))
    if m:
        mk["eua_price"] = float(m.group(2))
        tail = t[m.end(): m.end() + 8].lower()
        cur = m.group(1) or ("eur" if "eur" in tail or "€" in tail else ("usd" if "usd" in tail or "$" in tail else ""))
        mk["currency"] = "USD" if cur in ("$", "usd") else "EUR"
        if not cur:
            out["flags"].append("EUA price given without a currency: read as EUR.")
        t = _cut(t, m)
    m = re.search(r"(?:charter|hire|day[\s-]?rate|opex)[^.\d$€]{0,25}[$€]?\s*(\d[\d,\.]*)\s*(?:usd|eur|\$|€)?\s*(?:/|per|a)\s*day", t, re.I)
    if m:
        mk["charter_usd_per_day"] = _num(m.group(1))
        t = _cut(t, m)
    m = (re.search(r"(?:bunkers?|fuel(?:\s+price)?|vlsfo|hsfo|mgo|mdo|ifo|hfo|lng)[^.\d$€]{0,25}(?:is|at|of|=|:)?\s*[$€]?\s*(\d[\d,\.]*)\s*(?:usd|eur|\$|€)?\s*(?:/|per)\s*(?:t|tonne|tonnes|ton|tons|mt)\b(?!\s*/?\s*day)", t, re.I)
         or re.search(r"[$€]\s*(\d[\d,\.]*)\s*(?:/|per)\s*(?:t|tonne|ton|mt)\b", t, re.I)
         or re.search(r"(\d[\d,\.]*)\s*(?:usd|eur)\s*(?:/|per)\s*(?:t|tonne|ton|mt)\b", t, re.I))
    if m:
        mk["bunker_usd_per_t"] = _num(m.group(1))
        t = _cut(t, m)
    out["market"] = mk

    # --- daily consumption (and the speed it applies to)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:t|tonnes?|tons?|mt)(?:\s+of\s+[A-Za-z]+)?\s*(?:/|per|a)\s*day(?:\s+(?:[A-Za-z]{2,8}\s+){0,2}?at\s+(\d+(?:\.\d+)?)\s*(?:kn|kts?|knots?)\b)?", t, re.I)
    if m:
        out["daily_t"] = _num(m.group(1))
        if m.group(2):
            out["daily_speed_kn"] = _num(m.group(2))
        t = _cut(t, m)

    # --- what-if speed ("slow to 12 knots", "slow steaming at 10 kn") is not the base speed
    m = re.search(r"\bslow\w*(?:[\s-]steam\w*)?(?:\s+down)?\s+(?:to|at)\s+(\d+(?:\.\d+)?)\s*(?:kn|kts?|knots?)\b", t, re.I)
    if m:
        out["speed_option_kn"] = _num(m.group(1))
        t = _cut(t, m)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kn|kts?|knots?)\b", t, re.I)
    if m:
        out["speed_kn"] = _num(m.group(1))
        t = _cut(t, m)

    # --- distance
    m = re.search(r"(\d[\d,\.]*)\s*(?:nm|nautical\s+miles?)\b", t, re.I)
    if m:
        out["distance_nm"] = _num(m.group(1)); t = _cut(t, m)
    else:
        m = re.search(r"(\d[\d,\.]*)\s*km\b", t, re.I)
        if m:
            out["distance_nm"] = round(_num(m.group(1)) / 1.852, 1); out["flags"].append("Distance given in km, converted at 1.852 km per nautical mile."); t = _cut(t, m)
        else:
            m = re.search(r"(\d[\d,\.]*)\s*(?:statute\s+)?miles?\b", t, re.I)
            if m:
                out["distance_nm"] = round(_num(m.group(1)) * 0.868976, 1); out["flags"].append("Distance given in miles: read as statute miles (1 mile = 0.869 nm). Say 'nautical miles' if you meant those."); t = _cut(t, m)

    # --- ship size
    for key, rx in (("dwt", r"(\d[\d,\.]*)\s*(k?)\s*[-\s]?(?:dwt|deadweight)"), ("teu", r"(\d[\d,\.]*)\s*(k?)\s*[-\s]?teu"), ("gt", r"(\d[\d,\.]*)\s*(k?)\s*[-\s]?(?:gt|gross\s+ton\w*)"), ("cbm", r"(\d[\d,\.]*)\s*(k?)\s*[-\s]?(?:cbm|m3|m³)")):
        m = re.search(rx, t, re.I)
        if m:
            out[key] = _num(m.group(1)) * (1000.0 if m.group(2) else 1.0)
            t = _cut(t, m)

    # --- delay and diversion
    m = re.search(r"\bdelay(?:s|ed)?\b[^.\d]{0,25}(\d+(?:\.\d+)?)\s*days?|(\d+(?:\.\d+)?)\s*days?\s*(?:of\s+)?(?:delay|late|waiting|at anchor)", t, re.I)
    if m:
        out["delay_days"] = float(m.group(1) or m.group(2)); t = _cut(t, m)
    m = re.search(r"\b(?:divert\w*|detour|reroute\w*)[^.\d]{0,25}(\d+(?:\.\d+)?)\s*%", t, re.I)
    if m:
        out["diversion_pct"] = float(m.group(1)); t = _cut(t, m)

    m = re.search(r"\b(20[2-4]\d)\b", t)
    if m:
        out["year"] = int(m.group(1)); t = _cut(t, m)

    for rx, k in FUEL_WORDS:
        if re.search(rx, text, re.I):
            out["fuel"] = k
            if k == "lng_otto_medium":
                out["flags"].append("LNG engine type not given: a medium-speed Otto engine (methane slip default) is used. Give the engine type for a firm number.")
            if re.search(r"vlsfo|hsfo|ifo", text, re.I):
                out["flags"].append("VLSFO/HSFO/IFO grades are treated as the FuelEU 'HFO' class (RME-RMK); confirm for your grade.")
            break
    st = CL.classify(re.sub(r"\b(lng|lpg)\b(?!\s+(?:carrier|tanker))", "", text, flags=re.I))
    if st:
        out["ship_type"] = st

    # --- ship name: capitalised words after a ship word, up to the first word that cannot be part of a name
    name_end = None
    nm = re.search(rf"\b(?i:{SHIP_WORDS})\s+(?i:called\s+|named\s+)?[\"“']?({_TOK}(?:{_CONN}\s+{_TOK}|\s+{_TOK}){{0,3}})", t)
    if nm and re.match(r"(?i:ship|vessel|tanker|bulker|carrier|ferry)", nm.group(0)) and re.match(r"(?i:what|how|the|a|an|our|my|is|are)\b", nm.group(1)):
        nm = None
    if nm:
        words = nm.group(1).replace("’", "'").split()
        keep = []
        for i_, w in enumerate(words):
            lw = w.lower().strip(",.")
            if lw in ("of", "the", "de", "la", "del", "da", "do", "di", "el", "al", "von", "van") and keep and any(x[:1].isupper() for x in words[i_ + 1:i_ + 3]):
                keep.append(w)
                continue
            if lw in STOP or lw in NOTNAME or not re.search(r"[A-Za-z]", w):
                break
            keep.append(w.strip(","))
        while keep and keep[-1].lower() in ("of", "the", "de", "la", "del", "da", "do", "di", "el", "al", "von", "van"):
            keep.pop()
        while keep and keep[0] in ("MV", "M/V", "MT", "M/T"):
            keep = keep[1:]
        if keep:
            tail = t[nm.end(1):]
            followed_by_to = bool(re.match(r"\s+(?:to|for)\s+[A-Z]", tail)) or bool(re.match(r"\s+(?:to)\b", tail))
            if len(keep) >= 3 and followed_by_to and nm.group(1).split()[-len(keep)].strip(",") == keep[0]:
                # "MV Atlantic Star Singapore to Los Angeles": no 'from', so the last word is read as the origin port
                out["origin"] = keep[-1]
                keep = keep[:-1]
                out["flags"].append(f"No 'from' in the sentence, so I read '{out['origin']}' as the departure port and '{' '.join(keep)}' as the ship's name. Say 'from <port>' if that is wrong.")
            out["name"] = " ".join(keep)
            name_end = nm.start(1) + len(" ".join(nm.group(1).split()[:len(keep) + (1 if out.get("origin") else 0)]))
            t = t[:nm.start(1)] + " ; " + t[nm.start(1) + len(" ".join(nm.group(1).split()[:len(keep) + (1 if out.get("origin") else 0)])):]

    # --- ports
    def clean(s):
        words = s.split()
        keep = []
        for i_, w in enumerate(words):
            lw = w.lower().strip(",.")
            if lw in _CONNECT and keep and any(x[:1].isupper() for x in words[i_ + 1:i_ + 3]):
                keep.append(w)
                continue
            if lw in NOTNAME or lw in STOP:
                break
            keep.append(w.strip(","))
        while keep and keep[-1].lower() in _CONNECT:
            keep.pop()
        return " ".join(keep) or None

    def grab(rx, group):
        m_ = re.search(rx, t)
        return (clean(m_.group(group)), m_) if m_ else (None, None)

    o = d = None
    ctry = {}
    m = re.search(rf"\b(?:from|between)\s+{_PORTOF}({_PN}){_CTRY}\s+(?:to|for|towards?|into|and)\s+{_PORTOF}({_PN}){_CTRY}", t)
    if m:
        o, d = clean(m.group(1)), clean(m.group(3))
        ctry = {"origin_country": m.group(2), "dest_country": m.group(4)}
    if not (o and d):
        m = re.search(rf"(?<![\w']){_PORTOF}({_PN}){_CTRY}\s+to\s+{_PORTOF}({_PN}){_CTRY}", t)
        while m:
            a_, b_ = clean(m.group(1)), clean(m.group(3))
            if a_ and b_ and a_.lower() not in NOTNAME:
                o, d = o or a_, d or b_
                ctry = {"origin_country": m.group(2), "dest_country": m.group(4)}
                break
            m = re.search(rf"(?<![\w']){_PORTOF}({_PN}){_CTRY}\s+to\s+{_PORTOF}({_PN}){_CTRY}", t[m.end():]) if False else None
    if not d:
        m = re.search(rf"\b(?:to|for|towards?|into|reach|arrive\s+at|arriving\s+at)\s+(?:reach\s+|arrive\s+at\s+)?{_PORTOF}({_PN}){_CTRY}", t)
        if m:
            d = clean(m.group(1)); ctry["dest_country"] = m.group(2)
    if not o:
        m = (re.search(rf"\b(?:from|leaving|departing|leaves|departs|ex)\s+{_PORTOF}({_PN}){_CTRY}", t)
             or re.search(rf"\b(?:docked|berthed|moored|anchored|alongside|lying|located|sitting)\b[^.;]*?\b(?:in|at)\s+{_PORTOF}({_PN}){_CTRY}", t)
             or re.search(rf"\b(?:is|are|currently|now)?\s*\b(?:at|in)\s+{_PORTOF}({_PN}){_CTRY}(?=[\s,;]+(?:and|then|going|moving|heading|bound|needs|will|to|sail|sails|that)\b)", t))
        if m:
            o = clean(m.group(1)); ctry["origin_country"] = m.group(2)
    if o and not out.get("origin"):
        out["origin"] = re.sub(r"\s+Port$", "", o)
    if d:
        out["dest"] = re.sub(r"\s+Port$", "", d)
    for k, v in ctry.items():
        if v and v.lower() not in ("and", "the", "port", "moving", "heading", "then", "to"):
            out[k] = v
    if out.get("origin") and out.get("dest") and out["origin"].lower() == out["dest"].lower():
        out.pop("origin")
    return out
