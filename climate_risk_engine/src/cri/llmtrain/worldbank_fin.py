"""
cri/llmtrain/worldbank_fin.py — what the national economy did around each event outside the United States, from World Bank open data (CC BY 4.0).

For every non-US event it writes ONE source sentence per affected country: real GDP growth and consumer-price inflation in the year before the event,
the event year and the year after. Every figure is copied from the World Bank indicator API and rounded to one decimal by this code; nothing is estimated.
The sentence says the figures are country-wide, so the model learns not to attribute the movement to the event.

  collect(out_dir)             writes out_dir/macro_items.jsonl and macro_report.json (raw responses cached in out_dir/raw; resumable)
  build_rows(items, seed)      training rows ('macro', 'macro_unknown'); held-out events stay out of train
CLI:  python -m cri.llmtrain.worldbank_fin collect OUT_DIR
Source: World Bank, World Development Indicators: NY.GDP.MKTP.KD.ZG (GDP growth, annual %), FP.CPI.TOTL.ZG (inflation, consumer prices, annual %).
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

from .heldout import EVENTS as HELD_EVENTS

UA = {"User-Agent": "ClimRisk research (open-source climate-risk project; polite crawler) shrinivashdkannan@gmail.com", "Accept-Encoding": "identity"}
API = "https://api.worldbank.org/v2/country/{iso}/indicator/{ind}?format=json&date=1995:2025&per_page=100"
INDICATORS = {"real GDP growth": "NY.GDP.MKTP.KD.ZG", "consumer-price inflation": "FP.CPI.TOTL.ZG"}

# event key (SEC-collector key or curated-card id) -> ISO3 codes of the countries the event hit
COUNTRIES = {
    "thailand-2011": ["THA"], "jebi-2018": ["JPN"], "hagibis-2019": ["JPN"], "faxai-2019": ["JPN"], "japan-rain-2018": ["JPN"], "lekima-2019": ["CHN"], "yagi-2024": ["VNM", "PHL"],
    "doksuri-2023": ["CHN", "PHL"], "henan-2021": ["CHN"], "ketsana-2009": ["PHL", "VNM"], "odette-2021": ["PHL"], "haiyan-2013": ["PHL"], "hato-2017": ["CHN"], "mangkhut-2018": ["PHL", "CHN"],
    "jakarta-2020": ["IDN"], "chennai-2015": ["IND"], "kerala-2018": ["IND"], "amphan-2020": ["IND", "BGD"], "fani-2019": ["IND"], "biparjoy-2023": ["IND", "PAK"],
    "india-heatwave-2022": ["IND", "PAK"], "pakistan-floods-2022": ["PAK"], "pakistan-2022": ["PAK"], "china-drought-2022": ["CHN"],
    "yasi-2011": ["AUS"], "qld-floods-2011": ["AUS"], "seroja-2021": ["AUS"], "australia-floods-2022": ["AUS"], "australia-bushfire": ["AUS"], "suncorp-iag-2022": ["AUS"], "debbie-2017": ["AUS"],
    "gabrielle-2023": ["NZL"], "winston-2016": ["FJI"],
    "xynthia-2010": ["FRA"], "kyrill-2007": ["DEU", "GBR"], "ciara-2020": ["GBR", "DEU"], "eunice-2022": ["GBR", "DEU", "NLD"], "desmond-2015": ["GBR"], "boris-2024": ["POL", "CZE", "AUT"],
    "emilia-2023": ["ITA"], "daniel-2023": ["GRC", "LBY"], "valencia-2024": ["ESP"], "elbe-2013": ["DEU", "CZE"], "europe-floods-2021": ["DEU", "BEL"], "rhine-2018": ["DEU"], "basf-rhine-2018": ["DEU"],
    "germany-hail": ["DEU"], "europe-heat-2022": ["FRA", "ESP", "ITA"], "edf-2022": ["FRA"], "uk-ets": ["GBR"], "cbam": ["DEU"],
    "rs-brazil-2024": ["BRA"], "brazil-drought-2021": ["BRA"], "chile-megadrought": ["CHL"], "argentina-drought-2023": ["ARG"], "la-nina-colombia": ["COL"],
    "panama-2023": ["PAN"], "otis-2023": ["MEX"], "dean-2007": ["JAM", "MEX"], "maria-2017": ["PRI"], "fiona-2022": ["PRI"], "dorian-2019": ["BHS"],
    "day-zero-2018": ["ZAF"], "kzn-2022": ["ZAF"], "freddy-2023": ["MOZ", "MWI"], "idai-2019": ["MOZ", "ZWE"], "kariba-2024": ["ZMB", "ZWE"], "kenya-drought": ["KEN"],
    "dubai-2024": ["ARE"], "bc-floods-2021": ["CAN"], "lytton-2021": ["CAN"], "alberta-2013": ["CAN"], "calgary-hail-2020": ["CAN"], "heat-dome-2021": ["CAN"], "mcmurray-2016": ["CAN"], "canada-fires-2023": ["CAN"],
    "china-ets": ["CHN"], "tata-port-talbot": ["GBR"], "uk-freeze-2022": ["GBR"],
}


def _event_meta() -> dict:
    """key -> (display name, start year) from the SEC collector's event list and the curated cards."""
    from .edgar_cases import ALL_EVENTS
    from .cases_data import CASES
    meta = {e[0]: (e[1], int(e[3][:4])) for e in ALL_EVENTS}
    for c in CASES:
        meta.setdefault(c["id"], (c["title"].split(" (")[0], int(c["year"])))
    return meta


def _get(url: str, timeout: int = 40):
    last = None
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:                                       # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def fetch_country(iso: str, raw: Path) -> dict:
    """{label: {year: value}} plus the country's name."""
    f = raw / f"{iso}.json"
    if f.exists():
        return json.loads(f.read_text())
    out = {"name": None, "series": {}}
    for label, ind in INDICATORS.items():
        j = _get(API.format(iso=iso, ind=ind))
        rows = j[1] if isinstance(j, list) and len(j) > 1 and j[1] else []
        out["series"][label] = {r["date"]: r["value"] for r in rows if r.get("value") is not None}
        if rows and not out["name"]:
            out["name"] = rows[0]["country"]["value"]
        time.sleep(0.2)
    f.write_text(json.dumps(out))
    return out


def sentence(country: str, ev_name: str, year: int, series: dict):
    parts, amounts = [], []
    for label, ser in series.items():
        pts = [(y, ser.get(str(y))) for y in (year - 1, year, year + 1)]
        pts = [(y, v) for y, v in pts if v is not None]
        if len(pts) < 2:
            continue
        seg = ", ".join(f"{v:.1f}% in {y}" for y, v in pts)
        amounts += [f"{v:.1f}%" for _, v in pts]
        parts.append(f"{label} was {seg}")
    if not parts or not any(p.startswith("real GDP") for p in parts):
        return None, []
    return f"World Bank data for {country} around {ev_name}: " + "; ".join(parts) + ".", amounts


def collect(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    meta = _event_meta()
    items, errs = [], {}
    for ev, isos in COUNTRIES.items():
        if ev not in meta:
            continue
        name, year = meta[ev]
        for iso in isos:
            try:
                d = fetch_country(iso, raw)
                text, amounts = sentence(d["name"] or iso, name, year, d["series"])
                if text:
                    items.append({"event": ev, "event_name": name, "country": d["name"] or iso, "iso3": iso, "kind": "macro", "company": d["name"] or iso, "text": text, "amounts": amounts,
                                  "url": API.format(iso=iso, ind="NY.GDP.MKTP.KD.ZG"), "form": "World Bank WDI"})
            except Exception as e:                                   # noqa: BLE001
                errs[f"{ev}:{iso}"] = f"{type(e).__name__}: {str(e)[:80]}"
    with open(out / "macro_items.tmp", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "macro_items.tmp").replace(out / "macro_items.jsonl")
    rep = {"events": len({i["event"] for i in items}), "country_events": len(items), "errors": len(errs), "sample_errors": dict(list(errs.items())[:5])}
    (out / "macro_report.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return rep


# ── training rows ────────────────────────────────────────────────────────────

SYS = ("You are ClimRisk's climate-risk analyst. Answer using ONLY the numbered sources, which are official national statistics from the World Bank. Cite every factual sentence with its source number in square brackets, like [2]. "
       "If the sources do not contain the answer, say exactly what is missing instead of guessing. Keep any inference clearly separate by starting it with 'Analysis (not stated in the sources):'. Never add a number that is not in the sources.")
CAVEAT = "Analysis (not stated in the sources): these are country-wide annual figures, so they show the movement around the event but do not isolate how much of it the event caused."


def build_rows(path: Path, seed: int = 11, per_case: int = 1) -> list:
    rng = random.Random(seed)
    items = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    rows = []

    def row(q, srcs, ans_fn, split, meta, kind="case"):
        order = list(range(len(srcs)))
        rng.shuffle(order)
        pos = {k: n for n, k in enumerate(order, 1)}
        blk = "\n\n".join(f"[{n}] {srcs[k]['text']}" for n, k in sorted((pos[k], k) for k in range(len(srcs))))
        return {"kind": kind, "split": split, "meta": {"need": [pos[0]], "n": len(srcs), **meta},
                "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {q}\n\nSources:\n{blk}"}, {"role": "assistant", "content": ans_fn(pos)}]}

    for it in items:
        held = it["event"] in HELD_EVENTS
        split = "test" if held else ("valid" if rng.random() < 0.08 else "train")
        pool = [x for x in items if (x["event"] in HELD_EVENTS) == held and x is not it]
        if len(pool) < 2:
            continue
        for _ in range(per_case):
            srcs = [it] + rng.sample(pool, 2)
            q = rng.choice([f"How did {it['country']}'s economy move around {it['event_name']}, according to World Bank data?",
                            f"What do official statistics show for {it['country']} around {it['event_name']}?",
                            f"Give me the GDP growth and inflation figures for {it['country']} around {it['event_name']}."])
            rows.append(row(q, srcs, lambda pos, it=it: f"{it['text'].rstrip('.')} [{pos[0]}]. {CAVEAT}", split, {"type": "macro", "case": f"{it['event']}:{it['iso3']}"}))
        evs = {x["event"]: x["event_name"] for x in pool}
        mine = {x["event"] for x in items if x["iso3"] == it["iso3"]}
        free = [e for e in evs if e not in mine]
        if free and rng.random() < 0.4:
            e2 = rng.choice(free)
            srcs = [it] + rng.sample(pool, 2)
            rows.append(row(f"How did {it['country']}'s GDP growth move around {evs[e2]}?", srcs,
                            lambda pos, it=it, e2=e2: (f"The sources provided do not cover {it['country']} around {evs[e2]}. They only show {it['country']} around {it['event_name']}: "
                                                       f"{it['text'].split(': ', 1)[1].rstrip('.')} [{pos[0]}]. To answer I would need World Bank figures for the relevant years."),
                            "test" if held else "train", {"type": "macro_unknown", "case": f"{e2}:{it['iso3']}"}, kind="case_gap"))
    return rows


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "collect":
        collect(Path(a[1]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
