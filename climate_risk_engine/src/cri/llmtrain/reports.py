"""
cri/llmtrain/reports.py — what companies say in their own published climate reports (TCFD, IFRS S2, sustainability and resilience reports) about physical risk and how they adapt.

Same principle as the SEC collector: every passage is a sentence copied word for word from a named document at a stated address; nothing is paraphrased or filled in.
Only sentences that name a physical hazard AND state an amount, a scenario, or an action are kept, so the model learns from evidence, not from marketing text.

    python -m cri.llmtrain.reports collect OUT_DIR [--seeds extra_seeds.json] [--max-per-doc 12]

The output (OUT_DIR/report_cases.jsonl) has the same record layout as edgar_cases.jsonl (form = "climate report", filed = the report year), so the case index,
the training-set builder and the honesty check take it with no special handling. The address list is explicit (SEEDS plus any file you pass): the collector
fetches only those documents, one at a time, with a pause between requests. Reports are the companies' own publications; the quoted sentences are short excerpts with the source link.
A seeds file is a JSON list of {"company", "year", "url", "region"}.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from .edgar_cases import AMT, BOILER, classify, clean_text, sentences

UA = {"User-Agent": "ClimRisk research (open-source climate-risk project; polite crawler) shrinivashdkannan@gmail.com", "Accept-Encoding": "identity"}
EVENT = "climate-reports"
EVENT_NAME = "physical climate risk and adaptation"

# Documents found by search (company-published or hosted by the company's regulator or adviser). Add more with --seeds.
SEEDS = [
    {"company": "Duke Energy", "year": 2024, "region": "United States", "url": "https://guidehouse.com/-/media/new-library/industries/cei/documents/2025/duke-energy-climate-resilience-and-adaptation-study_december-2024.pdf"},
    {"company": "Duke Energy", "year": 2023, "region": "United States", "url": "https://www.duke-energy.com/-/media/pdfs/our-company/carolinsresiliencetransdiststudyfinal.pdf"},
    {"company": "Prologis", "year": 2024, "region": "Global", "url": "https://www.prologis.com/node/51746"},
    {"company": "BASF", "year": 2021, "region": "Germany", "url": "https://report.basf.com/2021/en/managements-report/forecast/opportunities-and-risks/operational-opportunities-and-risks.html"},
    {"company": "BASF", "year": 2024, "region": "Germany", "url": "https://report.basf.com/2024/en/combined-managements-report/consolidated-sustainability-statement/environment/e1-climate-change.html"},
    {"company": "Swiss Re", "year": 2021, "region": "Switzerland", "url": "https://www.swissre.com/sustainability/stories/further-information-tcfd-2021.html"},
    {"company": "Swiss Re", "year": 2017, "region": "Switzerland", "url": "https://reports.swissre.com/2017/financial-report/responsibility/climate-related-financial-disclosures-tcfd.html"},
    {"company": "SL Green", "year": 2023, "region": "United States", "url": "https://sustainability.slgreen.com/sl-green-tcfd-report-2"},
    {"company": "PSP Swiss Property", "year": 2025, "region": "Switzerland", "url": "https://www.psp.info/fileadmin/internet/nachhaltigkeit/pdf/2025/PSP_Nachhaltigkeit_TCFD_GB25_en.pdf"},
    {"company": "Borregaard", "year": 2023, "region": "Norway", "url": "https://borregaard.no/media/kcfdebzj/scenario-analysis-borregaard.pdf"},
]

PHYS = re.compile(r"flood|wildfire|hurricane|cyclone|typhoon|storm|heat ?wave|extreme heat|heat stress|drought|water (stress|scarcity|shortage)|low[- ]water|sea[- ]level|physical (climate )?risk|extreme weather|hail|substation|sea walls?", re.I)
ACT = re.compile(r"scenario|adapt|resilien|mitigat|invest|early[- ]warning|elevat|harden|relocat|diversif|insur|back-?up|barrier|defen[cs]e|vessel|capex|reduc|upgrad|retrofit|protect|plan\b|study|prioriti", re.I)
JUNK = re.compile(r"forward-looking|this report|table of contents|click here|copyright|all rights reserved|page \d|http|www\.", re.I)


def _fetch(url: str, timeout: int = 60) -> tuple:
    last = None
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                raw = r.read(25_000_000)
                return raw, (r.headers.get("Content-Type") or "").lower()
        except Exception as e:                                       # noqa: BLE001
            last = e
            time.sleep(2 * (k + 1))
    raise last


def _pdf_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from pdfminer.high_level import extract_text
            return extract_text(io.BytesIO(raw))
        except ImportError:
            raise RuntimeError("no PDF reader installed (pip install pypdf)")
    rd = PdfReader(io.BytesIO(raw))
    return "\n".join((p.extract_text() or "") for p in rd.pages[:250])


def doc_text(raw: bytes, ctype: str, url: str) -> str:
    if raw[:5] == b"%PDF-" or "pdf" in ctype or url.lower().endswith(".pdf"):
        t = _pdf_text(raw)
        t = re.sub(r"(?<=[a-z])-\s*\n\s*(?=[a-z])", "", t)          # words broken across lines
        return re.sub(r"\s*\n\s*", " ", t)                           # a PDF has line breaks mid-sentence
    return clean_text(raw.decode("utf-8", errors="replace"))


def _sentences_any(t: str) -> list:
    out = []
    for para in re.split(r"\n+", t):
        para = para.strip()
        if len(para) < 40:
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"(])", para):
            s = re.sub(r"\s+", " ", s).strip()
            if 8 <= len(s.split()) <= 60:
                out.append(s)
    return out


def extract(text: str, max_per_doc: int = 12) -> list:
    """Verbatim, complete sentences that name a physical hazard and state an amount, a scenario or an action."""
    seen, out = set(), []
    for s in _sentences_any(text):
        s = re.sub(r"^[••◦\-–\*\s]+", "", s)
        if not re.match(r"[A-Z0-9“\"$]", s) or not re.search(r"[.!?”\"]$", s):
            continue
        if not PHYS.search(s) or JUNK.search(s) or BOILER.search(s) or s.count("(") != s.count(")") or len(re.findall(r"[A-Za-z]", s)) < 0.6 * len(s):
            continue
        amts = AMT.findall(s) + re.findall(r"\d[\d,\.]*\s?%", s)
        cls = classify(s)
        if not (amts or cls or ACT.search(s)):
            continue
        key = re.sub(r"\W+", " ", s.lower())[:90]
        if key in seen:
            continue
        seen.add(key)
        out.append({"quote": s, "amounts": [a.strip() for a in amts][:6], "response": cls, "_score": (3 if amts else 0) + (2 if cls else 0) + (1 if re.search(r"scenario|adapt|resilien", s, re.I) else 0)})
    out.sort(key=lambda q: -q["_score"])
    return [{k: v for k, v in q.items() if k != "_score"} for q in out[:max_per_doc]]


def collect(out: Path, seeds: list = None, max_per_doc: int = 12) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds is not None else SEEDS
    recs, report = [], {}
    for sd in seeds:
        key = f"{sd['company']} {sd['year']}"
        try:
            raw, ct = _fetch(sd["url"])
            q = extract(doc_text(raw, ct, sd["url"]), max_per_doc)
            report[key] = {"quotes": len(q), "bytes": len(raw)}
            if q:
                slug = re.sub(r"[^a-z0-9]+", "-", sd["company"].lower()).strip("-")
                recs.append({"event": EVENT, "event_name": EVENT_NAME, "hazard": "physical risk disclosure", "region": sd.get("region") or "", "company": sd["company"], "cik": slug,
                             "form": "climate report", "filed": str(sd["year"]), "url": sd["url"], "quotes": q})
        except Exception as e:                                       # noqa: BLE001
            report[key] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        print(key, report[key], flush=True)
        time.sleep(1.5)
    (out / "report_cases.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs))
    summ = {"documents": len(seeds), "documents_with_quotes": len(recs), "companies": len({r["company"] for r in recs}), "quotes": sum(len(r["quotes"]) for r in recs),
            "failed": sorted(k for k, v in report.items() if "error" in v)}
    (out / "report_summary.json").write_text(json.dumps({"summary": summ, "report": report}, indent=1))
    print(summ)
    return summ


def main():
    a = sys.argv[1:]
    if len(a) >= 2 and a[0] == "collect":
        seeds = SEEDS
        if "--seeds" in a:
            seeds = SEEDS + json.loads(Path(a[a.index("--seeds") + 1]).read_text())
        mp = int(a[a.index("--max-per-doc") + 1]) if "--max-per-doc" in a else 12
        collect(Path(a[1]), seeds, mp)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
