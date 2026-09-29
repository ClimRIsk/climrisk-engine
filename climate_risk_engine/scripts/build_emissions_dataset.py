"""
Build a REAL company-level Scope 1 training set for the emissions model.

Replaces the synthetic data the EmissionsEstimator used to train on (rows
generated from its own sector-intensity table + noise), which made the "ML"
unable to be more accurate than the table it was sampled from.

Sources (all public, no keys)
─────────────────────────────
  EPA GHGRP (Envirofacts)  facility-level reported emissions + parent companies
                           with ownership shares — https://www.epa.gov/ghgreporting
  SEC EDGAR                company list, SIC industry code, and annual revenue
                           (XBRL companyfacts) — https://www.sec.gov/edgar
  Climate TRACE (v6 API)   global asset-level emissions with owners — used to
                           estimate how much of a company's global Scope 1 the
                           US-only GHGRP number captures.

Pipeline
────────
  1. GHGRP facilities + emissions for YEAR (direct-emitter sectors only; fuel
     *supplier* sectors and biogenic CO2 excluded — they are not the reporter's
     Scope 1).
  2. Attribute each facility's CO2e to parent companies by ownership share.
  3. Match parents to SEC registrants by normalised name.
  4. Pull SIC + calendar-year revenue from SEC.
  5. Write data/training/emissions_ghgrp_sec_<YEAR>.csv with provenance columns.

Important limitation — recorded in every row
────────────────────────────────────────────
GHGRP covers only US facilities above ~25 kt CO2e/yr, while SEC revenue is
global. For multinationals the observed intensity (US Scope 1 / global revenue)
understates true Scope 1 intensity. Rows carry `us_coverage_flag` so training
and evaluation can separate predominantly-domestic companies.

Usage
─────
  python scripts/build_emissions_dataset.py --year 2023 --stage all
  (stages: ghgrp, sec, merge, all — each stage caches to data/training/raw/)

SEC requires a descriptive User-Agent with contact info: set SEC_USER_AGENT.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "training"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)

EPA = "https://data.epa.gov/efservice"
SEC_UA = os.environ.get("SEC_USER_AGENT", "ClimRisk Intelligence research data@climrisk.io")

DIRECT_SECTOR_TYPES = {"E"}          # E = direct emitters; S = suppliers; I = injection
EXCLUDED_GAS_IDS = {8}               # biogenic CO2 (reported outside Scope 1)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get_json(url: str, headers: dict | None = None, timeout: int = 90, retries: int = 3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "ClimRisk-Engine/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as exc:     # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def _epa_table(table: str, year: int, page: int = 10000, max_new_pages: int | None = None) -> list[dict]:
    """
    Paged Envirofacts download, cached per page in raw/ so an interrupted run
    resumes (each 10k-row page takes ~1 min). Raises PartialDownload if
    max_new_pages is hit before the table is complete.
    """
    total = _get_json(f"{EPA}/{table}/YEAR/{year}/COUNT/JSON")[0]["TOTALQUERYRESULTS"]
    rows: list[dict] = []
    fetched = 0
    for start in range(0, total, page):
        pf = RAW / f"epa_{table}_{year}_{start}.json"
        if pf.exists():
            rows += json.loads(pf.read_text())
            continue
        if max_new_pages is not None and fetched >= max_new_pages:
            raise PartialDownload(f"{table}: stopped at row {start}/{total}; re-run to continue")
        chunk = _get_json(f"{EPA}/{table}/YEAR/{year}/ROWS/{start}:{start + page - 1}/JSON", timeout=80, retries=1)
        pf.write_text(json.dumps(chunk))
        rows += chunk
        fetched += 1
    return rows


class PartialDownload(RuntimeError):
    pass


# ── Stage 1: GHGRP → parent companies ─────────────────────────────────────────

_SHARE_RE = re.compile(r"^(.*?)\s*\(\s*([\d.]+)\s*%\s*\)\s*$")


def parse_parents(s: str | None) -> list[tuple[str, float]]:
    """'A Corp (60%); B Inc (40%)' → [('A Corp', .6), ('B Inc', .4)]; missing shares split evenly."""
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    out, unknown = [], []
    for p in parts:
        m = _SHARE_RE.match(p)
        if m:
            out.append((m.group(1).strip(), float(m.group(2)) / 100.0))
        else:
            unknown.append(p)
    if unknown:
        rest = max(0.0, 1.0 - sum(w for _, w in out))
        each = rest / len(unknown) if rest > 0 else 1.0 / (len(out) + len(unknown))
        out += [(u, each) for u in unknown]
    tot = sum(w for _, w in out) or 1.0
    return [(n, w / tot) for n, w in out]


def stage_ghgrp(year: int) -> Path:
    cache = RAW / f"ghgrp_parents_{year}.json"
    if cache.exists():
        return cache
    sectors = {r["sector_id"]: r for r in _get_json(f"{EPA}/PUB_DIM_SECTOR/JSON")}
    budget = int(os.environ.get("EPA_MAX_PAGES_PER_RUN", "0")) or None
    ems = _epa_table("PUB_FACTS_SECTOR_GHG_EMISSION", year, max_new_pages=budget)
    facs = _epa_table("PUB_DIM_FACILITY", year, page=500, max_new_pages=budget)

    fac_em: dict[int, float] = defaultdict(float)
    fac_sector_em: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for e in ems:
        sec = sectors.get(e["sector_id"], {})
        if sec.get("sector_type") not in DIRECT_SECTOR_TYPES or e["gas_id"] in EXCLUDED_GAS_IDS:
            continue
        v = float(e.get("co2e_emission") or 0.0)
        fac_em[e["facility_id"]] += v
        fac_sector_em[e["facility_id"]][sec.get("sector_code", "?")] += v

    parents: dict[str, dict] = {}
    for f in facs:
        fid = f["facility_id"]
        em = fac_em.get(fid, 0.0)
        if em <= 0:
            continue
        for name, share in parse_parents(f.get("parent_company")):
            p = parents.setdefault(name, {"parent": name, "scope1_t": 0.0, "n_facilities": 0,
                                          "states": set(), "naics": defaultdict(float),
                                          "ghgrp_sectors": defaultdict(float)})
            p["scope1_t"] += em * share
            p["n_facilities"] += 1
            p["states"].add(f.get("state"))
            p["naics"][str(f.get("naics_code") or "")[:3]] += em * share
            for sc, v in fac_sector_em[fid].items():
                p["ghgrp_sectors"][sc] += v * share

    rows = []
    for p in parents.values():
        rows.append({
            "parent": p["parent"], "scope1_t": round(p["scope1_t"], 1),
            "n_facilities": p["n_facilities"], "n_states": len(p["states"]),
            "top_naics3": max(p["naics"], key=p["naics"].get) if p["naics"] else "",
            "top_ghgrp_sector": max(p["ghgrp_sectors"], key=p["ghgrp_sectors"].get) if p["ghgrp_sectors"] else "",
        })
    cache.write_text(json.dumps({"year": year, "n_facility_rows": len(facs),
                                 "n_emission_rows": len(ems), "parents": rows}))
    return cache


# ── Stage 2: SEC matching, SIC, revenue ───────────────────────────────────────

_SUFFIX = re.compile(
    r"\b(the|inc|incorporated|corp|corporation|co|company|companies|llc|l\.l\.c|lp|l\.p|ltd|limited|plc|"
    r"holdings?|group|sa|nv|ag|se|na|usa|us|trust|partners)\b")


def norm(name: str) -> str:
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


REVENUE_CONCEPTS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "RegulatedAndUnregulatedOperatingRevenue",
    "ElectricUtilityRevenue",
    "OperatingRevenue",
]


def _annual_revenue(facts: dict, year: int) -> tuple[float | None, str | None]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    for concept in REVENUE_CONCEPTS:
        units = gaap.get(concept, {}).get("units", {}).get("USD", [])
        # Prefer SEC's calendar-aligned annual frame (e.g. "CY2023")
        for u in units:
            if u.get("frame") == f"CY{year}":
                return float(u["val"]), concept
        cands = [u for u in units if u.get("fy") == year and u.get("fp") == "FY" and u.get("form", "").startswith("10-K")]
        if cands:
            return float(max(cands, key=lambda u: u.get("filed", ""))["val"]), concept
    return None, None


def stage_sec(year: int) -> Path:
    cache = RAW / f"sec_matches_{year}.json"
    if cache.exists():
        return cache
    parents = json.loads(stage_ghgrp(year).read_text())["parents"]
    tcache = RAW / "sec_company_tickers.json"
    if not tcache.exists():
        tcache.write_text(json.dumps(_get_json("https://www.sec.gov/files/company_tickers.json",
                                               {"User-Agent": SEC_UA}, timeout=30)))
    tick = json.loads(tcache.read_text())
    sec = {}
    for v in tick.values():
        sec.setdefault(norm(v["title"]), v)   # first ticker per company
    # Bucket SEC names by first word so fuzzy matching compares ~10s, not ~10k, names
    by_first: dict[str, list[str]] = defaultdict(list)
    for k in sec:
        if k:
            by_first[k.split()[0]].append(k)

    matches = []
    for p in parents:
        n = norm(p["parent"])
        if not n:
            continue
        hit, how = sec.get(n), "exact"
        if not hit:
            cand = difflib.get_close_matches(n, by_first.get(n.split()[0], []), n=1, cutoff=0.92)
            if cand:
                hit, how = sec[cand[0]], f"fuzzy:{cand[0]}"
        if hit:
            matches.append({**p, "cik": int(hit["cik_str"]), "ticker": hit["ticker"],
                            "sec_title": hit["title"], "match": how})

    # Several GHGRP parent spellings can map to one CIK — merge emissions
    by_cik: dict[int, dict] = {}
    for m in matches:
        c = by_cik.setdefault(m["cik"], {**m, "scope1_t": 0.0, "n_facilities": 0, "ghgrp_names": []})
        c["scope1_t"] += m["scope1_t"]
        c["n_facilities"] += m["n_facilities"]
        c["ghgrp_names"].append(m["parent"])

    hdr = {"User-Agent": SEC_UA}

    cdir = RAW / "sec_cik"
    cdir.mkdir(exist_ok=True)
    budget = int(os.environ.get("SEC_MAX_NEW_PER_RUN", "0")) or None
    new_count = [0]

    def enrich(c):
        cf = cdir / f"{c['cik']}_{year}.json"
        if cf.exists():
            return {**c, **json.loads(cf.read_text())}
        if budget is not None and new_count[0] >= budget:
            c["pending"] = True
            return c
        new_count[0] += 1
        c = _enrich_live(c)
        keep = {k: c.get(k) for k in ("sic", "sic_desc", "state_of_inc", "hq_country", "foreign_filer",
                                      "sec_error")}
        if not c.get("sec_error"):
            cf.write_text(json.dumps(keep))
        return c

    # Revenue for every filer in one request per concept (SEC "frames" API,
    # calendar-aligned CY<year>). Far cheaper than per-company companyfacts.
    rev_by_cik: dict[int, tuple[float, str, str]] = {}
    for concept in reversed(REVENUE_CONCEPTS):          # later loop wins → first concept has priority
        fcache = RAW / f"sec_frame_{concept}_CY{year}.json"
        if not fcache.exists():
            try:
                fr = _get_json(f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/USD/CY{year}.json",
                               hdr, timeout=60)
                fcache.write_text(json.dumps(fr))
            except Exception:                            # noqa: BLE001 — concept may not exist as a frame
                continue
        for row in json.loads(fcache.read_text()).get("data", []):
            if row.get("val", 0) > 0:
                rev_by_cik[int(row["cik"])] = (float(row["val"]), concept, row.get("loc", ""))

    def _enrich_live(c):
        cik10 = f"{c['cik']:010d}"
        try:
            sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json", hdr, timeout=15, retries=2)
            c["sic"] = sub.get("sic")
            c["sic_desc"] = sub.get("sicDescription")
            c["state_of_inc"] = sub.get("stateOfIncorporation")
            addr = (sub.get("addresses") or {}).get("business") or {}
            c["hq_country"] = addr.get("stateOrCountryDescription") or addr.get("stateOrCountry")
            c["foreign_filer"] = "foreign" in str(sub.get("category", "")).lower()
        except Exception as exc:      # noqa: BLE001
            c["sec_error"] = f"submissions: {exc}"
        time.sleep(0.12)              # stay well under SEC's 10 req/s guidance
        return c

    def with_revenue(c):
        r = rev_by_cik.get(int(c["cik"]))
        if r:
            c["revenue_usd"], c["revenue_concept"], c["sec_loc"] = r
        return c

    with ThreadPoolExecutor(max_workers=4) as pool:
        enriched = [with_revenue(c) for c in pool.map(enrich, by_cik.values())]
    pending = sum(1 for c in enriched if c.get("pending"))
    if pending:
        raise PartialDownload(f"SEC: {pending} companies still to fetch; re-run to continue")
    cache.write_text(json.dumps({"year": year, "n_parents": len(parents),
                                 "n_matched_names": len(matches), "companies": enriched}))
    return cache


# ── SIC → engine sector taxonomy (ml.emissions_estimator._SECTOR_INTENSITY) ───

def sic_to_sector(sic: str | int | None, ghgrp_sector: str = "") -> str:
    try:
        s = int(sic)
    except (TypeError, ValueError):
        s = -1
    if 4910 <= s <= 4939 or 4960 <= s <= 4961 or s == 4991:  return "utilities"
    if 4920 <= s <= 4925:                                    return "utilities"      # gas distribution
    if 1220 <= s <= 1241:                                    return "coal"
    if 1310 <= s <= 1389 or 2910 <= s <= 2999 or s == 4610 or 4612 <= s <= 4619: return "oil_gas"
    if s == 4922 or s == 4923:                               return "oil_gas"        # pipelines
    if 3310 <= s <= 3329:                                    return "steel"
    if 3330 <= s <= 3399:                                    return "mining"         # non-ferrous metals
    if s == 3241 or 3270 <= s <= 3299:                       return "cement"
    if 2800 <= s <= 2899 or 3080 <= s <= 3089:               return "chemicals"
    if 1000 <= s <= 1499:                                    return "mining"
    if 2000 <= s <= 2099:                                    return "food_beverage"
    if 100 <= s <= 999:                                      return "agriculture"
    if 3710 <= s <= 3716:                                    return "automotive"
    if 4400 <= s <= 4499:                                    return "shipping"
    if 4510 <= s <= 4599:                                    return "aviation"
    if 4800 <= s <= 4899:                                    return "telecom"
    if 5000 <= s <= 5999:                                    return "retail"
    if 6000 <= s <= 6499 or 6700 <= s <= 6799:               return "financials"
    if 6500 <= s <= 6599 or s == 6798:                       return "real_estate"
    if 3570 <= s <= 3579 or 3670 <= s <= 3679 or 7370 <= s <= 7379: return "technology"
    if 2830 <= s <= 2836 or 8000 <= s <= 8099:               return "healthcare"
    if 1500 <= s <= 1799:                                    return "construction"
    if 2600 <= s <= 2699:                                    return "pulp_paper"
    return "other_industrial"


# ── Stage 3: merge → CSV ──────────────────────────────────────────────────────

def stage_merge(year: int) -> Path:
    data = json.loads(stage_sec(year).read_text())
    out = OUT / f"emissions_ghgrp_sec_{year}.csv"
    cols = ["cik", "ticker", "sec_title", "sic", "sic_desc", "sector", "hq_country", "foreign_filer",
            "revenue_usd_m", "revenue_concept", "scope1_t", "scope1_intensity_t_per_musd",
            "n_facilities", "top_ghgrp_sector", "top_naics3", "us_coverage_flag",
            "match", "ghgrp_names", "source_year", "source"]
    n = 0
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in data["companies"]:
            rev = c.get("revenue_usd")
            if not rev or rev <= 0 or c["scope1_t"] <= 0:
                continue
            rev_m = rev / 1e6
            loc = str(c.get("sec_loc") or "")
            foreign = bool(c.get("foreign_filer")) or (bool(loc) and not loc.startswith("US"))
            w.writerow({
                "cik": c["cik"], "ticker": c["ticker"], "sec_title": c["sec_title"],
                "sic": c.get("sic"), "sic_desc": c.get("sic_desc"),
                "sector": sic_to_sector(c.get("sic"), c.get("top_ghgrp_sector", "")),
                "hq_country": c.get("sec_loc") or c.get("hq_country"), "foreign_filer": foreign,
                "revenue_usd_m": round(rev_m, 2), "revenue_concept": c.get("revenue_concept"),
                "scope1_t": round(c["scope1_t"], 1),
                "scope1_intensity_t_per_musd": round(c["scope1_t"] / rev_m, 3),
                "n_facilities": c["n_facilities"], "top_ghgrp_sector": c.get("top_ghgrp_sector"),
                "top_naics3": c.get("top_naics3"),
                # filled later by the Climate TRACE coverage check; default unknown
                "us_coverage_flag": "foreign_hq" if foreign else "unknown",
                "match": c["match"], "ghgrp_names": " | ".join(c["ghgrp_names"])[:500],
                "source_year": year,
                "source": "EPA GHGRP (direct emitters, excl. suppliers & biogenic) + SEC EDGAR XBRL",
            })
            n += 1
    print(f"wrote {n} companies → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--stage", choices=["ghgrp", "sec", "merge", "all"], default="all")
    a = ap.parse_args()
    t0 = time.time()
    if a.stage in ("ghgrp", "all"):
        p = stage_ghgrp(a.year); d = json.loads(p.read_text())
        print(f"GHGRP {a.year}: {d['n_facility_rows']} facilities, {len(d['parents'])} parent companies → {p}")
    if a.stage in ("sec", "all"):
        try:
            p = stage_sec(a.year)
        except PartialDownload as e:
            print(e); sys.exit(3)
        d = json.loads(p.read_text())
        print(f"SEC: {d['n_matched_names']} parent names matched → {len(d['companies'])} registrants → {p}")
    if a.stage in ("merge", "all"):
        stage_merge(a.year)
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)
