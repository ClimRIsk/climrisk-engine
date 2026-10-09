"""
cri/riskmgmt.py — what a risk manager needs around the models: reproducibility, balance-sheet translation, input quality, disclosure drafts, validation evidence.

  audit_log / audit_recent / fingerprint   append-only record of every insurer run and case answer: input hash, seed, engine version, index version, model, result summary
  translate(fin, loss, covenants)          physical loss -> EBITDA, leverage, interest cover, equity ratio, covenant headroom and breach flags (arithmetic only; assumptions listed)
  exposure_quality(sites)                  data-quality report on an exposure file before it is modelled
  issb_s2_draft(outputs, org)              IFRS S2 / TCFD-style disclosure draft; each statement tagged with the engine output it came from, everything else "management to complete"
  validation_pack(...)                     model validation document built from the real test, evaluation and demo results, including what has NOT been validated

Nothing here estimates a number the engine did not compute or the user did not supply.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Optional

from . import __version__

TBC = "management to complete"


# ── 1. audit trail ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _home() -> Path:
    h = os.environ.get("CRI_HOME")
    return Path(h) if h else Path.home() / ".climrisk"


def fingerprint(obj) -> str:
    """Stable SHA-256 of any JSON-able object (key order and float repr do not matter)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def _index_version() -> Optional[str]:
    try:
        from .resident import casebase as CB
        p = CB.index_path()
        if p and Path(p).exists():
            st = Path(p).stat()
            return f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}"
    except Exception:                                                # noqa: BLE001
        pass
    return None


def audit_log(kind: str, inputs, summary: dict, **meta) -> Optional[dict]:
    """Append one record. Best effort: an audit failure never breaks the thing being audited."""
    try:
        rec = {"id": fingerprint([kind, time.time(), inputs])[:16], "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, "engine_version": __version__,
               "input_sha256": fingerprint(inputs), "case_index": _index_version(), "summary": summary, **meta}
        p = _home() / "audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec
    except Exception:                                                # noqa: BLE001
        return None


def audit_recent(n: int = 50, kind: Optional[str] = None) -> list:
    p = _home() / "audit.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines()[-5000:]:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if kind is None or r.get("kind") == kind:
            out.append(r)
    return out[-n:][::-1]


def audit_check(record_id: str, inputs) -> dict:
    """Does this input reproduce the recorded run? True only if the input hash matches."""
    for r in audit_recent(5000):
        if r.get("id") == record_id:
            ok = r.get("input_sha256") == fingerprint(inputs)
            return {"found": True, "same_input": ok, "record": r}
    return {"found": False, "same_input": False}


# ── 2. translation to balance sheet, covenants and KRIs ─────────────────────────────────────────────────────────────────────────

def _f(d: dict, k: str, default=None):
    v = d.get(k, default)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{k} is not a number: {v!r}")
    if math.isnan(x) or math.isinf(x):
        raise ValueError(f"{k} is not finite")
    return x


def _ratios(ebitda, net_debt, interest, equity, assets):
    r = {"net_debt_to_ebitda": (net_debt / ebitda) if ebitda and ebitda > 0 else None,
         "interest_cover": (ebitda / interest) if interest and interest > 0 else None,
         "equity_ratio": (equity / assets) if assets and assets > 0 else None}
    return r


def translate(fin: dict, loss: dict, covenants: Optional[dict] = None) -> dict:
    """
    fin       revenue, ebitda, net_debt, interest_expense, equity, total_assets   (annual, same currency)
    loss      asset_damage, insured_recovery (default 0), bi_revenue_lost (default 0), bi_margin (flow-through of lost revenue to EBITDA, default = ebitda/revenue),
              extra_costs (default 0), debt_funded_share (share of the uninsured cash cost funded by new debt, default 1), tax_rate (default 0)
    covenants max_net_debt_to_ebitda, min_interest_cover, min_equity_ratio  (any subset)
    All arithmetic; no loss figure is invented. Insurance recovery is taken as given by the user (the engine's insured loss is an input, not a forecast of payment timing).
    """
    revenue, ebitda, nd, intr = _f(fin, "revenue"), _f(fin, "ebitda"), _f(fin, "net_debt"), _f(fin, "interest_expense")
    eq, ta = _f(fin, "equity"), _f(fin, "total_assets")
    if ebitda is None or nd is None:
        raise ValueError("fin needs at least ebitda and net_debt")
    dmg, rec = _f(loss, "asset_damage", 0.0), _f(loss, "insured_recovery", 0.0)
    bi_rev, extra = _f(loss, "bi_revenue_lost", 0.0), _f(loss, "extra_costs", 0.0)
    tax, dfs = _f(loss, "tax_rate", 0.0), _f(loss, "debt_funded_share", 1.0)
    margin = _f(loss, "bi_margin")
    assum = []
    if margin is None:
        if revenue and revenue > 0:
            margin = ebitda / revenue
            assum.append("bi_margin not given: lost revenue is assumed to flow through at the company's average EBITDA margin (conservative companies use a higher incremental margin)")
        else:
            margin = 0.0
    if rec > dmg + extra + bi_rev * margin + 1e-9:
        assum.append("insured_recovery exceeds the modelled loss; check policy limits and what is actually insured")
    if not (0 <= tax < 1) or not (0 <= dfs <= 1):
        raise ValueError("tax_rate must be in [0,1) and debt_funded_share in [0,1]")
    uninsured_asset = max(0.0, dmg - rec)
    ebitda_hit = bi_rev * margin + extra
    ebitda2 = ebitda - ebitda_hit
    pre_tax_loss = uninsured_asset + ebitda_hit
    equity_hit = pre_tax_loss * (1 - tax)
    cash_out = uninsured_asset + ebitda_hit
    nd2 = nd + cash_out * dfs
    eq2 = None if eq is None else eq - equity_hit
    ta2 = None if ta is None else ta - dmg + rec * 0.0                  # assets written down by the damage; recoveries are cash, not restored assets
    before, after = _ratios(ebitda, nd, intr, eq, ta), _ratios(ebitda2, nd2, intr, eq2, ta2)
    cov = covenants or {}
    tests = []

    def test(name, key, thr, kind):                                  # kind: 'max' (must stay <=) or 'min' (must stay >=)
        if thr is None:
            return
        thr = float(thr)
        b, a = before.get(key), after.get(key)
        row = {"covenant": name, "threshold": thr, "before": b, "after": a, "rule": "<=" if kind == "max" else ">="}
        if a is None:
            row.update(status="breach" if (key == "net_debt_to_ebitda" and ebitda2 <= 0) else "not computable", headroom_pct=None)
        else:
            ok = a <= thr if kind == "max" else a >= thr
            head = (thr - a) / thr if kind == "max" else (a - thr) / thr
            row.update(status="pass" if ok else "breach", headroom_pct=round(100 * head, 1))
            if ok and head < 0.15:
                row["status"] = "pass, under 15% headroom"
        tests.append(row)

    test("Net debt / EBITDA", "net_debt_to_ebitda", cov.get("max_net_debt_to_ebitda"), "max")
    test("EBITDA / interest", "interest_cover", cov.get("min_interest_cover"), "min")
    test("Equity / total assets", "equity_ratio", cov.get("min_equity_ratio"), "min")
    # the EBITDA fall that would, on its own, breach the leverage covenant (closed form, before this loss)
    cush = None
    if cov.get("max_net_debt_to_ebitda") and ebitda > 0:
        need = nd / float(cov["max_net_debt_to_ebitda"])
        cush = {"ebitda_floor": need, "fall_to_breach_pct": round(100 * (ebitda - need) / ebitda, 1),
                "note": "EBITDA fall, with net debt unchanged, at which the leverage covenant would be breached"}
    kris = [{"kri": "Loss as % of EBITDA", "value": round(100 * pre_tax_loss / ebitda, 1) if ebitda else None, "unit": "%"},
            {"kri": "Uninsured share of the loss", "value": round(100 * (pre_tax_loss - min(rec, pre_tax_loss)) / pre_tax_loss, 1) if pre_tax_loss > 0 else 0.0, "unit": "%"},
            {"kri": "Liquidity need (uninsured cash cost)", "value": cash_out, "unit": "currency"}]
    return {"before": {"ebitda": ebitda, "net_debt": nd, "equity": eq, **before}, "after": {"ebitda": ebitda2, "net_debt": nd2, "equity": eq2, **after},
            "loss": {"asset_damage": dmg, "insured_recovery": rec, "uninsured_asset_loss": uninsured_asset, "ebitda_hit": ebitda_hit, "pre_tax_loss": pre_tax_loss},
            "covenants": tests, "breaches": [t["covenant"] for t in tests if t["status"] == "breach"], "ebitda_cushion": cush, "kris": kris,
            "assumptions": assum + ["the uninsured cash cost is funded by new debt in the share given (default all of it)", "interest expense is held at its current level (no repricing after a downgrade)",
                                    "no covenant waiver, cure right, equity injection or insurer prepayment is assumed"],
            "status": "arithmetic on the numbers supplied; not a forecast of timing or of lender behaviour"}


# ── 3. exposure data quality ────────────────────────────────────────────────────────────────────────────────────────────────────

def exposure_quality(sites: list) -> dict:
    issues, n = [], len(sites)
    tivs, seen = [], {}
    for i, s in enumerate(sites, 1):
        if not isinstance(s, dict):
            issues.append({"row": i, "severity": "error", "check": "row_type", "msg": "row is not an object"})
            continue

        def add(sev, check, msg):
            issues.append({"row": i, "name": s.get("name"), "severity": sev, "check": check, "msg": msg})
        lat, lon = s.get("lat"), s.get("lon")
        try:
            la, lo = float(lat), float(lon)
            if not (-90 <= la <= 90 and -180 <= lo <= 180):
                add("error", "coord_range", f"coordinates outside the valid range ({lat}, {lon})")
            elif la == 0 and lo == 0:
                add("error", "null_island", "coordinates are 0,0, which is a geocoder failure, not a place")
            else:
                dec = lambda v: len(str(v).split(".")[1]) if "." in str(v) else 0      # noqa: E731
                if dec(lat) < 2 and dec(lon) < 2:
                    add("warn", "coord_precision", "coordinates have under 2 decimals (about 1 km), likely a city centroid rather than the site")
                key = (round(la, 4), round(lo, 4))
                if key in seen:
                    add("warn", "duplicate_location", f"same location as row {seen[key]}; if these are different buildings, TIV may be double counted in one grid cell")
                else:
                    seen[key] = i
        except (TypeError, ValueError):
            add("error", "coord_missing", "latitude or longitude missing or not a number")
        tv = s.get("tiv_usd", s.get("value_usd"))
        try:
            t = float(tv)
            if t <= 0:
                add("error", "tiv_nonpositive", "total insured value must be above zero")
            else:
                tivs.append(t)
        except (TypeError, ValueError):
            add("error", "tiv_missing", "total insured value missing or not a number")
        try:
            lim, ded = s.get("limit_usd"), s.get("deductible_usd")
            if lim not in (None, "") and ded not in (None, "") and float(lim) <= float(ded):
                add("error", "limit_le_deductible", "limit must exceed the deductible")
            if lim not in (None, "") and tivs and float(lim) > 1.5 * float(tv):
                add("warn", "limit_above_tiv", "limit is more than 1.5 times the insured value")
        except (TypeError, ValueError):
            pass
        if not s.get("country"):
            add("info", "country_missing", "no country; used only for reporting, but needed for regional accumulation views")
        if not s.get("sector"):
            add("info", "sector_missing", "no sector or occupancy; damage curves fall back to a generic industrial curve")
        if not s.get("name"):
            add("info", "name_missing", "no name; rows will be shown as Site N")
    stats = {}
    if tivs:
        med = statistics.median(tivs)
        tot = sum(tivs)
        top = sorted(tivs, reverse=True)
        stats = {"tiv_total": tot, "tiv_median": med, "top10_share_pct": round(100 * sum(top[:10]) / tot, 1)}
        for i, s in enumerate(sites, 1):
            try:
                t = float(s.get("tiv_usd", s.get("value_usd")))
            except (TypeError, ValueError, AttributeError):
                continue
            if med > 0 and t > 50 * med:
                issues.append({"row": i, "name": s.get("name"), "severity": "warn", "check": "tiv_outlier", "msg": "value is over 50 times the median; check units (thousands vs units) and currency"})
    errs = sum(1 for x in issues if x["severity"] == "error")
    warns = sum(1 for x in issues if x["severity"] == "warn")
    rows_bad = len({x["row"] for x in issues if x["severity"] == "error"})
    score = round(100 * (1 - rows_bad / n), 1) if n else 0.0
    return {"n_sites": n, "errors": errs, "warnings": warns, "rows_with_errors": rows_bad, "usable_rows_pct": score, "stats": stats, "issues": issues[:500],
            "verdict": ("do not model: no usable rows" if n == 0 or rows_bad >= n else "fit to model" if errs == 0 and warns <= max(2, n // 20) else "model after fixing errors"),
            "note": "Checks are mechanical (ranges, missing fields, duplicates, outliers); they do not verify that a coordinate is the real building location."}


# ── 4. ISSB IFRS S2 / TCFD draft ──────────────────────────────────────────────────────────────────────────────────────────────────

def _g(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def _money(v):
    return "n/a" if v is None else f"{v:,.0f}"


def issb_s2_draft(outputs: dict, org: Optional[dict] = None) -> dict:
    """
    outputs   {insurance: run() result, translation: translate() result, assessment: site assessment dict, precedents: casebase passages}
    org       {name, reporting_year, currency, governance: {...}, transition: {...}, emissions: {scope1, scope2, scope3}, targets}  supplied by the company
    Each line carries 'source' (engine output path or 'supplied by the company') or the text 'management to complete'.
    """
    org = org or {}
    ins, tr = outputs.get("insurance") or {}, outputs.get("translation") or {}
    sections = []

    def sec(code, title, lines):
        sections.append({"ref": code, "title": title, "lines": lines})

    def ln(text, source):
        return {"text": text, "source": source}

    gov = org.get("governance") or {}
    sec("S2.5-7", "Governance", [ln(gov.get("board_oversight") or f"Board oversight of climate-related risks: {TBC}.", "supplied by the company" if gov.get("board_oversight") else TBC),
                                 ln(gov.get("management_role") or f"Management's role in assessing and managing climate-related risks: {TBC}.", "supplied by the company" if gov.get("management_role") else TBC)])
    phys = []
    gross = _g(ins, "stats", "gross")
    if gross:
        rp = {r["rp"]: r for r in gross.get("rows", [])}
        phys.append(ln(f"Modelled average annual insured loss across {ins.get('n_sites')} sites with total insured value {_money(ins.get('tiv'))} is {_money(gross.get('aal'))}, from simulated wind and flood perils.", "engine: insurance.stats.gross.aal"))
        if 100 in rp:
            phys.append(ln(f"The 1-in-100-year annual aggregate loss is {_money(rp[100].get('aep'))} and the occurrence loss {_money(rp[100].get('oep'))}.", "engine: insurance.stats.gross.rows[rp=100]"))
    cl = ins.get("climate")
    if cl:
        a = cl.get("aal", {})
        phys.append(ln(f"Under {cl.get('ssp')} to {cl.get('year')}, the modelled average annual loss changes from {_money(a.get('today'))} to {_money(a.get('scenario'))}"
                       + (f" ({a['change_pct']:+.1f}%)" if a.get("change_pct") is not None else "") + f"; evidence confidence: {cl.get('confidence') or 'see engine output'}.", "engine: insurance.climate.aal"))
    if not phys:
        phys.append(ln(f"Physical risk exposure and modelled loss: {TBC} (run the insurer or site assessment to populate this).", TBC))
    phys.append(ln(f"Perils not modelled by the engine (hail, convective storm, earthquake, wildfire, subsidence, separate business interruption values) are outside these figures: {TBC} for any that matter to the company.", "engine: insurance.limits"))
    tl = org.get("transition") or {}
    trans = [ln(tl.get("summary") or f"Transition risks (policy, technology, market, reputation) and their time horizons: {TBC}.", "supplied by the company" if tl.get("summary") else TBC)]
    sec("S2.9-12", "Strategy: climate-related risks and opportunities", phys + trans)
    fin = []
    if tr:
        a, b = tr.get("after", {}), tr.get("before", {})
        lo = tr.get("loss", {})
        fin.append(ln(f"For the modelled loss of {_money(lo.get('pre_tax_loss'))}, EBITDA moves from {_money(b.get('ebitda'))} to {_money(a.get('ebitda'))} and net debt from {_money(b.get('net_debt'))} to {_money(a.get('net_debt'))}.", "engine: translation"))
        for c in tr.get("covenants", []):
            fin.append(ln(f"{c['covenant']}: {c['status']} (threshold {c['rule']} {c['threshold']}; after the loss {c['after'] if c['after'] is None else round(c['after'], 2)}).", "engine: translation.covenants"))
    else:
        fin.append(ln(f"Anticipated financial effects on financial position, performance and cash flows: {TBC}.", TBC))
    sec("S2.15-21", "Financial effects", fin)
    sec("S2.24-26", "Risk management", [ln(org.get("risk_management") or f"How climate risks are identified, assessed and integrated into enterprise risk management: {TBC}.", "supplied by the company" if org.get("risk_management") else TBC),
                                        ln("Quantification of physical risk uses stochastic simulation with a fixed seed, so any figure here can be reproduced from the audit record of the run.", "engine: audit trail")])
    em = org.get("emissions") or {}
    mets = [ln(f"Scope {s} greenhouse-gas emissions: " + (f"{em[k]} (supplied by the company)." if em.get(k) is not None else f"{TBC}."), "supplied by the company" if em.get(k) is not None else TBC)
            for s, k in (("1", "scope1"), ("2", "scope2"), ("3", "scope3"))]
    mets.append(ln(org.get("targets") or f"Climate-related targets and progress: {TBC}.", "supplied by the company" if org.get("targets") else TBC))
    sec("S2.27-37", "Metrics and targets", mets)
    prec = outputs.get("precedents")
    if prec:
        sec("Context", "Precedents from real events (not part of the standard)", [ln(p["text"] if isinstance(p, dict) else str(p), (p.get("url") if isinstance(p, dict) else "case library")) for p in prec[:5]])
    filled = sum(1 for s in sections for l in s["lines"] if not l["source"].startswith("management") and l["source"] != TBC)
    todo = sum(1 for s in sections for l in s["lines"] if l["source"] == TBC)
    md = [f"# Climate-related financial disclosure: draft ({org.get('name') or 'company name: ' + TBC}, {org.get('reporting_year') or 'year: ' + TBC})", "",
          "Draft prepared from ClimRisk engine outputs. Lines marked \"management to complete\" were not available to the engine and must be written by the company. Not assured, not legal advice.", ""]
    for s in sections:
        md += [f"## {s['ref']}  {s['title']}", ""] + [f"- {l['text']}  _[{l['source']}]_" for l in s["lines"]] + [""]
    return {"sections": sections, "markdown": "\n".join(md), "lines_from_engine_or_company": filled, "lines_to_complete": todo,
            "standard": "IFRS S2 Climate-related Disclosures (paragraph references are to the four-pillar structure shared with TCFD)", "status": "draft"}


# ── 5. model validation pack ──────────────────────────────────────────────────────────────────────────────────────────────────────

def validation_pack(demo_report: Optional[str] = None, eval_report: Optional[str] = None, tests_summary: Optional[str] = None, extra: Optional[dict] = None) -> dict:
    """Built from files the pipeline actually wrote. A section with no evidence says so rather than being filled in."""
    def load(p):
        try:
            return json.loads(Path(p).read_text()) if p and Path(p).exists() else None
        except Exception:                                            # noqa: BLE001
            return None
    demo, ev = load(demo_report), load(eval_report)
    idx = None
    try:
        from .resident import casebase as CB
        idx = CB.stats() if CB.available() else None
    except Exception:                                                # noqa: BLE001
        idx = None
    md = [f"# ClimRisk model validation pack (engine {__version__})", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from the files named in each section. Where evidence is missing the section says so.", ""]
    md += ["## 1. Intended use", "", "Screening and explanation of climate-related physical and transition risk for assets, portfolios and financial statements; decision support for a risk manager. Not a replacement for a vendor catastrophe model when rating an individual risk.", ""]
    md += ["## 2. Data lineage", ""]
    if idx:
        md += [f"Case index: {json.dumps(idx, default=str)[:600]}", "", "Case facts are curated sentences from named sources or verbatim text from SEC filings, World Bank statistics and SEC XBRL structured data; each passage carries its URL.", ""]
    else:
        md += ["Case index not available in this environment.", ""]
    md += ["## 3. Conceptual soundness", "", "Hazard and damage components use published models (see insurance.METHOD and insurance.SOURCES in the engine). Correlation lengths are assumed, not fitted.", ""]
    md += ["## 4. Language model evaluation (held out by whole event)", ""]
    if ev:
        md += ["```", json.dumps(ev, indent=1)[:3500], "```", ""]
    else:
        md += ["No evaluation file supplied: run the evaluation stage and pass its JSON.", ""]
    md += ["## 5. End-to-end behaviour (scripted investor-demo questions)", ""]
    if demo and demo.get("summary"):
        s = demo["summary"]
        md += [f"{s.get('passed')}/{s.get('questions')} questions passed the honesty check (pass rate {s.get('pass_rate')}); median latency {s.get('median_s', s.get('p50_s', 'n/a'))} s, 95th percentile {s.get('p95_s', 'n/a')} s.", ""]
        if s.get("failures"):
            md += ["Failures:", ""] + [f"- {json.dumps(x, default=str)[:200]}" for x in s["failures"][:10]] + [""]
    else:
        md += ["No demo report supplied.", ""]
    md += ["## 6. Software tests", ""]
    md += [(tests_summary.strip() if tests_summary else "No test summary supplied.") , ""]
    md += ["## 7. Limitations", ""]
    try:
        from .insurance import LIMITS
        md += [f"- {x}" for x in LIMITS]
    except Exception:                                                # noqa: BLE001
        pass
    md += ["- Non-US structured company financials are not covered (SEC XBRL is US only); non-US company evidence comes from filings by foreign private issuers and curated cases.", ""]
    md += ["## 8. What has NOT been validated", "",
           "- Modelled loss against reported insured losses for a named event: needs a partner's exposure data. Until then the loss numbers are screening-grade.",
           "- Independent review by practising risk professionals: not yet done; this pack is evidence for such a review, not a substitute.", ""]
    if extra:
        md += ["## 9. Additional evidence", "", "```", json.dumps(extra, indent=1, default=str)[:3000], "```", ""]
    return {"markdown": "\n".join(md), "has_eval": bool(ev), "has_demo": bool(demo), "has_tests": bool(tests_summary)}
