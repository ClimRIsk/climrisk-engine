"""
cri/creditbridge.py — a bank-style schema (Entity -> Facility -> Collateral) over the catastrophe engine.

Input   entities, each with identifiers (lei / cik / ticker), annual financials, and facilities; each facility has a drawn amount, covenants and collateral (located assets with a value
        and, if the borrower insures them, deductible/limit terms).
Output  per facility and per entity: collateral value, expected annual climate loss, the loss in a 1-in-RP year, loan-to-value before and after that loss, and for the entity the
        EBITDA / leverage / covenant effect (cri.riskmgmt.translate). One simulation covers every collateral asset of every entity together, so losses are correlated across the book
        and the portfolio view shows concentration.

What it deliberately does NOT produce: probability of default, loss given default, exposure at default, ratings. Those need a validated credit model and the bank's own default history.
"collateral shortfall" below is the part of the drawn amount no longer covered by collateral if the borrower defaulted at the same time as the event: an indicator of how climate could widen
loss given default, not an LGD estimate. The loss in a 1-in-RP year is the average over simulated years whose total ground-up loss is within 5% of that return-period loss.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from . import insurance as I
from . import riskmgmt as RM

SCHEMA = {
    "entity": {"id": "str (your key)", "lei": "20-character LEI (check digits validated)", "cik": "SEC CIK (optional)", "ticker": "optional", "name": "str", "country": "optional", "sector": "optional",
               "financials": {"revenue": "num", "ebitda": "num", "net_debt": "num", "interest_expense": "num", "equity": "num", "total_assets": "num"}, "facilities": ["facility"]},
    "facility": {"id": "str", "type": "term_loan | revolver | bond | other", "committed": "num", "drawn": "num", "maturity": "YYYY-MM-DD (optional)",
                 "covenants": {"max_net_debt_to_ebitda": "num", "min_interest_cover": "num", "min_equity_ratio": "num"}, "collateral": ["collateral"]},
    "collateral": {"id": "str", "name": "str", "lat": "num", "lon": "num", "value_usd": "num (market or appraised value)", "sector": "optional", "insured": "bool (default false)",
                   "deductible_pct": "num 0-100", "limit_pct": "num 0-100"},
    "units": "all money in one currency (USD assumed by the hazard curves, which are ratios of value)",
}


def validate_lei(lei: str) -> bool:
    """ISO 17442: 18 alphanumerics + 2 check digits; the number formed by converting letters to 10..35 must leave remainder 1 modulo 97."""
    s = (lei or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{18}[0-9]{2}", s):
        return False
    return int("".join(str(int(c, 36)) for c in s)) % 97 == 1


def _num(d, k, default=None):
    v = d.get(k, default)
    if v in (None, ""):
        return default
    return float(v)


def flatten(entities: list) -> tuple:
    """-> (sites, index) where index[j] = (entity_i, facility_i, collateral_i); and a list of schema problems."""
    sites, index, problems = [], [], []
    for ei, e in enumerate(entities):
        eid = e.get("id") or e.get("name") or f"entity {ei + 1}"
        if e.get("lei") and not validate_lei(e["lei"]):
            problems.append({"entity": eid, "issue": "lei_check_digits_fail", "detail": f"{e['lei']} is not a valid LEI (length, characters or check digits)"})
        for fi, f in enumerate(e.get("facilities") or []):
            for ci, c in enumerate(f.get("collateral") or []):
                ref = f"{eid}/{f.get('id', fi + 1)}/{c.get('id', ci + 1)}"
                try:
                    val = _num(c, "value_usd")
                    if val is None or val <= 0:
                        raise ValueError("value_usd missing or not positive")
                    s = {"name": str(c.get("name") or ref), "lat": c.get("lat"), "lon": c.get("lon"), "tiv_usd": val, "sector": c.get("sector") or e.get("sector") or "other", "covers": ["wind", "flood"]}
                    if c.get("insured"):
                        if c.get("deductible_pct") is not None:
                            s["deductible_pct"] = float(c["deductible_pct"])
                        if c.get("limit_pct") is not None:
                            s["limit_pct"] = float(c["limit_pct"])
                    I.normalise_site(s, len(sites))                    # same validation as the insurer run (coordinates, ranges)
                    sites.append(s)
                    index.append((ei, fi, ci))
                except (ValueError, I.InsuranceError) as ex:
                    problems.append({"entity": eid, "issue": "collateral_invalid", "detail": f"{ref}: {ex}"})
    return sites, index, problems


def _scenario_loss(gu_tot: np.ndarray, ins_tot: np.ndarray, rp: float) -> dict:
    """Loss in a 1-in-rp year: ground-up quantile, and the mean ground-up / insured loss over the years within 5% of it."""
    if gu_tot.size == 0 or float(gu_tot.max()) <= 0:
        return {"ground_up": 0.0, "insured": 0.0, "uninsured": 0.0, "years_used": 0}
    q = float(np.quantile(gu_tot, 1.0 - 1.0 / rp))
    m = np.abs(gu_tot - q) <= 0.05 * max(q, 1e-9)
    if not m.any():
        m = gu_tot >= q
    g, i = float(gu_tot[m].mean()), float(ins_tot[m].mean())
    return {"ground_up": g, "insured": i, "uninsured": max(g - i, 0.0), "years_used": int(m.sum())}


def assess(entities: list, return_period: int = 100, n_years: int = 5000, seed: int = 11, corr_scale: float = 1.0, hazards: Optional[list] = None, flood_fn=None, tc_fn=None) -> dict:
    if not entities:
        raise ValueError("no entities")
    if not (10 <= return_period <= 500):
        raise ValueError("return_period must be between 10 and 500")
    sites_in, index, problems = flatten(entities)
    if not sites_in:
        raise ValueError("no valid collateral in any facility: " + "; ".join(p["detail"] for p in problems[:3]))
    sites = [I.normalise_site(s, k) for k, s in enumerate(sites_in)]
    hz = hazards if hazards is not None else I.fetch_hazards(sites, flood_fn=flood_fn, tc_fn=tc_fn)
    gu = I.ground_up(sites, hz, n_years, seed, corr_scale)
    L = I.apply_terms(gu, sites)
    ny = gu["n_years"]
    insured_flag = np.array([bool(c.get("insured")) for e, fi, ci in index for c in [entities[e]["facilities"][fi]["collateral"][ci]]])
    GU = L["ground_up_total"]                                              # years x collateral
    INS = L["total"] * insured_flag[None, :]
    out_entities, port_gu, port_ins = [], np.zeros(ny), np.zeros(ny)
    for ei, e in enumerate(entities):
        eid = e.get("id") or e.get("name") or f"entity {ei + 1}"
        idx_e = [j for j, (a, _, _) in enumerate(index) if a == ei]
        if not idx_e:
            out_entities.append({"id": eid, "lei": e.get("lei"), "cik": e.get("cik"), "ticker": e.get("ticker"), "name": e.get("name"), "facilities": [], "note": "no valid collateral"})
            continue
        g_e, i_e = GU[:, idx_e].sum(axis=1), INS[:, idx_e].sum(axis=1)
        port_gu += g_e
        port_ins += i_e
        sc = _scenario_loss(g_e, i_e, return_period)
        facs = []
        for fi, f in enumerate(e.get("facilities") or []):
            idx_f = [j for j, (a, b, _) in enumerate(index) if a == ei and b == fi]
            drawn = _num(f, "drawn", 0.0) or 0.0
            if not idx_f:
                facs.append({"id": f.get("id", fi + 1), "type": f.get("type"), "drawn": drawn, "collateral_value": 0.0, "note": "no valid collateral"})
                continue
            value = float(sum(sites[j]["tiv"] for j in idx_f))
            g_f, i_f = GU[:, idx_f].sum(axis=1), INS[:, idx_f].sum(axis=1)
            sf = _scenario_loss(g_f, i_f, return_period)
            after = max(value - sf["uninsured"], 0.0)
            facs.append({"id": f.get("id", fi + 1), "type": f.get("type"), "drawn": drawn, "committed": _num(f, "committed"), "maturity": f.get("maturity"), "collateral_assets": len(idx_f), "collateral_value": value,
                         "expected_annual_loss": float(g_f.mean()), "expected_annual_loss_pct_of_value": round(100 * float(g_f.mean()) / value, 3),
                         "expected_annual_uninsured_loss": float((g_f - i_f).mean()),
                         f"loss_1in{return_period}": sf, "ltv_before": (drawn / value) if value else None, "ltv_after": (drawn / after) if after > 0 else None,
                         "collateral_shortfall_if_default": max(drawn - after, 0.0), "collateral_shortfall_pct_of_drawn": round(100 * max(drawn - after, 0.0) / drawn, 1) if drawn > 0 else None,
                         "insured_share_of_value_pct": round(100 * float(sum(sites[j]["tiv"] for j in idx_f if insured_flag[j])) / value, 1)})
        fin, cov = e.get("financials") or {}, {}
        for f in e.get("facilities") or []:                                # the tightest covenant across the entity's facilities
            for k, mode in (("max_net_debt_to_ebitda", min), ("min_interest_cover", max), ("min_equity_ratio", max)):
                v = (f.get("covenants") or {}).get(k)
                if v is not None:
                    cov[k] = mode(cov[k], float(v)) if k in cov else float(v)
        tr = None
        if fin.get("ebitda") is not None and fin.get("net_debt") is not None:
            try:
                tr = RM.translate(fin, {"asset_damage": sc["ground_up"], "insured_recovery": sc["insured"]}, cov or None)
            except ValueError as ex:
                tr = {"error": str(ex)}
        out_entities.append({"id": eid, "lei": e.get("lei"), "lei_valid": validate_lei(e["lei"]) if e.get("lei") else None, "cik": e.get("cik"), "ticker": e.get("ticker"), "name": e.get("name"),
                             "collateral_value": float(sum(sites[j]["tiv"] for j in idx_e)), "expected_annual_loss": float(g_e.mean()), f"loss_1in{return_period}": sc, "facilities": facs,
                             "financial_effect": tr, "breaches": (tr or {}).get("breaches", [])})
    sp = _scenario_loss(port_gu, port_ins, return_period)
    tail = port_gu >= np.quantile(port_gu, 0.99) if port_gu.max() > 0 else np.zeros(ny, bool)
    summed_alone = float(sum(float(np.quantile(GU[:, [j for j, (a, _, _) in enumerate(index) if a == ei]].sum(axis=1), 1 - 1 / return_period)) for ei in range(len(entities))
                             if any(a == ei for a, _, _ in index)))
    portfolio = {"entities": len(entities), "collateral_assets": len(sites), "collateral_value": float(sum(s["tiv"] for s in sites)), "expected_annual_loss": float(port_gu.mean()),
                 f"loss_1in{return_period}": sp, "tail_1pct_mean_loss": float(port_gu[tail].mean()) if tail.any() else 0.0,
                 "diversification": {"sum_of_entity_1in_rp_losses": summed_alone, "portfolio_1in_rp_loss": float(np.quantile(port_gu, 1 - 1 / return_period)),
                                     "note": "the portfolio figure is below the sum when entity losses do not peak in the same years; the gap is the benefit of geographic spread"},
                 "entities_in_breach": [x["id"] for x in out_entities if x.get("breaches")]}
    return {"portfolio": portfolio, "entities": out_entities, "schema_problems": problems, "n_years": ny, "seed": seed, "return_period": return_period,
            "not_produced": ["probability of default", "loss given default", "exposure at default", "credit ratings"],
            "limits": I.LIMITS + ["collateral shortfall is an indicator, not an LGD estimate", "financial effect assumes the loss is funded by debt and no waiver, cure or equity injection (see riskmgmt.translate)"],
            "method": f"One correlated simulation of {ny} years over all collateral; scenario loss = mean over years within 5% of the 1-in-{return_period} ground-up loss; insured recovery only for collateral flagged insured."}
