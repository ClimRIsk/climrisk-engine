"""
finance/earnings.py — from carbon and energy prices to the numbers a lender or an auditor reads.

Optional: runs only when the request carries the asset's (or business unit's) financials. Everything is
real US$, per Monte Carlo draw, per scenario, 2026–2050, and measured against today's earnings held flat.

EBITDA(t) = EBITDA₀ − volume − carbon − abatement opex − power − fuel + commodity price
  volume     revenue₀ × (1 − min(A,1)) × EBITDA margin × operating leverage (A = NGFS sector output index;
             no credit for growth above today, consistent with the emissions side)
  carbon     residual emissions × carbon price faced net of free allocation and cost pass-through (loss model)
  abatement  operating part of the levelised abatement cost (finance/abatement.py)
  power      Δ NGFS electricity price × 3.6 GJ/MWh × MWh bought × (1 − pass-through)
  fuel       Δ NGFS fuel price × GJ burned × output × (1 − pass-through)
  commodity  producers only (coal mining, oil and gas, or `commodity` given): revenue × output ×
             (price ratio − 1) × realisation. Prices are Δ to the Current Policies path of the same
             NGFS model and region, so base-year calibration cancels.

Operating closure: once EBITDA turns negative the plant shuts — the rest of its margin is lost, carbon payments and
abatement spending stop, debt and interest remain.

Balance sheet and credit
  net debt(t)   = net debt₀ + cumulative after-tax EBITDA shortfall + abatement capital outlays
                  (baseline free cash flow is assumed to cover its own capex and distributions)
  interest(t)   = interest₀ + incremental net debt × implied rate (interest₀ / net debt₀)
  leverage      net debt / EBITDA;  coverage  (EBITDA − D&A) / interest  (EBIT basis, as the rating table)
  synthetic rating  Damodaran interest-coverage table for large non-financial firms (Jan 2026) →
                  S&P 1981–2024 average one-year default rate of the rating class, floored at the Basel 0.03 %
  ECL           PD × LGD × net debt (LGD default 45 %, Basel foundation-IRB senior unsecured)
  covenants     leverage and EBITDA/interest tested every year when thresholds are given
  impairment    value in use (PV of after-tax EBITDA − capex − sustaining capex to 2050, no terminal value)
                against the carrying amount; climate-attributable = impairment in the scenario minus
                impairment on the flat baseline (IAS 36 logic, simplified)

Limits: a through-the-cycle rating proxy, not a bank's internal model; financials must cover the same
perimeter as the emissions; debt dynamics ignore refinancing and covenant waivers; no terminal value, so
long-lived assets are conservative on both baseline and scenario (the climate-attributable figure is a difference).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

HORIZONS = (2030, 2035, 2040, 2050)
GJ_PER_MWH = 3.6
BASEL_PD_FLOOR = 0.0003
# Damodaran, "Ratings, Interest Coverage Ratios and Default Spread", large non-financial firms, Jan 2026:
# rating if coverage is in (lower bound, upper bound]. Index 0 = D … 14 = AAA.
ICR_BOUNDS = np.array([0.2, 0.65, 0.8, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.25, 5.5, 6.5, 8.5])
RATINGS = ["D", "C", "CC", "CCC", "B-", "B", "B+", "BB", "BB+", "BBB", "A-", "A", "A+", "AA", "AAA"]
# S&P Global Ratings, Annual Global Corporate Default and Rating Transition Study, 1981–2024 average
# one-year default rate by letter class (as reproduced in public summaries — verify against the latest study).
_PD_CLASS = {"AAA": 0.0000, "AA": 0.0002, "A": 0.0005, "BBB": 0.0016, "BB": 0.0062, "B": 0.0327, "CCC/C": 0.2830, "D": 1.0}
_CLASS_OF = {"AAA": "AAA", "AA": "AA", "A+": "A", "A": "A", "A-": "A", "BBB": "BBB", "BB+": "BB", "BB": "BB",
             "B+": "B", "B": "B", "B-": "B", "CCC": "CCC/C", "CC": "CCC/C", "C": "CCC/C", "D": "D"}
PD_BY_IDX = np.array([max(_PD_CLASS[_CLASS_OF[r]], BASEL_PD_FLOOR) if r != "D" else 1.0 for r in RATINGS])
RATING_SOURCES = ["Damodaran Online, Ratings and Coverage Ratios (Jan 2026, large non-financial firms)",
                  "S&P Global Ratings, Annual Global Corporate Default and Rating Transition Study (1981–2024 averages)",
                  "Basel PD floor 0.03 %"]


class FinancialsError(ValueError):
    pass


def _num(f: dict, key: str, lo=None, hi=None, required=False, default=None):
    v = f.get(key)
    if v is None or v == "":
        if required:
            raise FinancialsError(f"financials: {key} is required")
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise FinancialsError(f"financials: {key} must be a number")
    if not np.isfinite(x) or (lo is not None and x < lo) or (hi is not None and x > hi):
        raise FinancialsError(f"financials: {key} out of range" + (f" ({lo}…{hi})" if lo is not None else ""))
    return x


def normalise(f: dict, act_key: Optional[str]) -> dict:
    """Validate request financials (US$, annual, same perimeter as the emissions)."""
    rev = _num(f, "revenue_usd", lo=0, required=True)
    if rev <= 0:
        raise FinancialsError("financials: revenue_usd must be > 0")
    e0 = _num(f, "ebitda_usd", required=True)
    fuel = f.get("fuel")
    if fuel not in (None, "", "coal", "gas", "oil"):
        raise FinancialsError("financials: fuel must be coal, gas or oil")
    comm = f.get("commodity")
    if comm in (None, ""):
        comm = {"coal-mining": "coal", "oil-and-gas": "oil"}.get(act_key or "")
    elif comm == "none":
        comm = None
    elif comm not in ("coal", "gas", "oil"):
        raise FinancialsError("financials: commodity must be coal, gas, oil or none")
    out = {"revenue": rev, "ebitda": e0, "da": _num(f, "da_usd", lo=0), "interest": _num(f, "interest_usd", lo=0),
           "net_debt": _num(f, "net_debt_usd"), "book": _num(f, "book_value_usd", lo=0),
           "tax": _num(f, "tax_rate", lo=0, hi=0.6, default=0.0), "oplev": _num(f, "operating_leverage", lo=0.5, hi=3.0, default=1.0),
           "realisation": _num(f, "price_realisation", lo=0, hi=1.5, default=1.0),
           "mwh": _num(f, "electricity_mwh", lo=0), "fuel_gj": _num(f, "fuel_gj", lo=0), "fuel": fuel or None,
           "commodity": comm, "sustaining": _num(f, "sustaining_capex_usd", lo=0, default=0.0),
           "cov_lev": _num(f, "covenant_max_leverage", lo=0), "cov_icr": _num(f, "covenant_min_icr", lo=0),
           "lgd": _num(f, "lgd", lo=0, hi=1, default=0.45)}
    if out["fuel_gj"] and not out["fuel"]:
        raise FinancialsError("financials: fuel (coal, gas or oil) is required with fuel_gj")
    return out


def _stack(drv: dict, s: str, models: list, mi: np.ndarray, name: str, key: str, Y: int, fill: float):
    """(n, Y) array of an NGFS energy-price field per draw, or None if no model reports it."""
    per = (drv.get("energy") or {}).get(s) or {}
    rows, have = [], False
    for m in models:
        e = (per.get(m) or {}).get(name)
        have = have or e is not None
        rows.append(e[key] if e is not None else np.full(Y, fill))
    return np.stack(rows)[mi] if have else None


def _pct(x: np.ndarray, q):
    return np.percentile(x, q, axis=0)


def _r(a, d=0):
    return np.round(a, d).tolist()


def compute(F: dict, ctx: dict) -> dict:
    """F: normalised financials. ctx: arrays from the simulation for one scenario —
    Amin, pay, ab_opex, capex (n×Y), pt (n×1), disc (Y), years, discount, drv, scenario, models, mi."""
    Amin, pay, ab_opex, capex = ctx["Amin"], ctx["pay"], ctx["ab_opex"], ctx["capex"]
    n, Y = Amin.shape
    years = np.asarray(ctx["years"])
    disc = ctx["disc"][None, :]
    pt = ctx["pt"]
    e0, rev = F["ebitda"], F["revenue"]
    margin = e0 / rev
    notes = []
    alive = ctx.get("alive")                       # 1 while the plant is within its life / before legal retirement
    alive = np.ones((n, Y)) if alive is None else alive
    base = e0 * alive                              # today's EBITDA, ending when the plant does (not a climate effect)
    retiring = bool((alive < 1).any())

    def cq(x, q):                                  # percentile over the draws whose plant is still operating
        return np.array([np.percentile(x[alive[:, i] > 0, i], q) if (alive[:, i] > 0).any() else 0.0 for i in range(Y)])

    def cmean(x):
        return (x * alive).sum(axis=0) / np.maximum(alive.sum(axis=0), 1)

    vol = rev * (1 - Amin) * max(margin, 0.0) * F["oplev"] * alive
    elec = np.zeros((n, Y))
    if F["mwh"]:
        d = _stack(ctx["drv"], ctx["scenario"], ctx["models"], ctx["mi"], "electricity", "delta_usd_gj", Y, 0.0)
        if d is None:
            notes.append("No NGFS electricity price for this region: power-cost channel skipped.")
        else:
            elec = d * GJ_PER_MWH * F["mwh"] * Amin * (1 - pt) * alive
    else:
        notes.append("Electricity use (MWh) not given: indirect (Scope 2) power-price channel not modelled.")
    fuel = np.zeros((n, Y))
    if F["fuel_gj"]:
        d = _stack(ctx["drv"], ctx["scenario"], ctx["models"], ctx["mi"], F["fuel"], "delta_usd_gj", Y, 0.0)
        if d is None:
            notes.append(f"No NGFS {F['fuel']} price for this region: fuel-cost channel skipped.")
        else:
            fuel = d * F["fuel_gj"] * Amin * (1 - pt) * alive
    comm = np.zeros((n, Y))
    if F["commodity"]:
        ratio = _stack(ctx["drv"], ctx["scenario"], ctx["models"], ctx["mi"], F["commodity"], "ratio", Y, 1.0)
        if ratio is None:
            notes.append(f"No NGFS {F['commodity']} price for this region: commodity-price channel skipped.")
        else:
            comm = rev * Amin * (ratio - 1.0) * F["realisation"] * alive
    ebitda = base - vol - pay - ab_opex - elec - fuel + comm
    # Operating closure: a plant whose EBITDA turns negative shuts (it would not keep burning cash). From that
    # year the whole margin is lost, the other channels and capex stop; debt and interest remain.
    closed = np.maximum.accumulate(ebitda < 0, axis=1) if e0 > 0 else np.zeros((n, Y), bool)
    if closed.any():
        vol = np.where(closed, base, vol)
        pay, ab_opex, elec, fuel, comm, capex = (np.where(closed, 0.0, a) for a in (pay, ab_opex, elec, fuel, comm, capex))
        ebitda = base - vol - pay - ab_opex - elec - fuel + comm
    drag = (base - ebitda) * (1 - F["tax"]) + capex
    cum = np.cumsum(drag, axis=1)
    out = {"years": years.tolist(), "baseline_ebitda": e0, "margin": round(margin, 4),
           "ebitda": {"median": _r(cq(ebitda, 50)), "p5": _r(cq(ebitda, 5)), "p95": _r(cq(ebitda, 95))},
           "channels_pv_mean": {k: float((v * disc).sum(axis=1).mean()) for k, v in
                                (("volume", vol), ("carbon_payments", pay), ("abatement_opex", ab_opex), ("power", elec),
                                 ("fuel", fuel), ("commodity_price", -comm))},
           "channels_note": "present value, positive = reduces EBITDA; after an operating closure the lost margin is booked to volume",
           "prob_closed": [round(float(x), 4) for x in closed.mean(axis=0)],
           "prob_retired": [round(float(x), 4) for x in (1 - alive.mean(axis=0))], "notes": notes, "horizons": {}}
    if retiring:
        notes.append("The plant reaches the end of its life (or legal retirement) inside the horizon in some draws: figures "
                     "after retirement count only the draws still operating; the earnings that end with the plant are not a climate loss.")
    pv_loss = ((base - ebitda) * disc).sum(axis=1)
    out["ebitda_shortfall_pv"] = {"mean": float(pv_loss.mean()), "p50": float(np.percentile(pv_loss, 50)),
                                  "p95": float(np.percentile(pv_loss, 95))}
    out["abatement_capex_pv_mean"] = float((capex * disc).sum(axis=1).mean())
    for yr in HORIZONS:
        i = int(yr - years[0])
        if 0 <= i < Y:
            live = base[:, i] > 0
            chg = (ebitda[live, i] - base[live, i]) / base[live, i] if (e0 > 0 and live.any()) else None
            out["horizons"][str(yr)] = {
                "ebitda_change_pct_median": round(float(100 * np.median(chg)), 1) if chg is not None else None,
                "ebitda_change_pct_p5": round(float(100 * np.percentile(chg, 5)), 1) if chg is not None else None,
                "prob_closed": round(float(closed[:, i].mean()), 4), "prob_retired": round(float(1 - alive[:, i].mean()), 4)}
    # balance sheet
    nd0 = F["net_debt"]
    icr_ebitda = icr = lev = None
    interest = None
    if nd0 is not None:
        nd = nd0 + cum
        lev = np.where(ebitda > 0, nd / np.where(ebitda > 0, ebitda, 1.0), 99.0)
        lev = np.minimum(lev, 99.0)
        out["net_debt"] = {"baseline": nd0, "median": _r(cq(nd, 50)), "p95": _r(cq(nd, 95))}
        out["leverage"] = {"baseline": round(nd0 / e0, 2) if e0 > 0 else None, "median": np.round(cq(lev, 50), 2).tolist(),
                           "p95": np.round(cq(lev, 95), 2).tolist()}
        for yr in HORIZONS:
            if str(yr) in out["horizons"]:
                out["horizons"][str(yr)]["leverage_median"] = round(float(cq(lev, 50)[int(yr - years[0])]), 2)
    else:
        nd = None
        notes.append("Net debt not given: leverage, coverage trend and credit metrics not computed.")
    if F["interest"] is not None and F["interest"] > 0:
        rate = (F["interest"] / nd0) if (nd0 and nd0 > 0) else 0.0
        interest = F["interest"] + (np.maximum(nd - nd0, 0.0) * rate if nd is not None else 0.0)
        icr_ebitda = np.clip(ebitda / interest, -99, 99)
        out["coverage_ebitda"] = {"baseline": round(e0 / F["interest"], 2), "median": np.round(cq(icr_ebitda, 50), 2).tolist(),
                                  "p5": np.round(cq(icr_ebitda, 5), 2).tolist()}
        if F["da"] is not None:
            icr = np.clip((ebitda - F["da"]) / interest, -99, 99)
            base_icr = (e0 - F["da"]) / F["interest"]
            idx = np.searchsorted(ICR_BOUNDS, icr, side="left")
            pd_ = PD_BY_IDX[idx]
            pd_mean = cmean(pd_)
            b_idx = int(np.searchsorted(ICR_BOUNDS, base_icr, side="left"))
            ead = np.maximum(nd, 0.0) if nd is not None else None
            out["credit"] = {
                "basis": "EBIT / interest → Damodaran synthetic rating (large non-financial firms) → S&P class default rate",
                "baseline": {"coverage_ebit": round(base_icr, 2), "rating": RATINGS[b_idx], "pd": float(PD_BY_IDX[b_idx])},
                "rating_median": [RATINGS[int(cq(idx.astype(float), 50)[i])] for i in range(Y)],
                "rating_notches_vs_baseline_median": [int(cq(idx.astype(float), 50)[i]) - b_idx for i in range(Y)],
                "pd_mean": [round(float(x), 5) for x in pd_mean],
                "coverage_ebit_median": np.round(cq(icr, 50), 2).tolist(),
                "ecl_mean_usd": _r(cmean(pd_ * F["lgd"] * ead)) if ead is not None else None,
                "lgd": F["lgd"], "sources": RATING_SOURCES,
                "caveat": "Through-the-cycle rating proxy from coverage; not a substitute for an internal rating model."}
            for yr in HORIZONS:
                if str(yr) in out["horizons"]:
                    i = int(yr - years[0])
                    out["horizons"][str(yr)].update({"rating_median": RATINGS[int(cq(idx.astype(float), 50)[i])],
                                                     "pd_mean": round(float(pd_mean[i]), 5)})
        else:
            notes.append("D&A not given: coverage is EBITDA-based; the synthetic rating and PD need EBIT (give da_usd).")
    elif nd0 is not None:
        notes.append("Interest expense not given: coverage, rating and PD not computed.")
    # covenants
    if (F["cov_lev"] is not None and lev is not None) or (F["cov_icr"] is not None and icr_ebitda is not None):
        br = np.zeros((n, Y), bool)
        if F["cov_lev"] is not None and lev is not None:
            br |= lev > F["cov_lev"]
        if F["cov_icr"] is not None and icr_ebitda is not None:
            br |= icr_ebitda < F["cov_icr"]
        br = br & (alive > 0)
        ever = np.maximum.accumulate(br, axis=1)
        first = np.where(ever.any(axis=1), ever.argmax(axis=1), -1)
        out["covenants"] = {"max_leverage": F["cov_lev"], "min_coverage_ebitda": F["cov_icr"],
                            "prob_in_breach": [round(float(x), 4) for x in cmean(br.astype(float))],
                            "prob_breached_by": [round(float(x), 4) for x in ever.mean(axis=0)],
                            "first_breach_year_median": int(years[0] + np.median(first[first >= 0])) if (first >= 0).any() else None}
        out["covenants"]["by_horizon"] = {str(yr): round(float(ever[:, int(yr - years[0])].mean()), 4)
                                          for yr in HORIZONS if 0 <= int(yr - years[0]) < Y}
    elif F["cov_lev"] is not None or F["cov_icr"] is not None:
        notes.append("Covenant thresholds given but net debt or interest missing: covenants not tested.")
    # impairment
    if F["book"]:
        cf = ebitda * (1 - F["tax"]) - capex - F["sustaining"] * alive
        viu = (cf * disc).sum(axis=1)
        viu0_d = ((base * (1 - F["tax"]) - F["sustaining"] * alive) * disc).sum(axis=1)     # same life, today's earnings
        viu0 = float(viu0_d.mean())
        imp = np.maximum(0.0, F["book"] - np.maximum(viu, 0.0))
        imp0 = np.maximum(0.0, F["book"] - np.maximum(viu0_d, 0.0))
        clim = np.maximum(0.0, imp - imp0)
        tail = clim[clim >= np.percentile(clim, 95)]
        out["impairment"] = {"carrying_amount": F["book"], "value_in_use_baseline": viu0,
                             "value_in_use": {"mean": float(viu.mean()), "p50": float(np.percentile(viu, 50)),
                                              "p5": float(np.percentile(viu, 5))},
                             "climate_attributable": {"mean": float(clim.mean()), "p50": float(np.percentile(clim, 50)),
                                                      "p95": float(np.percentile(clim, 95)), "es95": float(tail.mean())},
                             "climate_attributable_pct_of_carrying": round(100 * float(clim.mean()) / F["book"], 2),
                             "prob_any": round(float((clim > 0).mean()), 4),
                             "baseline_already_impaired": bool(imp0.mean() > 0),
                             "method": "PV of after-tax EBITDA − abatement capex − sustaining capex to 2050 at the model discount rate, "
                                       "no terminal value; climate-attributable = impairment in scenario − impairment on flat baseline."}
        if not F["sustaining"]:
            notes.append("Sustaining capex not given: value in use is before maintenance spending (overstates it; "
                         "impairment is a lower bound).")
    return out
