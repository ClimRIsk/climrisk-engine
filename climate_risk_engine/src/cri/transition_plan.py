"""
Transition plan builder — targets → decarbonisation levers → capex/opex → year-by-year emissions path.

Structured to the disclosure requirements it feeds:
  ESRS E1-1 (transition plan for climate change mitigation, para 16 a–j)  ·  ESRS E1-4 (targets)
  UK Transition Plan Taskforce Disclosure Framework (Foundations, Implementation Strategy, Metrics & Targets)
  IFRS S2 para 14 (transition plan information)

What it computes
  • BAU path from the base year with an organic growth rate (per scope).
  • Planned path = BAU − Σ lever abatement, each lever ramped linearly over its ramp years, capped so a
    scope can never go below zero (overlaps are reported, not silently absorbed).
  • Target path (linear base → near-term target → long-term target) and the gap each year.
  • Marginal abatement cost per lever: (annualised capex + annual opex change) ÷ annual abatement,
    capex annualised with the capital recovery factor at the discount rate over the lever's life.
  • Capex and opex schedules (with like-for-like reinvestment at the end of each lever's life, since its
    abatement is assumed to persist); avoided carbon cost under NGFS Phase 5 shadow prices (median of models)
    for the plan's country, or a user-supplied internal carbon price.
  • SBTi check of the stated target and of the plan's own trajectory (cri.climate.sbti_alignment).
  • Residual emissions at the net-zero year vs the SBTi ≥ 90 % cut; offsets/removals are reported
    separately and never counted as abatement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SCOPES = ("scope1", "scope2", "scope3")


@dataclass
class Lever:
    name: str
    scope: str                      # scope1 | scope2 | scope3
    abatement_t_per_year: float     # at full implementation, relative to BAU
    start_year: int
    ramp_years: int = 1
    capex_usd: float = 0.0          # total, spread evenly over the ramp years
    opex_change_usd_per_year: float = 0.0   # at full implementation; negative = saving
    lifetime_years: int = 20
    category: str = "other"         # efficiency | electrification | renewables | fuel_switch | process | ccs | fleet | supplier | other
    status: str = "planned"         # committed | planned | exploratory
    notes: str = ""


@dataclass
class Plan:
    organisation: str
    base_year: int
    base_emissions_t: dict          # {"scope1": .., "scope2": .., "scope3": ..} (scope2 market-based)
    near_term_year: int
    near_term_reduction_pct: float
    near_term_scopes: tuple = ("scope1", "scope2")
    net_zero_year: Optional[int] = None
    net_zero_reduction_pct: Optional[float] = None
    growth_pct_per_year: dict = field(default_factory=dict)     # BAU organic growth per scope
    levers: list = field(default_factory=list)
    country: Optional[str] = None                                # ISO3 for NGFS carbon prices
    sector: str = ""
    internal_carbon_price_usd_t: Optional[dict] = None           # {year: price} overrides NGFS
    discount_rate: float = 0.07
    offsets_t_per_year: float = 0.0
    locked_in_t: Optional[float] = None                          # ESRS E1-1 16(b) — user estimate
    taxonomy_aligned_capex_pct: Optional[float] = None           # ESRS E1-1 16(e) — from taxonomy module
    end_year: int = 2050


class PlanError(ValueError):
    pass


def _crf(r: float, n: int) -> float:
    return r / (1 - (1 + r) ** -n) if r > 0 else 1.0 / n


def _ramp(l: Lever, y: int) -> float:
    """Share of full abatement in year y. Abatement persists after the lever's life on the assumption of
    like-for-like reinvestment, which is added to the capex schedule (see build)."""
    if y < l.start_year:
        return 0.0
    return min(1.0, (y - l.start_year + 1) / max(1, l.ramp_years))


def _validate(p: Plan) -> None:
    if not p.base_emissions_t or sum(float(p.base_emissions_t.get(s, 0) or 0) for s in SCOPES) <= 0:
        raise PlanError("base_emissions_t must contain positive emissions for at least one scope")
    if not p.base_year < p.near_term_year <= p.end_year:
        raise PlanError("near_term_year must be after base_year and no later than end_year")
    if not 0 < p.near_term_reduction_pct <= 100:
        raise PlanError("near_term_reduction_pct must be in (0, 100]")
    if p.net_zero_year is not None and not p.near_term_year < p.net_zero_year <= p.end_year:
        raise PlanError("net_zero_year must be after the near-term year and no later than end_year")
    for s in p.near_term_scopes:
        if s not in SCOPES:
            raise PlanError(f"unknown scope '{s}'")
    for l in p.levers:
        if l.scope not in SCOPES:
            raise PlanError(f"lever '{l.name}': scope must be scope1, scope2 or scope3")
        if l.abatement_t_per_year < 0 or l.capex_usd < 0 or l.ramp_years < 1 or l.lifetime_years < 1:
            raise PlanError(f"lever '{l.name}': abatement/capex ≥ 0, ramp and lifetime ≥ 1 year")


def _carbon_prices(p: Plan, years: list) -> dict:
    if p.internal_carbon_price_usd_t:
        pts = sorted((int(k), float(v)) for k, v in p.internal_carbon_price_usd_t.items())
        xs, ys = zip(*pts)
        return {"Internal carbon price": [float(np.interp(y, xs, ys)) for y in years]}
    try:
        from .finance import transition as T
        d = T.drivers(p.country, p.sector or "industry", "", None)
        out = {}
        for s, models in d["price"].items():
            arr = np.median(np.stack(list(models.values())), axis=0)
            out[T.label(s)] = [float(np.interp(y, d["years"], arr)) for y in years]
        return out
    except Exception:
        return {}


def build(p: Plan) -> dict:
    _validate(p)
    years = list(range(p.base_year, p.end_year + 1))
    base = {s: float(p.base_emissions_t.get(s, 0) or 0) for s in SCOPES}
    g = {s: float(p.growth_pct_per_year.get(s, 0) or 0) / 100 for s in SCOPES}
    bau = {s: [base[s] * (1 + g[s]) ** (y - p.base_year) for y in years] for s in SCOPES}

    # lever abatement, capped per scope at the BAU level (overlap reported)
    raw = {s: np.zeros(len(years)) for s in SCOPES}
    per_lever = []
    for l in p.levers:
        a = np.array([l.abatement_t_per_year * _ramp(l, y) for y in years])
        raw[l.scope] += a
        per_lever.append((l, a))
    planned, overlap_years = {}, []
    for s in SCOPES:
        b = np.array(bau[s])
        cut = np.minimum(raw[s], b)
        if np.any(raw[s] > b + 1e-6):
            overlap_years.append(s)
        planned[s] = (b - cut).tolist()

    target_scope_base = sum(base[s] for s in p.near_term_scopes)
    nt = target_scope_base * (1 - p.near_term_reduction_pct / 100)
    pts = [(p.base_year, target_scope_base), (p.near_term_year, nt)]
    if p.net_zero_year and p.net_zero_reduction_pct:
        # long-term target covers all scopes; express the target path for near-term scopes consistently
        pts.append((p.net_zero_year, target_scope_base * (1 - p.net_zero_reduction_pct / 100)))
    xs, ys = zip(*pts)
    target_path = [float(np.interp(y, xs, ys)) for y in years]
    plan_nt_scopes = [sum(planned[s][i] for s in p.near_term_scopes) for i in range(len(years))]
    gap = [plan_nt_scopes[i] - target_path[i] for i in range(len(years))]
    i_nt = years.index(p.near_term_year)

    # economics
    prices = _carbon_prices(p, years)
    r = p.discount_rate
    disc = np.array([(1 + r) ** -(y - p.base_year) for y in years])
    levers_out, capex_sched, opex_sched = [], np.zeros(len(years)), np.zeros(len(years))
    for l, a in per_lever:
        def _capex(y):
            k = y - l.start_year
            if k < 0:
                return 0.0
            return l.capex_usd / l.ramp_years if (k % l.lifetime_years) < l.ramp_years else 0.0
        per_year_capex = np.array([_capex(y) for y in years])   # includes reinvestment at end of life
        opex = np.array([l.opex_change_usd_per_year * _ramp(l, y) for y in years])
        capex_sched += per_year_capex
        opex_sched += opex
        ann = l.capex_usd * _crf(r, l.lifetime_years) + l.opex_change_usd_per_year
        mac = ann / l.abatement_t_per_year if l.abatement_t_per_year > 0 else None
        npv_cost = float((per_year_capex * disc).sum() + (opex * disc).sum())
        avoided = {k: float((a * np.array(v) * disc).sum()) for k, v in prices.items()}
        levers_out.append({
            "name": l.name, "scope": l.scope, "category": l.category, "status": l.status,
            "abatement_t_per_year": l.abatement_t_per_year, "start_year": l.start_year, "ramp_years": l.ramp_years,
            "capex_usd": l.capex_usd, "opex_change_usd_per_year": l.opex_change_usd_per_year,
            "lifetime_years": l.lifetime_years,
            "abatement_cost_usd_per_t": round(mac, 1) if mac is not None else None,
            "cumulative_abatement_t": float(a.sum()),
            "npv_cost_usd": round(npv_cost), "npv_avoided_carbon_cost_usd": {k: round(v) for k, v in avoided.items()},
            "notes": l.notes,
        })
    macc = sorted([x for x in levers_out if x["abatement_cost_usd_per_t"] is not None],
                  key=lambda x: x["abatement_cost_usd_per_t"])

    # checks
    checks = []
    if overlap_years:
        checks.append({"level": "warning", "message": f"lever abatement exceeds BAU emissions in {', '.join(overlap_years)} — "
                                                      "levers overlap; the excess is not counted"})
    if gap[i_nt] > 1e-6:
        checks.append({"level": "error", "message": f"the plan misses the {p.near_term_year} target by "
                                                    f"{gap[i_nt]:,.0f} t CO2e ({gap[i_nt] / target_scope_base:.1%} of base)"})
    if p.net_zero_year:
        i_nz = years.index(p.net_zero_year)
        tot_base = sum(base.values())
        resid = sum(planned[s][i_nz] for s in SCOPES)
        cut = 1 - resid / tot_base
        if cut < 0.90:
            checks.append({"level": "error", "message": f"residual emissions in {p.net_zero_year} are {resid:,.0f} t "
                                                        f"({cut:.0%} cut vs base) — SBTi net-zero needs ≥ 90 % abatement before neutralisation"})
    exploratory = sum(x["abatement_t_per_year"] for x in levers_out if x["status"] == "exploratory")
    total_ab = sum(x["abatement_t_per_year"] for x in levers_out) or 1
    if exploratory / total_ab > 0.3:
        checks.append({"level": "warning", "message": f"{exploratory / total_ab:.0%} of planned abatement relies on exploratory levers"})
    if p.offsets_t_per_year:
        checks.append({"level": "info", "message": "offsets/removals are reported separately and not counted towards targets"})
    if not prices:
        checks.append({"level": "info", "message": "no carbon price available — avoided carbon cost not computed"})

    from .climate.sbti_alignment import check as sbti_check
    reported = {years[i]: plan_nt_scopes[i] for i in range(len(years)) if years[i] in (p.near_term_year,)}
    scope_kind = "scope3" if tuple(p.near_term_scopes) == ("scope3",) else "scope12"
    sbti = sbti_check(p.base_year, target_scope_base, p.near_term_year, p.near_term_reduction_pct, scope_kind,
                      reported, p.net_zero_year, p.net_zero_reduction_pct)

    tot_capex = float(capex_sched.sum())
    return {
        "organisation": p.organisation, "base_year": p.base_year, "years": years,
        "bau_t": {s: [round(v, 1) for v in bau[s]] for s in SCOPES},
        "planned_t": {s: [round(v, 1) for v in planned[s]] for s in SCOPES},
        "target_path_t": [round(v, 1) for v in target_path], "target_scopes": list(p.near_term_scopes),
        "planned_target_scopes_t": [round(v, 1) for v in plan_nt_scopes],
        "gap_t": [round(v, 1) for v in gap],
        "near_term": {"year": p.near_term_year, "target_t": round(nt, 1), "planned_t": round(plan_nt_scopes[i_nt], 1),
                      "met": gap[i_nt] <= 1e-6},
        "levers": levers_out, "macc": [{k: x[k] for k in ("name", "abatement_t_per_year", "abatement_cost_usd_per_t")} for x in macc],
        "capex_usd_by_year": [round(v) for v in capex_sched], "opex_change_usd_by_year": [round(v) for v in opex_sched],
        "total_capex_usd": round(tot_capex),
        "carbon_price_paths_usd_t": {k: [round(v, 1) for v in vs] for k, vs in prices.items()},
        "carbon_price_basis": ("internal carbon price" if p.internal_carbon_price_usd_t else
                               "NGFS Phase 5 shadow carbon price, median of three models (a policy-stringency signal, "
                               "not the price the company pays today)"),
        "sbti": sbti, "checks": checks,
        "esrs_e1_1": {
            "a_compatibility_with_1_5C": {"target_ambition_aligned": sbti["ambition_aligned"],
                                          "plan_meets_near_term_target": gap[i_nt] <= 1e-6},
            "b_decarbonisation_levers": [{k: x[k] for k in ("name", "scope", "category", "abatement_t_per_year", "status")}
                                         for x in levers_out],
            "c_capex_opex": {"total_capex_usd": round(tot_capex), "capex_by_year": [round(v) for v in capex_sched]},
            "d_locked_in_emissions_t": p.locked_in_t,
            "e_taxonomy_aligned_capex_pct": p.taxonomy_aligned_capex_pct,
            "f_targets": {"near_term": {"year": p.near_term_year, "reduction_pct": p.near_term_reduction_pct,
                                        "scopes": list(p.near_term_scopes)},
                          "net_zero": {"year": p.net_zero_year, "reduction_pct": p.net_zero_reduction_pct}},
            "g_offsets_t_per_year_reported_separately": p.offsets_t_per_year,
            "h_governance": None, "i_progress": None,
            "note": "h (approval by administrative bodies) and i (progress) are for the company to complete.",
        },
        "tpt_mapping": {"Foundations": "targets + ambition (f, a)", "Implementation Strategy": "levers, capex schedule (b, c)",
                        "Metrics & Targets": "planned vs target path, gap by year", "Engagement Strategy": "scope 3 levers (supplier)",
                        "Governance": "to be completed by the company"},
    }


def from_dict(d: dict) -> Plan:
    levers = [Lever(**l) for l in d.get("levers", [])]
    keys = {k: d[k] for k in ("organisation", "base_year", "base_emissions_t", "near_term_year", "near_term_reduction_pct")}
    opt = {k: d[k] for k in ("net_zero_year", "net_zero_reduction_pct", "growth_pct_per_year", "country", "sector",
                             "internal_carbon_price_usd_t", "discount_rate", "offsets_t_per_year", "locked_in_t",
                             "taxonomy_aligned_capex_pct", "end_year") if k in d and d[k] is not None}
    if "near_term_scopes" in d:
        opt["near_term_scopes"] = tuple(d["near_term_scopes"])
    return Plan(levers=levers, **keys, **opt)
