"""
Climate-adjusted PD, LGD, ECL and IRB capital from a structural model: every number is computed from inputs the user supplies and published formulas. No fitted or tuned coefficients.

Replaces climate/credit_risk.py, whose sector sensitivities, regional coefficients, logistic betas and "ECB 2022 Table B.4 / C.3" multipliers could not be traced to any source and
were never tested against defaults. It is withdrawn.

Method (all standard, none calibrated here)
  PD   Merton (1974) one-factor structural model. Default if asset value V falls below debt D at horizon T:
         PD = N(-DD),  DD = [ln(V/D) + (r - sigma^2/2) T] / (sigma sqrt(T)).
       The obligor's baseline PD (from the bank's own rating system) and its leverage D/V fix the one unknown, asset volatility sigma (implied, solved by bisection).
       A climate shock then reduces asset value by a fraction s, V' = V (1 - s), with sigma unchanged, and PD' = N(-DD').
  s    Physical: the fraction of asset value the engine's hazard and depth-damage modules put at risk (you supply it, e.g. an annual average loss ratio or a 1-in-N event ratio).
       Transition: annual carbon cost (emissions x carbon price x (1 - pass-through)) divided by EBITDA, on the assumption that enterprise value is proportional to EBITDA.
       Combined: 1 - (1 - s_phys)(1 - s_trans). A shock that is not supplied is NOT ASSESSED (reported, not set to zero).
  LGD  LGD = LGD_unsecured x max(0, EAD - C_eff) / EAD,  C_eff = collateral x (1 - physical damage to collateral) x (1 - recovery haircut). The same event that lowers V lowers the
       collateral, so PD and LGD move together (wrong-way risk) without a correlation parameter.
  ECL  one-year PD x LGD x EAD. No IFRS 9 stage is assigned: significant increase in credit risk is the bank's own test.
  RWA  Basel IRB corporate formula (BCBS, "International Convergence of Capital Measurement", 2006, para 272 and following), maturity adjustment included. Basel II's 1.06 factor applies to aggregate RWA and is an input (default 1.0).

Limits (printed in every result)
  Structural PDs respond to value shocks as the Merton model implies; no test against observed climate-related defaults exists. Asset value shocks are the engine's own screening outputs
  (see the validation ledger), so the credit number is only as good as those. Carbon price, pass-through, recovery haircut and unsecured LGD are the user's inputs and are not defaulted.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional

_N = NormalDist()
METHOD = ("Merton structural PD with asset volatility implied from the baseline PD and leverage; climate enters as a fractional loss of asset value; "
          "LGD from collateral after damage; IRB capital from the Basel formula. No fitted coefficients.")
LIMITS = ("Not tested against observed climate-related defaults. Asset-value shocks are screening outputs from this engine. Carbon price, pass-through, haircut and unsecured LGD are inputs. "
          "Transition shock assumes enterprise value proportional to EBITDA. One-year horizon.")


class CreditError(ValueError):
    pass


def _dd(leverage: float, sigma: float, r: float, T: float) -> float:
    return (math.log(1.0 / leverage) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def implied_asset_vol(pd0: float, leverage: float, r: float = 0.0, T: float = 1.0) -> float:
    """Asset volatility that reproduces pd0 at the given leverage (D/V). Bisection; raises if no solution exists."""
    if not (0.0 < pd0 < 1.0):
        raise CreditError("baseline PD must be strictly between 0 and 1")
    if not (0.0 < leverage < 1.0):
        raise CreditError("leverage D/V must be strictly between 0 and 1 (debt below asset value)")
    if math.log(1.0 / leverage) + r * T <= 0:
        raise CreditError("leverage and rate leave no positive distance to default")
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _N.cdf(-_dd(leverage, mid, r, T)) < pd0:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    if abs(_N.cdf(-_dd(leverage, sigma, r, T)) - pd0) > 1e-6:
        raise CreditError("no asset volatility reproduces this PD at this leverage")
    return sigma


def shocked_pd(pd0: float, leverage: float, shock: float, r: float = 0.0, T: float = 1.0) -> dict:
    if not (0.0 <= shock < 1.0):
        raise CreditError("asset value shock must be in [0, 1)")
    sigma = implied_asset_vol(pd0, leverage, r, T)
    dd0 = _dd(leverage, sigma, r, T)
    lev1 = leverage / (1.0 - shock)
    if lev1 >= math.exp(r * T) * 1e9:
        pd1, dd1 = 1.0, float("-inf")
    else:
        dd1 = (math.log(1.0 / lev1) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        pd1 = _N.cdf(-dd1)
    return {"sigma": sigma, "dd0": dd0, "dd1": dd1, "pd0": _N.cdf(-dd0), "pd1": pd1}


def irb_rwa(pd: float, lgd: float, ead: float, maturity: float = 2.5, scaling: float = 1.0) -> float:
    """Basel IRB corporate risk-weighted assets (BCBS 2006 para 272), multiplied by the scaling input (Basel II used 1.06 on aggregate RWA; default 1.0)."""
    pd = min(max(pd, 1e-4), 0.9999)                         # floor of 0.01% only to keep the formula finite; supervisors set their own, higher floors, so apply yours
    e50 = math.exp(-50.0)
    w = (1.0 - math.exp(-50.0 * pd)) / (1.0 - e50)
    R = 0.12 * w + 0.24 * (1.0 - w)
    b = (0.11852 - 0.05478 * math.log(pd)) ** 2
    arg = (1.0 - R) ** -0.5 * _N.inv_cdf(pd) + (R / (1.0 - R)) ** 0.5 * _N.inv_cdf(0.999)
    K = (lgd * _N.cdf(arg) - pd * lgd) * (1.0 + (maturity - 2.5) * b) / (1.0 - 1.5 * b)
    return max(K, 0.0) * 12.5 * ead * scaling


def _frac(v, name, lo=0.0, hi=1.0):
    if v is None:
        return None
    v = float(v)
    if not (lo <= v <= hi):
        raise CreditError(f"{name} must be between {lo} and {hi}")
    return v


def assess(o: dict) -> dict:
    """One obligor. Required: ead, baseline_pd, leverage. Everything else is optional and its absence is reported as NOT ASSESSED."""
    for k in ("ead", "baseline_pd", "leverage"):
        if o.get(k) is None:
            raise CreditError(f"{k} is required")
    ead, pd0, lev = float(o["ead"]), float(o["baseline_pd"]), float(o["leverage"])
    if ead <= 0:
        raise CreditError("ead must be positive")
    r = float(o.get("risk_free_rate") or 0.0)
    T = 1.0
    na = []

    s_phys = _frac(o.get("physical_value_loss_fraction"), "physical_value_loss_fraction", 0.0, 0.999)
    if s_phys is None:
        na.append("physical shock: physical_value_loss_fraction not supplied")
    s_tr = None
    em, cp, ebitda = o.get("emissions_tco2e"), o.get("carbon_price_usd_per_t"), o.get("ebitda")
    if em is None or cp is None or ebitda is None:
        na.append("transition shock: needs emissions_tco2e, carbon_price_usd_per_t and ebitda")
    else:
        if float(ebitda) <= 0:
            na.append("transition shock: EBITDA is not positive, so a cost-to-EBITDA ratio is meaningless")
        else:
            pt = _frac(o.get("carbon_cost_pass_through"), "carbon_cost_pass_through") or 0.0
            cost = float(em) * float(cp) * (1.0 - pt)
            s_tr = min(0.999, cost / float(ebitda))
    comb = 1.0 - (1.0 - (s_phys or 0.0)) * (1.0 - (s_tr or 0.0))

    sp = shocked_pd(pd0, lev, comb, r, T)

    lgd_u = _frac(o.get("lgd_unsecured"), "lgd_unsecured")
    coll = o.get("collateral_value")
    hair = _frac(o.get("recovery_haircut"), "recovery_haircut")
    dmg = _frac(o.get("collateral_damage_fraction"), "collateral_damage_fraction")
    lgd0 = lgd1 = None
    if lgd_u is None:
        na.append("LGD: lgd_unsecured not supplied")
    elif coll is None:
        lgd0 = lgd1 = lgd_u
        if s_phys:
            na.append("LGD physical effect: no collateral supplied, so LGD is held at the unsecured value")
    elif hair is None:
        na.append("LGD: recovery_haircut not supplied with collateral")
    else:
        def lgd(d):
            ceff = float(coll) * (1.0 - d) * (1.0 - hair)
            return lgd_u * max(0.0, ead - ceff) / ead
        lgd0 = lgd(0.0)
        if dmg is None:
            lgd1 = lgd0
            if s_phys:
                na.append("collateral damage: collateral_damage_fraction not supplied, so collateral is not reduced")
        else:
            lgd1 = lgd(dmg)

    out = {"inputs": {k: o.get(k) for k in o}, "implied_asset_vol": sp["sigma"], "distance_to_default": {"baseline": sp["dd0"], "climate": sp["dd1"]},
           "shocks": {"physical": s_phys, "transition": s_tr, "combined": comb},
           "pd": {"baseline": sp["pd0"], "climate": sp["pd1"], "increase": sp["pd1"] - sp["pd0"]},
           "lgd": {"baseline": lgd0, "climate": lgd1},
           "not_assessed": na, "method": METHOD, "limits": LIMITS}
    if lgd0 is not None:
        e0, e1 = sp["pd0"] * lgd0 * ead, sp["pd1"] * lgd1 * ead
        out["ecl"] = {"baseline": e0, "climate": e1, "increase": e1 - e0}
        m = float(o.get("maturity_years") or 2.5)
        sc = float(o.get("irb_scaling") or 1.0)
        out["rwa"] = {"baseline": irb_rwa(sp["pd0"], lgd0, ead, m, sc), "climate": irb_rwa(sp["pd1"], lgd1, ead, m, sc), "maturity_years": m, "scaling": sc}
    else:
        out["ecl"] = out["rwa"] = None
    return out


def portfolio(obligors: list) -> dict:
    if not obligors:
        raise CreditError("no obligors")
    rows, tot = [], {"ead": 0.0, "ecl_b": 0.0, "ecl_c": 0.0, "rwa_b": 0.0, "rwa_c": 0.0}
    skipped = 0
    for i, o in enumerate(obligors):
        try:
            r = assess(o)
        except CreditError as e:
            raise CreditError(f"obligor {i + 1}: {e}")
        r["id"] = o.get("id") or o.get("name") or f"obligor {i + 1}"
        rows.append(r)
        tot["ead"] += float(o["ead"])
        if r["ecl"]:
            tot["ecl_b"] += r["ecl"]["baseline"]; tot["ecl_c"] += r["ecl"]["climate"]
            tot["rwa_b"] += r["rwa"]["baseline"]; tot["rwa_c"] += r["rwa"]["climate"]
        else:
            skipped += 1
    return {"n": len(rows), "ead": tot["ead"], "obligors_without_lgd": skipped,
            "ecl": None if skipped == len(rows) else {"baseline": tot["ecl_b"], "climate": tot["ecl_c"], "increase": tot["ecl_c"] - tot["ecl_b"]},
            "rwa": None if skipped == len(rows) else {"baseline": tot["rwa_b"], "climate": tot["rwa_c"]},
            "note": ("Totals cover obligors with an LGD only." if skipped else None), "obligors": rows, "method": METHOD, "limits": LIMITS}
