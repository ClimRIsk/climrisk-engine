"""
finance/distribution.py — Monte Carlo of an asset's climate-related financial loss, 2026–2050.

Every uncertain input is sampled, so the output is a distribution, not a point:

Physical (direct damage to the asset, repaired each year)
  cyclone   N ~ Poisson(λ) storms/yr at tropical-storm force or above; each storm's
            site wind v = 17.5 + Exp(σ) m/s (the exponential tail fitted to 45 years
            of IBTrACS storms at the site, capped at 95 m/s); damage from the
            regional Emanuel curve with v_half ~ U(TDR, RMSF) calibration
  river,    annual-maximum event with return period T = 1/U; footprint damage ratio
  coast     interpolated in log(T) between the modelled return periods; zero when T
            is within the FLOPROS protection standard (defences hold), full
            undefended depth when exceeded; coastal depths move from 2018 to 2050
            sea level linearly; depth-damage uncertainty × Triangular(0.6, 1, 1.4)
  value     asset replacement value ~ Triangular(low, mode, high)

Transition (per NGFS scenario; model drawn uniformly from those reporting the region)
  cost(t) = E·min(1, A(t)) · [p₀·c₀(t) + max(0, P(t) − p₀)·c_f] · (1 − pass-through)
            E emissions today, A sector activity index, p₀ explicit price today,
            P NGFS price, c₀/c_f charged shares today/future, pass-through ~ Tri
  stranded  V·f_econ·max(0, 1 − mean A(t) over remaining life) for fossil-linked
            assets (coal/gas/oil power, coal mining, oil & gas, refining), where
            f_econ ~ Tri(0.15, 0.35, 0.6) is the undepreciated share of replacement value (plant age unknown) and the
            remaining life ~ Tri(10, 20, 30) years

Operator response (finance/abatement.py): where the net carbon price reaches the levelised cost
of a sector lever, the plant abates (after a lead time, irreversibly). Loss = carbon paid on the
residual + levelised abatement cost. The carbon cost of doing nothing is kept as `transition_npv`
and `total_npv_unabated`, so both views are always available.
Closure option: transition loss (carbon cost + abatement + stranded capital) is capped at the asset's
value in each draw — an owner would shut the plant rather than pay more than it is worth;
`closure_share` reports how often that happens.
Plant life (optional): with a commissioning year the remaining life (design life ~ Tri by sector, less age) sets the
stranding horizon and undepreciated share and ends the plant's emissions; a retirement year (legal phase-out or
planned closure) ends operation on that date in every draw.
CBAM (optional): for products in EU CBAM scope exported from outside the EU, `export_share_eu` of the emissions pay
max(0, EU price − carbon price paid at home) × CBAM factor (1 − free allocation, 2026→2034) × (1 − pass-through),
using the NGFS price of the EU region or today's EU ETS price if higher; it adds to the price that triggers abatement.
Earnings layer (finance/earnings.py): with financials in the request, EBITDA, leverage, coverage,
rating/PD, covenants and impairment per scenario.

Present values use a real discount rate (default 7 %). Seeded → reproducible.
Not included: business interruption, supply-chain and market-wide effects,
climate change in cyclone and river-flood frequency, chronic heat and water stress.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import abatement, earnings
from .policy import EU_CBAM_FREE_SHARE
from .transition import FUTURE_CS, SCENARIOS, YEARS, label
from .vulnerability import V_HALF, tc_damage

PASS_THROUGH = {  # (low, mode, high) share of carbon cost passed on to customers
    "el": (0.5, 0.8, 1.0), "refining": (0.5, 0.8, 1.0), "cement": (0.2, 0.5, 0.8), "steel": (0.2, 0.4, 0.8),
    "chemicals": (0.2, 0.4, 0.7), "min": (0.0, 0.1, 0.4), "ind": (0.2, 0.4, 0.7), "was": (0.3, 0.6, 0.9),
}
# Design life of the asset class, years (low, mode, high) — analyst judgement from typical commercial lives
# (coal ~40, gas ~30, cement and steel ~40, refineries and mines longer-lived with rebuilds). Used only when the
# commissioning year is given; the remaining life then replaces the sampled 10–30 years.
DESIGN_LIFE = {"power:coal": (30, 40, 50), "power:gas": (25, 30, 40), "power:oil": (30, 40, 50), "cement": (30, 40, 50),
               "steel": (30, 40, 50), "refining": (30, 40, 55), "oil-and-gas": (20, 25, 35), "coal-mining": (20, 30, 40)}
DEFAULT_DESIGN_LIFE = (25, 35, 50)
STRANDING_KEYS = {"power:coal", "power:gas", "power:oil", "coal-mining", "oil-and-gas", "refining"}


def _tri(rng, lo, mo, hi, n):
    lo, mo, hi = float(lo), float(mo), float(hi)
    if hi <= lo:
        return np.full(n, mo)
    mo = min(max(mo, lo), hi)
    return rng.triangular(lo, mo, hi, n)


def _interp_rp(T: np.ndarray, rps, dmg) -> np.ndarray:
    """Damage for event return periods T by log-linear interpolation; 0 below 2 yr."""
    x = np.log(np.concatenate([[2.0], np.asarray(rps, float)]))
    y = np.concatenate([[0.0], np.asarray(dmg, float)])
    if rps[0] <= 2:
        x, y = np.log(np.asarray(rps, float)), np.asarray(dmg, float)
    return np.interp(np.log(np.clip(T, 1.0001, None)), x, y, left=0.0, right=float(y[-1]))


def _stats(x: np.ndarray, v_ref) -> dict:
    """Loss statistics. v_ref is either the value draws (same length as x) or a single value.
    With draws, % of value is taken per draw (loss / that draw's value) so it is bounded by the cap:
    an asset whose value is uncertain cannot show a loss above 100% of what it is worth."""
    q = np.percentile(x, [5, 50, 95, 99])
    tail = x[x >= q[2]]
    d = {"mean": float(x.mean()), "p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2]), "p99": float(q[3]),
         "es95": float(tail.mean()) if len(tail) else float(q[2])}
    if isinstance(v_ref, np.ndarray):
        r = 100 * x / np.where(v_ref > 0, v_ref, np.nan)
        r = r[np.isfinite(r)]
        if len(r):
            rq = np.percentile(r, [5, 50, 95, 99])
            rt = r[r >= rq[2]]
            d["pct_of_value"] = {"mean": round(float(r.mean()), 2), "p5": round(float(rq[0]), 2), "p50": round(float(rq[1]), 2),
                                 "p95": round(float(rq[2]), 2), "p99": round(float(rq[3]), 2),
                                 "es95": round(float(rt.mean()) if len(rt) else float(rq[2]), 2)}
    elif v_ref:
        d["pct_of_value"] = {k: round(100 * v / v_ref, 2) for k, v in d.items()}
    return d


def simulate(case: dict, n: int = 4000, discount: float = 0.07, seed: int = 7,
             scenarios: Optional[list] = None, common: Optional[dict] = None) -> dict:
    """`common` (portfolio runs): {"u_model": U(0,1) array of length n shared by all assets so the
    NGFS model drawn in each draw is the same everywhere, "seed": per-asset seed}."""
    rng = np.random.default_rng((common or {}).get("seed", seed))
    Y = len(YEARS)
    disc = 1.0 / (1.0 + discount) ** (YEARS - 2025)

    # ── asset value ──
    val = case.get("value") or {}
    has_value = val.get("mode") is not None
    V = _tri(rng, val["low"], val["mode"], val["high"], n) if has_value else np.zeros(n)

    # ── physical ──
    dmg = np.zeros((n, Y))
    parts = {}
    tc = case.get("tc") or {}
    if tc.get("lam", 0) > 0 and tc.get("sigma"):
        vh_c, vh_h = V_HALF[tc.get("region", "ROW")]
        vh = rng.uniform(min(vh_c, vh_h), max(vh_c, vh_h), n)[:, None]
        N = rng.poisson(tc["lam"], (n, Y))
        d_tc = np.zeros((n, Y))
        for k in range(int(N.max()) if N.size else 0):
            v = 17.5 + rng.exponential(tc["sigma"], (n, Y))
            v = np.minimum(v, 95.0)
            d_k = tc_damage(v, vh)                          # vh broadcasts per draw
            d_tc = 1 - (1 - d_tc) * (1 - np.where(N > k, d_k, 0.0))
        dmg = 1 - (1 - dmg) * (1 - d_tc)
        parts["cyclone"] = d_tc
    for key in ("river", "coast"):
        h = case.get(key) or {}
        any_dmg = np.any(np.asarray(h.get("damage") or [0]) > 0) or np.any(np.asarray(h.get("damage_2050") or [0]) > 0)
        if not h.get("rps") or not any_dmg:
            continue
        mult = np.clip(_tri(rng, 0.6, 1.0, 1.4, n), 0, None)[:, None]
        T = 1.0 / rng.uniform(1e-6, 1.0, (n, Y))
        if key == "coast" and h.get("damage_2050") is not None:
            w = np.clip((YEARS - 2018) / (2050 - 2018), 0, 1)[None, :]
            d0 = _interp_rp(T, h["rps"], h["damage"])
            d1 = _interp_rp(T, h["rps"], h["damage_2050"])
            d = d0 * (1 - w) + d1 * w
        else:
            d = _interp_rp(T, h["rps"], h["damage"])
        d = np.where(T > float(h.get("protection") or 0), np.minimum(d * mult, 1.0), 0.0)
        dmg = 1 - (1 - dmg) * (1 - d)
        parts[key] = d
    loss_phys = dmg * V[:, None]
    npv_phys = (loss_phys * disc).sum(axis=1)

    # ── transition ──
    tr = case.get("transition") or {}
    E = float(tr.get("emissions_t") or 0.0)
    drv = tr.get("drivers") or {"price": {}, "activity": {}}
    p0 = float(tr.get("explicit_price") or 0.0)
    cs_lo, cs_mo, cs_hi = [np.asarray(a, float) for a in tr.get("charged_share_path", (np.zeros(Y),) * 3)]
    u_cs = rng.uniform(0, 1, n)[:, None]
    cs0 = np.where(u_cs < 0.5, cs_lo + (cs_mo - cs_lo) * (u_cs / 0.5), cs_mo + (cs_hi - cs_mo) * ((u_cs - 0.5) / 0.5))
    csf = _tri(rng, *FUTURE_CS, n)[:, None]
    pt = _tri(rng, *PASS_THROUGH.get(tr.get("pass_through_key", "ind"), PASS_THROUGH["ind"]), n)[:, None]
    strand_on = tr.get("activity_key") in STRANDING_KEYS and has_value
    f_econ = _tri(rng, 0.15, 0.35, 0.6, n)          # undepreciated share of replacement value (age unknown)
    life = np.round(_tri(rng, 10, 20, 30, n)).astype(int)
    # Plant age and legal retirement: remaining life, undepreciated share, and the years the plant still runs.
    alive = np.ones((n, Y))
    al = case.get("asset_life") or {}
    if al.get("commissioned") or al.get("retire"):
        end = np.full(n, 1e9)
        if al.get("commissioned"):
            dl = _tri(np.random.default_rng((common or {}).get("seed", seed) + 104729), *al["design"], n)
            end = float(al["commissioned"]) + dl
            rem = np.clip(end - YEARS[0], 0, None)
            life = np.maximum(np.round(rem), 1).astype(int)
            f_econ = np.clip(rem / dl, 0.02, 1.0)
        if al.get("retire"):
            end = np.minimum(end, float(al["retire"]))
            life = np.minimum(life, max(int(al["retire"]) - int(YEARS[0]), 1)).astype(int)
        alive = (YEARS[None, :] < end[:, None]).astype(float)
    # Abatement levers use their own random stream so adding them never changes the physical or carbon draws.
    ab = tr.get("abatement") or {}
    levers = ab.get("levers") or []
    S_ab = None
    if ab.get("enabled") and levers and E > 0:
        S_ab = abatement.sample(levers, np.random.default_rng((common or {}).get("seed", seed) + 7919), n)
    fin = case.get("financials")
    cb = tr.get("cbam")                                      # {"share", "price": {scenario: {model: array}}, "p_now"}
    cb_factor = np.array([1.0 - EU_CBAM_FREE_SHARE.get(int(t), 0.0 if t > 2034 else 1.0) for t in YEARS])[None, :]

    per_scen = {}
    for s in (scenarios or SCENARIOS):
        prices = drv["price"].get(s)
        if not prices:
            continue
        models = list(prices.keys())
        u_m = (common or {}).get("u_model")
        mi = (np.minimum((u_m * len(models)).astype(int), len(models) - 1) if u_m is not None
              else rng.integers(0, len(models), n))
        P = np.stack([prices[m] for m in models])[mi]                         # n × Y
        acts = drv["activity"].get(s) or {}
        A = np.stack([acts[m] if m in acts else np.ones(Y) for m in models])[mi] if acts else np.ones((n, Y))
        Araw = np.minimum(A, 1.0)
        Amin = Araw * alive
        Ae = E * Amin                                                          # tonnes emitted today's way, per year
        qg = p0 * cs0 + np.maximum(P - p0, 0.0) * csf                          # carbon price paid at home, per tonne
        q = qg * (1 - pt)                                                      # net of cost pass-through
        cbam_unit = np.zeros((n, Y))
        if cb:
            eu = np.stack([np.maximum(cb["price"].get(s, {}).get(m, np.zeros(Y)), cb["p_now"]) for m in models])[mi]
            cbam_unit = cb["share"] * np.maximum(eu - qg, 0.0) * cb_factor * (1 - pt)      # EU border charge on exports
            q = q + cbam_unit
        cost = Ae * q                                                          # unabated: the cost of doing nothing
        npv_cost = (cost * disc).sum(axis=1)
        if strand_on and acts:
            idx = np.arange(Y)[None, :] < life[:, None]
            meanA = (Araw * idx).sum(axis=1) / np.maximum(idx.sum(axis=1), 1)
            strand = V * f_econ * np.clip(1 - meanA, 0, 1)
        else:
            strand = np.zeros(n)
        # Operator response: abate where the net carbon price reaches the levelised cost of a lever.
        if S_ab is not None:
            R = abatement.respond(q, Ae, S_ab, discount)
            pay = Ae * (1 - R["frac"]) * q
            resp = pay + R["cost"]
            npv_resp = (resp * disc).sum(axis=1)
        else:
            R = {"frac": np.zeros((n, Y)), "cost": np.zeros((n, Y)), "opex": np.zeros((n, Y)),
                 "capex": np.zeros((n, Y)), "starts": None}
            pay, resp, npv_resp = cost, cost, npv_cost
        # Closure option: an owner facing carbon costs and lost output worth more than the
        # asset would shut it — the transition loss to the asset is capped at its value.
        trans = np.minimum(npv_resp + strand, V) if has_value else npv_resp + strand
        trans_un = np.minimum(npv_cost + strand, V) if has_value else npv_cost + strand
        capped = float(np.mean(npv_resp + strand > V)) if has_value else 0.0
        # An asset cannot lose more than it is worth: once physical damage plus transition cost exceed
        # its value the rational owner closes it, so the total loss is capped at the asset value too.
        total = np.minimum(npv_phys + trans, V) if has_value else npv_phys + trans
        total_un = np.minimum(npv_phys + trans_un, V) if has_value else npv_phys + trans_un
        earn = None
        if fin is not None:
            earn = earnings.compute(fin, {"Amin": Araw, "alive": alive, "pay": pay, "ab_opex": R["opex"], "capex": R["capex"], "pt": pt,
                                          "disc": disc, "years": YEARS, "discount": discount, "drv": drv, "scenario": s,
                                          "models": models, "mi": mi})
        per_scen[s] = {"label": label(s), "total": total, "total_un": total_un, "cost": npv_cost, "strand": strand,
                       "trans": trans, "trans_un": trans_un, "npv_resp": npv_resp, "npv_pay": (pay * disc).sum(axis=1),
                       "npv_abate": (R["cost"] * disc).sum(axis=1), "R": R, "capped_share": capped, "earnings": earn,
                       "npv_cbam": (Ae * cbam_unit * disc).sum(axis=1) if cb else None,
                       "annual_cost": resp, "annual_cost_un": cost, "price": P, "activity": A}

    v_ref = V if has_value else None                       # per-draw value: % of value stays within the cap
    out = {"n_draws": n, "discount_rate": discount, "years": YEARS.tolist(), "asset_value": val,
           "physical": {
               "npv": _stats(npv_phys, v_ref),
               "aal_usd": float(loss_phys.mean()) if has_value else None,
               "aal_pct": float(dmg.mean() * 100),
               "annual_loss_rp": {str(rp): float(np.percentile(loss_phys, 100 * (1 - 1 / rp))) for rp in (10, 50, 100, 250)}
               if has_value else None,
               "annual_damage_ratio_rp": {str(rp): float(np.percentile(dmg, 100 * (1 - 1 / rp))) for rp in (10, 50, 100, 250)},
               "by_hazard_aal_pct": {k: float(v.mean() * 100) for k, v in parts.items()},
           },
           "scenarios": {}}
    for s, r in per_scen.items():
        ac = r["annual_cost"]
        hist_hi = float(np.percentile(r["total"], 99.5)) or 1.0
        counts, edges = np.histogram(np.clip(r["total"], 0, hist_hi), bins=40, range=(0, hist_hi))
        R = r["R"]
        out["scenarios"][s] = {
            "label": r["label"],
            "total_npv": _stats(r["total"], v_ref),
            "total_npv_unabated": _stats(r["total_un"], v_ref),
            "transition_npv": _stats(r["cost"], v_ref),                 # carbon cost if nothing is done (unabated)
            "transition_loss": _stats(r["trans"], v_ref),               # after the operator's response, capped at value
            "transition_loss_unabated": _stats(r["trans_un"], v_ref),
            "annual_cost_unabated_median": np.round(np.median(r["annual_cost_un"], axis=0), 0).tolist(),
            "abatement": ({
                "enabled": True,
                "carbon_paid_npv": _stats(r["npv_pay"], v_ref),
                "abatement_cost_npv": _stats(r["npv_abate"], v_ref),
                "response_cost_npv": _stats(r["npv_resp"], v_ref),
                "avoided_npv": _stats(r["cost"] - r["npv_resp"], v_ref),
                "abated_share_median": np.round(np.median(R["frac"], axis=0), 3).tolist(),
                "abated_share_by_year": {str(y): round(float(np.median(R["frac"][:, y - YEARS[0]])), 3) for y in (2030, 2035, 2040, 2050)},
                "capex_pv_mean": float(((R["capex"] * disc).sum(axis=1)).mean()),
                "levers": abatement.summarise(levers, S_ab, R, YEARS)} if S_ab is not None else {"enabled": False}),
            "earnings": r["earnings"],
            "cbam_npv": _stats(r["npv_cbam"], v_ref) if r["npv_cbam"] is not None else None,
            "retired_share_by_2050": round(float(1 - alive[:, -1].mean()), 3) if (alive < 1).any() else 0.0,
            "closure_share": round(r["capped_share"], 3),
            "stranded": _stats(r["strand"], v_ref) if strand_on else None,
            "annual_cost_median": np.round(np.median(ac, axis=0), 0).tolist(),
            "annual_cost_p5": np.round(np.percentile(ac, 5, axis=0), 0).tolist(),
            "annual_cost_p95": np.round(np.percentile(ac, 95, axis=0), 0).tolist(),
            "price_median": np.round(np.median(np.maximum(r["price"], p0), axis=0), 1).tolist(),   # effective (≥ today's price)
            "activity_median": np.round(np.median(r["activity"], axis=0), 3).tolist(),
            "histogram": {"edges": np.round(edges, 0).tolist(), "counts": counts.tolist()},
            "share_physical": float(npv_phys.mean() / max(r["total"].mean(), 1e-9)),
        }
        if common is not None:
            out["scenarios"][s]["_total_draws"] = r["total"]
    if common is not None:
        out["_value_draws"] = V if has_value else None
    return out
