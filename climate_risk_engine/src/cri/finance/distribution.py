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

Closure option: transition loss (carbon cost + stranded capital) is capped at the asset's
value in each draw — an owner would shut the plant rather than pay more than it is worth;
`closure_share` reports how often that happens. The uncapped carbon cost is reported as
`transition_npv` (the liability if the plant ran unabated).

Present values use a real discount rate (default 7 %). Seeded → reproducible.
Not included: business interruption, supply-chain and market-wide effects,
climate change in cyclone and river-flood frequency, chronic heat and water stress.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .transition import FUTURE_CS, SCENARIOS, YEARS, label
from .vulnerability import V_HALF, tc_damage

PASS_THROUGH = {  # (low, mode, high) share of carbon cost passed on to customers
    "el": (0.5, 0.8, 1.0), "refining": (0.5, 0.8, 1.0), "cement": (0.2, 0.5, 0.8), "steel": (0.2, 0.4, 0.8),
    "chemicals": (0.2, 0.4, 0.7), "min": (0.0, 0.1, 0.4), "ind": (0.2, 0.4, 0.7), "was": (0.3, 0.6, 0.9),
}
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


def _stats(x: np.ndarray, v_ref: Optional[float]) -> dict:
    q = np.percentile(x, [5, 50, 95, 99])
    tail = x[x >= q[2]]
    d = {"mean": float(x.mean()), "p5": float(q[0]), "p50": float(q[1]), "p95": float(q[2]), "p99": float(q[3]),
         "es95": float(tail.mean()) if len(tail) else float(q[2])}
    if v_ref:
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
        cost = E * np.minimum(A, 1.0) * (p0 * cs0 + np.maximum(P - p0, 0.0) * csf) * (1 - pt)
        npv_cost = (cost * disc).sum(axis=1)
        if strand_on and acts:
            idx = np.arange(Y)[None, :] < life[:, None]
            meanA = (np.minimum(A, 1.0) * idx).sum(axis=1) / np.maximum(idx.sum(axis=1), 1)
            strand = V * f_econ * np.clip(1 - meanA, 0, 1)
        else:
            strand = np.zeros(n)
        # Closure option: an owner facing carbon costs and lost output worth more than the
        # asset would shut it — the transition loss to the asset is capped at its value.
        trans = np.minimum(npv_cost + strand, V) if has_value else npv_cost + strand
        capped = float(np.mean(npv_cost + strand > V)) if has_value else 0.0
        # An asset cannot lose more than it is worth: once physical damage plus transition cost exceed
        # its value the rational owner closes it, so the total loss is capped at the asset value too.
        total = np.minimum(npv_phys + trans, V) if has_value else npv_phys + trans
        per_scen[s] = {"label": label(s), "total": total, "cost": npv_cost, "strand": strand, "trans": trans,
                       "capped_share": capped,
                       "annual_cost": cost, "price": P, "activity": A}

    v_ref = float(val["mode"]) if has_value else None
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
        out["scenarios"][s] = {
            "label": r["label"],
            "total_npv": _stats(r["total"], v_ref),
            "transition_npv": _stats(r["cost"], v_ref),
            "transition_loss": _stats(r["trans"], v_ref),
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
    return out
