"""Insurer view: a site list with policy terms -> average annual loss, exceedance-probability curves, probable maximum loss, accumulation, deductible options and an indicative technical premium, today and under a climate pathway.

It reuses the engine's own hazard inputs and damage functions (IBTrACS-based cyclone rate and wind tail, JRC river and Deltares coastal flood depths, FLOPROS defences, Emanuel/Eberenz wind and JRC depth-damage curves), then does what an insurer needs on top:

  1. simulates years of events at every site, with spatially correlated events (Gaussian copula, distance-decay correlation) so the portfolio tail is not the sum of independent sites;
  2. applies each policy's deductible and limit per peril (wind, flood), then optional quota-share and excess-of-loss reinsurance;
  3. reports AAL, occurrence (OEP) and aggregate (AEP) exceedance, TVaR, each site's contribution to the tail, accumulation by zone / country / sector;
  4. reruns the same random years under a climate pathway (the engine's own climate-scenario composer) so the change in AAL and PML is a like-for-like difference.

What it is not: a vendor catastrophe model. Damage functions are published, uncalibrated to claims; annual-maximum flood per site; terms are applied per site per peril per year (an annual approximation of per-event deductibles); business interruption is not separated; correlation scales are assumed. The result says all of this.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

RPS_OUT = (2, 5, 10, 20, 50, 100, 250, 500)
PERILS = ("wind", "river", "coast")
FLOOD = ("river", "coast")
CORR_KM = {"wind": 300.0, "river": 100.0, "coast": 60.0}      # assumed decay lengths of the event-severity correlation
MAX_CELLS = 6_000_000                                             # sites x years x perils stored
V_THRESH_TC = 17.5                                                # m/s: the rate lambda counts storms at or above tropical-storm force
LADDER_PCT = (0.0, 1.0, 2.0, 5.0, 10.0)
YEAR_NOW = 2026

# ── small numerics ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def _erf(x):
    # Abramowitz & Stegun 7.1.26, |error| < 1.5e-7: enough for turning copula normals into uniforms
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _ndtr(z):
    try:
        from scipy.special import ndtr
        return ndtr(z)
    except Exception:                                              # noqa: BLE001
        return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


def _km(lat1, lon1, lat2, lon2):
    p = math.pi / 180.0
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _tri(rng, lo, mo, hi, size):
    return rng.triangular(lo, mo, hi, size)


# ── input handling ───────────────────────────────────────────────────────────────────────────────────────────────────────────


class InsuranceError(ValueError):
    pass


def _num(v, what, k, lo=None, hi=None, allow_none=True):
    if v in (None, "") or (isinstance(v, float) and v != v):
        if allow_none:
            return None
        raise InsuranceError(f"site {k + 1}: {what} is required")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise InsuranceError(f"site {k + 1}: {what} is not a number: {v!r}")
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise InsuranceError(f"site {k + 1}: {what} must be between {lo} and {hi}")
    return x


def normalise_site(s: dict, k: int) -> dict:
    """One policy line. TIV is required; deductible and limit can be given in money or as % of TIV, and per peril group."""
    lat, lon = _num(s.get("lat"), "latitude", k, -90, 90, False), _num(s.get("lon"), "longitude", k, -180, 180, False)
    tiv = _num(s.get("tiv_usd", s.get("value_usd")), "tiv_usd (total insured value)", k, 1.0, None, False)

    def money(abs_key, pct_key, default_pct):
        a, p = _num(s.get(abs_key), abs_key, k, 0), _num(s.get(pct_key), pct_key, k, 0, 100)
        if a is not None:
            return a
        return (p if p is not None else default_pct) / 100.0 * tiv

    ded = money("deductible_usd", "deductible_pct", 0.0)
    lim = money("limit_usd", "limit_pct", 100.0)
    w_ded = _num(s.get("wind_deductible_usd"), "wind_deductible_usd", k, 0)
    f_ded = _num(s.get("flood_deductible_usd"), "flood_deductible_usd", k, 0)
    w_lim = _num(s.get("wind_limit_usd"), "wind_limit_usd", k, 0)
    f_lim = _num(s.get("flood_limit_usd"), "flood_limit_usd", k, 0)
    if lim <= ded:
        raise InsuranceError(f"site {k + 1}: limit must exceed the deductible")
    covers = s.get("covers") or ["wind", "flood"]
    covers = [str(c).lower() for c in (covers.split(",") if isinstance(covers, str) else covers)]
    bad = [c for c in covers if c not in ("wind", "flood")]
    if bad:
        raise InsuranceError(f"site {k + 1}: unknown peril {bad}; use wind and/or flood")
    return {"name": str(s.get("name") or f"Site {k + 1}"), "lat": lat, "lon": lon, "tiv": tiv, "sector": s.get("sector") or "other",
            "country": s.get("country") or None, "covers": covers,
            "terms": {"wind": {"ded": w_ded if w_ded is not None else ded, "lim": w_lim if w_lim is not None else lim},
                      "flood": {"ded": f_ded if f_ded is not None else ded, "lim": f_lim if f_lim is not None else lim},
                      "total_limit": lim},
            "hazard": s.get("hazard")}


# ── hazards ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def fetch_hazards(sites: list, workers: int = 8, flood_fn: Optional[Callable] = None, tc_fn: Optional[Callable] = None) -> list:
    """Per site: cyclone rate and wind tail, river and coastal depth lists, defences, damage-curve region. Sites that carry a
    'hazard' dict are used as given (offline runs, tests, a client's own hazard layer)."""
    from .finance import geo
    from .finance.analyst import _tc_params
    from .finance.flood_hazard import site_flood
    flood_fn = flood_fn or site_flood
    tc_fn = tc_fn or _tc_params

    def one(site):
        if site.get("hazard"):
            h = dict(site["hazard"])
            h.setdefault("notes", [])
            return h
        notes = []
        try:
            adm = geo.admin_at(site["lat"], site["lon"])
        except Exception:                                          # noqa: BLE001
            adm = {}
        iso3 = site.get("country") or adm.get("iso3")
        cont, tcr = geo.continent(iso3), geo.tc_region(iso3)
        try:
            tc = tc_fn(site["lat"], site["lon"])
        except Exception as e:                                     # noqa: BLE001
            tc, _ = {"lam": 0.0, "sigma": None}, notes.append(f"cyclone hazard unavailable: {str(e)[:100]}")
        try:
            fl = flood_fn(site["lat"], site["lon"], 250.0, None, None)
        except Exception as e:                                     # noqa: BLE001
            fl, _ = {"river": {"available": False}, "coastal": {"available": False}}, notes.append(f"flood maps unavailable: {str(e)[:100]}")
        riv, cst = fl.get("river") or {}, fl.get("coastal") or {}
        h = {"iso3": iso3, "admin1": adm.get("name"), "cont": cont, "tc_region": tcr,
             "wind": {"lam": float(tc.get("lam") or 0.0), "sigma": tc.get("sigma"), "rp100_ms": tc.get("rp100_ms")},
             "river": None, "coast": None, "notes": notes}
        if riv.get("available") and not riv.get("footprint_on_water") and riv.get("depths"):
            h["river"] = {"rps": [int(r) for r in riv["rps"]], "depths": {str(r): list(riv["depths"][str(r)]) for r in riv["rps"]},
                          "protection": float(adm.get("prot_river_rp") or 0.0), "source": riv.get("source")}
        elif riv.get("footprint_on_water"):
            notes.append("river flood: the site footprint falls on the mapped water body, so no land depth is available")
        elif not riv.get("available"):
            notes.append("river flood map not available here")
        if cst.get("available") and cst.get("depths"):
            h["coast"] = {"rps": [int(r) for r in cst["rps"]], "depths": {k: list(v) for k, v in cst["depths"].items()},
                          "protection": float(adm.get("prot_coast_rp") or 0.0), "source": cst.get("source")}
        return h

    with ThreadPoolExecutor(workers) as pool:
        futs = [pool.submit(one, s) for s in sites]
        out = []
        for f, s in zip(futs, sites):
            try:
                out.append(f.result(timeout=150))
            except Exception as e:                                 # noqa: BLE001
                out.append({"wind": {"lam": 0.0, "sigma": None}, "river": None, "coast": None, "cont": "oceania", "tc_region": "ROW",
                            "notes": [f"hazard fetch failed: {str(e)[:100]}"]})
    for h in out:
        h.setdefault("cont", "oceania")
        h.setdefault("tc_region", "ROW")
        h.setdefault("notes", [])
    return out


def _flood_damage_curve(h: dict, key: str, cont: str, depth_factor: float = 1.0, add_m: float = 0.0, now_year: int = YEAR_NOW) -> Optional[dict]:
    """Damage ratio at each mapped return period (footprint mean), with optional climate change of depth: river depth x factor,
    coastal depth + sea-level rise (only where the mapped footprint is already wet)."""
    from .finance import vulnerability
    spec = h.get(key)
    if not spec:
        return None
    rps = spec["rps"]
    if "damage" in spec and "depths" not in spec:                  # supplied as damage ratios (tests, own layer)
        return {"rps": rps, "damage": [float(x) for x in spec["damage"]], "protection": float(spec.get("protection") or 0.0)}
    dmg = []
    for rp in rps:
        if key == "river":
            d = np.asarray(spec["depths"].get(str(rp)) or [], float)
        else:
            w = float(np.clip((now_year - 2018) / (2050 - 2018), 0, 1))
            d0 = np.asarray(spec["depths"].get(f"2018_{rp}") or [], float)
            d1 = np.asarray(spec["depths"].get(f"2050_{rp}") or [], float)
            d = d0 * (1 - w) + d1 * w if len(d0) == len(d1) and len(d0) else (d0 if len(d0) else d1)
        if d.size == 0:
            dmg.append(0.0)
            continue
        wet = d > 0.05
        d = np.where(wet, d * depth_factor + add_m, d)
        dmg.append(float(np.mean(vulnerability.flood_damage(np.clip(d, 0, None), cont))))
    dmg = np.maximum.accumulate(np.asarray(dmg))                  # damage cannot fall with return period
    return {"rps": rps, "damage": dmg.tolist(), "protection": float(spec.get("protection") or 0.0)}


def climate_hazards(sites: list, hazards: list, ssp: str, year: int, band: str = "median", fetch=None, tile_fetch=None,
                    compose_fn: Optional[Callable] = None) -> tuple:
    """Hazards under a pathway: river depth x the composer's factor, coastal depth + AR6 uplift, wind x Knutson scale.
    Returns (hazards, per-site composer summaries). A site whose projection fails keeps today's hazard and is flagged."""
    from . import climate_scenario as CS
    compose_fn = compose_fn or CS.compose
    out, notes = [], []
    pb = {"low": "p10", "median": "median", "high": "p90"}[band]
    sb = {"low": "p17", "median": "median", "high": "p83"}[band]
    wb = band
    cache: dict = {}
    for s, h in zip(sites, hazards):
        key = (round(s["lat"], 1), round(s["lon"], 1))
        try:
            comp = cache.get(key) or compose_fn(s["lat"], s["lon"], ssp=ssp, year=year, fetch=fetch, tile_fetch=tile_fetch)
            cache[key] = comp
        except Exception as e:                                     # noqa: BLE001
            out.append(dict(h, _climate={"ok": False, "why": str(e)[:140]}))
            notes.append({"site": s["name"], "ok": False, "why": str(e)[:140]})
            continue
        f = comp["river_depth_factor"][pb]
        sea = comp["sea"]
        add = sea["add_m"][sb] if sea.get("available") else 0.0
        ws = comp["wind"]["scale"][wb] if comp["wind"]["tc_exposed"] else 1.0
        h2 = dict(h)
        h2["_climate"] = {"ok": True, "river_depth_factor": f, "sea_add_m": add, "wind_scale": ws, "rain_ratio": comp["rain"][pb],
                          "sea_available": bool(sea.get("available")), "warming_k": comp["warming_k"]}
        h2["_wind_scale"] = ws
        h2["river_cc"] = _flood_damage_curve(h, "river", h.get("cont", "oceania"), depth_factor=f)
        h2["coast_cc"] = _flood_damage_curve(h, "coast", h.get("cont", "oceania"), add_m=add)
        out.append(h2)
        notes.append({"site": s["name"], "ok": True, **h2["_climate"]})
    return out, notes, (next(iter(cache.values())) if cache else None)


# ── simulation ───────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _corr_chol(lat, lon, L_km: float, scale: float):
    n = len(lat)
    D = _km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    C = np.exp(-D / max(L_km * scale, 1e-6))
    C[np.diag_indices(n)] += 1e-6
    return np.linalg.cholesky(C)


def _uniforms(rng, n_years, lat, lon, L_km, scale, groups=None):
    """Spatially correlated uniforms (n_years x n_sites). Sites in the same group (large portfolios) share one draw."""
    if groups is None:
        z = rng.standard_normal((n_years, len(lat))) @ _corr_chol(lat, lon, L_km, scale).T
    else:
        ug, inv = np.unique(groups, return_inverse=True)
        glat = np.array([lat[inv == i].mean() for i in range(len(ug))])
        glon = np.array([lon[inv == i].mean() for i in range(len(ug))])
        zg = rng.standard_normal((n_years, len(ug))) @ _corr_chol(glat, glon, L_km, scale).T
        z = zg[:, inv]
    return np.clip(_ndtr(z), 1e-12, 1 - 1e-12)


def _wind_damage(seed, u, lam, sigma, vh_c, vh_h, uv, wind_scale):
    """Annual wind damage ratio, per site: the year's strongest storm from the correlated uniform u (Gumbel tail of a Poisson process of
    storms with exponential wind excess), plus the other storms of the year drawn independently (Poisson, truncated tail), combined
    as 1 - prod(1 - d). v_half is drawn once per year (same for all sites: vulnerability uncertainty is systemic)."""
    from .finance.vulnerability import tc_damage
    ny, ns = u.shape
    out = np.zeros((ny, ns))
    for j in range(ns):
        if not (lam[j] > 0 and sigma[j]):
            continue
        rng = np.random.default_rng([seed, 1000 + j])                 # per-site stream: a climate rerun keeps the other sites' extra storms unchanged
        vh = vh_h[j] + uv * (vh_c[j] - vh_h[j])                       # per year
        x1 = np.log(lam[j] / np.maximum(-np.log(u[:, j]), 1e-300))    # excess of the strongest storm in sigma units; <0: no storm
        has = x1 > 0
        v1 = np.minimum((V_THRESH_TC + sigma[j] * np.maximum(x1, 0)) * wind_scale[j], 95.0)
        d = np.where(has, tc_damage(v1, vh), 0.0)
        nx = rng.poisson(np.where(has, lam[j] * (1.0 - np.exp(-np.maximum(x1, 0))), 0.0))
        for k in range(int(min(nx.max(), 12)) if nx.size else 0):
            ue = rng.uniform(0, 1, ny)
            xe = -np.log(1.0 - ue * (1.0 - np.exp(-np.maximum(x1, 1e-9))))
            ve = np.minimum((V_THRESH_TC + sigma[j] * xe) * wind_scale[j], 95.0)
            d = 1.0 - (1.0 - d) * (1.0 - np.where(nx > k, tc_damage(ve, vh), 0.0))
        out[:, j] = d
    return out


def _flood_damage(u, curves, mult):
    """Annual flood damage ratio per site from correlated uniform u: return period T = 1/(1-u), curve interpolated in log T, zero inside the defence standard."""
    from .finance.distribution import _interp_rp
    ny, ns = u.shape
    out = np.zeros((ny, ns))
    T = 1.0 / (1.0 - u)
    for j in range(ns):
        c = curves[j]
        if not c or not np.any(np.asarray(c["damage"]) > 0):
            continue
        d = _interp_rp(T[:, j], c["rps"], c["damage"])
        out[:, j] = np.where(T[:, j] > c["protection"], np.minimum(d * mult[:, 0], 1.0), 0.0)
    return out


def ground_up(sites: list, hazards: list, n_years: int, seed: int, corr_scale: float = 1.0, climate: bool = False) -> dict:
    """Ground-up damage ratios (years x sites) by peril, for the same random years whatever the hazards (so a climate rerun is a paired difference)."""
    ns = len(sites)
    ny = int(max(200, min(n_years, MAX_CELLS // max(ns * 3, 1))))
    lat = np.array([s["lat"] for s in sites])
    lon = np.array([s["lon"] for s in sites])
    rng_w, rng_r, rng_c, rng_m = (np.random.default_rng([seed, i]) for i in range(4))
    groups = None
    if ns > 400:
        groups = np.floor(lat / 0.5).astype(int) * 100000 + np.floor(lon / 0.5).astype(int)
    lam = np.array([(h.get("wind") or {}).get("lam") or 0.0 for h in hazards])
    sig = np.array([(h.get("wind") or {}).get("sigma") or 0.0 for h in hazards])
    from .finance.vulnerability import V_HALF
    vhc = np.array([V_HALF.get(h.get("tc_region", "ROW"), V_HALF["ROW"])[0] for h in hazards])
    vhh = np.array([V_HALF.get(h.get("tc_region", "ROW"), V_HALF["ROW"])[1] for h in hazards])
    wscale = np.array([h.get("_wind_scale", 1.0) if climate else 1.0 for h in hazards])
    uv = rng_m.uniform(0, 1, ny)
    mult = np.clip(_tri(rng_m, 0.6, 1.0, 1.4, ny), 0, None)[:, None]
    u_w = _uniforms(rng_w, ny, lat, lon, CORR_KM["wind"], corr_scale, groups)
    u_r = _uniforms(rng_r, ny, lat, lon, CORR_KM["river"], corr_scale, groups)
    u_c = _uniforms(rng_c, ny, lat, lon, CORR_KM["coast"], corr_scale, groups)
    wind = _wind_damage(seed, u_w, lam, sig, vhc, vhh, uv, wscale)
    rcur = [(h.get("river_cc") if climate and h.get("river_cc") is not None else _flood_damage_curve(h, "river", h.get("cont", "oceania"))) for h in hazards]
    ccur = [(h.get("coast_cc") if climate and h.get("coast_cc") is not None else _flood_damage_curve(h, "coast", h.get("cont", "oceania"))) for h in hazards]
    river = _flood_damage(u_r, rcur, mult)
    coast = _flood_damage(u_c, ccur, mult)
    return {"wind": wind.astype(np.float32), "river": river.astype(np.float32), "coast": coast.astype(np.float32), "n_years": ny}


# ── policy terms, reinsurance, statistics ────────────────────────────────────────────────────────────────────────────────────


def apply_terms(gu: dict, sites: list, ded_override_pct: Optional[float] = None) -> dict:
    """Insured loss (years x sites), by wind and flood, from ground-up damage ratios and each policy's deductible and limit.
    Flood = river and coast combined as 1 - (1-dr)(1-dc). Terms are applied to the annual loss of a peril at a site."""
    tiv = np.array([s["tiv"] for s in sites])
    out = {}
    flood_d = 1.0 - (1.0 - gu["river"]) * (1.0 - gu["coast"])
    for peril, d in (("wind", gu["wind"]), ("flood", flood_d)):
        cov = np.array([peril in s["covers"] for s in sites], float)
        ded = np.array([(ded_override_pct / 100.0 * s["tiv"]) if ded_override_pct is not None else s["terms"][peril]["ded"] for s in sites])
        lim = np.array([min(s["terms"][peril]["lim"], s["terms"]["total_limit"]) for s in sites])
        gl = d.astype(np.float64) * tiv[None, :]
        out[peril] = np.clip(gl - ded[None, :], 0.0, lim[None, :]) * cov[None, :]
        out[peril + "_gu"] = gl * cov[None, :]
    tl = np.array([s["terms"]["total_limit"] for s in sites])
    tot = out["wind"] + out["flood"]
    over = np.maximum(tot - tl[None, :], 0.0)                      # cap the site-year total at the overall limit, scaling both perils
    scale = np.where(tot > 0, 1.0 - over / np.maximum(tot, 1e-12), 1.0)
    out["wind"] *= scale
    out["flood"] *= scale
    out["total"] = out["wind"] + out["flood"]
    out["ground_up_total"] = out["wind_gu"] + out["flood_gu"]
    # attribute flood to river vs coast by ground-up share
    r_share = np.where(flood_d > 0, gu["river"] / np.maximum(gu["river"] + gu["coast"], 1e-12), 0.0)
    out["river"] = out["flood"] * r_share
    out["coast"] = out["flood"] * (1.0 - r_share)
    return out


def reinsurance(loss_wind: np.ndarray, loss_flood: np.ndarray, program: Optional[dict]) -> dict:
    """Portfolio-level: quota share first, then each excess-of-loss layer on the net after quota share, applied per peril-year
    (one occurrence per peril per year), with an annual aggregate of limit x (1 + reinstatements)."""
    qs = float((program or {}).get("quota_share_pct") or 0.0) / 100.0
    pw, pf = loss_wind.sum(axis=1), loss_flood.sum(axis=1)       # portfolio occurrence loss per year and peril
    ceded_qs = qs * (pw + pf)
    nw, nf = (1 - qs) * pw, (1 - qs) * pf
    ceded_xl = np.zeros_like(pw)
    layers = []
    for L in (program or {}).get("xl") or []:
        a, lim = float(L["attach"]), float(L["limit"])
        if lim <= 0 or a < 0:
            raise InsuranceError("excess-of-loss layer needs attach >= 0 and limit > 0")
        hit = (nw > a) | (nf > a)
        rec = np.clip(nw - a, 0, lim) + np.clip(nf - a, 0, lim)
        cap = lim * (1.0 + float(L.get("reinstatements", 1)))
        rec = np.minimum(rec, cap)
        # recoveries come off the retained peril losses in proportion
        tot = nw + nf
        share_w = np.where(tot > 0, nw / np.maximum(tot, 1e-12), 0.5)
        nw = nw - rec * share_w
        nf = nf - rec * (1 - share_w)
        ceded_xl += rec
        layers.append({"attach": a, "limit": lim, "reinstatements": float(L.get("reinstatements", 1)), "expected_recovery": float(rec.mean()),
                       "prob_attach": float(np.mean(hit))})
    return {"net_wind": nw, "net_flood": nf, "ceded": ceded_qs + ceded_xl, "layers": layers, "quota_share": qs}


def _ep(occ: np.ndarray, agg: np.ndarray, n_years: int) -> dict:
    rows = []
    for rp in RPS_OUT:
        p = 1.0 - 1.0 / rp
        rows.append({"rp": rp, "oep": float(np.quantile(occ, p)), "aep": float(np.quantile(agg, p)), "reliable": n_years >= 10 * rp})
    return {"rows": rows, "aal": float(agg.mean()), "std": float(agg.std()), "se": float(agg.std() / math.sqrt(len(agg))),
            "tvar99": float(agg[agg >= np.quantile(agg, 0.99)].mean()) if np.any(agg > 0) else 0.0,
            "tvar995": float(agg[agg >= np.quantile(agg, 0.995)].mean()) if np.any(agg > 0) else 0.0,
            "prob_loss": float(np.mean(agg > 0))}


def portfolio_stats(L: dict, ri: Optional[dict] = None) -> dict:
    ny = L["wind"].shape[0]
    pw, pf = L["wind"].sum(axis=1), L["flood"].sum(axis=1)
    out = {"gross": _ep(np.maximum(pw, pf), pw + pf, ny), "by_peril": {"wind": float(pw.mean()), "flood": float(pf.mean()),
                                                                          "river": float(L["river"].sum(axis=1).mean()), "coast": float(L["coast"].sum(axis=1).mean())}}
    gu = L["ground_up_total"].sum(axis=1)
    out["ground_up"] = {"aal": float(gu.mean()), "wind": float(L["wind_gu"].sum(axis=1).mean()), "flood": float(L["flood_gu"].sum(axis=1).mean())}
    if ri is not None:
        out["net"] = _ep(np.maximum(ri["net_wind"], ri["net_flood"]), ri["net_wind"] + ri["net_flood"], ny)
        out["ceded_aal"] = float(ri["ceded"].mean())
    return out


# ── the full run ─────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _rate_str(aal, tiv):
    return round(1000.0 * aal / tiv, 3) if tiv else None            # per mille of TIV


def _site_rows(sites, hazards, L, g, cap_q, coc, exp_ratio, unc_load):
    agg = L["total"].sum(axis=1)
    thr = np.quantile(agg, cap_q)
    tail = agg >= thr
    ny = L["total"].shape[0]
    rows = []
    for j, (s, h) in enumerate(zip(sites, hazards)):
        loss = L["total"][:, j]
        aal = float(loss.mean())
        tail_c = float(loss[tail].mean()) if tail.any() else 0.0
        capital = max(tail_c - aal, 0.0)
        prem = (aal * (1.0 + unc_load) + coc * capital) / (1.0 - exp_ratio)
        w = (h.get("wind") or {})
        rows.append({"name": s["name"], "lat": s["lat"], "lon": s["lon"], "country": h.get("iso3") or s.get("country"), "sector": s["sector"], "tiv": s["tiv"],
                     "deductible": s["terms"]["wind"]["ded"], "limit": s["terms"]["total_limit"], "covers": s["covers"],
                     "ground_up_aal": float(L["ground_up_total"][:, j].mean()), "aal": aal, "aal_wind": float(L["wind"][:, j].mean()),
                     "aal_river": float(L["river"][:, j].mean()), "aal_coast": float(L["coast"][:, j].mean()),
                     "rate_permille": _rate_str(aal, s["tiv"]), "stand_alone_1in100": float(np.quantile(loss, 0.99)),
                     "stand_alone_1in250": float(np.quantile(loss, 1 - 1 / 250)), "tail_contribution": tail_c,
                     "capital_proxy": capital, "technical_premium": prem, "premium_rate_permille": _rate_str(prem, s["tiv"]),
                     "hazard": {"cyclone_rate": w.get("lam"), "cyclone_rp100_ms": w.get("rp100_ms"), "river_map": bool(h.get("river")),
                                "coast_map": bool(h.get("coast")), "river_protection_rp": (h.get("river") or {}).get("protection"),
                                "coast_protection_rp": (h.get("coast") or {}).get("protection")},
                     "notes": list(h.get("notes") or []), "climate": h.get("_climate")})
    return rows


def _accumulation(sites, hazards, rows, L):
    agg = L["total"]
    zones: dict = {}
    for j, r in enumerate(rows):
        z = (round(r["lat"] / 2.0) * 2, round(r["lon"] / 2.0) * 2)
        zones.setdefault(z, []).append(j)
    zl = []
    pf = agg.sum(axis=1)
    for z, js in zones.items():
        zl_loss = agg[:, js].sum(axis=1)
        zl.append({"zone": f"{z[0]:+d}°, {z[1]:+d}°", "sites": len(js), "tiv": float(sum(rows[j]["tiv"] for j in js)), "aal": float(zl_loss.mean()),
                   "worst_year_loss": float(zl_loss.max()), "pml100": float(np.quantile(zl_loss, 0.99)),
                   "share_of_tail": float(zl_loss[pf >= np.quantile(pf, 0.99)].mean() / max(pf[pf >= np.quantile(pf, 0.99)].mean(), 1e-9))})
    zl.sort(key=lambda r: -r["aal"])

    def by(key):
        d: dict = {}
        for r in rows:
            k = str(r.get(key) or "unknown")
            e = d.setdefault(k, {"key": k, "sites": 0, "tiv": 0.0, "aal": 0.0})
            e["sites"] += 1
            e["tiv"] += r["tiv"]
            e["aal"] += r["aal"]
        return sorted(d.values(), key=lambda e: -e["aal"])
    return {"zones": zl[:15], "countries": by("country"), "sectors": by("sector")}


def run(sites_in: list, hazards: Optional[list] = None, n_years: int = 5000, seed: int = 11, corr_scale: float = 1.0,
        reinsurance_program: Optional[dict] = None, climate: Optional[dict] = None, cost_of_capital: float = 0.08,
        expense_ratio: float = 0.25, uncertainty_load: float = 0.10, capital_quantile: float = 0.99,
        compose_fn: Optional[Callable] = None, flood_fn: Optional[Callable] = None, tc_fn: Optional[Callable] = None, fetch=None, tile_fetch=None) -> dict:
    t0 = time.time()
    if not sites_in:
        raise InsuranceError("no sites")
    if len(sites_in) > 1500:
        raise InsuranceError("at most 1500 sites per run")
    if not (0 <= expense_ratio < 0.9) or not (0 <= cost_of_capital <= 0.5) or not (0 <= uncertainty_load <= 1.0):
        raise InsuranceError("expense_ratio must be in [0, 0.9), cost_of_capital in [0, 0.5], uncertainty_load in [0, 1]")
    if not (0.9 <= capital_quantile < 1.0):
        raise InsuranceError("capital_quantile must be in [0.9, 1)")
    if not (0.25 <= corr_scale <= 4.0):
        raise InsuranceError("corr_scale must be between 0.25 and 4")
    sites = [normalise_site(s, k) for k, s in enumerate(sites_in)]
    hz = hazards if hazards is not None else fetch_hazards(sites, flood_fn=flood_fn, tc_fn=tc_fn)
    gu = ground_up(sites, hz, n_years, seed, corr_scale)
    ny = gu["n_years"]
    L = apply_terms(gu, sites)
    ri = reinsurance(L["wind"], L["flood"], reinsurance_program) if reinsurance_program else None
    stats = portfolio_stats(L, ri)
    rows = _site_rows(sites, hz, L, gu, capital_quantile, cost_of_capital, expense_ratio, uncertainty_load)
    acc = _accumulation(sites, hz, rows, L)
    ladder = []
    for p in LADDER_PCT:
        Lp = apply_terms(gu, sites, ded_override_pct=p)
        s = portfolio_stats(Lp)
        at100 = next(r for r in s["gross"]["rows"] if r["rp"] == 100)
        ladder.append({"deductible_pct": p, "aal": s["gross"]["aal"], "aep100": at100["aep"], "oep100": at100["oep"], "tvar99": s["gross"]["tvar99"]})
    tiv_total = float(sum(s["tiv"] for s in sites))
    prem_total = float(sum(r["technical_premium"] for r in rows))
    res = {"n_sites": len(sites), "tiv": tiv_total, "n_years": ny, "seed": seed, "stats": stats, "sites": rows, "accumulation": acc, "deductible_ladder": ladder,
           "premium": {"technical_total": prem_total, "rate_permille": _rate_str(prem_total, tiv_total),
                       "formula": f"(AAL x (1 + {uncertainty_load:.0%} model-uncertainty load) + {cost_of_capital:.0%} cost of capital x capital proxy) / (1 - {expense_ratio:.0%} expenses); "
                                  f"capital proxy = the site's mean loss in the worst {100 * (1 - capital_quantile):.0f}% of portfolio years minus its AAL",
                       "parameters": {"cost_of_capital": cost_of_capital, "expense_ratio": expense_ratio, "uncertainty_load": uncertainty_load, "capital_quantile": capital_quantile},
                       "status": "indicative technical premium for discussion, not a quote"},
           "reinsurance": ({"quota_share_pct": 100 * ri["quota_share"], "layers": ri["layers"], "ceded_aal": stats.get("ceded_aal")} if ri else None),
           "corr_scale": corr_scale, "correlation_km": {k: v * corr_scale for k, v in CORR_KM.items()}}
    if ny < n_years:
        res.setdefault("notes", []).append(f"years reduced from {n_years} to {ny} to keep memory bounded for {len(sites)} sites")
    if climate:
        ssp, year = climate.get("ssp", "ssp245"), int(climate.get("year", 2070))
        hz2, per_site, comp = climate_hazards(sites, hz, ssp, year, climate.get("band", "median"), fetch=fetch, tile_fetch=tile_fetch, compose_fn=compose_fn)
        gu2 = ground_up(sites, hz2, n_years, seed, corr_scale, climate=True)
        L2 = apply_terms(gu2, sites)
        ri2 = reinsurance(L2["wind"], L2["flood"], reinsurance_program) if reinsurance_program else None
        st2 = portfolio_stats(L2, ri2)
        g = lambda st, base, rp, k: next(r for r in st[base]["rows"] if r["rp"] == rp)[k]            # noqa: E731
        d = {}
        for rp in (20, 100, 250):
            for k in ("oep", "aep"):
                a, b = g(stats, "gross", rp, k), g(st2, "gross", rp, k)
                d[f"{k}{rp}"] = {"today": a, "scenario": b, "change": b - a, "change_pct": (100 * (b - a) / a) if a else None}
        a, b = stats["gross"]["aal"], st2["gross"]["aal"]
        rows2 = _site_rows(sites, hz2, L2, gu2, capital_quantile, cost_of_capital, expense_ratio, uncertainty_load)
        res["climate"] = {"ssp": ssp, "year": year, "band": climate.get("band", "median"), "stats": st2,
                          "aal": {"today": a, "scenario": b, "change": b - a, "change_pct": (100 * (b - a) / a) if a else None},
                          "tail": d, "premium_total_scenario": float(sum(r["technical_premium"] for r in rows2)),
                          "per_site": [{"name": r["name"], "aal_today": r0["aal"], "aal_scenario": r["aal"], "change_pct": (100 * (r["aal"] - r0["aal"]) / r0["aal"]) if r0["aal"] else None,
                                        "premium_today": r0["technical_premium"], "premium_scenario": r["technical_premium"], "climate": c}
                                       for r0, r, c in zip(rows, rows2, per_site)],
                          "confidence": (comp or {}).get("confidence"), "limits": (comp or {}).get("limits"), "attribution": (comp or {}).get("attribution"),
                          "method": "The same random years are replayed with the pathway's hazard changes (river depth from the CMIP6 rain ratio, coastal depth plus AR6 sea-level rise, "
                                    "cyclone wind scaled per Knutson et al. 2020), so the difference is paired, not two noisy runs. Event frequency is not changed."}
    try:                                                             # real past events with the same exposure, from the case library; context for the reader, never a model input
        from .resident import casebase as _cb
        if _cb.available():
            w_aal, f_aal = float(np.mean(L["wind"])), float(np.mean(L["flood"]))
            mix = [(h, v) for h, v in (("cyclone", w_aal), ("flood", f_aal)) if v > 0.2 * max(w_aal, f_aal, 1e-12)]
            sectors = ", ".join(sorted({str(s.get("sector")) for s in sites_in if isinstance(s, dict) and s.get("sector")})[:4])
            pre = _cb.precedents(mix, sectors, ["insurance_availability", "financing_liquidity", "asset_damage", "sovereign_backstop"], k=3)
            if pre:
                res["precedents"] = {"cases": pre, "basis": "closest real events by hazard and by how the loss reached insurers and balance sheets; each fact is quoted from its source",
                                     "status": "historical context, not an input to the numbers above"}
    except Exception:                                                # noqa: BLE001 — the case library must never break an insurer run
        pass
    res["method"] = METHOD
    res["limits"] = LIMITS
    res["sources"] = SOURCES
    res["elapsed_s"] = round(time.time() - t0, 1)
    return res


METHOD = [
    "Hazard: tropical-cyclone rate and wind tail from NOAA IBTrACS (Holland wind field), river flood depths from JRC/CEMS-GloFAS, coastal from Deltares with sea-level rise, defences from FLOPROS; per-site damage from the Emanuel (2011) wind curve (Eberenz et al. 2021 regions) and JRC industrial depth-damage curves.",
    "Years: each simulated year draws, per peril, one spatially correlated severity per site (Gaussian copula, correlation exp(-distance/L), L = 300 km wind, 100 km river, 60 km coast: assumed, not fitted). Cyclone years also add the other storms of the year, independently per site.",
    "Policy terms: deductible and limit per peril group (wind, flood) applied to each site's annual loss from that peril, then the overall limit. Flood combines river and coast. This approximates per-event terms.",
    "Reinsurance (optional): quota share first, then excess-of-loss layers on the retained peril-year loss, with annual aggregate = limit x (1 + reinstatements).",
    "Occurrence (OEP) = the year's largest peril loss across the portfolio; aggregate (AEP) = all perils in the year.",
    "Technical premium = AAL with a model-uncertainty load, plus cost of capital on each site's tail contribution, grossed up for expenses.",
]
LIMITS = [
    "Not a vendor catastrophe model: damage curves are published and not calibrated to claims; they are for screening and for explaining differences, not for rating a single risk unaided.",
    "Perils covered: tropical-cyclone wind, river flood, coastal flood. Not modelled: hail, tornado and convective storm, earthquake, wildfire, subsidence, storm surge beyond the mapped coastal floods, business interruption as a separate value.",
    "Correlation lengths are assumed. PML and tail contributions depend on them: run corr_scale 0.5 and 2 to see the range.",
    "Flood is one annual-maximum event per site per year; cyclone years stack the year's storms. Footprint mean damage uses a 250 m ring around the point; a precise building location can differ.",
    "Flood maps give no frequency change under climate: a pathway moves the depth of a mapped return-period event, not how often it happens.",
    "Return-period estimates need many years: values flagged not reliable have fewer than 10 simulated years per year of return period.",
]
SOURCES = ["NOAA IBTrACS v04r01", "JRC CEMS-GloFAS river flood hazard maps v2.1", "Deltares Global Flood Maps", "FLOPROS (Scussolini et al. 2016)",
           "Emanuel 2011; Eberenz et al. 2021 (wind damage)", "JRC depth-damage (Huizinga et al. 2017)", "CMIP6 via World Bank CCKP (CC0); IPCC AR6 sea level (CC BY 4.0); Knutson et al. 2020"]


# ── checks used by the tests ─────────────────────────────────────────────────────────────────────────────────────────────────


def analytic_aal(h: dict, n: int = 200_000) -> dict:
    """Mean annual damage ratio from the same hazard definition by numerical integration (no simulation), per peril. For wind: the Poisson stacking of storms is taken as the
    expectation of 1 - prod(1 - d); it is computed by quadrature over vh and a Monte Carlo of the storm count (independent of the copula code)."""
    from .finance.distribution import _interp_rp
    from .finance.vulnerability import V_HALF, tc_damage
    out = {"river": 0.0, "coast": 0.0, "wind": 0.0}
    for key in FLOOD:
        c = _flood_damage_curve(h, key, h.get("cont", "oceania"))
        if c:
            u = (np.arange(n) + 0.5) / n
            T = 1.0 / (1.0 - u)
            d = _interp_rp(T, c["rps"], c["damage"])
            out[key] = float(np.mean(np.where(T > c["protection"], d, 0.0)))
    w = h.get("wind") or {}
    if w.get("lam", 0) > 0 and w.get("sigma"):
        rng = np.random.default_rng(5)
        vhc, vhh = V_HALF.get(h.get("tc_region", "ROW"), V_HALF["ROW"])
        m = 200_000
        vh = rng.uniform(min(vhc, vhh), max(vhc, vhh), (m, 1))
        N = rng.poisson(w["lam"], m)
        d = np.zeros(m)
        for k in range(int(N.max())):
            v = np.minimum(V_THRESH_TC + rng.exponential(w["sigma"], m), 95.0)
            d = 1.0 - (1.0 - d) * (1.0 - np.where(N > k, tc_damage(v, vh[:, 0]), 0.0))
        out["wind"] = float(d.mean())
    return out
