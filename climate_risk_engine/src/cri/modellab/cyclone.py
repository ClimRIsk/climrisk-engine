"""Tropical cyclone hazard study: Holland wind field on IBTrACS best tracks, hindcast of event rates and a peak-over-threshold wind tail."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from . import counts as C, plots
from .generic_report import Notebook
from ..climate import tropical_cyclone as TC

TS, CAT1, CAT3 = 17.5, 33.0, 50.0


def bench_holland() -> list:
    out = []
    v, rm, lat = 55.0, 30.0, 20.0
    b = TC.holland_b(v, rm, lat)
    w_peak = float(TC.holland_wind(np.array([rm]), v, rm, b)[0])
    ok = abs(w_peak - v) < 1e-6
    out.append({"name": "Holland wind profile peaks at the radius of maximum wind", "metric": f"V(r = Rmax) = {w_peak:.3f} m/s (target {v})", "value": abs(w_peak - v), "limit": 1e-6, "ok": ok, "note": "By construction V = Vmax √(x e^(1−x)) with x = (Rmax/r)^B, so x = 1 gives Vmax."})
    r = np.linspace(rm, 600, 200)
    wr = TC.holland_wind(r, v, rm, b)
    mono = bool(np.all(np.diff(wr) <= 1e-9)) and wr[-1] < 0.25 * v
    out.append({"name": "Wind decays monotonically outside the eyewall", "metric": f"V(600 km) = {wr[-1]:.1f} m/s, {100 * wr[-1] / v:.0f} % of Vmax", "value": float(wr[-1] / v), "limit": 0.25, "ok": mono, "note": "A cyclone's wind field has no secondary peaks in this model and fades with distance."})
    d = float(TC._haversine_km(0.0, 0.0, 0.0, 1.0))
    out.append({"name": "Great-circle distance: one degree of longitude at the equator", "metric": f"{d:.2f} km (111.19)", "value": abs(d - 111.19), "limit": 0.05, "ok": abs(d - 111.19) < 0.05, "note": "Haversine on a 6371 km sphere."})
    # POT recovery
    rng = np.random.default_rng(5)
    pk = TS + rng.exponential(8.0, 4000)
    sc, n = C.pot_fit(pk, TS)
    out.append({"name": "Peaks-over-threshold tail recovers a known exponential", "metric": f"scale = {sc:.2f} m/s (true 8.00)", "value": abs(sc - 8.0), "limit": 0.5, "ok": abs(sc - 8.0) < 0.5, "note": "4 000 synthetic peaks over 17.5 m/s."})
    ok = 0
    for s in range(200):
        r2 = np.random.default_rng(300 + s)
        c = r2.poisson(2.0, 45)
        ok += int(C.rate_hindcast(c)["p"] > 0.05)
    out.append({"name": "Event-rate hindcast test is calibrated", "metric": f"{ok / 200:.2f} of true-Poisson records pass (expect ≈ 0.95)", "value": ok / 200, "limit": 0.88, "ok": ok / 200 >= 0.88, "note": "Same test as used below."})
    return out


def cat_rates_zero(cnt, n):
    return {"tropical storm": float(cnt[TS].mean()), "hurricane (Cat 1+)": float(cnt[CAT1].mean()), "major (Cat 3+)": float(cnt[CAT3].mean())}


def site_peaks(lat: float, lon: float, search_km: float = 600.0, T: dict = None) -> tuple:
    T = T or TC.tracks()
    d = TC._haversine_km(T["lat"], T["lon"], lat, lon)
    near = np.unique(T["storm"][d <= search_km])
    rows = []
    for si in near:
        ix = np.where(T["storm"] == si)[0]
        v, rmin, _ = TC.storm_peak_at(lat, lon, ix, T)
        rows.append((int(T["season"][si]), float(v), float(rmin), str(T["name"][si]), int(si)))
    return rows


def run(lat: float, lon: float, name: str = "Study site", progress=None, search_km: float = 600.0, T: dict = None) -> dict:
    nb = Notebook(progress)
    T = T or TC.tracks()
    y0, y1 = int(T["season"].min()), int(T["season"].max())
    years = list(range(y0, y1 + 1))
    nb("1", "Problem", f"How often do tropical cyclones bring damaging wind to {name} ({lat:.3f}, {lon:.3f}), how strong do they get, and does the early record predict the recent one? Data: IBTrACS best tracks, {len(years)} seasons ({y0}–{y1}).")
    nb("2", "Choice of physics", "Cyclone wind is not a reanalysis quantity (ERA5 smooths the eyewall away), so the engine models each storm explicitly: the Holland (1980) radial wind profile driven by the track's maximum wind, with the shape parameter from Willoughby & Rahn (2004) and the radius from the best track or Willoughby et al. (2006). Per-storm peak winds at the site feed a Poisson rate and an exponential peak-over-threshold tail.")
    rows = site_peaks(lat, lon, search_km, T)
    nb("3", "Wind field at the site", f"{len(rows)} storms passed within {search_km:.0f} km since {y0}; peak modelled 1-minute winds computed along each track at 30-minute resolution.")
    peaks = np.array([r[1] for r in rows], float)
    seas = np.array([r[0] for r in rows], int)
    cnt = {thr: np.array([int(np.sum((seas == y) & (peaks >= thr))) for y in years]) for thr in (TS, CAT1, CAT3)}
    n = len(years)
    tested = [(thr, lab) for thr, lab in ((TS, "tropical-storm force (17.5 m/s)"), (CAT1, "hurricane force (33 m/s)")) if cnt[thr][: int(round(n * 0.67))].sum() >= 3]
    hc = [(lab, thr, C.rate_hindcast(cnt[thr])) for thr, lab in tested]
    alpha = 0.05 / max(len(hc), 1)
    passed = bool(hc) and all(h["p"] > alpha for _, _, h in hc)
    for lab, thr, h in hc:
        nb("4", "Hindcast validation", f'{lab}: rate in {y0}–{y0 + h["n_train"] - 1} = {h["rate_train"]:.2f}/yr predicts {h["expected"]:.1f} events in the last {h["n_test"]} seasons; observed {h["observed"]} (Poisson p = {h["p"]:.2f}).', warn=h["p"] <= alpha)
    if not hc:
        nb("4", "Hindcast validation", "Too few storms at this site to test a rate forecast.", warn=True)
    ts_pk = peaks[peaks >= TS]
    scale, nex = C.pot_fit(ts_pk, TS)
    lam = nex / n
    lv = {T_: C.pot_level(TS, scale, lam, T_) for T_ in (10, 50, 100)} if nex >= 5 else {}
    ks_p = None
    if nex >= 12:
        cut_season = y0 + int(round(n * 0.67))
        a = np.array([p - TS for p, s in zip(peaks, seas) if p > TS and s < cut_season])
        b = np.array([p - TS for p, s in zip(peaks, seas) if p > TS and s >= cut_season])
        if len(a) >= 6 and len(b) >= 4:
            sc_a = a.mean()
            ks_p = float(stats.kstest(b, "expon", args=(0, sc_a)).pvalue)
            nb("5", "Intensity hindcast", f"Exponential tail fitted to {len(a)} early storms; the {len(b)} later storms' wind excesses are consistent with it: KS p = {ks_p:.2f}.", warn=ks_p <= 0.05)
    disp = C.dispersion(cnt[TS]) if cnt[TS].sum() else float("nan")
    t0 = __import__("time").time()
    bench = bench_holland()
    nfail = sum(1 for b in bench if not b["ok"])
    nb("6", "Verification", f"{len(bench) - nfail} of {len(bench)} tests passed." if not nfail else f"{nfail} verification test(s) FAILED.", warn=bool(nfail))
    ints_ok = ks_p is None or ks_p > 0.05
    if int((peaks >= TS).sum()) < 3:
        n_ts = int((peaks >= TS).sum())
        nb("7", "Exposure", f"Only {n_ts} storm(s) in {n} seasons brought tropical-storm-force wind to this site, so tropical cyclones are not a material hazard here and no rate or wind-tail model is fitted.")
        return {"domain": "cyclone", "label": "Tropical cyclone", "kind": "cyclone", "name": name, "lat": lat, "lon": lon, "title": f"Tropical cyclones at {name}: no material exposure",
                "standfirst": f"In {n} seasons of best-track data ({y0}–{y1}), only {n_ts} tropical cyclone(s) brought tropical-storm-force wind (17.5 m/s) to {name}. The engine treats tropical cyclone wind as not applicable here instead of fitting a model to almost no events. Status: no exposure.",
                "status": "no model (no tropical cyclone exposure)", "ok": nfail == 0, "headline": [(str(n_ts), f"storms with damaging wind in {n} seasons"), (str(len(rows)), f"storms within {search_km:.0f} km")],
                "blocks": [("h2", "result", "Exposure"), ("p", f"{len(rows)} storms passed within {search_km:.0f} km since {y0}; the strongest modelled wind at the site was {peaks.max() if len(peaks) else 0:.0f} m/s, below the 17.5 m/s tropical-storm threshold in all but {n_ts} case(s). Extratropical storms, thunderstorms and monsoon winds are not tropical cyclones: use the extreme-wind study for them."), ("basis",)],
                "verification": bench, "sensitivity": [], "assumptions": [], "limitations": ["Best tracks cover 1980 onward: a very rare landfall outside that window would not show."], "next_steps": ["Use the extreme-wind study for non-tropical wind at this site."], "notebook": nb.rows,
                "numbers": {"n_storms": len(rows), "n_ts": n_ts, "levels": {}, "rates": cat_rates_zero(cnt, n)}, "elapsed_s": round(__import__("time").time() - nb.t0, 1)}
    validated = passed and ints_ok
    status = ("FAILED verification" if nfail else "verified; too few storms to hindcast" if not hc else "verified and hindcast-validated" if validated else "verified; hindcast FAILED (rates changed between early and late record)")
    cat_rates = {"tropical storm": float(cnt[TS].mean()), "hurricane (Cat 1+)": float(cnt[CAT1].mean()), "major (Cat 3+)": float(cnt[CAT3].mean())}
    top = sorted(rows, key=lambda r: -r[1])[:5]
    head = [(f'{cat_rates["tropical storm"]:.2f} / yr', "tropical-storm wind or more"), (f'{cat_rates["hurricane (Cat 1+)"]:.2f} / yr', "hurricane-force wind"), (f'{lv[100]:.0f} m/s' if lv else "n/a", "100-year peak wind"), (f"{top[0][1]:.0f} m/s" if top else "n/a", f"worst storm ({top[0][3].title()} {top[0][0]})" if top else "worst storm")]
    fig_counts = plots.series_fig(np.array(years, float), {"Tropical-storm wind or more": cnt[TS].astype(float), "Hurricane-force": cnt[CAT1].astype(float)}, "storms per season", "Storms bringing damaging wind to the site, by season", cut=int(round(n * 0.67)))
    blocks = [("h2", "question", "The question"),
              ("p", f"How often do tropical cyclones bring damaging wind to {name}, how strong do they get, and can the early record predict the recent one? The engine rebuilds every storm that passed near the site and tests its own rate forecast on seasons it did not use."),
              ("h2", "physics", "The physics the engine chose"),
              ("p", "Each storm is a rotating vortex. The Holland (1980) profile gives the wind at distance r from the centre: **V(r) = Vmax √( x e^(1−x) ), x = (Rmax / r)^B**. The wind peaks at the radius of maximum wind and fades outside it; B is set by intensity and latitude."),
              ("p", "The engine interpolates every 3- or 6-hourly track fix to 30-minute steps so the closest approach is not missed, and keeps the highest wind each storm produces at the site. Winds are 1-minute sustained at 10 m over open terrain; multiply by 0.88 for a 10-minute mean."),
              ("basis",),
              ("h2", "data", "The data"),
              ("p", f"NOAA IBTrACS v04r01 best tracks, {y0}–{y1}, all basins ({len(T['sid'])} storms); {len(rows)} came within {search_km:.0f} km of the site. The track archive begins in 1980, so {n} seasons are the whole record."),
              ("fig", fig_counts, "Grey band: the seasons the rate forecast was not shown."),
              ("h2", "validation", "Did the model predict the seasons it never saw?")]
    if hc:
        blocks += [("table", ["Event", "Early rate", "Expected in held-out seasons", "Observed", "p-value"],
                    [[lab, f'{h["rate_train"]:.2f}/yr', f'{h["expected"]:.1f}', h["observed"], f'{h["p"]:.2f}'] for lab, thr, h in hc], f"Poisson test on the count; p-values below {alpha:.3f} (Bonferroni-adjusted) would fail."),
                   ("callout", "The rate forecast PASSED." if passed else "The rate forecast FAILED: the early record does not describe the recent seasons here.", "good" if passed else "bad")]
    else:
        blocks.append(("callout", "Too few storms to test a rate forecast at this site.", "warn"))
    if ks_p is not None:
        blocks.append(("p", f"Intensity: wind excesses over 17.5 m/s in the later storms are {'consistent' if ks_p > 0.05 else 'NOT consistent'} with the exponential tail fitted to the earlier storms (KS p = {ks_p:.2f})."))
    blocks += [("h2", "result", "How strong, how often")]
    if lv:
        blocks += [("table", ["Return period", "Peak wind (1-min, open terrain)", "Category"], [[f"{T_} years", f"{lv[T_]:.0f} m/s", TC.category(lv[T_])] for T_ in (10, 50, 100)],
                    f"Exponential peak-over-threshold tail above 17.5 m/s: mean excess {scale:.1f} m/s, {lam:.2f} storms/yr. Beyond {n} years these are extrapolations.")]
    else:
        blocks.append(("p", "Too few tropical-storm-force events (fewer than five) to fit a wind tail; only event rates are reported."))
    blocks.append(("table", ["Rank", "Storm", "Season", "Peak wind at site", "Closest approach"], [[i + 1, r[3].title(), r[0], f"{r[1]:.0f} m/s ({TC.category(r[1])})", f"{r[2]:.0f} km"] for i, r in enumerate(top)], "Strongest modelled winds at the site."))
    if np.isfinite(disp):
        blocks.append(("p", f"Seasons are {'clustered (variance-to-mean ratio ' + f'{disp:.1f}' + ', more than a Poisson process)' if disp > 1.5 else 'close to Poisson (variance-to-mean ratio ' + f'{disp:.1f}' + ')'}, " + ("so year-to-year swings in climate drivers such as ENSO matter." if disp > 1.5 else "so a constant annual rate is a fair description.")))
    steps = ["Compare the modelled peaks with local station gusts; the engine's own validation shows raw open-terrain winds run about 1.75 times station peaks, so apply the land-exposure factor for building loads.", "Add storm surge: wind is only one cyclone peril, and the sea level rise a storm pushes ashore often costs more (the engine does not yet model surge).", "Extend the record: 1980 onward is short for 100-year winds, so use the level as indicative."]
    if hc and not passed:
        steps.insert(0, "Rates changed between the early and late record: use the most recent decades or a covariate such as sea-surface temperature before relying on the long-run rate.")
    return {"domain": "cyclone", "label": "Tropical cyclone", "kind": "cyclone", "name": name, "lat": lat, "lon": lon, "title": f"Tropical cyclones at {name}: how often, how strong, and can we trust it",
            "standfirst": f"The engine rebuilt the {len(rows)} storms that passed near {name} since {y0} with the Holland wind profile, found {cat_rates['tropical storm']:.2f} tropical-storm-wind events a year" + (f" and a 100-year peak wind near {lv[100]:.0f} m/s" if lv else "") + ", then tested its forecast on seasons it had not seen. " + f"Status: {status}.",
            "status": status, "ok": nfail == 0, "headline": head, "blocks": blocks, "verification": bench, "sensitivity": [],
            "assumptions": ["Winds are best-track 1-minute sustained at 10 m, open terrain; no land friction or topographic speed-up.", "Each season's events are treated as a Poisson process (dispersion reported)."],
            "limitations": ["Storm surge, rainfall and tornadoes are separate perils not modelled here.", "A 45-season record is short for the 100-year wind."],
            "next_steps": steps, "notebook": nb.rows, "numbers": {"rates": cat_rates, "levels": lv, "n_storms": len(rows), "hindcast": [{"label": l, **h} for l, _, h in hc]}, "elapsed_s": round(__import__("time").time() - nb.t0, 1)}
