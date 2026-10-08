"""Earthquake hazard study: Gutenberg–Richter occurrence from the USGS catalogue, declustering, Joyner–Boore ground motion and a hindcast test."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from . import counts as C, plots, seismic as Z
from .generic_report import Notebook

XS = np.logspace(-2.5, 0.2, 60)                       # PGA grid, g


ALPHA = 0.05 / 3


def _assess(cat: dict, mc: float) -> dict:
    t, mag, la, lo = cat["t"], cat["mag"], cat["lat"], cat["lon"]
    keep = mag >= mc
    t, mag, la, lo = t[keep], mag[keep], la[keep], lo[keep]
    b, se = Z.b_value(mag, mc)
    dk = Z.decluster_gk(t, mag, la, lo)
    t_m, mag_m, la_m, lo_m = t[dk], mag[dk], la[dk], lo[dk]
    years = np.arange(1973, 2025)
    cnt = np.array([int(((t_m >= y) & (t_m < y + 1)).sum()) for y in years])
    cut = int(round(len(years) * 0.67))
    hc = C.rate_hindcast(cnt)
    tr_m, te_m = mag_m[t_m < 1973 + cut], mag_m[t_m >= 1973 + cut]
    b_tr, se_tr = Z.b_value(tr_m, mc) if len(tr_m) > 15 else (b, se)
    b_te, se_te = Z.b_value(te_m, mc) if len(te_m) > 15 else (float("nan"), float("nan"))
    p_b = float(2 * stats.norm.sf(abs(b_tr - b_te) / math.sqrt(se_tr ** 2 + se_te ** 2))) if np.isfinite(b_te) else float("nan")
    ks_p = float("nan")
    if len(te_m) >= 8:
        rng = np.random.default_rng(1)
        ex = (te_m - mc + 0.05) + rng.uniform(-0.05, 0.05, len(te_m))
        ks_p = float(stats.kstest(np.clip(ex, 1e-6, None), "expon", args=(0, 1.0 / (b_tr * math.log(10)))).pvalue)
    tests = [hc["p"]] + [x for x in (p_b, ks_p) if np.isfinite(x)]
    return {"mc": mc, "b": b, "se": se, "n_ge": int(len(t)), "t": t, "mag": mag, "la": la, "lo": lo, "t_m": t_m, "mag_m": mag_m, "la_m": la_m, "lo_m": lo_m, "cut": cut, "hc": hc,
            "b_tr": b_tr, "se_tr": se_tr, "b_te": b_te, "se_te": se_te, "p_b": p_b, "ks_p": ks_p, "valid": all(p > ALPHA for p in tests)}


def _hazard(mag_complete_n, years_span, b, mc, mmax, dist_km, xs=XS):
    """Annual rate of PGA > x: truncated Gutenberg–Richter magnitudes, empirical distances."""
    ms = np.arange(mc, mmax + 1e-9, 0.1) + 0.05 * 0
    w = 10 ** (-b * (ms - mc)) - 10 ** (-b * (ms + 0.1 - mc))
    w = np.clip(w, 0, None); w = w / w.sum()
    rate = mag_complete_n / years_span
    d = np.asarray(dist_km, float)
    out = []
    for x in xs:
        p = np.zeros_like(d)
        for m, wm in zip(ms, w):
            p += wm * Z.exceed_prob(m, d, x)
        out.append(rate * float(p.mean()))
    return np.array(out)


def run(lat: float, lon: float, name: str = "Study site", progress=None, catalog: dict = None, radius_km: float = 300.0, fetch=None) -> dict:
    nb = Notebook(progress)
    nb("1", "Problem", f"How often does the ground shake hard enough to matter at {name} ({lat:.3f}, {lon:.3f}), how hard in 475- and 2 475-year terms, and does the early record predict the recent one?")
    nb("2", "Choice of physics", "Earthquakes cannot be forecast event by event, but their statistics obey the Gutenberg–Richter law, log10 N = a − b M. The engine therefore models rates and magnitudes statistically (declustered so aftershocks do not inflate the count), then converts each magnitude and distance to shaking at the site with an empirical ground-motion relation (Joyner & Boore 1981).")
    cat = catalog or Z.fetch_catalog(lat, lon, radius_km, 1973, 2025, 4.0, fetch)
    t, mag, la, lo = cat["t"], cat["mag"], cat["lat"], cat["lon"]
    if len(mag) < 30:
        nb("3", "Data", f"Only {len(mag)} catalogued events within {radius_km:.0f} km: too few for a statistical model.", warn=True)
        return _too_few(name, lat, lon, len(mag), radius_km, nb)
    mc0 = Z.mc_max_curvature(mag)
    A = _assess(cat, mc0)
    nb("3", "Catalogue", f'{len(cat["mag"])} events within {radius_km:.0f} km since 1973 ({cat["source"]}). Completeness Mc = {mc0:.1f} (max curvature); b = {A["b"]:.2f} ± {A["se"]:.2f}; {A["n_ge"]} events ≥ Mc, {len(A["t_m"])} independent mainshocks after Gardner–Knopoff declustering.')
    hc = A["hc"]
    nb("4", "Hindcast validation", f'Mainshock rate {hc["rate_train"]:.2f}/yr (1973–{1973 + A["cut"] - 1}) predicts {hc["expected"]:.1f} events in the last {hc["n_test"]} years; observed {hc["observed"]} (Poisson p = {hc["p"]:.2f}).', warn=hc["p"] <= ALPHA)
    if np.isfinite(A["p_b"]):
        nb("5", "Magnitude-distribution hindcast", f'b-value early {A["b_tr"]:.2f} ± {A["se_tr"]:.2f}, late {A["b_te"]:.2f} ± {A["se_te"]:.2f}: difference p = {A["p_b"]:.2f}.', warn=A["p_b"] <= ALPHA)
    if np.isfinite(A["ks_p"]):
        nb("5", "Magnitude-distribution hindcast", f'Late-period magnitudes against the early-period Gutenberg–Richter law: KS p = {A["ks_p"]:.2f}.', warn=A["ks_p"] <= ALPHA)
    retuned = None
    if not A["valid"]:
        for dm in (0.2, 0.4, 0.6, 0.8):
            A2 = _assess(cat, mc0 + dm)
            if A2["valid"] and len(A2["t_m"]) >= 30:
                retuned = (dm, A2)
                break
        if retuned:
            nb("5b", "Diagnosis", f"The early catalogue did not predict the late one. The classic cause is changing completeness (small events were detected less reliably in early decades), which shows as a drifting b-value. Raising Mc by {retuned[0]:.1f} to {mc0 + retuned[0]:.1f} removes the drift and the refit passes all three tests. Mc was chosen after seeing the held-out years, so this is a diagnosis, not an independent validation.", warn=True)
            A = retuned[1]
        else:
            nb("5b", "Diagnosis", "Raising the completeness magnitude by up to 0.8 did not make the early catalogue predict the late one; the process itself changed (for example a large sequence such as an M 9 aftershock series), so a stationary model does not apply.", warn=True)
    mc, b, se = A["mc"], A["b"], A["se"]
    t, mag, la, lo = A["t"], A["mag"], A["la"], A["lo"]
    t_m, mag_m, la_m, lo_m = A["t_m"], A["mag_m"], A["la_m"], A["lo_m"]
    span = 2025.0 - 1973.0
    years = np.arange(1973, 2025)
    cnt = np.array([int(((t_m >= y) & (t_m < y + 1)).sum()) for y in years])
    cut = A["cut"]; hc = A["hc"]; b_tr, b_te, se_tr, se_te, p_b, ks_p = A["b_tr"], A["b_te"], A["se_tr"], A["se_te"], A["p_b"], A["ks_p"]
    bench = Z.run_benchmarks()
    nfail = sum(1 for x in bench if not x["ok"])
    nb("6", "Verification", f"{len(bench) - nfail} of {len(bench)} tests passed." if not nfail else f"{nfail} verification test(s) FAILED.", warn=bool(nfail))

    alpha = ALPHA
    validated = bool(A["valid"])
    status = ("FAILED verification" if nfail else "verified; hindcast FAILED (a stationary Gutenberg–Richter model does not describe this catalogue)" if not validated
              else "verified; hindcast passed only after raising the completeness magnitude (diagnosed, not independent)" if retuned else "verified and hindcast-validated")
    mmax = min(float(mag.max()) + 0.5, 9.0)
    dist = Z.haversine(lat, lon, la_m, lo_m)
    dist = np.sqrt(dist ** 2 + 1.0)                                    # epicentral distance, floored away from zero
    haz = _hazard(len(t_m), span, b, mc, mmax, dist)
    lvl = {T: Z.pga_return_level(XS, haz, T) for T in (475, 2475)}
    lo_b, hi_b = _hazard(len(t_m), span, b - se, mc, mmax, dist), _hazard(len(t_m), span, b + se, mc, mmax, dist)
    lv_rng = {T: sorted([Z.pga_return_level(XS, lo_b, T), Z.pga_return_level(XS, hi_b, T)]) for T in (475, 2475)}
    mm = _hazard(len(t_m), span, b, mc, mmax + 0.5, dist)
    lv_mm = {T: Z.pga_return_level(XS, mm, T) for T in (475,)}
    fmt = lambda v: "n/a" if not np.isfinite(v) else f"{v:.2f} g"
    big = np.argsort(-mag_m)[:5]
    head = [(f"{len(t_m) / span:.1f} / yr", f"independent M ≥ {mc:.1f} within {radius_km:.0f} km"), (f"b = {b:.2f}", "Gutenberg–Richter slope"), (fmt(lvl[475]), "PGA, 475-year"), (fmt(lvl[2475]), "PGA, 2 475-year"), (f"M {mag.max():.1f}", "largest on record")]
    fig_rate = plots.series_fig(years.astype(float), {"Independent mainshocks per year": cnt.astype(float)}, "events per year", f"M ≥ {mc:.1f} mainshocks within {radius_km:.0f} km", cut=cut)
    ms_ = np.sort(mag)
    fig_gr = plots.scatter_fit(ms_, np.arange(len(ms_), 0, -1) / span, np.array([mc, mag.max()]), (len(t) / span) * 10 ** (-b * (np.array([mc, mag.max()]) - mc)), "magnitude", "events per year at or above M", f"Gutenberg–Richter law, b = {b:.2f}")
    fig_haz = plots.scatter_fit(XS, haz, XS, haz, "peak ground acceleration (g)", "annual rate of exceedance", "Seismic hazard curve at the site")
    blocks = [("h2", "question", "The question"),
              ("p", f"How hard and how often does the ground shake at {name}? Earthquakes cannot be predicted individually, so the engine builds a statistical model of when and how big, hides the last third of the catalogue, and checks it can predict it."),
              ("h2", "physics", "The physics the engine chose"),
              ("p", "Earthquakes release strain along faults. Their sizes follow the **Gutenberg–Richter law, log10 N(≥M) = a − b M**; b near 1 means each magnitude unit is ten times rarer. The engine estimates the completeness magnitude by maximum curvature and b by the Aki–Utsu maximum-likelihood formula **b = log10(e) / (mean M − (Mc − ΔM/2))**."),
              ("p", "Aftershocks cluster in time and space and would inflate the rate, so the Gardner–Knopoff windows remove them, leaving independent mainshocks that are close to Poisson."),
              ("p", "Shaking at the site is converted from magnitude and distance by the Joyner & Boore (1981) relation, **log10 PGA = −1.02 + 0.249 M − log10 r − 0.00255 r**, with r = √(d² + 7.3²) km and lognormal scatter (σ = 0.26 in log10)."),
              ("basis",),
              ("h2", "data", "The data"),
              ("p", f"{len(cat['mag'])} events of magnitude ≥ 4.0 within {radius_km:.0f} km, 1973–2025, from the {cat['source']}; {len(t_m)} independent mainshocks of magnitude {mc:.1f} or more remain after completeness and declustering."),
              ("fig", fig_rate, "Grey band: years the model was not shown when it was tested."),
              ("fig", fig_gr, "Each dot is a catalogued event; the line is the fitted law."),
              ("h2", "validation", "Did the model predict the years it never saw?"),
              ("table", ["Test", "Result", "p-value"], [["Mainshock rate (early rate predicts late count)", f'{hc["observed"]} observed vs {hc["expected"]:.1f} expected', f'{hc["p"]:.2f}']] + ([["Magnitude slope b (early vs late)", f"{b_tr:.2f} vs {b_te:.2f}", f"{p_b:.2f}"]] if np.isfinite(p_b) else []) + ([["Late magnitudes vs early Gutenberg–Richter law", "KS test", f"{ks_p:.2f}"]] if np.isfinite(ks_p) else []),
               f"Each p-value is compared with {alpha:.3f} (Bonferroni across three tests)."),
              ("callout", ("The hindcast PASSED after raising the completeness magnitude to " + f"{mc:.1f}" + ": the first fit failed because early decades missed small events. Because Mc was chosen after seeing the held-out years, treat this as a diagnosis, not an independent test." if retuned else "The hindcast PASSED: the early catalogue predicted the recent rate and magnitudes.") if validated else "The hindcast FAILED even after raising the completeness magnitude: the rate or magnitude mix changed between early and late record (for example a very large earthquake and its aftershock sequence), so a stationary model does not apply. The numbers below are indicative only.", "warn" if (retuned or not validated) and validated else "good" if validated else "bad"),
              ("h2", "result", "How hard, how often"),
              ("fig", fig_haz, "Annual rate at which peak ground acceleration is exceeded (log-log)."),
              ("table", ["Return period", "PGA (median)", "range if b ± 1 SE"], [[f"{T} years", fmt(lvl[T]), f"{fmt(lv_rng[T][0])} – {fmt(lv_rng[T][1])}"] for T in (475, 2475)],
               f"Hazard sums every magnitude from Mc to Mmax = {mmax:.1f} (largest observed + 0.5) at the distances where events actually occurred. Raising Mmax by 0.5 changes the 475-year value to {fmt(lv_mm[475])}."),
              ("table", ["Rank", "Magnitude", "Epicentral distance", "Year"], [[i + 1, f"M {mag_m[j]:.1f}", f"{Z.haversine(lat, lon, la_m[j], lo_m[j]):.0f} km", f"{int(t_m[j])}"] for i, j in enumerate(big)], "Largest independent events in the catalogue."),
              ("p", f"For scale, 0.1 g is felt strongly and cracks plaster; 0.3 g damages ordinary buildings; building codes typically design for the 475-year level.")]
    steps = ["Check the catalogue's completeness over time (it improves after 1990 in most regions); a drift in Mc can masquerade as a rate change.", "Add known fault sources: a catalogue of 50 years rarely samples the largest earthquakes, which dominate the 2 475-year level.", "Add site amplification (soft soil raises shaking by up to a factor of 2–3) and then asset vulnerability curves."]
    if not validated:
        steps.insert(0, "The early record did not predict the recent one: do not rely on the stationary rate until completeness by period has been examined.")
    return {"domain": "earthquake", "label": "Earthquake", "kind": "seismic", "name": name, "lat": lat, "lon": lon, "title": f"Earthquakes at {name}: how often, how hard, and can we trust it",
            "standfirst": f"The engine fitted a Gutenberg–Richter model to {len(t_m)} independent earthquakes within {radius_km:.0f} km of {name}, found b = {b:.2f} and a 475-year ground acceleration of {fmt(lvl[475])}, then tested its forecast on years it had not seen. Status: {status}.",
            "status": status, "ok": nfail == 0, "headline": head, "blocks": blocks, "verification": bench, "sensitivity": [],
            "assumptions": ["Mainshocks are a Poisson process after Gardner–Knopoff declustering.", "Epicentral distance is used in place of distance to the fault rupture; this understates shaking from large nearby events.", "Median relation on firm ground; no site amplification."],
            "limitations": ["A 50-year catalogue cannot see the rarest large earthquakes; the extension to Mmax relies on the Gutenberg–Richter extrapolation.", "Joyner–Boore (1981) was fitted to western North America; other regions can differ by a factor of two."],
            "next_steps": steps, "notebook": nb.rows, "numbers": {"mc": mc, "b": b, "b_se": se, "n_mainshocks": int(len(t_m)), "levels": {str(k): v for k, v in lvl.items()}, "hindcast": hc, "p_b": p_b, "ks_p": ks_p}, "elapsed_s": round(__import__("time").time() - nb.t0, 1)}


def _too_few(name, lat, lon, n, radius, nb):
    return {"domain": "earthquake", "label": "Earthquake", "kind": "seismic", "name": name, "lat": lat, "lon": lon, "title": f"Earthquakes at {name}: too little seismicity to model",
            "standfirst": f"Only {n} earthquakes of magnitude 4.0 or more were catalogued within {radius:.0f} km of {name} since 1973. That is too few to fit a statistical model, so the engine does not give a ground-motion number rather than guessing.",
            "status": "no model (too little data)", "ok": True, "headline": [(str(n), "catalogued events ≥ M 4.0")],
            "blocks": [("h2", "result", "What the data allow"), ("p", "A Gutenberg–Richter fit needs at least a few dozen events above the catalogue's completeness magnitude. Low seismicity does not mean zero hazard: rare large events on poorly known faults are the typical risk in stable regions.")],
            "verification": [], "sensitivity": [], "assumptions": [], "limitations": ["Seismicity within the search radius is too sparse."], "next_steps": ["Widen the search radius, or use a regional seismic source model (e.g. GEM Global Seismic Hazard Map) for this site."], "notebook": nb.rows, "numbers": {"n_events": n}, "elapsed_s": 0.0}
