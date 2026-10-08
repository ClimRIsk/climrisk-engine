"""Hazards whose truth is a time series: drought (soil moisture) and river flow. The engine fits competing physical structures on the early record,
diagnoses what the best one misses, and scores it on years it never saw against baselines that need no physics."""
from __future__ import annotations

import math
import time

import numpy as np

from .dossier import NoModel
from . import bucket, calib, data as D, metrics, plots
from .generic_report import Notebook


def _doy(times):
    import datetime as dt
    return np.array([dt.date(int(t[:4]), int(t[5:7]), int(t[8:10])).timetuple().tm_yday for t in times], float)


def _fit_all(structs: dict, sim, obs, cut: int, burn: int, transform=lambda v: v, restarts: int = 2):
    fits = {}
    tr = slice(burn, cut)
    for key, sp in structs.items():
        def resid(x, key=key):
            return transform(sim(key, x)[tr]) - transform(obs[tr])
        f = calib.least_squares_fit(resid, sp["x0"], (sp["lo"], sp["hi"]), restarts=restarts)
        rss = float(np.sum(resid(f["x"]) ** 2))
        f["aic"] = calib.aic(cut - burn, rss, len(sp["lo"]))
        fits[key] = f
    return fits


def _doy_clim(obs, doy, cut, burn):
    clim = np.zeros(367)
    for d in range(1, 367):
        m = (np.abs(doy[burn:cut] - d) <= 7) | (np.abs(doy[burn:cut] - d) >= 358)
        clim[d] = np.mean(obs[burn:cut][m]) if m.any() else np.mean(obs[burn:cut])
    return clim


# ── drought ──────────────────────────────────────────────────────────────────

DR = {
    "M0": {"label": "One soil store (fixed drainage curvature)", "names": ["Sc", "kc", "kd", "a", "b"], "x0": [200, 0.8, 0.03, 0.3, 0.08], "lo": [40, 0.2, 0.001, 0.05, 0.0], "hi": [700, 1.6, 0.3, 0.7, 0.25]},
    "M1": {"label": "+ free drainage curvature and evaporation-stress threshold", "names": ["Sc", "kc", "kd", "a", "b", "m", "fc"], "x0": [200, 0.8, 0.03, 0.3, 0.08, 2.0, 0.6], "lo": [40, 0.2, 0.001, 0.05, 0.0, 1.0, 6e-1 * 0.3], "hi": [700, 1.6, 0.3, 0.7, 0.25, 6.0, 1.0]},
    "M2": {"label": "+ seasonal vegetation water use", "names": ["Sc", "kc", "kd", "a", "b", "m", "fc", "ka", "ph"], "x0": [200, 0.8, 0.03, 0.3, 0.08, 2.0, 0.6, 0.2, 150], "lo": [40, 0.2, 0.001, 0.05, 0.0, 1.0, 0.18, 0.0, 0], "hi": [700, 1.6, 0.3, 0.7, 0.25, 6.0, 1.0, 0.8, 365]},
}


def _soil_sim(key, x, P, E0, doy):
    p = dict(zip(DR[key]["names"], x))
    kc = p["kc"] * (1 + p.get("ka", 0.0) * np.cos(2 * math.pi * (doy - p.get("ph", 0.0)) / 365.0))
    S, *_ = bucket.soil(P, E0 * kc / p["kc"], p["Sc"], p["kc"], p["kd"], p.get("m", 2.0), p.get("fc", 0.6))
    return p["b"] + p["a"] * (S / p["Sc"])


def drought(lat: float, lon: float, name: str = "Study site", data: dict = None, fetch=None, progress=None) -> dict:
    nb = Notebook(progress)
    if data is None:
        v = "precipitation_sum,et0_fao_evapotranspiration,soil_moisture_7_to_28cm_mean"
        data = D.archive(lat, lon, 1991, 2023, daily=v) if fetch is None else D.archive(lat, lon, 1991, 2023, daily=v, fetch=fetch)
    dd = data["daily"]
    keep = [i for i in range(len(dd["time"])) if None not in (dd["precipitation_sum"][i], dd["et0_fao_evapotranspiration"][i], dd["soil_moisture_7_to_28cm_mean"][i])]
    if len(keep) < 365 * 6:
        raise NoModel(f"Too little soil-moisture data: {len(keep)} days (ERA5-Land has no land cell here, typical of open sea and some coasts).")
    t = [dd["time"][i] for i in keep]
    P = np.array([dd["precipitation_sum"][i] for i in keep], float); E0 = np.array([dd["et0_fao_evapotranspiration"][i] for i in keep], float)
    th = np.array([dd["soil_moisture_7_to_28cm_mean"][i] for i in keep], float)
    doy = _doy(t); n = len(th); cut = int(n * 0.6); burn = 365
    nb("1", "Problem", f'Can a water balance of the root zone reproduce the wet and dry spells of {name} ({lat:.3f}, {lon:.3f}) from rain and evaporative demand alone, and does it find the droughts? {n / 365.25:.0f} years of daily data.')
    nb("2", "Choice of physics", "Agricultural and hydrological drought is a deficit of soil water. The engine solves a conservative soil-store balance (rain in; evapotranspiration, drainage and runoff out) with water-stressed evaporation, and compares it with the ERA5-Land root-zone soil moisture. The measured quantity is not rain (a poor drought indicator on its own) but stored water.")
    fits = _fit_all({"M0": DR["M0"]}, lambda k, x: _soil_sim(k, x, P, E0, doy), th, cut, burn)
    f0 = fits["M0"]
    nb("3", "Model selection, round 1", f'Fitted on the first {cut / 365.25:.0f} years: M0 ({DR["M0"]["label"]}), training RMSE {f0["rms"]:.4f} m³/m³.')
    res0 = _soil_sim("M0", f0["x"], P, E0, doy)[burn:cut] - th[burn:cut]
    clim_tr = _doy_clim(th, doy, cut, burn)
    seas = float(np.corrcoef(res0, np.cos(2 * math.pi * (doy[burn:cut] - 1) / 365.0))[0, 1]) ** 2 + float(np.corrcoef(res0, np.sin(2 * math.pi * (doy[burn:cut] - 1) / 365.0))[0, 1]) ** 2
    nb("4", "Diagnosis", f'The first model\'s residuals carry a seasonal signal (R² ≈ {seas:.2f}) — vegetation water use, not captured by a constant coefficient. The engine adds free drainage curvature, then a seasonal evaporation term, and refits.' if seas > 0.05 else
       f'The first model\'s residuals have no strong seasonal signal (R² ≈ {seas:.2f}); the engine still tests the richer drainage structure.')
    more = _fit_all({k: DR[k] for k in ("M1", "M2")}, lambda k, x: _soil_sim(k, x, P, E0, doy), th, cut, burn)
    fits.update(more)
    best = min(fits, key=lambda k: fits[k]["aic"])
    nb("5", "Chosen structure", "AIC on the fitting period: " + "; ".join(f"{k} {fits[k]['aic']:.0f}" for k in fits) + f". Chosen: {best} ({DR[best]['label']}).")

    sim = _soil_sim(best, fits[best]["x"], P, E0, doy)
    te = slice(cut, n)
    o, s = th[te], sim[te]
    base = np.array([clim_tr[int(d)] for d in doy[te]])
    clim_sim = _doy_clim(sim, doy, cut, burn)
    anom_o = o - base
    anom_s = s - np.array([clim_sim[int(d)] for d in doy[te]])
    thr = np.percentile((th[burn:cut] - np.array([clim_tr[int(d)] for d in doy[burn:cut]])), 10)
    thr_s = np.percentile((sim[burn:cut] - np.array([clim_sim[int(d)] for d in doy[burn:cut]])), 10)
    cont = metrics.contingency(anom_o < thr, anom_s < thr_s)
    val = {"n_days": int(len(o)), "r": metrics.pearson_r(o, s), "rmse": metrics.rmse(o, s), "nse": metrics.nse(o, s), "bias": metrics.bias(o, s), "r_anom": metrics.pearson_r(anom_o, anom_s),
           "rmse_clim": metrics.rmse(o, base), "skill_vs_clim": metrics.skill_vs(o, s, base), "drought_csi": cont["csi"], "drought_pod": cont["pod"], "drought_far": cont["far"], "mean_obs": float(o.mean())}
    skilful = val["skill_vs_clim"] > 0.05 and val["r_anom"] > 0.5
    nb("6", "Validation on unseen years", f'Last {len(o) / 365.25:.0f} years: correlation {val["r"]:.2f}, RMSE {val["rmse"]:.3f} m³/m³, anomaly correlation {val["r_anom"]:.2f}, skill {val["skill_vs_clim"]:+.0%} against the calendar-day climatology. Drought days (driest 10 % of anomalies) were found with probability {val["drought_pod"]:.0%}, false-alarm ratio {val["drought_far"]:.0%}.'
       + (" The model beats climatology." if skilful else " The model does NOT clearly beat the climatology of the same calendar day, so its drought signal should not be trusted yet."), warn=not skilful)
    bench = bucket.run_benchmarks()
    nfail = sum(1 for b in bench if not b["ok"])
    nb("7", "Verification", f'{len(bench) - nfail} of {len(bench)} tests passed: water balance closes to machine precision, the store stays within bounds, steady rain settles at the analytic equilibrium, reservoir recession is geometric.' if not nfail else f"{nfail} verification test(s) FAILED.", warn=bool(nfail))
    pn = DR[best]["names"]; xb = fits[best]["x"]

    def f_(p):
        x = list(xb)
        for k in ("Sc", "kc"):
            x[pn.index(k)] = xb[pn.index(k)] * p[k]
        s2 = _soil_sim(best, x, P[te.start - 365:te.stop] * p["P"], E0[te.start - 365:te.stop] * p["E"], doy[te.start - 365:te.stop])[365:]
        return float(np.mean(s2))
    sens = calib.oat_sensitivity(f_, {"Sc": 1.0, "kc": 1.0, "P": 1.0, "E": 1.0}, {"Sc": (0.6, 1.4), "kc": (0.7, 1.3), "P": (0.8, 1.2), "E": (0.8, 1.2)})
    for r in sens:
        r["label"] = {"Sc": "Root-zone storage capacity", "kc": "Crop/vegetation coefficient", "P": "Rainfall", "E": "Evaporative demand"}[r["param"]]
    nb("8", "Sensitivity", f'Mean soil moisture on the test period moves most with {sens[0]["label"].lower()} (swing {sens[0]["swing"]:.3f} m³/m³).')
    x_ = np.arange(n) / 365.25 + 1991
    status = "FAILED verification" if nfail else ("verified, calibrated and validated against ERA5-Land soil moisture (a model product)" if skilful else "verified and calibrated; validation shows no skill over climatology")
    blocks = [("h2", "question", "The question"),
              ("p", f'Does {name} sit in a drought, how often, and can a physical water balance find the dry spells on its own? Drought is a shortage of water stored in the soil; the engine therefore models storage, not rain.'),
              ("h2", "physics", "The physics the engine chose"),
              ("p", "Soil store S with capacity Sc and wetness s = S/Sc: **ET = k_c E₀ min(1, s/f_c)**, **D = k_d Sc s^m**, **Q = max(0, S + P − ET − D − Sc)**, and **S′ = min(Sc, S + P − ET − D)**, so that **P = ET + D + Q + ΔS** exactly. Soil moisture θ is a linear map of s."),
              ("basis",),
              ("h2", "selection", "Competing structures"),
              ("table", ["Model", "Structure", "Parameters", "AIC (fitting years)"], [[k, DR[k]["label"], len(DR[k]["names"]), f'{fits[k]["aic"]:.0f}'] for k in fits], f"Chosen on the fitting period only: {best}. Fitted values: " + "; ".join(f"{a} = {b:.3g}" for a, b in zip(pn, xb)) + "."),
              ("h2", "validation", "How it performs on years it never saw"),
              ("fig", plots.series_fig(x_[te][:1500], {"ERA5-Land soil moisture": o[:1500], "Water-balance model": s[:1500], "Calendar-day climatology": base[:1500]}, "soil moisture (m³/m³)", "First four years of the held-out period"), "A model must do better than simply knowing the usual wetness of each calendar day."),
              ("table", ["Measure", "Model", "Climatology"], [["RMSE (m³/m³)", f'{val["rmse"]:.3f}', f'{val["rmse_clim"]:.3f}'], ["Anomaly correlation", f'{val["r_anom"]:.2f}', "0 (by construction)"], ["Drought-day detection (driest 10 %)", f'POD {val["drought_pod"]:.0%} · FAR {val["drought_far"]:.0%} · CSI {val["drought_csi"]:.2f}', "–"]], "Anomaly = departure from the calendar-day climatology; a drought day is one in the driest 10 % of anomalies.")]
    head = [(f'{val["r_anom"]:.2f}', "anomaly correlation on unseen years"), (f'{val["skill_vs_clim"]:+.0%}', "skill vs climatology"), (f'{val["drought_pod"]:.0%}', "drought days detected"), (f'{val["drought_far"]:.0%}', "false-alarm ratio"), (f'{val["rmse"]:.3f}', "RMSE (m³/m³)")]
    steps = ["Supply soil-moisture observations (in-situ probes, SMAP satellite) so the model is validated against measurements rather than a reanalysis."]
    if not skilful:
        steps.append("The model does not beat climatology: add the missing process (irrigation, groundwater, vegetation phenology) before using it for drought early warning.")
    steps.append(f"Sensitivity says the result depends most on {sens[0]['label'].lower()}; measure or constrain it first.")
    return {"domain": "drought", "label": "Drought", "kind": "series", "name": name, "lat": lat, "lon": lon, "title": f"Drought model for {name}: soil water, not rain", "status": status, "ok": nfail == 0,
            "standfirst": f'The engine built a soil water balance for {name}, chose among three structures on the early years, and tested it on the last {len(o) / 365.25:.0f}: anomaly correlation {val["r_anom"]:.2f}, skill {val["skill_vs_clim"]:+.0%} against climatology. Status: {status}.',
            "headline": head, "blocks": blocks, "verification": bench, "sensitivity": sens, "sens_unit": "m³/m³ of mean soil moisture",
            "assumptions": ["One bucket for the root zone; no irrigation, groundwater, snow or lateral flow.", "Evaporative demand is FAO-56 reference ET from ERA5; actual vegetation use is calibrated through the coefficients."],
            "limitations": ["The 'observations' are ERA5-Land soil moisture, itself a land-surface model, not probes or satellites.", "Agricultural drought in irrigated areas is decided by water management, which this model cannot see."],
            "next_steps": steps, "notebook": nb.rows, "numbers": {"validation": val, "structure": best, "params": dict(zip(pn, xb))}}


# ── river flow ───────────────────────────────────────────────────────────────

RV = {
    "R0": {"label": "One soil store, one river reservoir", "names": ["Sc", "kc", "kd", "g", "k", "A"], "x0": [200, 0.8, 0.03, 0.5, 0.1, 1.0], "lo": [40, 0.2, 0.001, 0.0, 0.005, 1e-3], "hi": [700, 1.6, 0.3, 1.0, 0.9, 1e5]},
    "R1": {"label": "Fast and slow reservoirs in parallel", "names": ["Sc", "kc", "kd", "g", "kf", "ks", "wf", "A"], "x0": [200, 0.8, 0.03, 0.5, 0.4, 0.02, 0.5, 1.0], "lo": [40, 0.2, 0.001, 0.0, 0.1, 0.002, 0.05, 1e-3], "hi": [700, 1.6, 0.3, 1.0, 0.95, 0.2, 0.95, 1e5]},
}


def _flow_sim(key, x, P, E0):
    p = dict(zip(RV[key]["names"], x))
    S, ET, Dr, Q = bucket.soil(P, E0, p["Sc"], p["kc"], p["kd"], 2.0, 0.6)
    if key == "R0":
        return bucket.route(Q, Dr, p["g"], p["k"], p["A"])
    return bucket.route2(Q, Dr, p["g"], p["kf"], p["ks"], p["wf"], p["A"])


def river(lat: float, lon: float, name: str = "Study site", data: dict = None, flow: dict = None, fetch=None, progress=None) -> dict:
    nb = Notebook(progress)
    obs_src = "GloFAS v4 river-discharge reanalysis (Copernicus/ECMWF), a model product, not a gauge"
    if data is None:
        v = "precipitation_sum,et0_fao_evapotranspiration"
        data = D.archive(lat, lon, 1991, 2023, daily=v) if fetch is None else D.archive(lat, lon, 1991, 2023, daily=v, fetch=fetch)
    if flow is None:
        g = D.usgs_gauge_near(lat, lon, 1991, 2023)
        if g:
            flow = {"series": g["series"]}; obs_src = f'{g["source"]}: gauge {g["site"]} {g["name"].title()}, {g["km"]:.0f} km from the site'
        else:
            j = D.river_discharge(lat, lon, 1991, 2023)
            if "daily" not in j:
                raise NoModel("No river discharge data exist for this exact location (the GloFAS river network has no river cell here, typical of deserts and sites away from a river): " + str(j.get("reason", "no data"))[:120])
            flow = {"series": {t: v for t, v in zip(j["daily"]["time"], j["daily"]["river_discharge"]) if v is not None}}
    dd = data["daily"]
    idx = [i for i, t in enumerate(dd["time"]) if t in flow["series"] and dd["precipitation_sum"][i] is not None and dd["et0_fao_evapotranspiration"][i] is not None]
    if len(idx) < 365 * 6:
        raise NoModel(f"Too little overlapping data to calibrate: only {len(idx)} days with both rainfall and river discharge.")
    t = [dd["time"][i] for i in idx]
    P = np.array([dd["precipitation_sum"][i] for i in idx], float); E0 = np.array([dd["et0_fao_evapotranspiration"][i] for i in idx], float)
    q = np.array([flow["series"][x] for x in t], float)
    doy = _doy(t); n = len(q); cut = int(n * 0.6); burn = 365
    nb("1", "Problem", f'Can rain falling at {name} ({lat:.3f}, {lon:.3f}), routed through soil and a river reservoir, reproduce the river\'s flow and its floods? {n / 365.25:.0f} years of daily data; observed flow: {obs_src}.')
    nb("2", "Choice of physics", "A conceptual rainfall–runoff model: a soil store partitions rain into evaporation, drainage and saturation-excess runoff (water balance exact), and linear reservoirs delay the water on its way to the river. "
                                 "Calibration works on the square root of flow so that floods and low flows both count.")
    tf = np.sqrt
    import copy
    rv = copy.deepcopy(RV)
    a0 = float(np.mean(q[burn:cut]) / max(np.mean(P[burn:cut]) * 0.5, 0.1))
    for k_ in rv:
        i_ = rv[k_]["names"].index("A")
        rv[k_]["x0"][i_] = a0; rv[k_]["lo"][i_] = a0 / 300.0; rv[k_]["hi"][i_] = a0 * 300.0
    RV_ = rv
    fits = _fit_all({"R0": RV_["R0"]}, lambda k, x: _flow_sim(k, x, P, E0), q, cut, burn, transform=tf, restarts=1)
    nb("3", "Model selection, round 1", f'R0 ({RV["R0"]["label"]}) fitted on the first {cut / 365.25:.0f} years; training RMSE {fits["R0"]["rms"]:.2f} √(m³/s).')
    r0 = tf(_flow_sim("R0", fits["R0"]["x"], P, E0)[burn:cut]) - tf(q[burn:cut])
    ac = metrics.pearson_r(r0[1:], r0[:-1])
    nb("4", "Diagnosis", f'Day-to-day residual autocorrelation is {ac:.2f}: ' + ("the single reservoir cannot reproduce both the quick flood response and the slow recession, so a fast-and-slow structure is added." if ac > 0.5 else "the single reservoir already captures the timing; the two-reservoir structure is still tested."))
    fits.update(_fit_all({"R1": RV_["R1"]}, lambda k, x: _flow_sim(k, x, P, E0), q, cut, burn, transform=tf, restarts=1))
    best = min(fits, key=lambda k: fits[k]["aic"])
    nb("5", "Chosen structure", "AIC on the fitting period: " + "; ".join(f"{k} {fits[k]['aic']:.0f}" for k in fits) + f". Chosen: {best} ({RV[best]['label']}).")
    sim = _flow_sim(best, fits[best]["x"], P, E0)
    te = slice(cut, n)
    o, s = q[te], sim[te]
    clim = _doy_clim(q, doy, cut, burn)
    base = np.array([clim[int(d)] for d in doy[te]])
    thr = float(np.percentile(q[burn:cut], 95))
    cont = metrics.contingency(o > thr, s > thr)
    yrs = np.array([int(x[:4]) for x in t])[te]
    ay = sorted(set(yrs.tolist()))
    pk_o = np.array([o[yrs == y].max() for y in ay]); pk_s = np.array([s[yrs == y].max() for y in ay])
    val = {"n_days": int(len(o)), "nse": metrics.nse(o, s), "kge": metrics.kge(o, s), "r": metrics.pearson_r(o, s), "pbias": metrics.nmb(o, s), "nse_clim": metrics.nse(o, base),
           "skill_vs_clim": metrics.skill_vs(o, s, base), "flood_csi": cont["csi"], "flood_pod": cont["pod"], "flood_far": cont["far"], "flood_thr": thr, "peak_r": metrics.pearson_r(pk_o, pk_s), "peak_bias": metrics.nmb(pk_o, pk_s)}
    skilful = val["skill_vs_clim"] > 0.05 and val["nse"] > 0.3
    nb("6", "Validation on unseen years", f'Last {len(o) / 365.25:.0f} years: NSE {val["nse"]:.2f}, KGE {val["kge"]:.2f}, volume bias {val["pbias"]:+.0%}; skill {val["skill_vs_clim"]:+.0%} against the calendar-day climatology of flow. Flood days (flow above the fitting-period 95th percentile, {thr:.0f} m³/s): detected {val["flood_pod"]:.0%}, false-alarm ratio {val["flood_far"]:.0%}; annual-peak correlation {val["peak_r"]:.2f}.'
       + (" The model beats climatology." if skilful else " The model does NOT clearly beat climatology. For a river this size local rain is the wrong forcing: the water comes from the catchment upstream."), warn=not skilful)
    bench = bucket.run_benchmarks()
    nfail = sum(1 for b in bench if not b["ok"])
    nb("7", "Verification", f'{len(bench) - nfail} of {len(bench)} tests passed (water balance, bounds, equilibrium, recession).' if not nfail else f"{nfail} verification test(s) FAILED.", warn=bool(nfail))
    pn = RV[best]["names"]; xb = fits[best]["x"]

    def f_(p):
        x = list(xb)
        x[pn.index("Sc")] = xb[pn.index("Sc")] * p["Sc"]
        sm = _flow_sim(best, x, P[te.start - 365:te.stop] * p["P"], E0[te.start - 365:te.stop])[365:]
        return float(np.percentile(sm, 99))
    sens = calib.oat_sensitivity(f_, {"Sc": 1.0, "P": 1.0}, {"Sc": (0.6, 1.4), "P": (0.8, 1.2)})
    for r in sens:
        r["label"] = {"Sc": "Soil storage capacity", "P": "Rainfall"}[r["param"]]
    nb("8", "Sensitivity", f'The 99th-percentile (flood-level) flow moves most with {sens[0]["label"].lower()} (swing {sens[0]["swing"]:.0f} m³/s).')
    x_ = np.arange(n) / 365.25 + 1991
    gauge = "USGS" in obs_src
    status = "FAILED verification" if nfail else (("verified, calibrated and validated against a gauge" if gauge else "verified, calibrated and validated against GloFAS (a model product, not a gauge)") if skilful else "verified and calibrated; validation shows no skill over climatology")
    blocks = [("h2", "question", "The question"),
              ("p", f'How does the river near {name} respond to rain, and can the engine reproduce its floods from physics and rainfall alone? A flood forecast is only as good as its hindcast, so the engine scores itself on years it never saw.'),
              ("h2", "physics", "The physics the engine chose"),
              ("p", "A soil store partitions rain (**P = ET + D + Q + ΔS**, exact) into runoff Q and drainage D; the water reaches the river through linear reservoirs, **R′ = R + (Q + gD) − kR, flow = A k R**. A second, slower reservoir is tried when one cannot fit both floods and recession."),
              ("basis",),
              ("h2", "selection", "Competing structures"),
              ("table", ["Model", "Structure", "Parameters", "AIC (fitting years)"], [[k, RV[k]["label"], len(RV[k]["names"]), f'{fits[k]["aic"]:.0f}'] for k in fits], f"Chosen on the fitting period only: {best}."),
              ("h2", "validation", "How it performs on years it never saw"),
              ("fig", plots.series_fig(x_[te][:1100], {f"Observed ({'gauge' if gauge else 'GloFAS'})": o[:1100], "Model": s[:1100], "Climatology": base[:1100]}, "discharge (m³/s)", "Held-out period: the first three years"), "Observed flow against the engine's model and against the usual flow of each calendar day."),
              ("table", ["Measure", "Model", "Climatology"], [["Nash–Sutcliffe efficiency", f'{val["nse"]:.2f}', f'{val["nse_clim"]:.2f}'], ["Kling–Gupta efficiency", f'{val["kge"]:.2f}', "–"], ["Volume bias", f'{val["pbias"]:+.0%}', "–"],
                                                          [f"Flood days (> {thr:.0f} m³/s)", f'POD {val["flood_pod"]:.0%} · FAR {val["flood_far"]:.0%} · CSI {val["flood_csi"]:.2f}', "–"], ["Annual-peak correlation", f'{val["peak_r"]:.2f}', "–"]], "NSE: 1 is perfect, 0 is no better than the mean flow.")]
    head = [(f'{val["nse"]:.2f}', "Nash–Sutcliffe on unseen years"), (f'{val["kge"]:.2f}', "Kling–Gupta efficiency"), (f'{val["skill_vs_clim"]:+.0%}', "skill vs climatology"), (f'{val["flood_pod"]:.0%}', "flood days detected"), (f'{val["peak_r"]:.2f}', "annual-peak correlation")]
    steps = []
    if not gauge:
        steps.append("Supply gauge observations so the model is validated against measured discharge, not a reanalysis.")
    if not skilful:
        steps.append("Use catchment-average rainfall (the whole upstream area) instead of rain at one point; for a large river this alone decides the skill.")
    steps.append("Add a flood-frequency analysis of the simulated annual peaks and connect it to the 2-D flood model for inundation.")
    return {"domain": "river", "label": "River flood", "kind": "series", "name": name, "lat": lat, "lon": lon, "title": f"River-flow model for {name}: can rain at one place explain the river?", "status": status, "ok": nfail == 0,
            "standfirst": f'The engine built a rainfall–runoff model for the river near {name} and tested it on {len(o) / 365.25:.0f} unseen years: Nash–Sutcliffe {val["nse"]:.2f}, flood days detected {val["flood_pod"]:.0%}. Status: {status}.',
            "headline": head, "blocks": blocks, "verification": bench, "sensitivity": sens, "sens_unit": "m³/s at the 99th percentile",
            "assumptions": ["Rain and evaporative demand at the site stand for the whole contributing area.", "Flow is generated by one soil store and linear reservoirs; no dams, abstraction, floodplain storage or snowmelt."],
            "limitations": [f"Observed flow: {obs_src}.", "At a large river the local rainfall explains little of the flow; the result then says so rather than hiding it."],
            "next_steps": steps, "notebook": nb.rows, "numbers": {"validation": val, "structure": best, "params": dict(zip(pn, xb)), "source": obs_src}}
