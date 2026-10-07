"""
Air-quality study: the engine builds a PM2.5 / AQI model of a city, chooses between competing model structures, and tests it on data it never saw.

Competing structures are fitted on the first 60 % of the record and ranked by AIC there (no peeking); the winner is then scored on the last 40 %
against two naive baselines (the mean diurnal cycle, and persistence). A model that cannot beat climatology is reported as not skilful.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

from . import air, calib, metrics

NOW = time.time

STRUCTURES = {
    "M0": {"label": "Constant emission", "params": ["E", "Cb"], "x0": [0.05, 15.0], "lo": [0.0, 0.0], "hi": [2.0, 120.0]},
    "M1": {"label": "Diurnal emission (traffic/cooking cycle)", "params": ["E", "Cb", "amp", "phase_h"], "x0": [0.05, 15.0, 0.4, 8.0], "lo": [0.0, 0.0, 0.0, 0.0], "hi": [2.0, 120.0, 0.95, 24.0]},
    "M2": {"label": "Diurnal emission + mixing-height correction", "params": ["E", "Cb", "amp", "phase_h", "fH"], "x0": [0.05, 15.0, 0.4, 8.0, 1.0], "lo": [0.0, 0.0, 0.0, 0.0, 0.4], "hi": [2.0, 120.0, 0.95, 24.0, 2.5]},
}

# Added only when the residuals of the first round show the missing process (see diagnose()).
EXTRA = {
    "M3": {"label": "M2 + slowly drifting regional background", "params": ["E", "Cb", "amp", "phase_h", "fH", "bslope"], "x0": [0.05, 15.0, 0.4, 8.0, 1.0, 0.0],
           "lo": [0.0, 0.0, 0.0, 0.0, 0.4, -15.0], "hi": [2.0, 150.0, 0.95, 24.0, 2.5, 15.0]},
    "M4": {"label": "M2 + wind-dependent ventilation (flushing ∝ u^p)", "params": ["E", "Cb", "amp", "phase_h", "fH", "p"], "x0": [0.05, 15.0, 0.4, 8.0, 1.0, 1.0],
           "lo": [0.0, 0.0, 0.0, 0.0, 0.4, 0.0], "hi": [2.0, 150.0, 0.95, 24.0, 2.5, 2.0]},
    "M5": {"label": "M2 + drifting background + wind-dependent ventilation", "params": ["E", "Cb", "amp", "phase_h", "fH", "bslope", "p"], "x0": [0.05, 15.0, 0.4, 8.0, 1.0, 0.0, 1.0],
           "lo": [0.0, 0.0, 0.0, 0.0, 0.4, -15.0, 0.0], "hi": [2.0, 150.0, 0.95, 24.0, 2.5, 15.0, 2.0]},
}
ALL = {**STRUCTURES, **EXTRA}


def _simulate(key: str, x, hours: np.ndarray, H: np.ndarray, u: np.ndarray, L: float, vd: float, c0: float, idx0: int = 0) -> np.ndarray:
    p = dict(zip(ALL[key]["params"], x))
    amp, ph = p.get("amp", 0.0), p.get("phase_h", 0.0)
    E = p["E"] * (1 + amp * np.cos(2 * math.pi * (hours - ph) / 24.0))
    Cb = np.maximum(p["Cb"] + p.get("bslope", 0.0) * (idx0 + np.arange(len(hours))) / 24.0, 0.0)
    ue = u if "p" not in p else 2.5 * (np.maximum(u, 0.3) / 2.5) ** p["p"]
    return air.box_series(E, H * p.get("fH", 1.0), ue, L, Cb, vd, c0=c0)


def diagnose(resid: np.ndarray, u: np.ndarray) -> dict:
    """What process is the first-round model missing? Look at the structure left in its training residuals."""
    n = len(resid)
    third = max(n // 3, 1)
    sd = float(np.std(resid)) or 1.0
    drift = float(np.mean(resid[:third]) - np.mean(resid[-third:])) / sd
    rw = metrics.pearson_r(resid, u)
    ac24 = metrics.pearson_r(resid[24:], resid[:-24]) if n > 48 else float("nan")
    return {"drift_sd": drift, "r_wind": rw, "ac24": ac24, "needs_drift": abs(drift) > 0.5, "needs_wind": bool(np.isfinite(rw) and abs(rw) > 0.3)}


def run(lat: float, lon: float, name: str = "Study city", city_km: float = 20.0, vd: float = 0.002, data: Optional[dict] = None, fetch: Optional[Callable] = None,
        obs: Optional[dict] = None, progress: Optional[Callable] = None, days: int = 60) -> dict:
    nb: list = []
    t_start = NOW()

    def note(step, title, detail, **kw):
        nb.append({"step": step, "title": title, "detail": detail, "t": round(NOW() - t_start, 1), **kw})
        if progress:
            try:
                progress(f"{step}: {title}")
            except Exception:                                         # noqa: BLE001
                pass

    if data is None:
        from . import data as D
        data = D.hourly_met_and_pm(lat, lon, days=days, fetch=fetch) if fetch else D.hourly_met_and_pm(lat, lon, days=days)
    rows = data["rows"]
    t = [r["t"] for r in rows]
    hours = np.array([int(s[11:13]) for s in t], float)
    H = np.array([r["blh"] for r in rows], float)
    u = np.array([r["wind"] for r in rows], float)
    pm = np.array([r["pm25"] for r in rows], float)
    obs_source = data.get("source", "")
    if obs and obs.get("pm25"):
        pm = np.asarray(obs["pm25"], float)
        obs_source = obs.get("source", "supplied ground-station series")
        truth_is_station = True
    else:
        truth_is_station = False
    n = len(pm)
    cut = int(n * 0.6)
    L = city_km * 1000.0
    note("1", "Problem and scale", f"Can a physical box model of {name} reproduce its PM2.5 and the AQI category from wind and boundary-layer depth alone? {n} hourly values ({n / 24:.0f} days); the city is treated as a well-mixed box {city_km:g} km long.")
    note("2", "Choice of physics", "Mass balance of a well-mixed urban box: emissions into a layer as deep as the boundary layer, flushed by the wind over the city's length, removed by deposition. "
                                   "A Gaussian plume describes the near-field of one stack but cannot describe a whole city; a chemical-transport model needs an emission inventory this study does not have. "
                                   "Three nested structures are compared first; the engine then diagnoses what the best one gets wrong on the training data and adds exactly that process, so the data, not the author, decide how much physics is needed.")

    # build + calibrate
    fits = {}
    tr = slice(0, cut)

    def fit_structures(keys):
        for key in keys:
            spec = ALL[key]

            def resid(x, key=key):
                return _simulate(key, x, hours[tr], H[tr], u[tr], L, vd, pm[0]) - pm[tr]
            f = calib.least_squares_fit(resid, spec["x0"], (spec["lo"], spec["hi"]))
            rss = float(np.sum(resid(f["x"]) ** 2))
            f["aic"] = calib.aic(cut, rss, len(spec["params"]))
            fits[key] = f
    fit_structures(list(STRUCTURES))
    best1 = min(fits, key=lambda k: fits[k]["aic"])
    note("3", "Model selection, round 1", "Fitted on the first 60 % of the record and ranked by AIC (no look at the held-out 40 %): " +
         "; ".join(f"{k} {STRUCTURES[k]['label']} AIC {fits[k]['aic']:.0f}" for k in fits) + f". Best so far: {best1}.")

    # round 2: diagnose what the best model is missing, then add exactly that process
    res1 = _simulate(best1, fits[best1]["x"], hours[tr], H[tr], u[tr], L, vd, pm[0]) - pm[tr]
    dg = diagnose(res1, u[tr])
    add = []
    if dg["needs_drift"] and dg["needs_wind"]:
        add = ["M3", "M4", "M5"]
    elif dg["needs_drift"]:
        add = ["M3"]
    elif dg["needs_wind"]:
        add = ["M4"]
    why = (f"the training residuals drift by {dg['drift_sd']:+.1f} standard deviations from start to end (the regional background is not constant)" if dg["needs_drift"] else "") + \
          (" and " if dg["needs_drift"] and dg["needs_wind"] else "") + \
          (f"correlate with wind speed (r = {dg['r_wind']:+.2f}; the flushing is not simply proportional to wind)" if dg["needs_wind"] else "")
    if add:
        fit_structures(add)
        note("4", "Diagnosis and model revision", f"Looking at what the best model gets wrong on the training data: {why}. The engine added " + ", ".join(f"{k} ({ALL[k]['label']})" for k in add) +
             f" and refit. New AICs: " + "; ".join(f"{k} {fits[k]['aic']:.0f}" for k in add) + ".")
    else:
        note("4", "Diagnosis", f"The training residuals show no drift ({dg['drift_sd']:+.1f} sd) and no dependence on wind (r = {dg['r_wind']:+.2f}); no structure was added.")
    best = min(fits, key=lambda k: fits[k]["aic"])
    note("5", "Chosen structure", f"Lowest AIC on the training data: {best} ({ALL[best]['label']}).")

    # validation on held-out
    te = slice(cut, n)
    sim_all = _simulate(best, fits[best]["x"], hours, H, u, L, vd, pm[0])
    sim_te = _simulate(best, fits[best]["x"], hours[te], H[te], u[te], L, vd, sim_all[cut - 1], idx0=cut)      # restart from the model state, never from observations
    o_te = pm[te]
    clim = np.array([np.mean(pm[:cut][hours[:cut] == h]) for h in range(24)])
    base_clim = clim[hours[te].astype(int)]
    base_pers = pm[cut - 24:n - 24]                                         # same hour yesterday: the fair no-model forecast over a day or more
    val = {"n": int(o_te.size), "r": metrics.pearson_r(o_te, sim_te), "rmse": metrics.rmse(o_te, sim_te), "nmb": metrics.nmb(o_te, sim_te), "fac2": metrics.fac2(o_te, sim_te),
           "mean_obs": float(np.mean(o_te)), "mean_sim": float(np.mean(sim_te)), "rmse_clim": metrics.rmse(o_te, base_clim), "rmse_pers": metrics.rmse(o_te, base_pers),
           "skill_vs_clim": metrics.skill_vs(o_te, sim_te, base_clim), "skill_vs_pers": metrics.skill_vs(o_te, sim_te, base_pers)}
    # daily AQI agreement
    nd = o_te.size // 24
    if nd >= 2:
        od = o_te[:nd * 24].reshape(nd, 24).mean(1); sd = sim_te[:nd * 24].reshape(nd, 24).mean(1)
        oc = [air.aqi_category(air.aqi_pm25(v)) for v in od]; sc = [air.aqi_category(air.aqi_pm25(v)) for v in sd]
        val["aqi_days"] = nd
        val["aqi_category_hit"] = float(np.mean([a == b for a, b in zip(oc, sc)]))
        val["aqi_obs_mean"] = float(np.mean([air.aqi_pm25(v) for v in od])); val["aqi_sim_mean"] = float(np.mean([air.aqi_pm25(v) for v in sd]))
    cand = {}
    for k, f in fits.items():
        s_k = _simulate(k, f["x"], hours[te], H[te], u[te], L, vd, _simulate(k, f["x"], hours[:cut], H[:cut], u[:cut], L, vd, pm[0])[-1], idx0=cut)
        cand[k] = {"test_rmse": metrics.rmse(o_te, s_k), "test_r": metrics.pearson_r(o_te, s_k)}
    val["candidates"] = cand
    mse = float(np.mean((sim_te - o_te) ** 2))
    val["bias_share"] = float(np.mean(sim_te - o_te) ** 2 / mse) if mse > 0 else 0.0
    skilful = val["skill_vs_clim"] > 0.05 and val["skill_vs_pers"] > 0 and val["r"] > 0.5
    note("6", "Validation on unseen data", f"Last {val['n']} hours, never used in fitting: r = {val['r']:.2f}, RMSE {val['rmse']:.1f} µg/m³, bias {val['nmb']:+.0%}, {val['fac2']:.0%} of hours within a factor of two. "
                                          f"Skill against the mean diurnal cycle {val['skill_vs_clim']:+.0%}, against yesterday-same-hour persistence {val['skill_vs_pers']:+.0%}. "
                                          + ("The model beats the naive baselines." if skilful else "The model does NOT clearly beat the naive baselines, so it should not be used to forecast."), warn=not skilful)

    if val["bias_share"] > 0.4:
        note("6b", "Anatomy of the error", f"{val['bias_share']:.0%} of the squared error on unseen hours is a plain level offset (bias {val['nmb']:+.0%}), not a wrong pattern (r = {val['r']:.2f}). "
                                          "That is the signature of a regional background that changed between the fitting and the test period: smoke or haze arriving from upwind, which a box with a constant background cannot know about. "
                                          "The missing physics is regional transport; a longer record or an upwind concentration would address it.", warn=True)
    # verification
    t0 = NOW()
    bench = air.run_benchmarks()
    verification = bench
    nfail = sum(1 for v in verification if not v["ok"])
    note("7", "Verification", f"{len(verification) - nfail} of {len(verification)} physics tests passed: the plume formula matches its closed form and conserves mass, the box model reaches its analytic steady state and decays at the ventilation rate, the AQI reproduces the EPA breakpoints." if not nfail else f"{nfail} verification test(s) FAILED.", warn=bool(nfail))

    # sensitivity (mean PM2.5 over the held-out period)
    xb = fits[best]["x"]; pn = ALL[best]["params"]

    def f_(p):
        x = list(xb)
        for k in ("E", "Cb"):
            x[pn.index(k)] = xb[pn.index(k)] * p[k]
        s = _simulate(best, x, hours[te], H[te] * p["H"], u[te] * p["u"], p["L"], p["vd"], sim_all[cut - 1])
        return float(np.mean(s))
    sens = calib.oat_sensitivity(f_, {"E": 1.0, "Cb": 1.0, "H": 1.0, "u": 1.0, "L": L, "vd": vd},
                                 {"E": (0.7, 1.3), "Cb": (0.7, 1.3), "H": (0.7, 1.3), "u": (0.8, 1.2), "L": (L * 0.5, L * 2.0), "vd": (vd * 0.25, vd * 4.0)})
    names = {"E": "Emission rate", "Cb": "Regional background", "H": "Mixing-layer depth", "u": "Wind speed", "L": "City length", "vd": "Deposition velocity"}
    for r in sens:
        r["label"] = names[r["param"]]
    note("8", "Sensitivity", f"Mean PM2.5 over the test period is most sensitive to {sens[0]['label'].lower()} (swing {sens[0]['swing']:.1f} µg/m³, elasticity {sens[0]['elasticity']:+.2f}).")

    # near-field plume illustration at the latest stability
    last = rows[min(len(rows) - 1, n - 1)]
    cls = air.stability_class(max(last["wind"], 0.5), last["day"], "moderate", last["cloud"])
    xs = np.array([100, 250, 500, 1000, 2000, 4000, 8000.0])
    plume_gm3 = air.plume(xs, np.zeros_like(xs), 1.5, 100.0, max(last["wind"], 1.0), 40.0, cls)
    plume = {"class": cls, "wind": last["wind"], "x_m": xs.tolist(), "ugm3": (plume_gm3 * 1e6).tolist(), "Q_g_s": 100.0, "H_m": 40.0}

    status = ("FAILED verification" if nfail else ("verified, calibrated and validated against ground stations" if (truth_is_station and skilful) else
              "verified, calibrated and validated against the CAMS analysis (a model product, not stations)" if skilful else "verified and calibrated; validation shows no skill over baselines"))
    assumptions = [f"The city behaves as one well-mixed box {city_km:g} km long; spatial gradients inside it are not resolved.",
                   f"PM2.5 is treated as chemically inert (secondary aerosol formation is absorbed into the emission and background terms); deposition velocity {vd * 100:.1f} cm/s.",
                   "Emission follows a smooth daily cycle (structures M1/M2); episodic sources (fires, dust storms, festivals) are not modelled.",
                   "The regional background is constant over the record."]
    limits = ["The 'observations' are the Copernicus CAMS analysis, itself a model constrained by satellites, not a network of ground monitors. Skill here shows the box model reproduces CAMS, not that it matches the air at street level." if not truth_is_station else
              "Ground-station series supplied by the user were used as truth.",
              f"A {n / 24:.0f}-day record cannot capture seasonal change in emissions or meteorology; repeat the study over a full year before relying on it."]
    return {"domain": "air", "name": name, "lat": lat, "lon": lon, "status": status, "ok": nfail == 0, "skilful": bool(skilful), "structure": best, "structures": {k: {**ALL[k], "fit": fits[k], **cand[k]} for k in fits}, "diagnosis": dg,
            "validation": val, "verification": verification, "sensitivity": sens, "assumptions": assumptions, "limitations": limits, "notebook": nb, "plume": plume, "source": obs_source,
            "series": {"t": t, "obs": pm.tolist(), "sim": sim_all.tolist(), "cut": cut, "test_sim": sim_te.tolist(), "hours": hours.tolist(), "clim": base_clim.tolist()},
            "params": dict(zip(pn, xb)), "city_km": city_km, "elapsed_s": round(NOW() - t_start, 1)}
