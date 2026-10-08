"""Hazards defined by 'the worst value in a year': extreme heat, rainfall, wind and fire weather. Built and hindcast-validated from ERA5."""
from __future__ import annotations

import numpy as np

from . import data as D, evt, fire, heat, plots
from .generic_report import Notebook


def _annual_max_daily(lat, lon, var, y0, y1, scale=1.0, fetch=None):
    d = D.archive(lat, lon, y0, y1, daily=var) if fetch is None else D.archive(lat, lon, y0, y1, daily=var, fetch=fetch)
    by: dict = {}
    for t, v in zip(d["daily"]["time"], d["daily"][var]):
        if v is not None:
            by[int(t[:4])] = max(by.get(int(t[:4]), -1e9), float(v) * scale)
    ys = sorted(by)
    return ys, [by[y] for y in ys]


def _heat_series(lat, lon, fetch=None):
    y0, y1 = 1981, 2023
    d = D.archive(lat, lon, y0, y1, hourly="temperature_2m,relative_humidity_2m") if fetch is None else D.archive(lat, lon, y0, y1, hourly="temperature_2m,relative_humidity_2m", fetch=fetch)
    t = np.array([np.nan if v is None else v for v in d["hourly"]["temperature_2m"]], float)
    rh = np.array([np.nan if v is None else v for v in d["hourly"]["relative_humidity_2m"]], float)
    w = heat.wbgt_shade(t, np.clip(rh, 1, 100))
    yrs = np.array([int(s[:4]) for s in d["hourly"]["time"]])
    ys = sorted(set(yrs.tolist()))
    xs = [float(np.nanmax(w[yrs == y])) for y in ys]
    hrs28 = [int(np.nansum(w[yrs == y] >= 28.0)) for y in ys]
    return ys, xs, {"hours_over_28": hrs28}


def _fire_series(lat, lon, fetch=None):
    y0, y1 = 1981, 2023
    v = "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum"
    d = D.archive(lat, lon, y0, y1, daily=v) if fetch is None else D.archive(lat, lon, y0, y1, daily=v, fetch=fetch)
    dd = d["daily"]
    ok = [i for i in range(len(dd["time"])) if None not in (dd["temperature_2m_max"][i], dd["relative_humidity_2m_min"][i], dd["wind_speed_10m_max"][i], dd["precipitation_sum"][i])]
    months = [int(dd["time"][i][5:7]) for i in ok]
    out = fire.run_series([dd["temperature_2m_max"][i] for i in ok], [dd["relative_humidity_2m_min"][i] for i in ok], [dd["wind_speed_10m_max"][i] for i in ok],
                          [dd["precipitation_sum"][i] for i in ok], months, tropical=abs(lat) < 25)
    yrs = np.array([int(dd["time"][i][:4]) for i in ok]); f = np.array(out["fwi"])
    ys = [y for y in sorted(set(yrs.tolist())) if y > y0]                 # first year is spin-up
    xs = [float(f[yrs == y].max()) for y in ys]
    ex = {"days_over_30": [int((f[yrs == y] >= 30).sum()) for y in ys]}
    return ys, xs, ex


HAZ = {
    "heat": {"label": "Extreme heat", "unit": "WBGT °C", "var": "hourly peak wet-bulb globe temperature (shade)",
             "physics": ["Heat stress is set by how well the body can lose heat to the air, which depends on temperature and humidity together. The wet-bulb globe temperature (WBGT) blends them; the engine uses the Australian Bureau of Meteorology shade form **WBGT = 0.567 T + 0.393 e + 3.94**, where e is the vapour pressure (hPa) from the Magnus relation **e = RH · 6.1094 exp(17.625 T / (T + 243.04))**.",
                         "Work-safety limits are written in WBGT: about 28 °C ends heavy outdoor labour on most guidance, and above about 32 °C even light work is unsafe."],
             "extract": _heat_series, "benchmarks": heat.run_benchmarks, "fmt": "{:.1f}",
             "assumptions": ["Shade, light-wind conditions; direct sun adds several degrees (the engine does not model radiation).", "ERA5 is a 25 km grid cell: urban street canyons run hotter than the cell average."],
             "limits": ["ERA5 humidity and temperature are modelled, not measured; peaks in dry-season heat waves are typically under-stated."],
             "meaning": "Each hour above 28 °C WBGT is an hour in which heavy outdoor work should stop; the count per year is shown above."},
    "rain": {"label": "Extreme rainfall", "unit": "mm per day", "var": "annual maximum daily rainfall",
             "physics": ["Extreme rainfall is a statistical hazard: the largest daily total in each year is a block maximum, and the Fisher–Tippett–Gnedenko theorem says block maxima follow a GEV distribution (Gumbel when the shape parameter is zero). Return level for period T: **x_T = μ − β ln(−ln(1 − 1/T))**.",
                         "Rainfall depth drives the flood model (cri.modellab flood study) through the SCS curve-number runoff relation."],
             "extract": lambda la, lo, fetch=None: (*_annual_max_daily(la, lo, "precipitation_sum", 1950, 2023, 1.0, fetch), {}), "benchmarks": lambda: [], "fmt": "{:.0f}",
             "assumptions": ["Daily totals; short, intense bursts (an hour of cloudburst) are not resolved."], "limits": ["ERA5 smooths convective extremes: the numbers are a floor."], "meaning": "Daily rain beyond the 100-year level overwhelms most urban drainage designed for 10 to 25 years."},
    "wind": {"label": "Extreme wind", "unit": "m/s gust", "var": "annual maximum daily 10 m gust",
             "physics": ["Strong winds also follow a block-maxima law. A structure's wind load scales with the square of speed (**q = ½ ρ V²**), so a 10 % higher gust means about 21 % more load, which is why return levels matter more than averages.",
                         "Tropical cyclones are poorly resolved by a 25 km reanalysis; the cyclone study (cri.modellab cyclone) is the better source where they dominate."],
             "extract": lambda la, lo, fetch=None: (*_annual_max_daily(la, lo, "wind_gusts_10m_max", 1950, 2023, 1 / 3.6, fetch), {}), "benchmarks": lambda: [], "fmt": "{:.1f}",
             "assumptions": ["Gusts at 10 m over open terrain; local exposure and building height change the design wind."], "limits": ["Tornadoes, downbursts and the core of tropical cyclones are not in ERA5."], "meaning": "Design standards use the 50-year gust; assets built to 25 m/s are outside their design envelope whenever a gust passes it."},
    "fire": {"label": "Fire weather", "unit": "FWI", "var": "annual maximum Fire Weather Index",
             "physics": ["The Canadian Fire Weather Index system (Van Wagner 1987) turns daily weather into fuel dryness and spread potential: three moisture codes (fine fuels FFMC, duff DMC, deep drought DC) each dry with heat and wind and wet with rain, combine into the Initial Spread Index (**ISI = 0.208 e^{0.05039 W} f(FFMC)**), the Buildup Index and finally the FWI.",
                         "Values above about 30 mean extreme fire danger in most regimes."],
             "extract": _fire_series, "benchmarks": fire.run_benchmarks, "fmt": "{:.0f}",
             "assumptions": ["The index is driven by daily maximum temperature, minimum humidity, maximum wind and rainfall as a proxy for the noon observation, so it runs slightly harsher than the official noon index.", "Tropical day-length constants are used inside 25° of the equator."],
             "limits": ["Fire weather is not fire: ignition and fuel load decide whether it burns. The index measures how the weather would let a fire spread."], "meaning": "The number of days above FWI 30 each year is the practical exposure."},
}


def run(hazard: str, lat: float, lon: float, name: str = "Study site", series: tuple = None, fetch=None, progress=None, bootstrap: int = 200) -> dict:
    cfg = HAZ[hazard]
    nb = Notebook(progress)
    if series is None:
        got = cfg["extract"](lat, lon, fetch)
        years, x, extra = got[0], np.array(got[1], float), got[2]
        src = "ERA5 reanalysis (ECMWF) via Open-Meteo"
    else:
        years, x, extra = list(series[0]), np.array(series[1], float), (series[2] if len(series) > 2 else {})
        src = "supplied series"
    n = len(x)
    fmt = cfg["fmt"]
    nb("1", "Problem", f'How bad does {cfg["label"].lower()} get at {name} ({lat:.3f}, {lon:.3f}) once in 10, 50 and 100 years, is that estimate trustworthy, and is it getting worse? Measure: {cfg["var"]}, {n} years ({years[0]}–{years[-1]}).')
    nb("2", "Choice of physics", "The hazard is physical (the index is computed from temperature, humidity, wind and rain) but its largest values are rare, so the engine models the annual maxima statistically, with a Generalised Extreme Value family. A Gumbel is kept unless the GEV's extra shape parameter is statistically justified.")
    bm = evt.best_model(x)
    model = bm["params"]
    nb("3", "Model fit", f'{bm["kind"]} fitted by maximum likelihood to {n} annual maxima' + (f' (GEV rejected, LRT p = {bm["lrt_p"]:.2f})' if bm["kind"] == "Gumbel" and bm["lrt_p"] is not None else "") + ".")
    ci = evt.bootstrap_levels(x, n=bootstrap)
    lv = {T: evt.level(model, T) for T in (10, 50, 100)}
    hc = evt.hindcast(x)
    tr = evt.trend_test(years, x)
    cut = hc.get("n_train", int(n * 0.67))
    if hc["ok"]:
        nb("4", "Hindcast validation", f'Fitted on {hc["n_train"]} years ({years[0]}–{years[hc["n_train"] - 1]}), asked to predict {hc["n_test"]} years it never saw. Uniformity test p = {hc["ks_p"]:.2f}; exceedances of the 2-, 5- and 10-year levels: ' +
           "; ".join(f'{r["observed"]} seen vs {r["expected"]:.1f} expected (p = {r["p"]:.2f})' for r in hc["exceedances"]) + ". " + ("The model predicted the held-out years." if hc["passed"] else "The model FAILED to predict the held-out years: " + (hc["bias_note"] or "the distribution is not stable over the record") ) , warn=not hc["passed"])
    nb("5", "Trend", f'Location parameter changes {tr["slope_per_decade"]:+.2g} {cfg["unit"]} per decade (LRT p = {tr["p"]:.3f}); ' + ("significant, so a trend-adjusted 2050 level is reported (an extrapolation)." if tr["significant"] else "not significant: the stationary model is used."))
    t0 = __import__("time").time()
    bench = evt.run_benchmarks() + cfg["benchmarks"]()
    nfail = sum(1 for b in bench if not b["ok"])
    nb("6", "Verification", f'{len(bench) - nfail} of {len(bench)} tests passed: the fitter recovers known parameters, the return-level formula is exact, the hindcast test is calibrated on synthetic stationary records and detects a strong trend' + ("; domain physics checked against hand values and limits." if cfg["benchmarks"]() else ".") if not nfail else f"{nfail} verification test(s) FAILED.", warn=bool(nfail))

    validated = hc["ok"] and hc["passed"]
    status = ("FAILED verification" if nfail else "verified and hindcast-validated" + (", trend detected" if tr["significant"] else "") if validated else "verified; hindcast FAILED (the early record does not predict the recent one)" if hc["ok"] else "verified only (record too short to hindcast)")
    L50 = lv[50]
    tl = evt.trend_level(tr, 2050, 100) if tr["significant"] else None
    head = [(fmt.format(lv[10]) + f' {cfg["unit"]}', "10-year level"), (fmt.format(lv[50]) + f' {cfg["unit"]}', "50-year level"), (fmt.format(lv[100]) + f' {cfg["unit"]}', "100-year level"),
            (f'{tr["slope_per_decade"]:+.2g} per decade', "trend (significant)" if tr["significant"] else "trend (not significant)"), (f'p = {hc["ks_p"]:.2f}' if hc["ok"] else "n/a", "hindcast uniformity test")]
    blocks = [("h2", "question", "The question"),
              ("p", f'How severe does {cfg["label"].lower()} get at {name}, how often, can the estimate be trusted, and is the hazard changing? Instead of fitting a curve and stopping, the engine hides the most recent third of the record, predicts it, and only then reports what it learned.'),
              ("h2", "physics", "The physics the engine chose")] + [("p", t) for t in cfg["physics"]] + [("basis",),
              ("h2", "data", "The data"),
              ("p", f'{n} annual maxima, {years[0]}–{years[-1]}, from {src}.'),
              ("fig", plots.hindcast_plot(years, x, cut, {T: hc["exceedances"][i]["level"] for i, T in enumerate((2, 5, 10))} if hc["ok"] else {}, cfg["unit"], f'{cfg["label"]}: annual maximum, and the levels predicted from the fitting period'),
               "Grey: the fitting period. Orange: the years the model was not shown. Dashed lines: the 2-, 5- and 10-year levels it predicted from the fitting period alone."),
              ("h2", "validation", "Did the model predict the years it never saw?")]
    if hc["ok"]:
        blocks += [("table", ["Level (from fitting period)", "Value", "Exceeded in held-out years", "Expected", "p-value"],
                    [[f'{r["T"]}-year', fmt.format(r["level"]) + f' {cfg["unit"]}', r["observed"], f'{r["expected"]:.1f}', f'{r["p"]:.2f}'] for r in hc["exceedances"]],
                    f'Probability-integral-transform uniformity test p = {hc["ks_p"]:.2f}; p-values below 0.05 would mean the model failed. ' + (hc["bias_note"] or "")),
                   ("callout", ("The hindcast PASSED: the early record predicted the recent years within sampling error." if hc["passed"] else "The hindcast FAILED. " + (hc["bias_note"] or "The early record does not describe the recent years.") + " Do not use the stationary estimate without adjustment."), "good" if hc["passed"] else "bad")]
    else:
        blocks += [("callout", f'Hindcast not possible: {hc["reason"]}.', "warn")]
    blocks += [("h2", "result", "Return levels"),
               ("fig", plots.return_level_plot(x, lambda T: evt.level(model, T), ci, cfg["unit"], f'{cfg["label"]}: how large, how often', (lambda T: evt.trend_level(tr, 2050, T)) if tr["significant"] else None), "Black dots: observed maxima by their empirical return period. Band: 90 % bootstrap range."),
               ("table", ["Return period", "Level", "90 % range"], [[f"{T} years", fmt.format(lv[T]) + f' {cfg["unit"]}', f'{fmt.format(ci[T][0])} – {fmt.format(ci[T][1])}' if T in ci else "n/a"] for T in (10, 50, 100)],
                "Return periods beyond the length of the record are extrapolations; the range widens accordingly." + (" 100 years exceeds the record: treat it as indicative." if n < 100 else "")),
               ("p", f'Trend: the typical annual maximum changes by {tr["slope_per_decade"]:+.2g} {cfg["unit"]} per decade (p = {tr["p"]:.3f}). ' + (f'Extending that trend to 2050 would put the 100-year level at {fmt.format(tl)} {cfg["unit"]}, against {fmt.format(lv[100])} now; this is an extrapolation of a linear trend, not a climate projection.' if tr["significant"] else "The change is not distinguishable from noise in this record, so the stationary model stands.")),
               ("p", cfg["meaning"])]
    if extra:
        k, v = next(iter(extra.items()))
        blocks.append(("p", f'Supporting count ({k.replace("_", " ")} per year): mean {np.mean(v):.1f}, highest {max(v)} in {years[int(np.argmax(v))]}, last 10 years mean {np.mean(v[-10:]):.1f}.'))
    steps = []
    if hc["ok"] and not hc["passed"]:
        steps.append("The hazard is not stationary here: fit a model with a time-varying location, or restrict the record to the most recent decades, before using the return levels.")
    if n < 60:
        steps.append(f"The record has only {n} years, so the 100-year level is mostly extrapolation; add a longer or station record.")
    steps.append("Compare with local station observations: reanalysis gives the shape of the hazard; stations anchor its absolute level.")
    steps.append("Feed the return levels into the asset's damage or downtime curves to turn the hazard into money.")
    return {"domain": hazard, "label": cfg["label"], "kind": "evt", "name": name, "lat": lat, "lon": lon, "title": f'{cfg["label"]} at {name}: how bad, how often, and can we trust it',
            "standfirst": f'The engine built an extreme-value model of {cfg["var"]} at {name} from {n} years of data, hid the last third to test it, and found a 100-year level of {fmt.format(lv[100])} {cfg["unit"]} (50-year {fmt.format(lv[50])}). '
                          + ("It predicted the held-out years." if validated else "It did NOT predict the held-out years." if hc["ok"] else "The record was too short to test it.") + f' Status: {status}.',
            "status": status, "ok": nfail == 0, "headline": head, "blocks": blocks, "verification": bench, "sensitivity": [], "assumptions": cfg["assumptions"] + [f"Annual maxima are treated as independent and identically distributed (tested by the hindcast and trend tests)."],
            "limitations": cfg["limits"] + [f"{src}: a 25 km grid-cell model, not a point measurement."], "next_steps": steps, "notebook": nb.rows, "numbers": {"levels": lv, "ci": {str(k): v for k, v in ci.items()}, "trend": tr, "hindcast": {k: v for k, v in hc.items() if k != "pit"}},
            "elapsed_s": round(__import__("time").time() - nb.t0, 1)}
