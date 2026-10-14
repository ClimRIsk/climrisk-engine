"""Hazards defined by 'the worst value in a year': extreme heat, rainfall, wind and fire weather. Built and hindcast-validated from ERA5."""
from __future__ import annotations

import numpy as np

from . import data as D, evt, fire, heat, plots
SCN_NOTE = 'CMIP6 multi-model ensemble (World Bank CCKP, NASA NEX-GDDP-CMIP6 downscaled, 0.25°). CC0 model output; commercial use allowed with acknowledgement.'
from .generic_report import Notebook
from .dossier import NoModel


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
    ex["_rain_mm"] = float(sum(dd["precipitation_sum"][i] for i in ok) / (len(ok) / 365.25))
    ex["_inputs"] = ([dd["temperature_2m_max"][i] for i in ok], [dd["relative_humidity_2m_min"][i] for i in ok], [dd["wind_speed_10m_max"][i] for i in ok], [dd["precipitation_sum"][i] for i in ok], months, yrs.tolist())
    return ys, xs, ex


# ── NASA POWER versions (MERRA-2 / IMERG): the engine's second source, used when the first is out of quota or on request ──────────

def _power_max(var, scale=1.0):
    def f(lat, lon, fetch=None):
        d = D.power_daily(lat, lon, 1981, 2023, var) if fetch is None else D.power_daily(lat, lon, 1981, 2023, var, fetch=fetch)
        by: dict = {}
        for t, v in zip(d["time"], d[var]):
            if v is not None:
                by[int(t[:4])] = max(by.get(int(t[:4]), -1e9), float(v) * scale)
        ys = sorted(by)
        return ys, [by[y] for y in ys], {}
    return f


def _power_heat(lat, lon, fetch=None):
    d = D.power_daily(lat, lon, 1981, 2023, "T2M_MAX,T2MDEW") if fetch is None else D.power_daily(lat, lon, 1981, 2023, "T2M_MAX,T2MDEW", fetch=fetch)
    ok = [i for i in range(len(d["time"])) if d["T2M_MAX"][i] is not None and d["T2MDEW"][i] is not None]
    t = np.array([d["T2M_MAX"][i] for i in ok]); td = np.array([d["T2MDEW"][i] for i in ok])
    w = 0.567 * t + 0.393 * heat.sat_vapour_pressure_hpa(td) + 3.94          # daily-maximum temperature with the day's dew-point vapour pressure
    yrs = np.array([int(d["time"][i][:4]) for i in ok]); ys = sorted(set(yrs.tolist()))
    return ys, [float(w[yrs == y].max()) for y in ys], {"days_over_28": [int((w[yrs == y] >= 28.0).sum()) for y in ys]}


def _power_fire(lat, lon, fetch=None):
    v = "T2M_MAX,T2MDEW,WS10M_MAX,PRECTOTCORR"
    d = D.power_daily(lat, lon, 1981, 2023, v) if fetch is None else D.power_daily(lat, lon, 1981, 2023, v, fetch=fetch)
    ok = [i for i in range(len(d["time"])) if None not in (d["T2M_MAX"][i], d["T2MDEW"][i], d["WS10M_MAX"][i], d["PRECTOTCORR"][i])]
    t = np.array([d["T2M_MAX"][i] for i in ok]); td = np.array([d["T2MDEW"][i] for i in ok])
    rh = np.clip(100.0 * heat.sat_vapour_pressure_hpa(td) / heat.sat_vapour_pressure_hpa(t), 1, 100)   # afternoon RH from Tmax and the day's dew point
    months = [int(d["time"][i][5:7]) for i in ok]
    out = fire.run_series(t.tolist(), rh.tolist(), [d["WS10M_MAX"][i] * 3.6 for i in ok], [d["PRECTOTCORR"][i] for i in ok], months, tropical=abs(lat) < 25)
    yrs = np.array([int(d["time"][i][:4]) for i in ok]); f = np.array(out["fwi"])
    ys = [y for y in sorted(set(yrs.tolist())) if y > 1981]
    return ys, [float(f[yrs == y].max()) for y in ys], {"days_over_30": [int((f[yrs == y] >= 30).sum()) for y in ys], "_rain_mm": float(sum(d["PRECTOTCORR"][i] for i in ok) / (len(ok) / 365.25)), "_inputs": (t.tolist(), rh.tolist(), [d["WS10M_MAX"][i] * 3.6 for i in ok], [d["PRECTOTCORR"][i] for i in ok], months, yrs.tolist())}


POWER = {"rain": _power_max("PRECTOTCORR"), "wind": _power_max("WS10M_MAX"), "heat": _power_heat, "fire": _power_fire}
POWER_NOTE = {"rain": "NASA POWER daily precipitation (MERRA-2 / IMERG-corrected), 1981 onward",
              "wind": "NASA POWER daily maximum hourly 10 m wind (MERRA-2), a smoother quantity than an instantaneous gust, so levels read lower than gusts",
              "heat": "NASA POWER daily maximum temperature with the day's dew-point vapour pressure (MERRA-2), a daily proxy for the hourly peak",
              "fire": "NASA POWER daily maximum temperature, dew point, maximum wind and rain (MERRA-2); afternoon humidity is inferred from the dew point, so absolute index levels differ from the ERA5-based index (compare shape and trend, not level)"}


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
             "extract": lambda la, lo, fetch=None: (*_annual_max_daily(la, lo, "precipitation_sum", 1979, 2023, 1.0, fetch), {}), "benchmarks": lambda: [], "fmt": "{:.0f}",
             "assumptions": ["Daily totals; short, intense bursts (an hour of cloudburst) are not resolved.", "The record starts in 1979, the satellite era: ERA5 before 1979 has a documented step in tropical rainfall (found by the engine's own hindcast at Singapore, where 1950–78 annual maxima average 35 mm and 1979 onward 65 mm)."], "limits": ["ERA5 smooths convective extremes: the numbers are a floor."], "meaning": "Daily rain beyond the 100-year level overwhelms most urban drainage designed for 10 to 25 years."},
    "wind": {"label": "Extreme wind", "unit": "m/s gust", "var": "annual maximum daily 10 m gust",
             "physics": ["Strong winds also follow a block-maxima law. A structure's wind load scales with the square of speed (**q = ½ ρ V²**), so a 10 % higher gust means about 21 % more load, which is why return levels matter more than averages.",
                         "Tropical cyclones are poorly resolved by a 25 km reanalysis; the cyclone study (cri.modellab cyclone) is the better source where they dominate."],
             "extract": lambda la, lo, fetch=None: (*_annual_max_daily(la, lo, "wind_gusts_10m_max", 1979, 2023, 1 / 3.6, fetch), {}), "benchmarks": lambda: [], "fmt": "{:.1f}",
             "assumptions": ["Gusts at 10 m over open terrain; local exposure and building height change the design wind."], "limits": ["Tornadoes, downbursts and the core of tropical cyclones are not in ERA5."], "meaning": "Design standards use the 50-year gust; assets built to 25 m/s are outside their design envelope whenever a gust passes it."},
    "fire": {"label": "Fire weather", "unit": "FWI", "var": "annual maximum Fire Weather Index",
             "physics": ["The Canadian Fire Weather Index system (Van Wagner 1987) turns daily weather into fuel dryness and spread potential: three moisture codes (fine fuels FFMC, duff DMC, deep drought DC) each dry with heat and wind and wet with rain, combine into the Initial Spread Index (**ISI = 0.208 e^{0.05039 W} f(FFMC)**), the Buildup Index and finally the FWI.",
                         "Values above about 30 mean extreme fire danger in most regimes."],
             "extract": _fire_series, "benchmarks": fire.run_benchmarks, "fmt": "{:.0f}",
             "assumptions": ["The index is driven by daily maximum temperature, minimum humidity, maximum wind and rainfall as a proxy for the noon observation, so it runs slightly harsher than the official noon index.", "Tropical day-length constants are used inside 25° of the equator."],
             "limits": ["Fire weather is not fire: ignition and fuel load decide whether it burns. The index measures how the weather would let a fire spread."], "meaning": "The number of days above FWI 30 each year is the practical exposure."},
}


def _scenario_section(hazard, lat, lon, lv, fmt, unit, scenario_fetch=None):
    """Forward-looking CMIP6 projection of the design levels plus its backtest. Returns (blocks, numbers, limitation_lines). Never raises: a failure becomes a plain note."""
    from . import scenarios as SC
    try:
        fac = SC.change_factors(hazard, lat, lon, fetch=scenario_fetch)
    except Exception as e:                                             # noqa: BLE001
        return [], {"error": str(e)[:200]}, [f"Climate-scenario projections were not available for this run ({str(e)[:120]})."]
    tab = SC.projection_table(fac, lv, fmt, unit, wbgt=(hazard == "heat"))
    try:
        oy, ox = SC.observed_series(hazard, lat, lon)
        bt = SC.backtest(hazard, lat, lon, oy, ox, fetch=scenario_fetch)
    except Exception as e:                                             # noqa: BLE001
        bt = {"ok": False, "reason": str(e)[:160]}
    cfg = SC.VARS[hazard]
    short = cfg["label"].split(" (")[0]
    if hazard == "rain":
        how = f"The projected change in the {cfg['label']} is applied as a ratio to this site's own return levels (a delta method: the site's observed level is kept and only the change comes from the models)."
    else:
        how = (f"The projected change in the {cfg['label']} is added to this site's own return levels. The wet-bulb-globe temperature rises by 0.57 of the air-temperature change if humidity stays as it is, "
               "and by about 1.4 of it if relative humidity stays as it is; the table shows the midpoint of the two.")
    blocks = [("h2", "scenarios", "Where the hazard is heading: climate scenarios to 2099"),
              ("p", f"The records above describe the last four decades. For the decades an asset will live through, the engine reads the CMIP6 multi-model ensemble (World Bank Climate Change Knowledge Portal; four twenty-year horizons, four emissions pathways) at the grid cell {fac['cell'][0]:.2f}°, {fac['cell'][1]:.2f}°. {how}"),
              ("table", ["Scenario", "Horizon", f"Change in {short} (median; 10th–90th percentile of models)", f"100-year level ({unit}), median (range)"], tab["rows"],
               "The range is the spread across climate models for the same pathway. It excludes the uncertainty of the extreme-value fit (the 90 % range in the return-level table) and the choice of emissions pathway.")]
    if bt and bt.get("ok"):
        col = {"pass": "good", "partial": "warn", "fail": "bad"}[bt["verdict"]]
        weak = "" if bt["trend_resolvable"] else " The observed record is too short to separate a climate trend from natural variability, so the trend comparison is weak evidence either way."
        blocks += [("h2", "backtest", "Can the climate models be trusted here? A backtest"),
                   ("p", f"Before using a projection, the engine asks whether the same models reproduce what was observed at this site over 1981–2014: {bt['bias_text']}; trend over the same years {bt['obs_slope_decade']:+.2g} per decade observed (90 % range {bt['obs_slope_range'][0]:+.2g} to {bt['obs_slope_range'][1]:+.2g}) against {bt['model_slope_decade']:+.2g} for the ensemble median."),
                   ("callout", bt["summary"] + " The observations are a reanalysis (NASA POWER), not a weather station, so a mismatch can come from the reanalysis rather than the climate models." + weak, col)]
    else:
        blocks += [("callout", "Backtest not possible: " + (bt or {}).get("reason", "no observed series") + ". The projection above has not been checked against observations here.", "warn")]
    lim = ["Climate-model projections are scenarios, not forecasts: they depend on the emissions pathway society follows, and the 10th–90th percentile range covers model differences, not every uncertainty.",
           "Downscaled models smooth the extreme tail; the change in the annual maximum is taken as the change in the design levels (a delta method). " + ("Short convective storms may intensify faster than the annual-maximum change." if hazard == "rain" else "Humidity changes are uncertain, so the wet-bulb-globe change is a range."),
           "Scenario data: " + SC.ATTRIBUTION]
    return blocks, {"table": tab["numbers"], "baseline": fac["baseline_1995_2014"], "cell": fac["cell"], "backtest": bt, "mode": fac["mode"], "source": SC.NOTE}, lim


def _station_section(hazard, lat, lon, years, x, station_fetch=None):
    """Compare the gridded series with the nearest long-record weather stations (NOAA GHCN-Daily). Returns (blocks, crosscheck dict or None, limitation lines). Never raises."""
    from . import stationcheck as SK, scenarios as SC
    try:
        if hazard == "heat":
            years, x = SC.observed_series("heat", lat, lon)            # the station measures air temperature, not WBGT
        res = SK.compare(hazard, lat, lon, years, list(x), fetch=station_fetch)
    except Exception as e:                                             # noqa: BLE001
        return [], None, [f"The weather-station check could not run ({str(e)[:100]})."]
    if not res.get("ok"):
        return [("h2", "stations", "Compared with a weather station"), ("callout", "Not checked: " + res["reason"] + ".", "warn")], None, []
    col = {"consistent": "good", "inconsistent": "bad"}.get(res["verdict"], "warn")
    what = "daily rainfall" if hazard == "rain" else "the hottest day of each year"
    blocks = [("h2", "stations", "Compared with a weather station"),
              ("p", f"The gridded series above is a model product. As an independent check, the engine compared {what} with the nearest long-record weather stations (NOAA GHCN-Daily): " + res["detail"]),
              ("callout", res["note"] + (" A grid cell averages over tens of kilometres, so point rainfall extremes are expected to exceed it; a ratio far outside the band means the product misses local extremes." if hazard == "rain" else " A grid cell is cooler than a built-up station on the hottest days."), col)]
    if hazard == "rain" and res.get("station_levels"):
        blocks.append(("table", ["Return period", "Gridded series (mm/day)", "Weather stations, median (mm/day)"], [[f"{T} years", f"{res['grid_levels'][T]:.0f}", f"{res['station_levels'][T]:.0f}"] for T in (10, 50)],
                       "For an asset at a point, the station value is the better estimate of what falls on the ground when the two disagree; the gridded value is an area average."))
    return blocks, {"agree": res["agree"], "note": res["note"], "verdict": res["verdict"], "stations": res["stations"], "rho": res["rho"], "station_levels": res.get("station_levels"), "grid_levels": res.get("grid_levels")}, ["Station data: " + SK.ATTRIBUTION]


def run(hazard: str, lat: float, lon: float, name: str = "Study site", series: tuple = None, fetch=None, progress=None, bootstrap: int = 200, scenarios=None, scenario_fetch=None, stations=None, station_fetch=None) -> dict:
    cfg = HAZ[hazard]
    nb = Notebook(progress)
    if series is None:
        import os
        # default: NASA POWER (CC BY 4.0, commercial use permitted, no tight quota). "auto"/"openmeteo": try ERA5 via Open-Meteo first (free tier is non-commercial: development only).
        prov = os.environ.get("CRI_DATA_PROVIDER", "power")
        src = "ERA5 reanalysis (ECMWF) via Open-Meteo"
        if prov == "power":
            got = POWER[hazard](lat, lon, fetch)
            src = POWER_NOTE[hazard]
            nb("2b", "Data source", f"Data: {src}. Licence: CC BY 4.0, commercial use permitted with attribution.")
        else:
            try:
                got = cfg["extract"](lat, lon, fetch)
                nb("2b", "Data source", "Data: ERA5 via the Open-Meteo free tier. That tier is licensed for non-commercial use only: fine for development, replace before commercial release.", warn=True)
            except D.QuotaExceeded:
                got = POWER[hazard](lat, lon, fetch)
                src = POWER_NOTE[hazard]
                nb("2b", "Data source", f"The primary source (ERA5 via Open-Meteo) is unavailable (daily allowance used), so the engine used a second independent source: {src}.", warn=True)
        years, x, extra = got[0], np.array(got[1], float), got[2]
        rain_mm = extra.pop("_rain_mm", None)
        fire_in = extra.pop("_inputs", None)
        if hazard == "fire" and rain_mm is not None and rain_mm < 150.0:
            raise NoModel(f"Hyper-arid site (about {rain_mm:.0f} mm of rain a year): there is too little vegetation to carry a fire, so a fire weather index would measure dry air, not fire hazard.")
    else:
        years, x, extra = list(series[0]), np.array(series[1], float), (series[2] if len(series) > 2 else {})
        src = "supplied series"
    fire_in = locals().get("fire_in")
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
    # a failed test is judged by size as well as significance: how far do the 2-, 10- and 50-year levels move when the recent years are included?
    drift = None
    if hc["ok"] and not validated:
        try:
            ptr, pall = evt.best_model(x[:hc["n_train"]])["params"], model
            sh = [(evt.level(pall, T_) - evt.level(ptr, T_)) / abs(evt.level(ptr, T_)) for T_ in (2, 10, 50)]
            drift = max(sh, key=abs)
        except Exception:                                              # noqa: BLE001
            drift = None
    minor = drift is not None and abs(drift) < 0.10
    stp = evt.step_change(years, x) if (hc["ok"] and not validated) or tr["significant"] else {"ok": False}
    step_txt = ""
    if stp.get("ok") and stp["step_like"]:
        step_txt = (f'The change looks like one abrupt step around {stp["year"]} (mean {fmt.format(stp["mean_before"])} before, {fmt.format(stp["mean_after"])} after) rather than a steady trend. '
                    "An abrupt step usually means the data changed (instrument, reanalysis input or production stream) or the regime shifted; check the source before reading it as climate change.")
        nb("5b", "Diagnosis", step_txt, warn=True)
    elif stp.get("ok") and (not validated) and tr["significant"]:
        step_txt = "The change fits a steady trend better than a single step, which is what a gradually changing climate would give."
        nb("5b", "Diagnosis", step_txt)
    status = ("FAILED verification" if nfail else "verified and hindcast-validated" + (", trend detected" if tr["significant"] else "") if validated else
              f"verified; hindcast detects a small drift ({drift:+.0%} in the design levels): statistically real, practically minor" if minor else
              "verified; hindcast FAILED (the early record does not predict the recent one)" if hc["ok"] else "verified only (record too short to hindcast)")
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
                   ("callout", ("The hindcast PASSED: the early record predicted the recent years within sampling error." if hc["passed"] else
                                f"The hindcast detected a drift, but it is small: including the recent years moves the design levels by up to {drift:+.0%}. The stationary estimate is usable; the trend is reported above." if minor else
                                "The hindcast FAILED. " + (hc["bias_note"] or "The early record does not describe the recent years.") + (f" Including the recent years moves the design levels by up to {drift:+.0%}." if drift is not None else "") + " Do not use the stationary estimate without adjustment."), "good" if hc["passed"] else "warn" if minor else "bad")] + ([("p", step_txt)] if step_txt else [])
    else:
        blocks += [("callout", f'Hindcast not possible: {hc["reason"]}.', "warn")]
    blocks += [("h2", "result", "Return levels"),
               ("fig", plots.return_level_plot(x, lambda T: evt.level(model, T), ci, cfg["unit"], f'{cfg["label"]}: how large, how often', (lambda T: evt.trend_level(tr, 2050, T)) if tr["significant"] else None), "Black dots: observed maxima by their empirical return period. Band: 90 % bootstrap range."),
               ("table", ["Return period", "Level (average of the record)"] + ([f"Level today ({years[-1]}, trend-adjusted)"] if tr["significant"] else []) + ["90 % range (average)"],
                [[f"{T} years", fmt.format(lv[T]) + f' {cfg["unit"]}'] + ([fmt.format(evt.trend_level(tr, years[-1], T)) + f' {cfg["unit"]}'] if tr["significant"] else []) + [f'{fmt.format(ci[T][0])} – {fmt.format(ci[T][1])}' if T in ci else "n/a"] for T in (10, 50, 100)],
                "Return periods beyond the length of the record are extrapolations; the range widens accordingly." + (" 100 years exceeds the record: treat it as indicative." if n < 100 else "")),
               ("p", f'Trend: the typical annual maximum changes by {tr["slope_per_decade"]:+.2g} {cfg["unit"]} per decade (p = {tr["p"]:.3f}). ' + (f'Extending that trend to 2050 would put the 100-year level at {fmt.format(tl)} {cfg["unit"]}, against {fmt.format(lv[100])} now; this is an extrapolation of a linear trend, not a climate projection.' if tr["significant"] else "The change is not distinguishable from noise in this record, so the stationary model stands.")),
               ("p", cfg["meaning"])]
    if extra:
        k, v = next(iter(extra.items()))
        blocks.append(("p", f'Supporting count ({k.replace("_", " ")} per year): mean {np.mean(v):.1f}, highest {max(v)} in {years[int(np.argmax(v))]}, last 10 years mean {np.mean(v[-10:]):.1f}.'))
    scen_lim: list = []
    scen_num = None
    want = scenarios if scenarios is not None else (series is None and fetch is None)
    if want and hazard in ("rain", "heat"):
        nb("7", "Climate scenarios", "Reading the CMIP6 multi-model ensemble at this grid cell (World Bank CCKP, CC0 model output) and backtesting it against the observed record.")
        sb, scen_num, scen_lim = _scenario_section(hazard, lat, lon, lv, fmt, cfg["unit"], scenario_fetch)
        blocks += sb
    if want and hazard == "fire" and fire_in is not None:
        from . import futureclimate as FC
        try:
            dr = FC.drivers(lat, lon, scenario_fetch)
            fp = FC.fire_projection(*fire_in[:5], fire_in[5], dr, tropical=abs(lat) < 25)
            blocks += [("h2", "scenarios", "Fire weather in a warmer climate"),
                       ("p", f"The engine re-runs the Fire Weather Index on this site's own 1981–2023 daily weather after adding the CMIP6 ensemble's change in annual-mean temperature and scaling rainfall by its change in annual-mean precipitation (grid cell {dr['cell'][0]:.2f}°, {dr['cell'][1]:.2f}°). Humidity is bounded two ways: relative humidity unchanged, and specific humidity unchanged (the air gets drier in relative terms as it warms). Wind is unchanged. Today (baseline weather) the mean annual maximum is {fp['baseline_annual_max']:.0f} and the mean number of days above FWI 30 is {fp['baseline_days30']:.1f} a year."),
                       ("table", ["Scenario", "Horizon", "Warming", "Annual rain", "Mean annual-maximum FWI", "Days above FWI 30 per year"], fp["rows"], "Ranges span the two humidity assumptions. Ensemble medians. The change in how rain is spread through the year, and in wind, is not captured."),
                       ("callout", "These projections apply climate-model changes in the mean to observed daily weather. They are not backtested at this site (the backtest in the rain and heat studies covers extremes of those variables only), so treat the direction as informative and the size as indicative.", "warn")]
            scen_num = {"fire": fp["numbers"], "baseline_annual_max": fp["baseline_annual_max"], "baseline_days30": fp["baseline_days30"], "source": SCN_NOTE}
            scen_lim += ["Fire projections: temperature and mean-rain changes from the CMIP6 ensemble applied to observed daily weather; vegetation, fuel load and ignitions are not modelled. " + FC.ATTRIBUTION]
        except Exception as e:                                         # noqa: BLE001
            scen_lim += [f"Fire-weather projections were not available for this run ({str(e)[:120]})."]
    xcheck = None
    want_st = stations if stations is not None else (series is None and fetch is None)
    if want_st and hazard in ("rain", "heat"):
        nb("8", "Weather-station check", "Comparing with the nearest long-record weather stations (NOAA GHCN-Daily).")
        stb, xcheck, stl = _station_section(hazard, lat, lon, years, x, station_fetch)
        blocks += stb; scen_lim += stl
    steps = []
    if hc["ok"] and not hc["passed"]:
        steps.append("The hazard is not stationary here: fit a model with a time-varying location, or restrict the record to the most recent decades, before using the return levels.")
    if n < 60:
        steps.append(f"The record has only {n} years, so the 100-year level is mostly extrapolation; add a longer or station record.")
    steps.append("Compare with local station observations: reanalysis gives the shape of the hazard; stations anchor its absolute level.")
    steps.append("Feed the return levels into the asset's damage or downtime curves to turn the hazard into money.")
    return {"domain": hazard, "label": cfg["label"], "kind": "evt", "name": name, "lat": lat, "lon": lon, "title": f'{cfg["label"]} at {name}: how bad, how often, and can we trust it',
            "standfirst": f'The engine built an extreme-value model of {cfg["var"]} at {name} from {n} years of data, hid the last third to test it, and found a 100-year level of {fmt.format(lv[100])} {cfg["unit"]} (50-year {fmt.format(lv[50])}). '
                          + ("It predicted the held-out years." if validated else "It predicted the held-out years only approximately (a small, practically minor drift)." if (hc["ok"] and minor) else "It did NOT predict the held-out years." if hc["ok"] else "The record was too short to test it.") + f' Status: {status}.',
            "status": status, "ok": nfail == 0, "headline": head, "blocks": blocks, "verification": bench, "sensitivity": [], "assumptions": cfg["assumptions"] + [f"Annual maxima are treated as independent and identically distributed (tested by the hindcast and trend tests)."],
            "limitations": cfg["limits"] + scen_lim + [f"{src}: a gridded model, not a point measurement."] + (["Attribution: These data were obtained from the NASA Langley Research Center POWER Project funded through the NASA Earth Science Directorate Applied Science Program."] if "POWER" in src else []), "next_steps": steps, "notebook": nb.rows, "numbers": {"n_years": n, "crosscheck": xcheck, "scenarios": scen_num, "levels_today": ({T: evt.trend_level(tr, years[-1], T) for T in (10, 50, 100)} if tr["significant"] else None), "source": src, "drift50": drift, "step": stp if stp.get("ok") else None, "levels": lv, "ci": {str(k): v for k, v in ci.items()}, "trend": tr, "hindcast": {k: v for k, v in hc.items() if k != "pit"}},
            "elapsed_s": round(__import__("time").time() - nb.t0, 1)}
