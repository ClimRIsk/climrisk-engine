"""
Offshore wind: weather hazard, access windows and storm damage for a turbine site at sea (or on the coast), built for any coordinate.

What this module can and cannot say
  * It builds the WEATHER hazard that drives structural storm loss and, more importantly for insurers, the weather windows that decide how long a repair
    takes (and so the size of a business-interruption claim): annual-maximum gust and significant wave height with return levels, the joint behaviour of the
    two, and the share of time vessels can reach the site.
  * Its damage curve for extreme wind and waves is a PLACEHOLDER built from design practice (turbines and foundations are designed to the 50-year load with
    partial safety factors, so structural failure from weather alone starts well above it). Nobody publishes offshore claims, so the curve is not calibrated:
    it is returned as a bracket from benign to severe, and the output says so.
  * Most offshore wind insurance loss is NOT weather-driven (subsea cables, installation, blades, drivetrain, electrical). That part cannot be built from
    hazard data. cri.analysis.claims_credibility turns an insurer's own claims history into expected loss per cause; without claims, it is reported as
    NOT ASSESSED, and the premium becomes a lower bound or is withheld.

Sources: ERA5 via Open-Meteo (daily maximum 10 m gust, archive 1950 onward; NASA POWER MERRA-2 daily maximum 10 m wind is the fallback and reads lower),
ERA5 ocean-wave reanalysis (0.5 degree, model era5_ocean) via the Open-Meteo Marine API, significant wave height and period, 1950 onward (the default marine model only starts in 2021). Extremes: annual maxima, Gumbel by L-moments.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np

EULER = 0.5772156649015329
RPS = (10, 50, 100, 250, 500)
MARINE = "https://marine-api.open-meteo.com/v1/marine"
UA = {"User-Agent": "ClimRisk/1.0 (climate risk research)"}
# Conversion of ERA5 3-second-style daily gust at 10 m to a 10-minute mean wind at 10 m over open sea, and height scaling by the log law (Charnock roughness).
GUST_FACTOR = (1.20, 1.25, 1.35)               # low, central, high gust / 10-minute mean ratio
Z0_SEA = 2.0e-4
# Placeholder damage curve in load-ratio terms: load = max((V/Vref)^2, (Hs/Hs50)^2). r_half is where mean damage reaches half its ceiling.
CURVE = {"dmax": 0.60, "width": 0.25, "r_half_severe": 1.2, "r_half_central": 1.8, "r_half_benign": 2.4}
ACCESS_HS = (("small crew transfer vessel", 1.5), ("service operation vessel", 2.5), ("jack-up or heavy-lift vessel", 2.0))


def _cache_dir() -> Path:
    d = Path(os.environ.get("CRI_CACHE_DIR", Path.home() / ".climrisk" / "cache")) / "offshore"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, timeout: float = 120.0, waits: tuple = (8, 20, 45)) -> dict:
    """GET JSON; the free services answer 429 when several requests arrive together, so back off and retry before giving up."""
    import time
    import urllib.error
    for i in range(len(waits) + 1):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < len(waits) and not os.environ.get("CRI_NO_RETRY_SLEEP"):
                time.sleep(waits[i])
                continue
            raise


def _cached(tag: str, url: str, fetch: Callable) -> dict:
    f = _cache_dir() / (hashlib.md5(tag.encode()).hexdigest() + ".json")
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:                                          # noqa: BLE001
            pass
    d = fetch(url)
    if isinstance(d, dict) and d.get("error"):
        raise RuntimeError(str(d.get("reason") or d)[:160])
    try:
        f.write_text(json.dumps(d))
    except Exception:                                              # noqa: BLE001
        pass
    return d


def _annual_max(times: list, vals: list) -> dict:
    by: dict = {}
    for t, v in zip(times, vals):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            y = int(t[:4])
            by[y] = max(by.get(y, -1e9), float(v))
    return by


def gumbel_lmom(x) -> tuple:
    x = np.sort(np.asarray(x, float)); n = len(x)
    b0 = x.mean(); b1 = float(np.sum((np.arange(n) / (n - 1)) * x) / n)
    a = (2 * b1 - b0) / math.log(2.0)
    return float(b0 - EULER * a), float(a)


def return_level(mu: float, beta: float, T: float) -> float:
    return mu - beta * math.log(-math.log(1.0 - 1.0 / T))


def wave_daily(lat: float, lon: float, y0: int, y1: int, fetch: Callable = _get) -> dict:
    q = urllib.parse.urlencode({"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31", "daily": "wave_height_max,wave_period_max", "timezone": "UTC", "models": "era5_ocean"})
    return _cached(f"wave_daily_era5ocean:{lat:.2f}:{lon:.2f}:{y0}:{y1}", MARINE + "?" + q, fetch)


def wave_hourly(lat: float, lon: float, y0: int, y1: int, fetch: Callable = _get) -> dict:
    times: list = []; hs: list = []
    for a in range(y0, y1 + 1, 5):
        b = min(a + 4, y1)
        q = urllib.parse.urlencode({"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": f"{a}-01-01", "end_date": f"{b}-12-31", "hourly": "wave_height", "timezone": "UTC", "models": "era5_ocean"})
        d = _cached(f"wave_hourly_era5ocean:{lat:.2f}:{lon:.2f}:{a}:{b}", MARINE + "?" + q, fetch)
        times += d["hourly"]["time"]; hs += d["hourly"]["wave_height"]
    return {"time": times, "hs": hs}


def gust_daily(lat: float, lon: float, y1: int, fetch: Callable = _get) -> dict:
    """Daily maximum 10 m gust. ERA5 (Open-Meteo archive) first; NASA POWER daily maximum wind (MERRA-2, a smoother quantity) when ERA5 cannot be read."""
    try:
        q = urllib.parse.urlencode({"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": "1950-01-01", "end_date": f"{y1}-12-31", "daily": "wind_gusts_10m_max", "wind_speed_unit": "ms", "timezone": "UTC"})
        d = _cached(f"gust:{lat:.2f}:{lon:.2f}:{y1}", "https://archive-api.open-meteo.com/v1/archive?" + q, fetch)
        return {"time": d["daily"]["time"], "gust": d["daily"]["wind_gusts_10m_max"], "source": "ERA5 daily maximum 10 m gust (Open-Meteo archive)", "kind": "gust"}
    except Exception as e:                                         # noqa: BLE001
        err = str(e)[:100]
    q = urllib.parse.urlencode({"parameters": "WS10M_MAX", "community": "RE", "longitude": round(lon, 3), "latitude": round(lat, 3), "start": "19810101", "end": f"{y1}1231", "format": "JSON"}, safe=",")
    d = _cached(f"power:{lat:.2f}:{lon:.2f}:{y1}", "https://power.larc.nasa.gov/api/temporal/daily/point?" + q, fetch)
    par = (d.get("properties") or {}).get("parameter", {}).get("WS10M_MAX")
    if not par:
        raise RuntimeError("no wind record from ERA5 or NASA POWER (" + err + ")")
    days = sorted(par)
    return {"time": [f"{x[:4]}-{x[4:6]}-{x[6:]}" for x in days], "gust": [None if par[x] in (-999, -999.0) else float(par[x]) for x in days],
            "source": f"NASA POWER MERRA-2 daily maximum hourly 10 m wind (ERA5 unavailable: {err}); this is a mean wind, not a gust, so it reads low", "kind": "mean_hourly_max"}


def hub_wind(v10_gust: float, hub_m: float, gust_factor: float = GUST_FACTOR[1], kind: str = "gust") -> float:
    """10-minute mean wind at hub height from the 10 m daily maximum: gust to mean, then log-law shear over sea."""
    mean10 = v10_gust / gust_factor if kind == "gust" else v10_gust * 1.05          # hourly-mean maxima sit a little below the 10-minute maximum
    return mean10 * math.log(hub_m / Z0_SEA) / math.log(10.0 / Z0_SEA)


def load_ratio(v_hub, hs, vref: float, hs50: float):
    return np.maximum((np.asarray(v_hub, float) / vref) ** 2, (np.asarray(hs, float) / hs50) ** 2)


def storm_damage(r, r_half: float, dmax: float = CURVE["dmax"], width: float = CURVE["width"]):
    """Placeholder mean damage ratio against the load ratio r (see module note): a logistic centred on r_half, forced to zero at and below the design load r = 1."""
    r = np.asarray(r, float)
    d = dmax / (1.0 + np.exp(-(r - r_half) / width))
    d0 = dmax / (1.0 + math.exp(-(1.0 - r_half) / width))
    return np.where(r <= 1.0, 0.0, np.clip((d - d0) / (1.0 - d0 / dmax), 0.0, dmax))


def access_stats(h: dict) -> dict:
    t, hs = h["time"], np.array([np.nan if v is None else v for v in h["hs"]], float)
    mon = np.array([int(x[5:7]) for x in t])
    ok = np.isfinite(hs)
    out = {"years": [t[0][:4], t[-1][:4]], "share_of_hours_hs_below": {}, "worst_month_share_below_1_5m": None}
    for name, th in ACCESS_HS:
        out["share_of_hours_hs_below"][f"{th} m ({name})"] = round(float((hs[ok] < th).mean()), 3)
    by_m = {m: float((hs[ok & (mon == m)] < 1.5).mean()) for m in range(1, 13)}
    out["monthly_share_below_1_5m"] = {str(m): round(v, 2) for m, v in by_m.items()}
    worst = min(by_m, key=by_m.get)
    out["worst_month_share_below_1_5m"] = {"month": worst, "share": round(by_m[worst], 2)}
    # waiting time for a continuous 48 h window below 1.5 m starting at a random hour (a typical cable-repair or blade-exchange access window)
    win = 48
    good = (hs < 1.5) & ok
    c = np.concatenate([[0], np.cumsum(good)])
    full = (c[win:] - c[:-win]) == win
    n = len(full)
    nxt = np.full(n, np.inf)
    last = np.inf
    for i in range(n - 1, -1, -1):
        if full[i]:
            last = i
        nxt[i] = last - i if np.isfinite(last) else np.inf
    w = nxt[np.isfinite(nxt)]
    out["wait_hours_for_48h_window_below_1_5m"] = {"median": round(float(np.median(w)), 0), "p90": round(float(np.percentile(w, 90)), 0), "p99": round(float(np.percentile(w, 99)), 0)} if len(w) else None
    wm = {}
    for m in (1, 7):
        idx = np.where(mon[:n] == m)[0]
        ww = nxt[idx][np.isfinite(nxt[idx])]
        wm[str(m)] = {"median": round(float(np.median(ww)), 0), "p90": round(float(np.percentile(ww, 90)), 0)} if len(ww) else None
    out["wait_hours_48h_window_january_and_july"] = wm
    return out


def _cold_daily(lat: float, lon: float, y0: int, y1: int, fetch: Callable = _get) -> dict:
    """Daily air temperature (max, min, mean), mean relative humidity and rain: ERA5 (Open-Meteo archive) first, NASA POWER (MERRA-2) when ERA5 cannot be read."""
    v = "temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum"
    try:
        q = urllib.parse.urlencode({"latitude": round(lat, 3), "longitude": round(lon, 3), "start_date": f"{y0}-01-01", "end_date": f"{y1}-12-31", "daily": v, "timezone": "UTC"})
        d = _cached(f"cold:{lat:.2f}:{lon:.2f}:{y0}:{y1}", "https://archive-api.open-meteo.com/v1/archive?" + q, fetch)["daily"]
        return {"time": d["time"], "tmax": d["temperature_2m_max"], "tmin": d["temperature_2m_min"], "tmean": d["temperature_2m_mean"], "rh": d["relative_humidity_2m_mean"], "rain": d["precipitation_sum"], "source": "ERA5 (Open-Meteo archive)"}
    except Exception as e:                                         # noqa: BLE001
        err = str(e)[:100]
    q = urllib.parse.urlencode({"parameters": "T2M_MAX,T2M_MIN,T2M,RH2M,PRECTOTCORR", "community": "RE", "longitude": round(lon, 3), "latitude": round(lat, 3), "start": f"{y0}0101", "end": f"{y1}1231", "format": "JSON"}, safe=",")
    d = _cached(f"coldpower:{lat:.2f}:{lon:.2f}:{y0}:{y1}", "https://power.larc.nasa.gov/api/temporal/daily/point?" + q, fetch)
    par = (d.get("properties") or {}).get("parameter", {})
    if not par.get("T2M"):
        raise RuntimeError("no temperature record from ERA5 or NASA POWER (" + err + ")")
    days = sorted(par["T2M"])
    g = lambda k: [None if par[k][x] in (-999, -999.0) else float(par[k][x]) for x in days]  # noqa: E731
    return {"time": [f"{x[:4]}-{x[4:6]}-{x[6:]}" for x in days], "tmax": g("T2M_MAX"), "tmin": g("T2M_MIN"), "tmean": g("T2M"), "rh": g("RH2M"), "rain": g("PRECTOTCORR"),
            "source": f"NASA POWER MERRA-2 (ERA5 unavailable: {err}); smoother, so cold extremes read milder"}


def cold_screen(lat: float, lon: float, today: Optional[dt.date] = None, fetch: Callable = _get, years: int = 35) -> dict:
    """Cold-season screen for a coastal or offshore point: freezing degree-days (the driver of sea-ice formation in shallow seas), days with icing-prone weather, and
    air freeze-thaw cycle days. It is a SCREEN of how cold the site gets, not a loss model: sea-ice loads, blade-ice downtime and freeze-thaw durability are not priced from it.
    Annual values are per winter (July to June) so a cold spell is not split across two calendar years."""
    today = today or dt.date.today()
    y1 = today.year - 1
    y0 = y1 - years + 1
    try:
        d = _cold_daily(lat, lon, y0, y1, fetch)
    except Exception as e:                                         # noqa: BLE001
        return {"status": "null", "reason": f"temperature record unreadable ({type(e).__name__}: {str(e)[:140]}); cold-season screen not assessed (not zero)"}
    t = np.array([np.nan if x is None else x for x in d["tmean"]], float); tx = np.array([np.nan if x is None else x for x in d["tmax"]], float)
    tn = np.array([np.nan if x is None else x for x in d["tmin"]], float); rh = np.array([np.nan if x is None else x for x in d["rh"]], float)
    rn = np.array([np.nan if x is None else x for x in d["rain"]], float)
    times = np.array(d["time"])
    yr = np.array([int(x[:4]) for x in times]); mo = np.array([int(x[5:7]) for x in times])
    win = np.where(mo >= 7, yr, yr - 1)                            # winter label = the year it starts
    fdd, icing, ft, cold_days = {}, {}, {}, {}
    for w in sorted(set(win)):
        m = (win == w)
        if m.sum() < 330 or np.isnan(t[m]).mean() > 0.05:
            continue
        tm = t[m]
        fdd[w] = float(np.nansum(np.where(tm < 0, -tm, 0.0)))
        cold_days[w] = int(np.nansum(tx[m] < 0))                   # ice days: the air never rose above freezing
        ft[w] = int(np.nansum((tx[m] > 0) & (tn[m] < 0)))
        wet = (rh[m] >= 90) | (rn[m] > 0.1)
        icing[w] = int(np.nansum((tm <= 0.5) & (tm >= -10) & wet))
    if len(fdd) < 20:
        return {"status": "null", "reason": f"too few complete winters ({len(fdd)})"}
    ws = sorted(fdd)
    f = np.array([fdd[w] for w in ws]); ic = np.array([icing[w] for w in ws]); fz = np.array([ft[w] for w in ws]); cd = np.array([cold_days[w] for w in ws])
    q = lambda a, p: round(float(np.percentile(a, p)), 1)  # noqa: E731
    worst = ws[int(np.argmax(f))]
    return {"status": "constructed", "winters": [ws[0], ws[-1]], "n_winters": len(ws), "weather_source": d["source"],
            "freezing_degree_days": {"median": q(f, 50), "p90": q(f, 90), "max": round(float(f.max()), 1), "worst_winter": f"{worst}/{worst + 1}", "winters_with_any_freezing_share": round(float((f > 0).mean()), 2)},
            "ice_days_tmax_below_0": {"median": q(cd, 50), "p90": q(cd, 90), "max": int(cd.max())},
            "icing_prone_days_proxy": {"median": q(ic, 50), "p90": q(ic, 90), "max": int(ic.max()), "definition": "mean air temperature between -10 and +0.5 C with relative humidity of 90 % or more or rain on the day: a screen for supercooled-water icing, not a measurement of ice on a blade"},
            "freeze_thaw_air_cycle_days": {"median": q(fz, 50), "p90": q(fz, 90), "max": int(fz.max()), "definition": "days with maximum above 0 C and minimum below 0 C"},
            "loss_effect": None,
            "reading": ("Cold perils are screened, not priced. Sea ice (foundation loads, vessel access) follows freezing degree-days in shallow, low-salinity water; blade icing is rare offshore because salt "
                        "depresses the freezing point of spray but sea-spray icing on platforms and drifting ice at foundations remain; freeze-thaw acts on concrete and grout in the tidal and splash zone "
                        "as slow deterioration, which property policies normally treat as wear and tear rather than sudden damage. Their loss effect is therefore carried only through the insurer's own claims."),
            "limits": ["Air temperature at a 25 to 50 km reanalysis cell; sea-surface freezing, salinity, fetch and ice drift are not modelled.",
                       "No claims or loss data link these counts to money, and the research found no public offshore claims for icing or freeze-thaw.",
                       "The splash zone is wetted by seawater (freezing near -1.9 C), so air freeze-thaw cycles are an upper-level proxy for the cycles the concrete itself sees."]}


def hazard(lat: float, lon: float, hub_m: float = 120.0, vref_ms: float = 50.0, hs_design_m: Optional[float] = None, today: Optional[dt.date] = None,
           fetch: Callable = _get, access: bool = True, cold: bool = True) -> dict:
    """Weather hazard for an offshore wind turbine site. `vref_ms` is the turbine's design 50-year 10-minute mean wind at hub height (IEC class I = 50 m/s);
    `hs_design_m` the foundation's design 50-year significant wave height (default: this site's own 50-year value, i.e. a foundation designed to the site)."""
    today = today or dt.date.today()
    y1 = today.year - 1
    try:
        gd = gust_daily(lat, lon, y1, fetch)
    except Exception as e:                                         # noqa: BLE001
        return {"status": "null", "reason": f"wind record unreadable ({type(e).__name__}: {str(e)[:140]})"}
    try:
        wd = wave_daily(lat, lon, 1950, y1, fetch)
    except Exception as e:                                         # noqa: BLE001
        return {"status": "null", "reason": f"wave record unreadable ({type(e).__name__}: {str(e)[:140]})"}
    gm = _annual_max(gd["time"], gd["gust"])
    hm = _annual_max(wd["daily"]["time"], wd["daily"]["wave_height_max"])
    yrs = sorted(set(gm) & set(hm))
    if len(yrs) < 25:
        return {"status": "null", "reason": f"too few matching years of wind and wave record ({len(yrs)}); open water cells sometimes lack one of them"}
    g = np.array([gm[y] for y in yrs]); h = np.array([hm[y] for y in yrs])
    if float(h.max()) < 0.5:
        return {"status": "null", "reason": "no significant wave signal at this point (it is probably on land or in a closed basin)"}
    kind = gd["kind"]
    gf = GUST_FACTOR[1]
    v = np.array([hub_wind(x, hub_m, gf, kind) for x in g])
    mu_v, b_v = gumbel_lmom(v); mu_h, b_h = gumbel_lmom(h)
    rl_v = {T: round(return_level(mu_v, b_v, T), 1) for T in RPS}
    rl_h = {T: round(return_level(mu_h, b_h, T), 1) for T in RPS}
    rl_g = {T: round(return_level(*gumbel_lmom(g), T), 1) for T in RPS}
    hs50 = hs_design_m or rl_h[50]
    from scipy import stats
    rho = float(stats.spearmanr(g, h)[0])
    # analytic annual loss ratio for each placeholder curve: integrate over the annual-maximum return periods (wind and waves comonotone: one storm)
    u = (np.arange(1, 4001) - 0.5) / 4000.0
    vq = mu_v - b_v * np.log(-np.log(u)); hq = mu_h - b_h * np.log(-np.log(u))
    r = load_ratio(vq, hq, vref_ms, hs50)
    aal = {k: float(np.mean(storm_damage(r, CURVE[f"r_half_{k}"]))) for k in ("benign", "central", "severe")}
    n = len(yrs)
    a: dict = {}
    if access:
        try:
            a = access_stats(wave_hourly(lat, lon, max(1990, y1 - 29), y1, fetch))
        except Exception as e:                                     # noqa: BLE001
            a = {"status": "null", "reason": f"hourly wave record unreadable ({type(e).__name__}: {str(e)[:100]})"}
    return {"status": "constructed", "years": [yrs[0], yrs[-1]], "n_years": n, "hub_height_m": hub_m, "wind_source": gd["source"], "wave_source": "ERA5 ocean wave reanalysis (0.5 degree) via the Open-Meteo Marine API, daily maximum significant wave height, 1950 onward",
            "return_levels": {"hub_mean_wind_ms": rl_v, "gust_10m_ms": rl_g, "hs_m": rl_h, "hmax_m_approx": {T: round(1.86 * x, 1) for T, x in rl_h.items()}},
            "design": {"vref_ms": vref_ms, "hs50_m": round(hs50, 1), "site_50yr_over_vref": round(rl_v[50] / vref_ms, 2), "site_50yr_hs_over_design": round(rl_h[50] / hs50, 2)},
            "gumbel": {"wind_hub": [round(mu_v, 3), round(b_v, 3)], "hs": [round(mu_h, 3), round(b_h, 3)]},
            "wind_wave_rank_correlation": round(rho, 2), "extrapolated_return_periods": [T for T in RPS if T > 1.3 * n],
            "storm_damage_ratio_aal": {"benign": aal["benign"], "central": aal["central"], "severe": aal["severe"]},
            "storm_damage_note": "Placeholder curve (design practice, not claims): shows how little or how much structural loss weather alone produces under a benign, central and severe reading. Not calibrated.",
            "access": a, "cold": cold_screen(lat, lon, today, fetch) if cold else None, "assumptions": {"gust_to_10min_mean": f"{gf} (range {GUST_FACTOR[0]} to {GUST_FACTOR[2]})", "shear": f"log law over sea, roughness {Z0_SEA} m", "design_wind_vref_ms": vref_ms,
                                         "hs_design": "site 50-year significant wave height unless given", "curve": CURVE},
            "limits": ["ERA5 and its wave model are 25 to 50 km analyses: nearshore shoaling, wakes inside the array and local seabed effects are not resolved.",
                       "Annual maxima over 70 years support 50-year levels; 100 to 500-year levels are extrapolations and flagged.",
                       "Wave loads, scour, foundation fatigue, sea-ice loads and marine growth need site surveys and a structural model; none is built here (the cold-season block only screens how cold the site gets).",
                       "Cable faults, installation accidents, blade and drivetrain failures are not weather hazards and are not assessed here."]}
