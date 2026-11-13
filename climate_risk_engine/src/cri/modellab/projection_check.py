"""Second, independent look at how much heavier extreme rain gets: HighResMIP models (via Open-Meteo) against the CMIP6 ensemble the engine normally uses (World Bank CCKP / NASA NEX-GDDP).  (13 Nov 2026)

Why: the engine's future-rain number is one product (`scenarios.change_factors`: CMIP6 ensemble, statistically downscaled to 0.25 degrees). A second product that comes from a
different model set and a different downscaling shows whether that number is robust or an artefact of one pipeline.

What is compared: the change in the ANNUAL MAXIMUM ONE-DAY RAINFALL (Rx1day), 2031-2050 over 1995-2014, as a ratio of 20-year means. The CCKP ensemble gives windows 2020-2039 and
2040-2059; the HighResMIP window 2031-2050 sits between them, so the CCKP ratio is interpolated linearly in time (stated in the output, not hidden).

What the second source is, plainly: seven HighResMIP models (20 to 51 km) run for CMIP6 under a high-emission forcing close to SSP5-8.5; Open-Meteo bias-corrects each to ERA5-Land
with a LINEAR monthly correction, which leaves the climate-change signal alone, and interpolates to about 10 km. It is therefore compared with the CCKP SSP5-8.5 numbers. Different
experiment, different models, different downscaling: that is the point. It is NOT a validation, and 20-year samples of annual maxima are noisy, so the spread is shown, not hidden.
Licence: Open-Meteo's free tier is non-commercial; commercial use needs their key (CRI_OPEN_METEO_KEY is not wired here).
"""
from __future__ import annotations

import json
import math
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

MODELS = ("CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR", "MPI_ESM1_2_XR", "NICAM16_8S")
HIST = (1995, 2014)
FUT = (2031, 2050)
URL = "https://climate-api.open-meteo.com/v1/climate?latitude={lat}&longitude={lon}&start_date={a}-01-01&end_date={b}-12-31&models={m}&daily=precipitation_sum"


def _get(url: str) -> dict:
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.loads(r.read())
        except Exception as e:                                       # noqa: BLE001
            last = e
            time.sleep(4.0 * (_ + 1) if "429" in str(e) else 1.0)       # the free tier rate-limits bursts
    raise RuntimeError(f"{type(last).__name__}: {str(last)[:100]}")


def annual_maxima(daily: dict) -> dict:
    """{year: max daily precipitation} from an Open-Meteo daily block; a year with any missing day is dropped (never filled)."""
    by: dict = {}
    for t, v in zip(daily.get("time", []), daily.get("precipitation_sum", [])):
        by.setdefault(int(t[:4]), []).append(v)
    return {y: max(vs) for y, vs in by.items() if vs and all(v is not None for v in vs) and len(vs) >= 365}


def rx1day_ratio(hist: dict, fut: dict, min_years: int = 15) -> Optional[float]:
    """Ratio of the 20-year mean of annual maxima, future over historical; None if either window has too few complete years."""
    h, f = [v for v in hist.values()], [v for v in fut.values()]
    if len(h) < min_years or len(f) < min_years or statistics.fmean(h) <= 0:
        return None
    return statistics.fmean(f) / statistics.fmean(h)


def cckp_interpolated(factors: dict, ssp: str = "ssp585", mid: float = 0.55) -> Optional[dict]:
    """CCKP change factor for the HighResMIP window: linear in time between the 2020-2039 and 2040-2059 windows (2031-2050 is 55 % of the way)."""
    s = (factors or {}).get("ssp", {}).get(ssp, {})
    a, b = s.get("2020-2039"), s.get("2040-2059")
    if not a or not b:
        return None
    return {k: a[k] + mid * (b[k] - a[k]) for k in ("p10", "median", "p90")}


def rain_projection_sources(lat: float, lon: float, factors: Optional[dict] = None, fetch: Optional[Callable[[str], dict]] = None, models=MODELS) -> dict:
    """Per-model Rx1day ratios from HighResMIP via Open-Meteo, next to the CCKP ensemble. `factors` is `scenarios.change_factors("rain", lat, lon)` output (fetched if omitted)."""
    fetch = fetch or _get

    def one(m):
        try:
            h = annual_maxima(fetch(URL.format(lat=lat, lon=lon, a=HIST[0], b=HIST[1], m=m))["daily"])
            f = annual_maxima(fetch(URL.format(lat=lat, lon=lon, a=FUT[0], b=FUT[1], m=m))["daily"])
            r = rx1day_ratio(h, f)
            return m, (round(r, 3) if r is not None else None), (None if r is not None else "too few complete years")
        except Exception as e:                                       # noqa: BLE001
            return m, None, f"{type(e).__name__}: {str(e)[:80]}"

    with ThreadPoolExecutor(3) as ex:
        res = list(ex.map(one, models))
    per = {m: r for m, r, _ in res if r is not None}
    failed = {m: why for m, r, why in res if r is None}
    if factors is None:
        try:
            from . import scenarios as SC
            factors = SC.change_factors("rain", lat, lon)
        except Exception as e:                                       # noqa: BLE001
            factors = None
            failed["CCKP ensemble"] = f"{type(e).__name__}: {str(e)[:80]}"
    cck = cckp_interpolated(factors)
    out = {"available": len(per) >= 3, "per_model": per, "failed": failed, "windows": {"historical": HIST, "future": FUT}, "cckp_ssp585_interpolated": cck,
           "source": "HighResMIP models via Open-Meteo (linear bias correction to ERA5-Land, ~10 km) versus World Bank CCKP / NASA NEX-GDDP-CMIP6 (0.25 deg)",
           "note": "Different model sets, experiments and downscaling, so agreement is a robustness check, not a validation. Rx1day over 20 years is noisy."}
    if len(per) >= 3:
        v = sorted(per.values())
        out.update(low=v[0], high=v[-1], median=round(statistics.median(v), 3))
        if cck:
            out["highres_median_inside_cckp_p10_p90"] = bool(cck["p10"] <= out["median"] <= cck["p90"])
            out["highres_overlaps_cckp"] = bool(v[0] <= cck["p90"] and v[-1] >= cck["p10"])
    else:
        out["reason"] = "fewer than three HighResMIP models returned enough complete years"
    return out
