"""Forward-looking climate scenarios (2020-2099) for the hazards a global climate-model ensemble can speak to, and a backtest of those projections.

Source: World Bank Climate Change Knowledge Portal (CCKP) CMIP6 collection, 0.25 degree, statistically downscaled and bias-corrected from NASA NEX-GDDP-CMIP6.
Licence: the CMIP6 model output is CC0; CCKP content is under the Mozilla Public License; both allow commercial use with acknowledgement
("World Bank Climate Change Knowledge Portal: https://climateknowledgeportal.worldbank.org" and the NEX-GDDP-CMIP6 / CMIP6 acknowledgements).
Files are uncompressed classic NetCDF on the AWS open-data bucket, so one grid cell is read with a few HTTP range requests (a handful of bytes each); nothing global is downloaded.

What is projected: rain -> the ensemble change in the annual-maximum one-day rainfall (Rx1day), applied as a ratio to the site's return levels;
heat -> the change in the annual maximum of daily maximum temperature (TXx), applied as a shift (and converted to a wet-bulb-globe range).
Wind, tropical cyclones, earthquake: no ensemble product is used (said plainly in the report).

The backtest asks of every site: does the model ensemble reproduce what was actually observed there?  (a) level: the CMIP6 historical-run mean of 1995-2014 against the observed mean of the same years;
(b) trend: the ensemble's 1981-2014 trend against the observed trend and its bootstrap range.  A projection is only as credible as its hindcast at the same place.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

BASE = "https://wbg-cckp.s3.amazonaws.com/data/cmip6-x0.25/"
UA = {"User-Agent": "ClimRisk-ModelLab/1.0 (climate risk research prototype)"}
SSPS = ("ssp126", "ssp245", "ssp370", "ssp585")
SSP_LABEL = {"ssp126": "SSP1-2.6 (strong mitigation)", "ssp245": "SSP2-4.5 (middle road)", "ssp370": "SSP3-7.0 (high emissions)", "ssp585": "SSP5-8.5 (very high emissions)"}
PERIODS = ("2020-2039", "2040-2059", "2060-2079", "2080-2099")
VARS = {"rain": {"cckp": "rx1day", "mode": "ratio", "unit": "mm/day", "label": "annual maximum one-day rainfall (Rx1day)"},
        "heat": {"cckp": "txx", "mode": "shift", "unit": "°C", "label": "annual maximum daily temperature (TXx)"}}
WBGT_PER_K = (0.567, 1.36)        # d(WBGT)/dT: humidity unchanged (0.567) ... relative humidity unchanged (0.567 + 0.393 de/dT with de/dT ~ 2 hPa/K near 33 C)
ATTRIBUTION = ("World Bank Climate Change Knowledge Portal (https://climateknowledgeportal.worldbank.org), CMIP6 0.25° ensemble statistically downscaled from NASA NEX-GDDP-CMIP6 (Thrasher et al. 2021). "
               "We acknowledge the World Climate Research Programme, which, through its Working Group on Coupled Modelling, coordinated and promoted CMIP6, and the climate modelling groups who produced and shared their output (CC0).")
NOTE = "CMIP6 multi-model ensemble (World Bank CCKP, NASA NEX-GDDP-CMIP6 downscaled, 0.25°). Licence: CC0 model output / MPL portal content, commercial use allowed with acknowledgement."

Fetch = Callable[[str, int, int], bytes]


def _range(url: str, a: int, b: int) -> bytes:
    last = None
    for i in range(4):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except Exception as e:                                         # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"could not read the climate-projection file ({last})")


# ── classic NetCDF (CDF-1) header reader: just enough to find a variable and its record layout ────────────────────────────────
_T = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}
_HDR: dict = {}


class _P:
    def __init__(self, b): self.b, self.i = b, 0

    def u32(self):
        v = int.from_bytes(self.b[self.i:self.i + 4], "big"); self.i += 4
        if self.i > len(self.b):
            raise IndexError
        return v

    def name(self):
        n = self.u32(); s = self.b[self.i:self.i + n].decode("latin1"); self.i += (n + 3) // 4 * 4
        if self.i > len(self.b):
            raise IndexError
        return s

    def attrs(self):
        tag, n = self.u32(), self.u32()
        for _ in range(n):
            self.name(); t = self.u32(); c = self.u32(); self.i += (c * _T[t] + 3) // 4 * 4
            if self.i > len(self.b):
                raise IndexError


def _parse(b: bytes) -> dict:
    if b[:3] != b"CDF" or b[3] != 1:
        raise ValueError("unexpected file format (expected classic NetCDF)")
    p = _P(b); p.i = 4; p.u32()
    p.u32(); n = p.u32(); dims = [(p.name(), p.u32()) for _ in range(n)]
    p.attrs()
    p.u32(); n = p.u32(); vs = {}
    for _ in range(n):
        nm = p.name(); nd = p.u32(); ids = [p.u32() for _ in range(nd)]; p.attrs(); t = p.u32(); size = p.u32(); begin = p.u32()
        vs[nm] = {"rec": bool(ids) and dims[ids[0]][1] == 0, "type": t, "vsize": size, "begin": begin}
    rec = sum(v["vsize"] for v in vs.values() if v["rec"])
    nlat = dict(dims).get("lat", 721); nlon = dict(dims).get("lon", 1440)
    return {"vars": vs, "recsize": rec, "nlat": nlat, "nlon": nlon}


def _header(url: str, fetch: Fetch) -> dict:
    if url not in _HDR:
        for size in (16384, 131072):
            try:
                _HDR[url] = _parse(fetch(url, 0, size - 1)); break
            except (IndexError, ValueError, KeyError):
                if size == 131072:
                    raise
    return _HDR[url]


def grid_ij(lat: float, lon: float, nlat: int = 721, nlon: int = 1440) -> tuple:
    """Index of the grid cell containing (lat, lon) on the CCKP 0.25° grid (lat -90..90, lon -180..179.75)."""
    i = min(nlat - 1, max(0, int(round((lat + 90.0) / 0.25))))
    j = int(round(((lon + 180.0) % 360.0) / 0.25)) % nlon
    return i, j


def _read(url: str, k: int, i: int, j: int, fetch: Fetch) -> Optional[float]:
    h = _header(url, fetch)
    name = next(n for n, v in h["vars"].items() if v["rec"] and n != "time" and not n.startswith("anomalysignificance") and not n.endswith("_bnds"))
    v = h["vars"][name]
    off = v["begin"] + k * h["recsize"] + (i * h["nlon"] + j) * 4
    x = float(np.frombuffer(fetch(url, off, off + 3), ">f4")[0])
    return None if (not math.isfinite(x) or abs(x) > 1e19) else x


def _cell_value(url: str, k: int, i: int, j: int, fetch: Fetch) -> Optional[float]:
    """Value at the cell; if it is masked (open sea, ice), the nearest valid neighbour within one cell."""
    x = _read(url, k, i, j, fetch)
    if x is not None:
        return x
    for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        try:
            x = _read(url, k, min(720, max(0, i + di)), (j + dj) % 1440, fetch)
        except Exception:                                              # noqa: BLE001
            x = None
        if x is not None:
            return x
    return None


def _urls(var: str, kind: str, scen: str, stat: str, period: str) -> str:
    """kind: anomaly | climatology (period files) ; timeseries (yearly records)."""
    if kind == "timeseries":
        return f"{BASE}{var}/ensemble-all-{scen}/timeseries-{var}-annual-mean_cmip6-x0.25_ensemble-all-{scen}_timeseries_{stat}_{period}.nc"
    return f"{BASE}{var}/ensemble-all-{scen}/{kind}-{var}-annual-mean_cmip6-x0.25_ensemble-all-{scen}_climatology_{stat}_{period}.nc"


def _cache_get(key: str):
    from . import data as D
    p = D._cache_path(key)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:                                              # noqa: BLE001
            return None
    return None


def _cache_put(key: str, obj) -> None:
    from . import data as D
    try:
        D._cache_path(key).write_text(json.dumps(obj))
    except Exception:                                                  # noqa: BLE001
        pass


def _hist_series(var: str, i: int, j: int, fetch: Fetch) -> dict:
    """Ensemble-median CMIP6 historical-run annual values 1981-2014 at the cell: {"1981": v, ...}."""
    key = f"cckp-hist:{var}:{i}:{j}"
    ms = _cache_get(key) if fetch is _range else None
    if ms is None:
        url = _urls(var, "timeseries", "historical", "median", "1950-2014")
        with ThreadPoolExecutor(12) as ex:
            vals = list(ex.map(lambda yy: _cell_value(url, yy - 1950, i, j, fetch), range(1981, 2015)))
        ms = {str(1981 + k): v for k, v in enumerate(vals)}
        if fetch is _range:
            _cache_put(key, ms)
    return ms


# ── projected change factors ───────────────────────────────────────────────────────────────────────────────────────────────────
def change_factors(hazard: str, lat: float, lon: float, fetch: Fetch = None, ssps=SSPS) -> dict:
    """Ensemble change in the hazard variable at the grid cell, by scenario and 20-year horizon, relative to 1995-2014.
    rain: ratio (1.12 = 12 % heavier extremes); heat: shift in °C. Each entry has p10 / median / p90 across the CMIP6 models."""
    if hazard not in VARS:
        raise KeyError(hazard)
    cfg = VARS[hazard]; var = cfg["cckp"]
    fetch = fetch or _range
    i, j = grid_ij(lat, lon)
    key = f"cckp-factors:{var}:{i}:{j}:{','.join(ssps)}"
    got = _cache_get(key) if fetch is _range else None
    if got:
        return got
    jobs = [(s, st, p) for s in ssps for st in ("p10", "median", "p90") for p in PERIODS]

    def one(job):
        s, st, p = job
        return job, _cell_value(_urls(var, "anomaly", s, st, p), 0, i, j, fetch)

    with ThreadPoolExecutor(12) as ex:
        res = dict(ex.map(one, jobs))
    hs = _hist_series(var, i, j, fetch)
    bv = [v for y, v in hs.items() if int(y) >= 1995 and v is not None]
    base = float(np.mean(bv)) if len(bv) >= 10 else None
    if base is None:
        raise RuntimeError("the climate-projection grid has no value at this location (open sea or polar ice)")
    out = {"hazard": hazard, "var": var, "mode": cfg["mode"], "unit": cfg["unit"], "label": cfg["label"], "baseline_1995_2014": base, "cell": [-90 + 0.25 * i, -180 + 0.25 * j], "periods": list(PERIODS), "ssp": {}, "source": NOTE, "attribution": ATTRIBUTION}
    for s in ssps:
        out["ssp"][s] = {}
        for p in PERIODS:
            vals = {st: res[(s, st, p)] for st in ("p10", "median", "p90")}
            if any(v is None for v in vals.values()):
                continue
            if cfg["mode"] == "ratio":
                vals = {st: 1.0 + v / base for st, v in vals.items()}
            lo, md, hi = sorted((vals["p10"], vals["median"], vals["p90"]))
            out["ssp"][s][p] = {"p10": lo, "median": md, "p90": hi}
    if fetch is _range:
        _cache_put(key, out)
    return out


def apply(factor: dict, level: float, mode: str, wbgt: bool = False):
    """Future level from a baseline level and a change factor {p10, median, p90}. Returns (p10, median, p90); for heat on a WBGT scale returns the humidity-unchanged and humidity-held bounds at the median change."""
    if mode == "ratio":
        return tuple(level * factor[k] for k in ("p10", "median", "p90"))
    if wbgt:
        lo, hi = WBGT_PER_K
        d = factor["median"]
        return (level + lo * d, level + d * (lo + hi) / 2, level + hi * d)
    return tuple(level + factor[k] for k in ("p10", "median", "p90"))


def projection_table(fac: dict, levels: dict, fmt: str, unit: str, wbgt: bool = False) -> dict:
    """Rows for the report: for each scenario and horizon, the projected levels of the design return periods."""
    mode = fac["mode"]; rows = []; numbers = {}
    for s in fac["ssp"]:
        for p in fac["periods"]:
            f = fac["ssp"][s].get(p)
            if not f:
                continue
            cells = []
            nums = {}
            for T in (10, 50, 100):
                a, b, c = apply(f, levels[T], mode, wbgt)
                nums[T] = [a, b, c]
            chg = (f"{(f['median'] - 1) * 100:+.0f} % ({(f['p10'] - 1) * 100:+.0f} to {(f['p90'] - 1) * 100:+.0f})" if mode == "ratio" else f"{f['median']:+.1f} °C ({f['p10']:+.1f} to {f['p90']:+.1f})")
            lo, mid, hi = nums[100]
            rows.append([SSP_LABEL[s].split(" (")[0], p, chg, (f"{fmt.format(mid)} ({fmt.format(lo)} – {fmt.format(hi)})")])
            numbers[f"{s}:{p}"] = {"change": f, "levels": nums}
    return {"rows": rows, "numbers": numbers}


# ── backtest: does the ensemble reproduce what was observed here? ─────────────────────────────────────────────────────────────
def _slope_decade(y, x) -> float:
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float(np.polyfit(y - y.mean(), x, 1)[0] * 10.0)


def backtest(hazard: str, lat: float, lon: float, years, x, fetch: Fetch = None, boot: int = 400) -> dict:
    """Compare the CMIP6 historical-run ensemble (median) with observed annual maxima of the same variable at the site.
    years, x: observed annual maxima of Rx1day (mm/day) or TXx (°C), ideally 1981-2014 or longer."""
    cfg = VARS[hazard]; var = cfg["cckp"]; fetch = fetch or _range
    i, j = grid_ij(lat, lon)
    yrs = np.array(years, int); xs = np.array(x, float)
    sel = (yrs >= 1981) & (yrs <= 2014)
    if sel.sum() < 20:
        return {"ok": False, "reason": f"only {int(sel.sum())} observed years overlap the historical model run (1981-2014); at least 20 are needed"}
    ms = _hist_series(var, i, j, fetch)
    my = np.array([y for y in range(1981, 2015) if ms.get(str(y)) is not None], int)
    mx = np.array([ms[str(y)] for y in my], float)
    if len(my) < 20:
        return {"ok": False, "reason": "the climate-model grid has no historical value at this location"}
    oy, ox = yrs[sel], xs[sel]
    lvl_o = float(ox[oy >= 1995].mean()) if (oy >= 1995).sum() >= 10 else float(ox.mean())
    lvl_m = float(mx[my >= 1995].mean()) if (my >= 1995).sum() >= 10 else float(mx.mean())
    so = _slope_decade(oy, ox); sm = _slope_decade(my, mx)
    rng = np.random.default_rng(11)
    bs = []
    for _ in range(boot):
        idx = rng.integers(0, len(oy), len(oy))
        if len(set(oy[idx])) > 3:
            bs.append(_slope_decade(oy[idx], ox[idx]))
    lo, hi = (float(np.percentile(bs, 5)), float(np.percentile(bs, 95))) if bs else (float("nan"), float("nan"))
    if cfg["mode"] == "ratio":
        bias = lvl_o / lvl_m if lvl_m else float("nan"); level_ok = 0.7 <= bias <= 1.43; bias_txt = f"observed {lvl_o:.0f} against model {lvl_m:.0f} {cfg['unit']} (ratio {bias:.2f})"
    else:
        bias = lvl_o - lvl_m; level_ok = abs(bias) <= 2.0; bias_txt = f"observed {lvl_o:.1f} against model {lvl_m:.1f} {cfg['unit']} (difference {bias:+.1f})"
    trend_ok = lo <= sm <= hi
    same_sign = (so == 0) or (sm * so > 0)
    # can the observed record tell trends apart at all? (width of its range against the size of the model trend)
    resolvable = (hi - lo) < 4 * max(abs(sm), 1e-9)
    if level_ok and trend_ok:
        verdict, tag = "pass", "The ensemble reproduces both the level and the trend that were observed here."
    elif level_ok:
        verdict, tag = "partial", f"The ensemble matches the observed level but its trend ({sm:+.2g} per decade) lies outside the observed range ({lo:+.2g} to {hi:+.2g})."
    elif trend_ok:
        verdict, tag = "partial", "The ensemble's trend is consistent with the observed one but its level differs from the observed level."
    else:
        verdict, tag = "fail", "The ensemble does not reproduce the observed level or trend here."
    return {"ok": True, "verdict": verdict, "summary": tag, "level_ok": level_ok, "trend_ok": trend_ok, "same_sign": same_sign, "trend_resolvable": resolvable,
            "bias": bias, "bias_text": bias_txt, "obs_slope_decade": so, "obs_slope_range": [lo, hi], "model_slope_decade": sm, "n_obs": int(sel.sum()), "unit": cfg["unit"], "mode": cfg["mode"]}


def observed_series(hazard: str, lat: float, lon: float, fetch=None):
    """Observed annual maxima of the variable the climate model projects (precipitation or air temperature) from NASA POWER, cached with the study data."""
    from . import data as D
    par = "PRECTOTCORR" if hazard == "rain" else "T2M_MAX"
    d = D.power_daily(lat, lon, 1981, 2023, par) if fetch is None else D.power_daily(lat, lon, 1981, 2023, par, fetch=fetch)
    by: dict = {}
    for t, v in zip(d["time"], d[par]):
        if v is not None:
            by[int(t[:4])] = max(by.get(int(t[:4]), -1e9), float(v))
    ys = sorted(by)
    return ys, [by[y] for y in ys]
