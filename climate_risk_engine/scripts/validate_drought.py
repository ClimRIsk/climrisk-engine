"""
Validate the engine's drought model against OBSERVED drought: the US Drought Monitor (USDM).

The engine's drought study fits a soil water balance to ERA5-Land soil moisture (a land-surface model, not an observation). That tests the
engine against another model. This script tests it against the thing that is used to declare drought: the weekly USDM maps, which
blend rain gauges, streamflow, soil probes, satellites and local expert reports into drought categories by county.

  truth   USDM weekly share of each county in drought (D1 or worse, D2 or worse), 2000 to 2023
  model   the engine's soil-water-balance (same code, structure chosen by AIC) fitted on 2000-2010 only; scored on 2011-2023
  score   ROC area (does a drier modelled soil rank drought weeks above non-drought weeks?) and drought-week detection, against
          two baselines that need no physics: 30-day rain deficit and ERA5-Land soil moisture itself.

  python scripts/validate_drought.py --stage fetch
  python scripts/validate_drought.py --stage evaluate     (resumable; run until it says complete)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "backtest" / "raw" / "drought"
OUT = ROOT / "data" / "backtest" / "drought_validation.json"
SITES = [("Story IA", "19169", 42.03, -93.62, "rainfed"), ("Lancaster NE", "31109", 40.81, -96.70, "rainfed"), ("Ford KS", "20057", 37.75, -100.02, "irrigated plains"),
         ("Lubbock TX", "48303", 33.58, -101.86, "irrigated plains"), ("Travis TX", "48453", 30.27, -97.74, "semi-humid"), ("Fresno CA", "06019", 36.74, -119.79, "irrigated"),
         ("Yakima WA", "53077", 46.60, -120.51, "irrigated"), ("Cook IL", "17031", 41.88, -87.63, "urban humid"), ("Hennepin MN", "27053", 44.98, -93.27, "humid"),
         ("Fulton GA", "13121", 33.75, -84.39, "humid"), ("Orange FL", "12095", 28.54, -81.38, "humid subtropical"), ("Maricopa AZ", "04013", 33.45, -112.07, "desert"),
         ("Bernalillo NM", "35001", 35.08, -106.65, "arid"), ("Larimer CO", "08069", 40.59, -105.08, "semi-arid"), ("Cascade MT", "30013", 47.50, -111.30, "semi-arid"),
         ("Ada ID", "16001", 43.62, -116.20, "semi-arid irrigated"), ("Franklin OH", "39049", 39.96, -83.00, "humid"), ("Dauphin PA", "42043", 40.26, -76.88, "humid"),
         ("Wake NC", "37183", 35.78, -78.64, "humid"), ("East Baton Rouge LA", "22033", 30.45, -91.15, "humid subtropical"), ("Greene MO", "29077", 37.21, -93.29, "humid"),
         ("Cleveland OK", "40027", 35.22, -97.44, "semi-humid"), ("Weld CO", "08123", 40.42, -104.71, "semi-arid"), ("Marion IN", "18097", 39.77, -86.16, "humid"),
         ("Hinds MS", "28049", 32.30, -90.18, "humid"), ("Salt Lake UT", "49035", 40.76, -111.89, "arid irrigated")]
SPLIT = "2011-01-01"
UA = {"User-Agent": "climrisk-research/0.1 (drought validation)", "Accept": "application/json"}


def _usdm(fips: str) -> list:
    f = RAW / f"usdm_{fips}.json"
    if f.exists():
        return json.loads(f.read_text())
    u = f"https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent?aoi={fips}&startdate=1/1/2000&enddate=12/31/2023&statisticsType=1"
    d = json.loads(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=90).read())
    rows = [{"date": r["mapDate"][:10], "d1": r["d1"], "d2": r["d2"]} for r in d]
    f.write_text(json.dumps(rows))
    return rows


def stage_fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    from cri.modellab import data as D
    t0 = time.time()
    for name, fips, lat, lon, _ in SITES:
        if time.time() - t0 > 120:
            print("budget reached; run again")
            return
        _usdm(fips)
        D.archive(lat, lon, 1999, 2023, daily="precipitation_sum,et0_fao_evapotranspiration,soil_moisture_7_to_28cm_mean")
    print("fetch complete")


def _auc(score: np.ndarray, truth: np.ndarray) -> float:
    pos, neg = score[truth], score[~truth]
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _site(name, fips, lat, lon, kind) -> dict:
    from cri.modellab import data as D, studies_series as SS
    d = D.archive(lat, lon, 1999, 2023, daily="precipitation_sum,et0_fao_evapotranspiration,soil_moisture_7_to_28cm_mean")["daily"]
    keep = [i for i in range(len(d["time"])) if None not in (d["precipitation_sum"][i], d["et0_fao_evapotranspiration"][i], d["soil_moisture_7_to_28cm_mean"][i])]
    t = [d["time"][i] for i in keep]
    P = np.array([d["precipitation_sum"][i] for i in keep], float); E0 = np.array([d["et0_fao_evapotranspiration"][i] for i in keep], float)
    th = np.array([d["soil_moisture_7_to_28cm_mean"][i] for i in keep], float)
    doy = SS._doy(t)
    cut = next(i for i, x in enumerate(t) if x >= SPLIT); burn = 365
    fits = SS._fit_all({k: SS.DR[k] for k in ("M0", "M1", "M2")}, lambda k, x: SS._soil_sim(k, x, P, E0, doy), th, cut, burn, restarts=1)
    best = min(fits, key=lambda k: fits[k]["aic"])
    sim = SS._soil_sim(best, fits[best]["x"], P, E0, doy)
    cs, ct = SS._doy_clim(sim, doy, cut, burn), SS._doy_clim(th, doy, cut, burn)
    a_sim = sim - np.array([cs[int(x)] for x in doy]); a_th = th - np.array([ct[int(x)] for x in doy])
    r30 = np.convolve(P, np.ones(30), "full")[: len(P)]
    r30a = r30 - np.array([np.mean(r30[burn:cut][np.abs(doy[burn:cut] - x) <= 7]) for x in doy])
    thr_s = np.percentile(a_sim[burn:cut], 10)
    idx = {x: i for i, x in enumerate(t)}
    rows = [r for r in _usdm(fips) if r["date"] in idx and r["date"] >= SPLIT]
    ii = np.array([idx[r["date"]] for r in rows])
    out = {"site": name, "fips": fips, "kind": kind, "structure": best, "n_weeks_test": int(len(rows)), "train_rmse": round(float(fits[best]["rms"]), 4)}
    for lab, th_ in (("D1+_50pct", ("d1", 50.0)), ("D2+_25pct", ("d2", 25.0))):
        tr = np.array([r[th_[0]] >= th_[1] for r in rows])
        res = {"drought_weeks": int(tr.sum()), "auc_engine_model": round(_auc(-a_sim[ii], tr), 3), "auc_era5land_soil_moisture": round(_auc(-a_th[ii], tr), 3), "auc_30day_rain_deficit": round(_auc(-r30a[ii], tr), 3)}
        flag = a_sim[ii] < thr_s
        res["pod"] = round(float((flag & tr).sum() / max(tr.sum(), 1)), 2); res["far"] = round(float((flag & ~tr).sum() / max(flag.sum(), 1)), 2)
        out[lab] = res
    return out


def stage_evaluate() -> None:
    f = RAW / "results.json"
    done = json.loads(f.read_text()) if f.exists() else {}
    t0 = time.time()
    for name, fips, lat, lon, kind in SITES:
        if name in done:
            continue
        if time.time() - t0 > 110:
            print("budget reached; run again:", len(done), "of", len(SITES))
            f.write_text(json.dumps(done))
            return
        try:
            done[name] = _site(name, fips, lat, lon, kind)
        except Exception as e:                                     # noqa: BLE001
            done[name] = {"site": name, "error": f"{type(e).__name__}: {str(e)[:100]}"}
    f.write_text(json.dumps(done))
    rows = [r for r in done.values() if "auc_engine_model" in r.get("D1+_50pct", {})]
    summ = {}
    for lab in ("D1+_50pct", "D2+_25pct"):
        ok = [r for r in rows if r[lab]["drought_weeks"] >= 10]
        summ[lab] = {"sites_scored": len(ok), "median_auc_engine": round(float(np.nanmedian([r[lab]["auc_engine_model"] for r in ok])), 3), "median_auc_era5land": round(float(np.nanmedian([r[lab]["auc_era5land_soil_moisture"] for r in ok])), 3),
                     "median_auc_rain_deficit": round(float(np.nanmedian([r[lab]["auc_30day_rain_deficit"] for r in ok])), 3), "median_pod": round(float(np.nanmedian([r[lab]["pod"] for r in ok])), 2), "median_far": round(float(np.nanmedian([r[lab]["far"] for r in ok])), 2),
                     "sites_engine_auc_above_0.7": int(sum(r[lab]["auc_engine_model"] > 0.7 for r in ok)), "sites_engine_beats_rain_deficit": int(sum(r[lab]["auc_engine_model"] > r[lab]["auc_30day_rain_deficit"] for r in ok))}
    OUT.write_text(json.dumps({"sites": list(done.values()), "summary": summ, "split": SPLIT,
                               "limits": ["USDM is an expert-blended product, not a gauge: it is the operational definition of drought in the US but not independent of rainfall and soil data.",
                                          "Irrigated sites (Fresno, Yakima, Ford, Lubbock, Ada, Salt Lake) are decided by water management that the model cannot see.",
                                          "US only: no open drought-declaration archive of this quality exists elsewhere. Tropical, monsoon and snow-fed systems are untested."]}, indent=1))
    print("complete")
    print(json.dumps(summ, indent=1))
    for r in done.values():
        if "error" in r:
            print(r)
        else:
            print(f'{r["site"]:20s} {r["kind"]:20s} {r["structure"]} D1+ auc eng {r["D1+_50pct"]["auc_engine_model"]} era5l {r["D1+_50pct"]["auc_era5land_soil_moisture"]} rain {r["D1+_50pct"]["auc_30day_rain_deficit"]} weeks {r["D1+_50pct"]["drought_weeks"]}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "evaluate"])
    a = ap.parse_args()
    {"fetch": stage_fetch, "evaluate": stage_evaluate}[a.stage]()
