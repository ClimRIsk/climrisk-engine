"""
Validate the engine's constructed hourly UTCI (heat AND cold stress) against weather stations, on held-out sites, for the regions our
clients are in (Europe, US, Canada) and the regions their industries sit in (South and Southeast Asia, the Gulf, Africa, South America).

Truth chain   NOAA ISD-Lite station air temperature and dew point (+ ERA5 wind, pressure, sunshine, identical in both chains)
Engine chain  ERA5 2 m temperature and humidity (+ the same wind, pressure, sunshine)
Both run through cri.modellab.utci.hourly_utci (Liljegren globe temperature -> mean radiant temperature -> Broede UTCI polynomial).
This tests the reanalysis inputs and how errors show up in the stress categories; it does not test UTCI against human thermal
sensation, which is what the UTCI model itself was validated for in the literature (Broede et al. 2012; Blazejczyk et al. 2013).

Split: sites alternate calibration / held-out by list order within each region. Two additive corrections (one for the hottest 5 % of hours, one for the
coldest 5 %) are fitted on calibration sites and scored on held-out sites.

  python scripts/validate_utci.py --stage fetch      (resumable)
  python scripts/validate_utci.py --stage evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_wbgt as VW                                         # noqa: E402  (ISD-Lite readers, station list, cache folder)

ROOT = VW.ROOT
sys.path.insert(0, str(ROOT / "src"))
RAW, YEARS = VW.RAW, VW.YEARS
OUT = ROOT / "data" / "backtest" / "utci_validation.json"
# name, lat, lon, region
SITES = [("Madrid ES", 40.47, -3.56, "Europe"), ("Seville ES", 37.42, -5.90, "Europe"), ("Athens GR", 37.94, 23.94, "Europe"), ("Rome IT", 41.80, 12.24, "Europe"), ("Paris FR", 49.01, 2.55, "Europe"),
         ("Berlin DE", 52.37, 13.50, "Europe"), ("London UK", 51.47, -0.45, "Europe"), ("Oslo NO", 60.19, 11.10, "Europe"), ("Stockholm SE", 59.65, 17.93, "Europe"), ("Helsinki FI", 60.32, 24.96, "Europe"),
         ("Warsaw PL", 52.17, 20.97, "Europe"), ("Bucharest RO", 44.57, 26.08, "Europe"), ("Lisbon PT", 38.77, -9.13, "Europe"), ("Vienna AT", 48.11, 16.57, "Europe"), ("Amsterdam NL", 52.31, 4.76, "Europe"),
         ("Milan IT", 45.63, 8.72, "Europe"),
         ("Phoenix US", 33.43, -112.01, "US"), ("Houston US", 29.98, -95.36, "US"), ("Miami US", 25.79, -80.29, "US"), ("Chicago US", 41.98, -87.90, "US"), ("New York US", 40.64, -73.78, "US"),
         ("Seattle US", 47.45, -122.31, "US"), ("Denver US", 39.86, -104.67, "US"), ("Atlanta US", 33.64, -84.43, "US"), ("Los Angeles US", 33.94, -118.41, "US"), ("Minneapolis US", 44.88, -93.22, "US"),
         ("Dallas US", 32.90, -97.04, "US"),
         ("Toronto CA", 43.68, -79.63, "Canada"), ("Calgary CA", 51.11, -114.02, "Canada"), ("Edmonton CA", 53.31, -113.58, "Canada"), ("Winnipeg CA", 49.91, -97.24, "Canada"),
         ("Yellowknife CA", 62.46, -114.44, "Canada"), ("Montreal CA", 45.47, -73.74, "Canada"), ("Vancouver CA", 49.19, -123.18, "Canada"), ("Halifax CA", 44.88, -63.51, "Canada")]
SITES += [(n, la, lo, "Industrial regions elsewhere") for n, la, lo, _ in VW.SITES if n not in ("Oslo NO", "Phoenix US", "Houston US", "Miami US")]
STATIONS = RAW / "stations_utci.json"


def _once(url: str, timeout: float = 150.0) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=VW.UA), timeout=timeout).read()


def stage_fetch(budget_s: float = 150.0) -> None:
    t0 = time.time()
    hist = VW._isd_history()
    sel = json.loads(STATIONS.read_text()) if STATIONS.exists() else {}
    for name, lat, lon, _ in SITES:
        if name in sel:
            continue
        cand = [s for s in hist if s["icao"] and s["begin"] <= 2017 and s["end"] >= 2023 and VW._km(lat, lon, s["lat"], s["lon"]) < 40]
        cand.sort(key=lambda s: VW._km(lat, lon, s["lat"], s["lon"]))
        sel[name] = cand[:3]
    STATIONS.write_text(json.dumps(sel, indent=1))
    for name, lat, lon, _ in SITES:
        if time.time() - t0 > budget_s:
            print("budget reached; run again")
            return
        for st in sel.get(name, [])[:3]:
            for y in YEARS:
                f = RAW / f"isd_{st['usaf']}_{st['wban']}_{y}.gz"
                if not f.exists():
                    try:
                        f.write_bytes(_once(f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{y}/{st['usaf']}-{st['wban']}-{y}.gz", 60))
                    except Exception:                              # noqa: BLE001
                        f.write_bytes(b"")
        f = RAW / f"era5_{lat:.2f}_{lon:.2f}.json"
        if not f.exists():
            q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": f"{YEARS[0]}-01-01", "end_date": f"{YEARS[-1]}-12-31", "timezone": "UTC", "wind_speed_unit": "ms",
                                        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,shortwave_radiation"})
            try:
                f.write_bytes(_once("https://archive-api.open-meteo.com/v1/archive?" + q))
            except urllib.error.HTTPError as e:
                print(name, "ERA5 HTTP", e.code, "(rate limit: stop and rerun later)" if e.code == 429 else "")
                if e.code == 429:
                    return
            except Exception as e:                                 # noqa: BLE001
                print(name, "ERA5 error", str(e)[:80])
    print("fetch complete")


def _pair(name, lat, lon, st):
    from cri.modellab import utci as U
    f = RAW / f"era5_{lat:.2f}_{lon:.2f}.json"
    if not f.exists():
        return None
    e = json.loads(f.read_bytes())["hourly"]
    et = np.array(e["time"], dtype="datetime64[h]")
    A = {k: np.array([np.nan if v is None else v for v in e[k]], float) for k in ("temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "shortwave_radiation")}
    o = VW._isd(st)
    if len(o["t"]) < 2000:
        return None
    idx = {t: i for i, t in enumerate(et)}
    keep = [(i, idx.get(np.datetime64(t, "h"))) for i, t in enumerate(o["t"])]
    keep = [(i, j) for i, j in keep if j is not None and np.isfinite(A["temperature_2m"][j]) and np.isfinite(A["shortwave_radiation"][j]) and np.isfinite(A["wind_speed_10m"][j])]
    if len(keep) < 2000:
        return None
    oi = np.array([k[0] for k in keep]); ej = np.array([k[1] for k in keep])
    tmid = et[ej].astype("datetime64[s]") - np.timedelta64(1800, "s")
    p = np.where(np.isfinite(A["surface_pressure"][ej]), A["surface_pressure"][ej], 1000.0)
    sw = np.nan_to_num(A["shortwave_radiation"][ej], nan=0.0)
    w = A["wind_speed_10m"][ej]
    stn = U.hourly_utci(o["T"][oi], VW._rh(o["T"][oi], o["Td"][oi]), p, w, sw, tmid, lat, lon)["utci"]
    era = U.hourly_utci(A["temperature_2m"][ej], np.clip(A["relative_humidity_2m"][ej], 1, 100), p, w, sw, tmid, lat, lon)["utci"]
    ok = np.isfinite(stn) & np.isfinite(era)
    return {"stn": stn[ok], "era": era[ok], "t_bias": float(np.mean(A["temperature_2m"][ej] - o["T"][oi])), "station": st}


def stage_evaluate() -> None:
    sel = json.loads(STATIONS.read_text())
    rows, data = [], {}
    k_in_region: dict = {}
    for name, lat, lon, reg in SITES:
        got = None
        for st in sel.get(name, []):
            got = _pair(name, lat, lon, st)
            if got is not None:
                break
        if got is None:
            rows.append({"site": name, "region": reg, "status": "no usable station record or ERA5 file"})
            continue
        s, e = got["stn"], got["era"]
        n = k_in_region.get(reg, 0); k_in_region[reg] = n + 1
        split = "calibration" if n % 2 == 0 else "held-out"
        hot, cold = s >= np.nanpercentile(s, 95), s <= np.nanpercentile(s, 5)
        yrs = len(YEARS)
        r = {"site": name, "region": reg, "station": f'{got["station"]["name"]} ({VW._km(lat, lon, got["station"]["lat"], got["station"]["lon"]):.0f} km)', "split": split, "n_hours": int(len(s)), "t2m_bias_c": round(got["t_bias"], 2),
             "utci_bias_all_c": round(float(np.mean(e - s)), 2), "utci_rmse_c": round(float(np.sqrt(np.mean((e - s) ** 2))), 2), "utci_corr": round(float(np.corrcoef(e, s)[0, 1]), 3),
             "bias_hottest5pct_c": round(float(np.mean((e - s)[hot])), 2), "bias_coldest5pct_c": round(float(np.mean((e - s)[cold])), 2),
             "hours_per_year": {lab: {"station": round(float((s >= th).sum() / yrs * (8766 / len(s)) * yrs), 0), "era5": round(float((e >= th).sum() / yrs * (8766 / len(s)) * yrs), 0)} for lab, th in (("heat_ge_32", 32.0), ("strong_heat_ge_38", 38.0))},
             "cold_hours_per_year": {lab: {"station": round(float((s <= th).sum() * 8766 / len(s)), 0), "era5": round(float((e <= th).sum() * 8766 / len(s)), 0)} for lab, th in (("cold_le_minus13", -13.0), ("very_strong_cold_le_minus27", -27.0))}}
        rows.append(r)
        data[name] = (s, e, hot, cold, split)
    cal = [t for t in data.values() if t[4] == "calibration"]
    c_hot = -float(np.mean([np.mean((e - s)[h]) for s, e, h, c, _ in cal])) if cal else 0.0
    c_cold = -float(np.mean([np.mean((e - s)[c]) for s, e, h, c, _ in cal])) if cal else 0.0
    held = [t for t in data.values() if t[4] == "held-out"]
    sc = lambda f, sel_i: float(np.mean([abs(np.mean((e + f - s)[t[sel_i]])) for t in held for s, e in [(t[0], t[1])]])) if held else None   # noqa: E731
    by_reg = {}
    for reg in sorted({r["region"] for r in rows if "utci_bias_all_c" in r}):
        sub = [r for r in rows if r.get("region") == reg and "utci_bias_all_c" in r]
        by_reg[reg] = {"sites": len(sub), "mean_bias_c": round(float(np.mean([r["utci_bias_all_c"] for r in sub])), 2), "mean_rmse_c": round(float(np.mean([r["utci_rmse_c"] for r in sub])), 2),
                       "hottest5pct_bias_c": round(float(np.mean([r["bias_hottest5pct_c"] for r in sub])), 2), "coldest5pct_bias_c": round(float(np.mean([r["bias_coldest5pct_c"] for r in sub])), 2)}
    summ = {"sites_with_data": len(data), "by_region": by_reg, "calibration_correction_hot_c": round(c_hot, 2), "calibration_correction_cold_c": round(c_cold, 2),
            "held_out_hot_mean_abs_bias_before_c": round(sc(0.0, 2), 2) if held else None, "held_out_hot_mean_abs_bias_after_c": round(sc(c_hot, 2), 2) if held else None,
            "held_out_cold_mean_abs_bias_before_c": round(sc(0.0, 3), 2) if held else None, "held_out_cold_mean_abs_bias_after_c": round(sc(c_cold, 3), 2) if held else None}
    OUT.write_text(json.dumps({"sites": rows, "summary": summ, "years": YEARS, "limits": [
        "Truth is ISD-Lite airport air temperature and dew point; wind, pressure and sunshine are ERA5 in both chains, so their errors are not tested.",
        "No open archive of measured UTCI or mean radiant temperature exists; the globe-to-radiant step and UTCI itself rest on ISO 7726 and on Broede et al. 2012.",
        "Airport stations are open and flat; city centres and industrial yards run hotter at night and in the sun."]}, indent=1))
    print(json.dumps(summ, indent=1))
    for r in rows:
        if "utci_bias_all_c" in r:
            print(f'{r["site"]:18s} {r["region"][:10]:10s} {r["split"][:5]} bias {r["utci_bias_all_c"]:+.2f} rmse {r["utci_rmse_c"]:.2f} hot5 {r["bias_hottest5pct_c"]:+.2f} cold5 {r["bias_coldest5pct_c"]:+.2f}')
        else:
            print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "evaluate"])
    a = ap.parse_args()
    {"fetch": stage_fetch, "evaluate": stage_evaluate}[a.stage]()
