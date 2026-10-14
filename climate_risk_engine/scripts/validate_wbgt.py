"""
Validate the engine's heat-stress input chain against real weather-station observations, on held-out sites.

Question: if the engine builds hourly WBGT from reanalysis (ERA5: temperature, humidity, wind), does it get the same heat stress as the
same method driven by what a weather station actually measured at that place and hour?

  truth chain   NOAA ISD-Lite station air temperature, dew point, 10 m wind  (+ ERA5 pressure and sunshine, identical in both chains)
  engine chain  ERA5 2 m temperature, humidity, 10 m wind                    (+ the same ERA5 pressure and sunshine)
  old method    shade shortcut sWBGT = 0.567 T + 0.393 e + 3.94 on ERA5      (what the engine used before)

Both chains use the Liljegren (2008) outdoor WBGT in cri.modellab.wbgt (a port checked against the reference C code). The comparison
therefore tests the INPUTS (reanalysis vs station) and quantifies the old shortcut. It does not test Liljegren itself against
black-globe measurements: no open archive of measured WBGT exists. Liljegren et al. (2008) and Kong & Huber (2022) report that test.

Method: sites are split into a calibration half and a held-out half (alternating by list order). A single additive correction is fitted
on the hot hours of the calibration sites and scored on the held-out sites. The result says whether a correction helps at all.

  python scripts/validate_wbgt.py --stage fetch      (resumable)
  python scripts/validate_wbgt.py --stage evaluate
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "backtest" / "raw" / "wbgt"
OUT = ROOT / "data" / "backtest" / "wbgt_validation.json"
YEARS = list(range(2018, 2023))
UA = {"User-Agent": "climrisk-research/0.1 (wbgt validation)"}
# name, lat, lon, climate
SITES = [("Virudhunagar/Madurai IN", 9.83, 78.09, "hot-humid"), ("Dhaka BD", 23.84, 90.40, "hot-humid"), ("Kolkata IN", 22.65, 88.45, "hot-humid"),
         ("Houston US", 29.98, -95.36, "hot-humid"), ("Phoenix US", 33.43, -112.01, "hot-dry"), ("Jacobabad PK", 28.28, 68.45, "hot-dry"),
         ("Riyadh SA", 24.96, 46.70, "hot-dry"), ("Oslo NO", 60.19, 11.10, "cool"), ("Bangkok TH", 13.69, 100.75, "hot-humid"), ("Dubai AE", 25.25, 55.36, "hot-humid"),
         ("Delhi IN", 28.57, 77.12, "hot-semi-arid"), ("Chennai IN", 12.99, 80.18, "hot-humid"), ("Lagos NG", 6.58, 3.32, "hot-humid"), ("Miami US", 25.79, -80.29, "hot-humid"),
         ("Karachi PK", 24.91, 67.16, "hot-humid"), ("Brisbane AU", -27.38, 153.12, "warm-humid"), ("Manaus BR", -3.04, -60.05, "hot-humid"), ("Singapore SG", 1.36, 103.99, "hot-humid"),
         ("Basra IQ", 30.55, 47.66, "hot-humid"), ("Lahore PK", 31.52, 74.40, "hot-semi-arid")]


def _get(url: str, timeout: float = 120.0) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def _isd_history() -> list:
    f = RAW / "isd-history.csv"
    if not f.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        f.write_bytes(_get("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"))
    out = []
    for r in csv.DictReader(io.StringIO(f.read_text(errors="ignore"))):
        try:
            out.append({"usaf": r["USAF"], "wban": r["WBAN"], "name": r["STATION NAME"], "lat": float(r["LAT"]), "lon": float(r["LON"]), "elev": float(r["ELEV(M)"] or 0),
                        "begin": int(r["BEGIN"][:4]), "end": int(r["END"][:4]), "icao": r["ICAO"]})
        except (ValueError, KeyError):
            continue
    return out


def _km(a, b, c, d):
    return float(np.hypot((c - a) * 111.32, (d - b) * 111.32 * np.cos(np.radians(a))))


def stage_fetch(budget_s: float = 150.0) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hist = _isd_history()
    chosen = RAW / "stations.json"
    sel = json.loads(chosen.read_text()) if chosen.exists() else {}
    for name, lat, lon, clim in SITES:
        if name in sel:
            continue
        cand = [s for s in hist if s["icao"] and s["begin"] <= 2017 and s["end"] >= 2023 and _km(lat, lon, s["lat"], s["lon"]) < 40]
        cand.sort(key=lambda s: _km(lat, lon, s["lat"], s["lon"]))
        sel[name] = cand[:3]
    chosen.write_text(json.dumps(sel, indent=1))
    for name, lat, lon, clim in SITES:
        for st in sel.get(name, [])[:3]:
            for y in YEARS:
                f = RAW / f"isd_{st['usaf']}_{st['wban']}_{y}.gz"
                if f.exists():
                    continue
                try:
                    f.write_bytes(_get(f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{y}/{st['usaf']}-{st['wban']}-{y}.gz", 60))
                except Exception:                                  # noqa: BLE001
                    f.write_bytes(b"")
                if time.time() - t0 > budget_s:
                    print("budget reached; run again")
                    return
        f = RAW / f"era5_{lat:.2f}_{lon:.2f}.json"
        if not f.exists():
            q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": f"{YEARS[0]}-01-01", "end_date": f"{YEARS[-1]}-12-31", "timezone": "UTC", "wind_speed_unit": "ms",
                                        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,shortwave_radiation"})
            for wait in (0, 20, 60):
                time.sleep(wait)
                try:
                    f.write_bytes(_get("https://archive-api.open-meteo.com/v1/archive?" + q, 150))
                    break
                except Exception as e:                             # noqa: BLE001
                    print(name, "ERA5 retry:", str(e)[:80])
            if time.time() - t0 > budget_s:
                print("budget reached; run again")
                return
    print("fetch complete")


def _isd(st: dict) -> dict:
    rows = []
    for y in YEARS:
        f = RAW / f"isd_{st['usaf']}_{st['wban']}_{y}.gz"
        if not f.exists() or f.stat().st_size == 0:
            continue
        for ln in gzip.decompress(f.read_bytes()).decode().splitlines():
            p = ln.split()
            if len(p) < 11:
                continue
            t, td, ws = int(p[4]), int(p[5]), int(p[9])
            if t == -9999 or td == -9999:
                continue
            rows.append((np.datetime64(f"{p[0]}-{p[1]}-{p[2]}T{p[3]}:00"), t / 10.0, td / 10.0, ws / 10.0 if ws != -9999 else np.nan))
    return {"t": np.array([r[0] for r in rows]), "T": np.array([r[1] for r in rows]), "Td": np.array([r[2] for r in rows]), "ws": np.array([r[3] for r in rows])}


def _rh(t, td):
    return np.clip(100.0 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t)), 1.0, 100.0)


def _pair(name, lat, lon, st, W, heat):
    f = RAW / f"era5_{lat:.2f}_{lon:.2f}.json"
    if not f.exists():
        return None
    e = json.loads(f.read_bytes())["hourly"]
    et = np.array(e["time"], dtype="datetime64[h]")
    arr = {k: np.array([np.nan if v is None else v for v in e[k]], float) for k in ("temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "shortwave_radiation")}
    o = _isd(st)
    if len(o["t"]) < 2000:
        return None
    idx = {t: i for i, t in enumerate(et)}
    keep = [(i, idx.get(np.datetime64(t, "h"))) for i, t in enumerate(o["t"])]
    keep = [(i, j) for i, j in keep if j is not None and np.isfinite(arr["temperature_2m"][j]) and np.isfinite(arr["shortwave_radiation"][j])]
    if len(keep) < 2000:
        return None
    oi = np.array([k[0] for k in keep]); ej = np.array([k[1] for k in keep])
    tt = et[ej]
    half = np.timedelta64(30, "m")
    tmid = tt.astype("datetime64[s]") - half                          # hourly-mean radiation is centred half an hour earlier
    p = np.where(np.isfinite(arr["surface_pressure"][ej]), arr["surface_pressure"][ej], 1000.0)
    sw = np.nan_to_num(arr["shortwave_radiation"][ej], nan=0.0)
    ws_o = np.where(np.isfinite(o["ws"][oi]), o["ws"][oi], arr["wind_speed_10m"][ej])
    stn = W.wbgt(o["T"][oi], _rh(o["T"][oi], o["Td"][oi]), p, arr["wind_speed_10m"][ej], sw, tmid, lat, lon)["wbgt"]   # primary truth: station T and humidity, ERA5 wind
    stn_w = W.wbgt(o["T"][oi], _rh(o["T"][oi], o["Td"][oi]), p, ws_o, sw, tmid, lat, lon)["wbgt"]                       # sensitivity: station wind too
    era = W.wbgt(arr["temperature_2m"][ej], np.clip(arr["relative_humidity_2m"][ej], 1, 100), p, arr["wind_speed_10m"][ej], sw, tmid, lat, lon)["wbgt"]
    old = heat.wbgt_shade(arr["temperature_2m"][ej], np.clip(arr["relative_humidity_2m"][ej], 1, 100))
    ok = np.isfinite(stn) & np.isfinite(era) & np.isfinite(old) & np.isfinite(stn_w)
    return {"stn": stn[ok], "stn_w": stn_w[ok], "wind_missing_share": float(np.isnan(o["ws"][oi]).mean()), "wind_station_mean": float(np.nanmean(o["ws"][oi])), "wind_era5_mean": float(np.mean(arr["wind_speed_10m"][ej])), "era": era[ok], "old": old[ok], "t_stn": o["T"][oi][ok], "t_era": arr["temperature_2m"][ej][ok], "time": tt[ok], "station": st}


def stage_evaluate() -> None:
    from cri.modellab import wbgt as W, heat
    sel = json.loads((RAW / "stations.json").read_text())
    res, data = [], {}
    for k, (name, lat, lon, clim) in enumerate(SITES):
        got = None
        for st in sel.get(name, []):
            got = _pair(name, lat, lon, st, W, heat)
            if got is not None:
                break
        if got is None:
            res.append({"site": name, "climate": clim, "status": "no usable station record"})
            continue
        s, e, o = got["stn"], got["era"], got["old"]
        hot = s >= np.nanpercentile(s, 95)                      # hottest 5% by the station chain (biased low for the ERA5 chain by selection)
        hot_e = e >= np.nanpercentile(e, 95)                    # hottest 5% by the ERA5 chain (biased high by selection); the truth lies between the two
        sw_ = got["stn_w"]
        thr = {}
        for th in (28.0, 30.0, 32.0):
            thr[str(th)] = {"station_share_%": round(float((s >= th).mean() * 100), 3), "era5_share_%": round(float((e >= th).mean() * 100), 3), "old_shade_share_%": round(float((o >= th).mean() * 100), 3)}
        r = {"site": name, "climate": clim, "station": f'{got["station"]["name"]} ({got["station"]["usaf"]}-{got["station"]["wban"]}), {_km(lat, lon, got["station"]["lat"], got["station"]["lon"]):.0f} km', "n_hours": int(len(s)),
             "split": "calibration" if k % 2 == 0 else "held-out", "t_bias_c": round(float(np.mean(got["t_era"] - got["t_stn"])), 2),
             "era5_minus_station_wbgt": {"bias_c": round(float(np.mean(e - s)), 2), "rmse_c": round(float(np.sqrt(np.mean((e - s) ** 2))), 2), "hot_hours_bias_c": round(float(np.mean((e - s)[hot])), 2), "hot_hours_bias_selected_on_era5_c": round(float(np.mean((e - s)[hot_e])), 2), "corr": round(float(np.corrcoef(e, s)[0, 1]), 3)},
             "old_shade_minus_station_wbgt": {"bias_c": round(float(np.mean(o - s)), 2), "hot_hours_bias_c": round(float(np.mean((o - s)[hot])), 2), "rmse_c": round(float(np.sqrt(np.mean((o - s) ** 2))), 2)},
             "station_wind_sensitivity": {"station_wind_missing_share": round(got["wind_missing_share"], 2), "station_wind_mean_ms": round(got["wind_station_mean"], 2), "era5_wind_mean_ms": round(got["wind_era5_mean"], 2), "wbgt_shift_if_station_wind_used_c": round(float(np.mean(sw_ - s)), 2)},
             "max_wbgt_c": {"station_chain": round(float(np.max(s)), 1), "era5_chain": round(float(np.max(e)), 1), "old_shade": round(float(np.max(o)), 1)}, "exceedance": thr}
        res.append(r)
        data[name] = (s, e, hot, k % 2 == 0, hot_e)
    # one additive correction on the hot hours of the calibration sites, scored on the held-out sites
    cal = [np.mean((e - s)[h]) for (s, e, h, c, he) in data.values() if c]
    corr = -float(np.mean(cal)) if cal else 0.0
    before, after = [], []
    for name, (s, e, hot, c, he) in data.items():
        if not c:
            before.append(abs(float(np.mean((e - s)[hot])))); after.append(abs(float(np.mean((e + corr - s)[hot]))))
    held = [r for r in res if r.get("split") == "held-out"]
    summ = {"sites_with_data": len(data), "calibration_correction_c": round(corr, 2),
            "held_out_hot_hours_mean_abs_bias_before_c": round(float(np.mean(before)), 2) if before else None, "held_out_hot_hours_mean_abs_bias_after_c": round(float(np.mean(after)), 2) if after else None,
            "era5_hot_hours_bias_c_all_sites": [round(float(np.mean((e - s)[h])), 2) for (s, e, h, c, he) in data.values()],
            "old_shade_hot_hours_bias_c_all_sites": {r["site"]: r["old_shade_minus_station_wbgt"]["hot_hours_bias_c"] for r in res if "old_shade_minus_station_wbgt" in r},
            "held_out_sites": [r["site"] for r in held]}
    OUT.write_text(json.dumps({"sites": res, "summary": summ, "years": YEARS,
                               "limits": ["No open archive of measured WBGT exists: Liljegren itself is not tested against black-globe data here (see Liljegren 2008, Kong & Huber 2022).",
                                          "Sunshine and pressure are ERA5 in both chains, so radiation error is not tested.",
                                          "Wind is ERA5 in the primary comparison: ISD-Lite wind is missing for 0 to 75% of hours and is measured at sheltered airport masts, so it is reported only as a sensitivity. Calm wind raises WBGT strongly, so this is a real uncertainty.",
                                          "ISD-Lite reports at 1 to 3 hourly intervals at some stations; only reported hours are compared."]}, indent=1))
    for r in res:
        if "era5_minus_station_wbgt" in r:
            print(f'{r["site"]:26s} {r["split"]:12s} n={r["n_hours"]:6d} ERA5 bias {r["era5_minus_station_wbgt"]["bias_c"]:+.2f} hot {r["era5_minus_station_wbgt"]["hot_hours_bias_c"]:+.2f} rmse {r["era5_minus_station_wbgt"]["rmse_c"]:.2f} | old shade bias hot {r["old_shade_minus_station_wbgt"]["hot_hours_bias_c"]:+.2f}')
        else:
            print(r)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fetch", "evaluate"])
    a = ap.parse_args()
    {"fetch": stage_fetch, "evaluate": stage_evaluate}[a.stage]()
