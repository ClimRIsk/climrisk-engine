"""
Validate the tropical-cyclone wind model against MEASURED station winds.

Truth: NOAA ISD-Lite hourly observations (global, public) —
  https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/<year>/<USAF>-<WBAN>-<year>.gz
For each storm, every station within 150 km of the track (while the storm is
at least tropical-storm strength) is compared: max observed hourly sustained
wind during the storm's passage vs the model's peak wind at that station.

Averaging: model output is 1-min sustained. US ASOS reports 2-min means
(×0.93), most other networks 10-min means (×0.88, WMO 2010 conversion).
Hourly snapshots can miss the true peak and anemometers often fail in the
strongest winds, so observations are biased low; stations with < 60 % of
hours present during the passage are dropped.

Writes src/cri/data/tc_validation.json (read by the engine) and
data/backtest/tc_validation_pairs.csv.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cri.climate import tropical_cyclone as tc      # noqa: E402

RAW = ROOT / "data" / "backtest" / "raw" / "isd"
RAW.mkdir(parents=True, exist_ok=True)

STORMS = [  # (name, season) — major landfalls in every basin
    ("ANDREW", 1992), ("KATRINA", 2005), ("WILMA", 2005), ("IKE", 2008), ("SANDY", 2012),
    ("HARVEY", 2017), ("IRMA", 2017), ("MARIA", 2017), ("MICHAEL", 2018), ("LAURA", 2020),
    ("IDA", 2021), ("IAN", 2022),
    ("HAIYAN", 2013), ("RAMMASUN", 2014), ("MERANTI", 2016), ("HATO", 2017), ("JEBI", 2018),
    ("MANGKHUT", 2018), ("HAGIBIS", 2019), ("FAXAI", 2019),
    ("PHAILIN", 2013), ("HUDHUD", 2014), ("FANI", 2019), ("AMPHAN", 2020), ("TAUKTAE", 2021),
    ("BIPARJOY", 2023),
    ("YASI", 2011), ("DEBBIE", 2017), ("WINSTON", 2016), ("IDAI", 2019), ("FREDDY", 2023),
]
MAX_STATIONS = 15
NEAR_KM = 150.0


def _get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ClimRisk-Engine/1.0"}),
                                timeout=timeout) as r:
        return r.read()


def stations() -> list[dict]:
    f = RAW / "isd-history.csv"
    if not f.exists():
        f.write_bytes(_get("https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"))
    out = []
    with f.open(newline="", encoding="latin-1") as fh:
        for r in csv.DictReader(fh):
            try:
                la, lo = float(r["LAT"]), float(r["LON"])
            except ValueError:
                continue
            if la == 0 and lo == 0:
                continue
            out.append({"usaf": r["USAF"], "wban": r["WBAN"], "name": r["STATION NAME"], "ctry": r["CTRY"],
                        "lat": la, "lon": lo, "begin": r["BEGIN"], "end": r["END"]})
    return out


def isd_lite(usaf, wban, year) -> list[tuple]:
    f = RAW / f"{usaf}-{wban}-{year}.txt"
    if not f.exists():
        try:
            raw = _get(f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz")
            f.write_text(gzip.decompress(raw).decode())
        except Exception:                      # noqa: BLE001
            f.write_text("")
    rows = []
    for ln in f.read_text().splitlines():
        p = ln.split()
        if len(p) >= 9:
            rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[8])))
    return rows


def storm_index(T, name, season):
    ok = np.where((T["name"] == name) & (T["season"] == season))[0]
    return int(ok[0]) if len(ok) else None


def main():
    T = tc.tracks()
    st = stations()
    st_lat = np.array([s["lat"] for s in st]); st_lon = np.array([s["lon"] for s in st])
    t0 = np.datetime64("1980-01-01T00:00")
    pairs = []
    jobs = []
    for name, season in STORMS:
        si = storm_index(T, name, season)
        if si is None:
            print("not found", name, season); continue
        ix = np.where(T["storm"] == si)[0]
        s = tc._interp_storm(ix, T)
        strong = s["vmax"] * tc.KT >= 17.5
        cand = []
        for k in np.where(strong)[0][::2]:
            d = tc._haversine_km(st_lat, st_lon, s["lat"][k], s["lon"][k])
            cand += list(np.where(d <= NEAR_KM)[0])
        cand = sorted(set(cand))
        ymd0, ymd1 = f"{season}0101", f"{season}1231"
        cand = [c for c in cand if st[c]["begin"] <= ymd0 and st[c]["end"] >= ymd1]
        dmin = {c: float(np.nanmin(tc._haversine_km(s["lat"], s["lon"], st[c]["lat"], st[c]["lon"]))) for c in cand}
        cand = sorted(cand, key=lambda c: dmin[c])[:MAX_STATIONS]
        jobs.append((name, season, si, ix, s, cand, dmin))

    # download station-years in parallel
    need = {(st[c]["usaf"], st[c]["wban"], season) for _, season, _, _, _, cand, _ in jobs for c in cand}
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda a: isd_lite(*a), need))

    for name, season, si, ix, s, cand, dmin in jobs:
        for c in cand:
            S = st[c]
            obs = isd_lite(S["usaf"], S["wban"], season)
            if not obs:
                continue
            # passage window: hours when the centre is within 400 km
            d = tc._haversine_km(s["lat"], s["lon"], S["lat"], S["lon"])
            win = s["t"][d <= 400.0]
            if not len(win):
                continue
            h0, h1 = win.min() - 3, win.max() + 3
            vals, n_hours = [], int(h1 - h0) + 1
            for (y, m, dd, hh, ws) in obs:
                th = (np.datetime64(f"{y:04d}-{m:02d}-{dd:02d}T{hh:02d}:00") - t0) / np.timedelta64(1, "h")
                if h0 <= th <= h1 and ws != -9999:
                    vals.append(ws / 10.0)
            if len(vals) < 0.6 * n_hours:
                continue
            conv = 0.93 if S["ctry"] == "US" else 0.88
            p_asym, _, _ = tc.storm_peak_at(S["lat"], S["lon"], ix, T, asym=tc.ASYM)
            p_sym, _, _ = tc.storm_peak_at(S["lat"], S["lon"], ix, T, asym=0.0)
            pairs.append({"storm": name, "season": season, "basin": str(T["basins"][T["basin"][si]]),
                          "station": S["name"], "ctry": S["ctry"], "dist_km": round(dmin[c], 1),
                          "obs_ms": max(vals), "model_ms": round(p_asym * conv, 2),
                          "model_sym_ms": round(p_sym * conv, 2)})

    def metrics(key):
        o = np.array([p["obs_ms"] for p in pairs]); m = np.array([p[key] for p in pairs])
        ok = (o > 0) & (m > 0)
        o, m = o[ok], m[ok]
        ratio = m / o
        return {"n": int(len(o)), "pearson_r": round(float(np.corrcoef(o, m)[0, 1]), 3),
                "median_ratio_model_over_obs": round(float(np.median(ratio)), 3),
                "within_30pct": round(float(np.mean(np.abs(ratio - 1) <= 0.3)), 3),
                "rmse_ms": round(float(np.sqrt(np.mean((m - o) ** 2))), 2),
                "gale_hit_rate": round(float(np.mean(m[o >= 17.2] >= 17.2)), 3) if (o >= 17.2).any() else None,
                "gale_false_alarm": round(float(np.mean(o[m >= 17.2] < 17.2)), 3) if (m >= 17.2).any() else None}

    # ── Land/exposure calibration, leave-one-storm-out ───────────────────────
    # One multiplicative factor k (median obs/model, log space) bundles land
    # roughness, station exposure and hourly sampling. Each storm is predicted
    # with k fitted on all OTHER storms, so the metrics below are held-out.
    keys = sorted({p["storm"] + str(p["season"]) for p in pairs})
    def k_fit(ps):
        r = [math.log(p["obs_ms"] / p["model_sym_ms"]) for p in ps if p["obs_ms"] > 0 and p["model_sym_ms"] > 0]
        return math.exp(float(np.median(r)))
    for key in keys:
        k = k_fit([p for p in pairs if p["storm"] + str(p["season"]) != key])
        for p in pairs:
            if p["storm"] + str(p["season"]) == key:
                p["model_cal_ms"] = round(p["model_sym_ms"] * k, 2)
    k_all = k_fit(pairs)

    res = {"n_storms": len({p["storm"] + str(p["season"]) for p in pairs}),
           "calibration": {"k_land_exposure": round(k_all, 3),
                           "note": "applied to symmetric Holland winds; fitted on all storms (reported "
                                   "metrics use leave-one-storm-out k)"},
           "model_calibrated_heldout": metrics("model_cal_ms"),
           "model_with_asymmetry": metrics("model_ms"),
           "model_symmetric": metrics("model_sym_ms"),
           "by_basin": {}}
    for b in sorted({p["basin"] for p in pairs}):
        sub = [p for p in pairs if p["basin"] == b]
        o = np.array([p["obs_ms"] for p in sub]); m = np.array([p["model_cal_ms"] for p in sub])
        res["by_basin"][b] = {"n": len(sub), "pearson_r": round(float(np.corrcoef(o, m)[0, 1]), 3) if len(sub) > 3 else None,
                              "median_ratio": round(float(np.median(m / np.maximum(o, 0.1))), 3)}
    res["notes"] = ["Observations are hourly snapshots (2-min US, 10-min elsewhere); they miss short peaks and "
                    "anemometers often fail in the strongest winds, so observed peaks are biased low.",
                    "Station exposure (buildings, trees) lowers measured winds relative to open terrain."]
    (ROOT / "src/cri/data/tc_validation.json").write_text(json.dumps(res, indent=1))
    with (ROOT / "data/backtest/tc_validation_pairs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys())); w.writeheader(); w.writerows(pairs)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    t = time.time(); main(); print(f"done in {time.time() - t:.0f}s")
