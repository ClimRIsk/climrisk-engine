"""
Validate the engine's measured-air-quality layer (CAMS PM2.5 through the Open-Meteo air-quality API) against real ground monitors.

The engine reports, for any place: annual-mean PM2.5, the 98th-percentile day, and days above the WHO 24 h guideline (15 ug/m3) and the India
NAAQS 24 h standard (60). All of these come from the CAMS analysis (roughly a 10 to 40 km cell). This script asks how far a regulatory
or reference monitor, which is the truth people are held to, agrees, at sites the model was not tuned on.

  truth    US EPA AQS daily PM2.5 (parameter 88101), 2023, plus OpenAQ archive hourly PM2.5 (reference grade where flagged), 2023
  model    Open-Meteo CAMS hourly pm2_5 at the monitor's coordinates, daily means on the local calendar
  split    sites alternate calibration / held-out by list order after sorting by longitude (a spatial split, not a random one)
  fit      one multiplicative factor (ratio of annual means) fitted on calibration sites, applied to the held-out sites

  python scripts/validate_air.py --stage select
  python scripts/validate_air.py --stage fetch       (resumable, run until it says complete)
  python scripts/validate_air.py --stage evaluate
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "backtest" / "raw" / "aq"
OUT = ROOT / "data" / "backtest" / "air_validation.json"
YEAR = 2023
UA = {"User-Agent": "climrisk-research/0.1 (air validation)"}
OAQ = "https://openaq-data-archive.s3.amazonaws.com/records/csv.gz/locationid={i}/year={y}/month={m:02d}/location-{i}-{y}{m:02d}{d:02d}.csv.gz"


def _get(url: str, timeout: float = 60.0) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def _km(a, b, c, d):
    return float(np.hypot((c - a) * 111.32, (d - b) * 111.32 * np.cos(np.radians(a))))


def _region(la, lo):
    if 24 < la < 50 and -125 < lo < -66:
        return "United States"
    if 35 < la < 70 and -12 < lo < 40:
        return "Europe"
    if 6 < la < 36 and 68 < lo < 98:
        return "South Asia"
    if 18 < la < 54 and 73 < lo < 135:
        return "East Asia"
    if -35 < la < 37 and -20 < lo < 52:
        return "Africa"
    if -45 < la < -10 and 110 < lo < 155:
        return "Australia"
    return "Other"


def stage_select() -> None:
    z = zipfile.ZipFile(RAW / f"daily_88101_{YEAR}.zip")
    sites: dict = {}
    with z.open(z.namelist()[0]) as fh:
        for r in csv.DictReader(io.TextIOWrapper(fh)):
            if r["Event Type"] not in ("None", "Included") or r["Sample Duration"] not in ("24 HOUR", "1 HOUR", "24-HR BLK AVG"):
                continue
            k = (r["State Code"], r["County Code"], r["Site Num"], r["POC"])
            s = sites.setdefault(k, {"lat": float(r["Latitude"]), "lon": float(r["Longitude"]), "name": r["Local Site Name"] or r["City Name"], "state": r["State Name"], "days": {}})
            try:
                s["days"][r["Date Local"]] = float(r["Arithmetic Mean"])
            except ValueError:
                pass
    good = [(k, s) for k, s in sites.items() if len(s["days"]) >= 330]
    # one POC per site, then greedy spacing of 250 km so that neighbouring monitors in one CAMS cell are not counted twice
    best: dict = {}
    for k, s in good:
        kk = k[:3]
        if kk not in best or len(s["days"]) > len(best[kk]["days"]):
            best[kk] = s
    random.seed(7)
    pool = list(best.values())
    random.shuffle(pool)
    chosen: list = []
    for s in pool:
        if all(_km(s["lat"], s["lon"], c["lat"], c["lon"]) > 250 for c in chosen):
            chosen.append(s)
        if len(chosen) >= 45:
            break
    out = [{"id": f'AQS-{i}', "source": "US EPA AQS", "lat": s["lat"], "lon": s["lon"], "name": f'{s["name"]}, {s["state"]}', "region": "United States", "days": s["days"]} for i, s in enumerate(chosen)]
    probe = json.loads((RAW / "probe.json").read_text())
    cand = [(k, v) for k, v in probe.items() if v and v["n"] >= 20 and _region(v["lat"], v["lon"]) != "United States"]
    random.shuffle(cand)
    cand.sort(key=lambda kv: _region(kv[1]["lat"], kv[1]["lon"]) == "Europe")     # regions the US sample does not cover come first; Europe fills the rest
    pick: list = []
    for k, v in cand:
        if all(_km(v["lat"], v["lon"], c["lat"], c["lon"]) > 60 for c in pick):
            pick.append({"id": f"OAQ-{k}", "oaq": int(k), "source": "OpenAQ", "lat": v["lat"], "lon": v["lon"], "name": v["name"], "region": _region(v["lat"], v["lon"]), "days": None})
        if len(pick) >= 90:
            break
    (RAW / "sites.json").write_text(json.dumps(out + pick))
    print(len(out), "AQS sites,", len(pick), "OpenAQ candidates")


def _oaq_day(i: int, d) -> list:
    try:
        b = _get(OAQ.format(i=i, y=d.year, m=d.month, d=d.day), 20)
        return [(r["datetime"], float(r["value"])) for r in csv.DictReader(io.StringIO(gzip.decompress(b).decode())) if r["parameter"] == "pm25" and float(r["value"]) >= 0]
    except Exception:                                              # noqa: BLE001
        return []


def _oaq_year(i: int) -> dict:
    import datetime as dt
    days = [dt.date(YEAR, 1, 1) + dt.timedelta(k) for k in range(365)]
    with ThreadPoolExecutor(32) as ex:
        rows = list(ex.map(lambda d: _oaq_day(i, d), days))
    by: dict = {}
    for r in rows:
        for t, v in r:
            by.setdefault(t[:10], []).append(v)
    return {d: float(np.mean(v)) for d, v in by.items() if len(v) >= 16 and d.startswith(str(YEAR))}


def _cams(lat, lon) -> dict:
    q = urllib.parse.urlencode({"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": f"{YEAR}-01-01", "end_date": f"{YEAR}-12-31", "hourly": "pm2_5", "timezone": "auto"})
    for wait in (0, 15, 45):
        time.sleep(wait)
        try:
            d = json.loads(_get("https://air-quality-api.open-meteo.com/v1/air-quality?" + q, 120))
            by: dict = {}
            for t, v in zip(d["hourly"]["time"], d["hourly"]["pm2_5"]):
                if v is not None:
                    by.setdefault(t[:10], []).append(v)
            return {"days": {k: float(np.mean(v)) for k, v in by.items() if len(v) >= 18}, "cell": [d.get("latitude"), d.get("longitude")]}
        except Exception as e:                                     # noqa: BLE001
            err = str(e)[:80]
    return {"error": err}


def stage_fetch(budget_s: float = 150.0) -> None:
    t0 = time.time()
    sites = json.loads((RAW / "sites.json").read_text())
    f = RAW / "fetched.json"
    got = json.loads(f.read_text()) if f.exists() else {}
    for s in sites:
        if s["id"] in got:
            continue
        if time.time() - t0 > budget_s:
            print("budget reached; run again")
            break
        truth = s["days"] if s["days"] is not None else _oaq_year(s["oaq"])
        if len(truth) < 250:
            got[s["id"]] = {"skip": f"only {len(truth)} valid days of monitor data"}
        else:
            c = _cams(s["lat"], s["lon"])
            got[s["id"]] = {"obs": truth, "cams": c.get("days"), "cell": c.get("cell"), "error": c.get("error")}
        f.write_text(json.dumps(got))
    else:
        print("fetch complete:", len(got), "sites,", sum(1 for v in got.values() if v.get("obs")), "usable")


def _stats(o: np.ndarray, m: np.ndarray) -> dict:
    return {"obs_mean": round(float(o.mean()), 1), "cams_mean": round(float(m.mean()), 1), "bias": round(float((m - o).mean()), 1), "ratio": round(float(m.mean() / o.mean()), 2),
            "daily_corr": round(float(np.corrcoef(o, m)[0, 1]), 2), "daily_rmse": round(float(np.sqrt(((m - o) ** 2).mean())), 1),
            "obs_days_over_15": int((o > 15).sum()), "cams_days_over_15": int((m > 15).sum())}


def stage_evaluate() -> None:
    sites = {s["id"]: s for s in json.loads((RAW / "sites.json").read_text())}
    got = json.loads((RAW / "fetched.json").read_text())
    rows = []
    for k, v in got.items():
        if k not in sites or not v.get("obs") or not v.get("cams"):
            continue
        ds = sorted(set(v["obs"]) & set(v["cams"]))
        if len(ds) < 230:
            continue
        o = np.array([v["obs"][d] for d in ds]); m = np.array([v["cams"][d] for d in ds])
        s = sites[k]
        r = {"id": k, "name": s["name"], "region": s["region"], "lat": round(s["lat"], 2), "lon": round(s["lon"], 2), "n_days": len(ds), **_stats(o, m),
             "daily_p98_obs": round(float(np.percentile(o, 98)), 1), "daily_p98_cams": round(float(np.percentile(m, 98)), 1)}
        rows.append((r, o, m))
    rows.sort(key=lambda t: t[0]["lon"])
    cal = [t for i, t in enumerate(rows) if i % 2 == 0]
    held = [t for i, t in enumerate(rows) if i % 2 == 1]
    for r, _, _ in cal:
        r["split"] = "calibration"
    for r, _, _ in held:
        r["split"] = "held-out"
    fac = float(np.mean([t[1].mean() for t in cal]) / np.mean([t[2].mean() for t in cal])) if cal else 1.0
    fac_med = float(np.median([t[1].mean() / t[2].mean() for t in cal])) if cal else 1.0

    def score(set_, f):
        ab = [abs(float((m * f).mean() - o.mean())) for _, o, m in set_]
        rel = [abs(float((m * f).mean() / o.mean() - 1)) for _, o, m in set_]
        exc = [abs(int(((m * f) > 15).sum()) - int((o > 15).sum())) for _, o, m in set_]
        within = [bool(abs((m * f).mean() / o.mean() - 1) <= 0.3) for _, o, m in set_]
        return {"mean_abs_bias_ugm3": round(float(np.mean(ab)), 2), "median_abs_rel_error": round(float(np.median(rel)), 2), "mean_abs_error_in_days_over_15": round(float(np.mean(exc)), 1), "share_of_sites_within_30pct_of_monitor": round(float(np.mean(within)), 2)}
    allr = [t[0] for t in rows]
    by_region = {}
    for reg in sorted(set(r["region"] for r in allr)):
        sub = [r for r in allr if r["region"] == reg]
        by_region[reg] = {"sites": len(sub), "median_ratio_cams_over_monitor": round(float(np.median([r["ratio"] for r in sub])), 2), "median_daily_corr": round(float(np.median([r["daily_corr"] for r in sub])), 2),
                          "median_bias_ugm3": round(float(np.median([r["bias"] for r in sub])), 1)}
    pol = [r for r in allr if r["obs_mean"] >= 20]
    clean = [r for r in allr if r["obs_mean"] < 20]
    split_conc = {"monitor_annual_ge_20": {"sites": len(pol), "cams_over_monitor_ratios": sorted(r["ratio"] for r in pol), "median_ratio": round(float(np.median([r["ratio"] for r in pol])), 2) if pol else None},
                  "monitor_annual_lt_20": {"sites": len(clean), "median_ratio": round(float(np.median([r["ratio"] for r in clean])), 2), "share_cams_high_by_20pct_or_more": round(float(np.mean([r["ratio"] >= 1.2 for r in clean])), 2)}}
    lo, la_ = np.log([r["obs_mean"] for r in allr]), np.log([r["cams_mean"] for r in allr])
    slope = float(np.polyfit(lo, la_, 1)[0])
    summ = {"year": YEAR, "log_log_slope_cams_vs_monitor": round(slope, 2), "by_monitor_concentration": split_conc, "sites_used": len(rows), "calibration_factor_ratio_of_means": round(fac, 2), "calibration_factor_median_ratio": round(fac_med, 2),
            "held_out_uncorrected": score(held, 1.0), "held_out_corrected_with_one_factor": score(held, fac), "calibration_sites_corrected": score(cal, fac), "by_region": by_region,
            "overall_median_daily_corr": round(float(np.median([r["daily_corr"] for r in allr])), 2), "overall_median_ratio": round(float(np.median([r["ratio"] for r in allr])), 2)}
    limits = ["CAMS is a regional analysis (roughly 10 to 40 km); a monitor beside a road or a stack reads higher than its cell.",
              "Open-Meteo provides CAMS from 2022-08; one calendar year (2023) is tested, which includes the June 2023 Canadian wildfire smoke over the eastern US.",
              "OpenAQ coverage of South Asia in the open archive is very thin (one Kathmandu site was found); the national CPCB feed needs a login, so the region where the user lives is the least tested.",
              "Monitors are not random: they sit in towns. Rural cells are under-represented."]
    OUT.write_text(json.dumps({"sites": allr, "summary": summ, "limits": limits}, indent=1))
    print(json.dumps(summ, indent=1))
    for r in allr:
        print(f'{r["id"]:12s} {r["region"]:14s} {r["split"]:12s} obs {r["obs_mean"]:5.1f} cams {r["cams_mean"]:5.1f} ratio {r["ratio"]:.2f} corr {r["daily_corr"]:.2f}  {r["name"][:40]}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "fetch", "evaluate"])
    a = ap.parse_args()
    {"select": stage_select, "fetch": stage_fetch, "evaluate": stage_evaluate}[a.stage]()
