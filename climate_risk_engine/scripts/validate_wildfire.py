"""
Validate the engine's fire-weather index against OBSERVED wildfire: the Canadian National Fire Database (NFDB), 2000 to 2022.

The engine's fire study computes the Canadian Fire Weather Index (FWI) from ERA5 daily weather and checks it against ERA5 itself
(hindcast of the index's own annual maxima). That says nothing about whether high FWI days are the days that fires actually start. NFDB lists
every fire reported by the provincial, territorial and parks agencies, with date, position and final size.

  truth   days on which at least one fire of 20 ha or more was reported in a 1-degree box, May to September
  model   engine FWI (cri.modellab.fire.run_series on ERA5 daily T max, RH min, wind max, rain), computed with no tuning
  split   a logistic link from log(1+FWI) to P(large-fire day) is fitted on 2000-2010 and scored on 2011-2022
  scores  ROC area on unseen years; Brier skill against the box's own season-day base rate; share of large-fire days with FWI above 30
          against the share of all days above 30 (enrichment); baseline: hottest-day temperature alone

  python scripts/validate_wildfire.py --stage select | fetch | evaluate
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "backtest" / "raw" / "fire"
OUT = ROOT / "data" / "backtest" / "wildfire_validation.json"
Y0, Y1, SPLIT = 2000, 2022, 2011
MIN_HA = 20.0


def _km(a, b, c, d):
    return float(np.hypot((c - a) * 111.32, (d - b) * 111.32 * np.cos(np.radians(a))))


def stage_select() -> None:
    import pandas as pd
    f = glob.glob(str(RAW / "x" / "NFDB_point_*.txt"))[0]
    d = pd.read_csv(f, encoding="latin1", low_memory=False, usecols=["LATITUDE", "LONGITUDE", "YEAR", "MONTH", "DAY", "SIZE_HA", "PRESCRIBED"])
    d = d[(d.YEAR >= Y0) & (d.YEAR <= Y1) & (d.SIZE_HA >= MIN_HA) & d.MONTH.between(5, 9) & (d.DAY > 0) & (d.PRESCRIBED.fillna("").astype(str).str.strip().isin(["", "0", "No"]))]
    d["la"] = np.floor(d.LATITUDE); d["lo"] = np.floor(d.LONGITUDE)
    g = d.groupby(["la", "lo"]).size().sort_values(ascending=False)
    chosen = []
    for (la, lo), n in g.items():
        if n < 50:
            break
        if all(_km(la + .5, lo + .5, c["lat"], c["lon"]) > 300 for c in chosen):
            sub = d[(d.la == la) & (d.lo == lo)]
            chosen.append({"lat": float(la + .5), "lon": float(lo + .5), "n_large_fires": int(n), "days": sorted({f"{int(r.YEAR)}-{int(r.MONTH):02d}-{int(r.DAY):02d}" for r in sub.itertuples()})})
        if len(chosen) >= 16:
            break
    (RAW / "boxes.json").write_text(json.dumps(chosen))
    print(len(chosen), "boxes:", [(c["lat"], c["lon"], c["n_large_fires"]) for c in chosen])


def _series(lat, lon):
    """Daily T max, RH min, wind max (km/h), rain: ERA5 through Open-Meteo when its allowance lets us, else NASA POWER (the engine's own fallback)."""
    from cri.modellab import data as D, heat
    v = "temperature_2m_max,relative_humidity_2m_min,wind_speed_10m_max,precipitation_sum"
    def once(url, timeout=150):                                      # one try: the library's own back-off can sleep for minutes
        import urllib.error, urllib.request
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "climrisk-research/0.1"}), timeout=timeout).read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise D.QuotaExceeded("rate limited") from e
            raise
    try:
        dd = D.archive(lat, lon, 1981, 2023, daily=v, fetch=once)["daily"]
        ok = [i for i in range(len(dd["time"])) if None not in (dd[k][i] for k in v.split(","))]
        return ([dd["time"][i] for i in ok], np.array([dd["temperature_2m_max"][i] for i in ok], float), np.array([dd["relative_humidity_2m_min"][i] for i in ok], float),
                np.array([dd["wind_speed_10m_max"][i] for i in ok], float), np.array([dd["precipitation_sum"][i] for i in ok], float), "ERA5 (Open-Meteo)")
    except D.QuotaExceeded:
        d = D.power_daily(lat, lon, 1981, 2023, "T2M_MAX,T2MDEW,WS10M_MAX,PRECTOTCORR")
        ok = [i for i in range(len(d["time"])) if None not in (d["T2M_MAX"][i], d["T2MDEW"][i], d["WS10M_MAX"][i], d["PRECTOTCORR"][i])]
        t = np.array([d["T2M_MAX"][i] for i in ok]); td = np.array([d["T2MDEW"][i] for i in ok])
        rh = np.clip(100.0 * heat.sat_vapour_pressure_hpa(td) / heat.sat_vapour_pressure_hpa(t), 1, 100)
        return ([d["time"][i] for i in ok], t, rh, np.array([d["WS10M_MAX"][i] * 3.6 for i in ok]), np.array([d["PRECTOTCORR"][i] for i in ok]), "NASA POWER (MERRA-2)")


def stage_fetch() -> None:
    t0 = time.time()
    for b in json.loads((RAW / "boxes.json").read_text()):
        if time.time() - t0 > 130:
            print("budget reached; run again")
            return
        _series(b["lat"], b["lon"])
    print("fetch complete")


def _auc(score, truth):
    pos, neg = score[truth], score[~truth]
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stage_evaluate() -> None:
    from sklearn.linear_model import LogisticRegression
    from cri.modellab import fire
    res = []
    for b in json.loads((RAW / "boxes.json").read_text()):
        t, T, RH, W, P, src = _series(b["lat"], b["lon"])
        mo = [int(x[5:7]) for x in t]
        fwi = np.array(fire.run_series(T, RH, W, P, mo, tropical=abs(b["lat"]) < 25)["fwi"])
        yr = np.array([int(x[:4]) for x in t]); month = np.array(mo)
        sel = (yr >= Y0) & (yr <= Y1) & (month >= 5) & (month <= 9)
        truth_days = set(b["days"])
        y = np.array([x in truth_days for x in t])
        doy = np.array([int(np.datetime64(x).astype("datetime64[D]").astype(int) - np.datetime64(x[:4] + "-01-01").astype(int)) + 1 for x in t])
        tr, te = sel & (yr < SPLIT), sel & (yr >= SPLIT)
        if y[tr].sum() < 15 or y[te].sum() < 15:
            res.append({"lat": b["lat"], "lon": b["lon"], "skip": "too few large-fire days"})
            continue
        X = np.log1p(np.maximum(fwi, 0))[:, None]
        lr = LogisticRegression(C=100.0).fit(X[tr], y[tr])
        p = lr.predict_proba(X[te])[:, 1]
        # base rate by season day, fitted on the training years (the no-physics reference)
        base = np.zeros(367)
        for d_ in range(1, 367):
            m = tr & (np.abs(doy - d_) <= 10)
            base[d_] = y[m].mean() if m.any() else y[tr].mean()
        pb = base[doy[te]]
        brier = float(np.mean((p - y[te]) ** 2)); brier_b = float(np.mean((pb - y[te]) ** 2))
        ex = float((fwi[te][y[te]] >= 30).mean() / max((fwi[te] >= 30).mean(), 1e-9)) if (fwi[te] >= 30).any() else None
        res.append({"lat": b["lat"], "lon": b["lon"], "weather_source": src, "large_fire_days_test": int(y[te].sum()), "days_test": int(te.sum()), "auc_fwi": round(_auc(fwi[te], y[te]), 3), "auc_tmax_only": round(_auc(T[te], y[te]), 3),
                    "auc_season_day_base_rate": round(_auc(pb, y[te]), 3), "brier_skill_vs_season_base_rate": round(1 - brier / brier_b, 3),
                    "share_large_fire_days_fwi_ge_30": round(float((fwi[te][y[te]] >= 30).mean()), 2), "share_all_days_fwi_ge_30": round(float((fwi[te] >= 30).mean()), 3), "enrichment_at_fwi_30": round(ex, 1) if ex else None})
    ok_ = [r for r in res if "auc_fwi" in r]
    summ = {"boxes_scored": len(ok_), "median_auc_fwi": round(float(np.median([r["auc_fwi"] for r in ok_])), 3), "median_auc_tmax_only": round(float(np.median([r["auc_tmax_only"] for r in ok_])), 3),
            "median_auc_season_day_base_rate": round(float(np.median([r["auc_season_day_base_rate"] for r in ok_])), 3),
            "boxes_fwi_beats_tmax": int(sum(r["auc_fwi"] > r["auc_tmax_only"] for r in ok_)), "boxes_with_positive_brier_skill": int(sum(r["brier_skill_vs_season_base_rate"] > 0 for r in ok_)),
            "median_brier_skill": round(float(np.median([r["brier_skill_vs_season_base_rate"] for r in ok_])), 3),
            "median_enrichment_at_fwi_30": round(float(np.median([r["enrichment_at_fwi_30"] for r in ok_ if r["enrichment_at_fwi_30"]])), 1)}
    OUT.write_text(json.dumps({"boxes": res, "summary": summ, "years": [Y0, Y1], "split_year": SPLIT, "min_fire_ha": MIN_HA,
                               "limits": ["Canada only (boreal and montane forest). Mediterranean, tropical and savanna fire regimes are untested.",
                                          "A fire is reported on the day found; many ignitions are human-caused and independent of weather.",
                                          "ERA5 daily maxima stand in for the noon observation the official index uses, so levels run slightly high.",
                                          "Large-fire days cluster (one lightning outbreak starts many fires), so ROC area is optimistic about independent skill."]}, indent=1))
    print(json.dumps(summ, indent=1))
    for r in res:
        print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "fetch", "evaluate"])
    a = ap.parse_args()
    {"select": stage_select, "fetch": stage_fetch, "evaluate": stage_evaluate}[a.stage]()
