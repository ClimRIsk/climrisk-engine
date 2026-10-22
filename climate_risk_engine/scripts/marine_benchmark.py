"""Tests for cri.marine.
  python scripts/marine_benchmark.py fetch            # resumable: caches the OISST series for the event sites (call until it reports done)
  python scripts/marine_benchmark.py analyse out.json # detection vs documented events, base rates, and a reference-code cross-check
Event windows and criteria were written down before any series was read (see EVENTS and CRITERIA)."""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cri import marine as M  # noqa: E402

# (name, lat, lon, window start, window end, source of the event)
EVENTS = [
    ("Ningaloo Nino 2011, Western Australia", -27.0, 113.0, "2010-12-01", "2011-04-30", "Pearce and Feng 2013; Feng et al. 2013"),
    ("Northeast Pacific 'Blob' 2014-16", 50.0, -145.0, "2014-01-01", "2016-12-31", "Bond et al. 2015; Di Lorenzo and Mantua 2016"),
    ("Northeast Pacific 2019", 50.0, -145.0, "2019-07-01", "2019-10-31", "Amaya et al. 2020"),
    ("Tasman Sea 2015/16", -42.5, 149.5, "2015-09-01", "2016-06-30", "Oliver et al. 2017"),
    ("Great Barrier Reef 2016", -16.5, 146.0, "2016-02-01", "2016-04-30", "Hughes et al. 2017"),
    ("Mediterranean 2003", 42.5, 6.5, "2003-06-01", "2003-08-31", "Olita et al. 2007; Hobday et al. 2018"),
    ("Mediterranean 2022", 41.0, 4.0, "2022-06-01", "2022-08-31", "Marba et al. 2022; Garrabou et al. 2022"),
    ("Northwest Atlantic 2012", 41.5, -68.0, "2012-03-01", "2012-08-31", "Mills et al. 2013; Chen et al. 2014"),
    ("Peru coastal El Nino 2017", -10.0, -80.0, "2017-01-01", "2017-04-30", "Garreaud 2018; Echevin et al. 2018"),
]
# Pre-specified: an event counts as detected if a marine heatwave overlapping the window has a peak anomaly of at least 2.0 C and lasts 30 days or more.
CRITERIA = {"min_peak_anomaly_c": 2.0, "min_duration_days": 30}
END = dt.date(2025, 4, 30)          # the benchmark stops where NOAA's Zarr copy stops; the live module reads newer days one by one


def cells():
    out = []
    for n, la, lo, *_ in EVENTS:
        oc = M.ocean_cell(la, lo)
        out.append((oc["i"], oc["j"]))
    return sorted(set(out))


def fetch(budget=150.0):
    t0 = time.time()
    c = cells()
    res = M.fetch_cells(c, budget_s=budget - (time.time() - t0), progress=lambda d, s: print("  reached", d, round(s), "s", flush=True), end=END)
    full = (END - M.FIRST_DAY).days + 1
    print({f"{k[0]},{k[1]}": v for k, v in res.items()}, "of", full, "days")
    print("done" if all(v >= full for v in res.values()) else "not done")


def overlap(e, a, b):
    return e["start"] <= b and e["end"] >= a


def analyse(out):
    res = {"criteria": CRITERIA, "events": [], "controls": {}}
    detected = 0
    base = []
    for n, la, lo, a, b, src in EVENTS:
        s = M.sst_series(la, lo, end=END)
        d = M.detect(s["time"], np.array([np.nan if x is None else x for x in s["sst"]]))
        ev = d["events"]
        hit = [e for e in ev if overlap(e, a, b)]
        best = max(hit, key=lambda e: e["peak_anomaly_c"]) if hit else None
        ok = bool(best and best["peak_anomaly_c"] >= CRITERIA["min_peak_anomaly_c"] and best["duration_days"] >= CRITERIA["min_duration_days"])
        detected += ok
        # control: share of the other full years at the same cell with an event meeting the same criteria
        yrs = sorted({int(t[:4]) for t in s["time"] if 1983 <= int(t[:4]) <= 2025})
        ey = int(a[:4])
        wy = {y for y in range(int(a[:4]), int(b[:4]) + 1)}
        other = [y for y in yrs if y not in wy]
        big = [e for e in ev if e["peak_anomaly_c"] >= CRITERIA["min_peak_anomaly_c"] and e["duration_days"] >= CRITERIA["min_duration_days"]]
        yrs_big = {y for e in big for y in range(int(e["start"][:4]), int(e["end"][:4]) + 1)}
        base_rate = len([y for y in other if y in yrs_big]) / max(len(other), 1)
        base.append(base_rate)
        res["events"].append({"name": n, "source": src, "cell": [s["cell_lat"], s["cell_lon"]], "window": [a, b], "detected": ok,
                              "best_event": None if not best else {k: best[k] for k in ("start", "end", "duration_days", "peak_anomaly_c", "mean_anomaly_c", "category")},
                              "events_overlapping_window": len(hit), "share_of_other_years_with_such_an_event": round(base_rate, 3),
                              "total_events_1982_2025": len(ev)})
    res["detected"], res["of"] = int(detected), len(EVENTS)
    res["mean_share_of_other_years_with_such_an_event"] = round(float(np.mean(base)), 3)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


def crosscheck(ref_path="/tmp/marineHeatWaves.py", out="docs/marine_crosscheck.json"):
    """Compare cri.marine.detect with the published reference implementation (Oliver, marineHeatWaves.py) on the cached cells."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mhw_ref", ref_path)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    rows = []
    for n, la, lo, *_ in EVENTS:
        s = M.sst_series(la, lo, end=END)
        v = np.array([np.nan if x is None else x for x in s["sst"]], dtype=float)
        t = np.array(s["time"], dtype="datetime64[D]")
        ordinal = np.array([x.astype(dt.date).toordinal() for x in t])
        mine = M.detect(s["time"], v)["events"]
        mhw, _ = R.detect(ordinal, v, climatologyPeriod=[1982, 2011], pctile=90, windowHalfWidth=5, smoothPercentile=True,
                          smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2)
        ref = list(zip([x.isoformat() for x in mhw["date_start"]],
                       [x.isoformat() for x in mhw["date_end"]],
                       [float(x) for x in mhw["intensity_max"]]))
        mine_se = {(e["start"], e["end"]): e["peak_anomaly_c"] for e in mine}
        same = sum(1 for a, b, _ in ref if (a, b) in mine_se)
        big = [(a, b, p) for a, b, p in ref if p >= 2.0]
        same_big = sum(1 for a, b, _ in big if (a, b) in mine_se)
        rows.append({"site": n, "events_mine": len(mine), "events_reference": len(ref), "identical_start_end": same,
                     "reference_events_peak_ge_2C": len(big), "of_which_identical": same_big})
        print(rows[-1], flush=True)
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch()
    elif sys.argv[1] == "crosscheck":
        crosscheck()
    else:
        analyse(sys.argv[2])
