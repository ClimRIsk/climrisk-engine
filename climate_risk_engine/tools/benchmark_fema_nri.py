"""Compare the engine's insurer-view loss rates with FEMA's National Risk Index (county building expected annual loss).

NRI is the one free, US-wide, peril-split loss benchmark. It is a COUNTY average of all buildings; the engine value is for one pin, so
expect scatter. The question is order of magnitude by peril, not agreement to a percent.

Usage:  python tools/benchmark_fema_nri.py /path/to/NRI_Table_Counties.csv [tests/golden/insurer_snapshot_2026-10-19.json]
Get the CSV from https://hazards.fema.gov/nri/data-resources  (Data Download -> Counties table).
Columns used: STCOFIPS, BUILDVALUE, HRCN_EALB (hurricane), RFLD_EALB (riverine flood), CFLD_EALB (coastal flood).
"""
import csv
import json
import sys

SITES = {"miami": "12086", "new_orleans": "22071", "houston": "48201"}      # Miami-Dade FL, Orleans Parish LA, Harris TX


def main(csv_path, snap_path="tests/golden/insurer_snapshot_2026-10-19.json"):
    snap = json.load(open(snap_path))["sites"]
    rows = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["STCOFIPS"] in SITES.values():
                rows[r["STCOFIPS"]] = r
    print(f"{'site':12s} {'peril':8s} {'engine per mille':>17s} {'NRI per mille':>14s} {'engine/NRI':>11s}")
    for name, fips in SITES.items():
        if fips not in rows or name not in snap:
            print(f"{name:12s} missing in inputs"); continue
        r, s = rows[fips], snap[name]
        bv = float(r["BUILDVALUE"]) or float("nan")
        tiv = 10_000_000.0                                                   # the snapshot's TIV per site
        for lab, eng_key, nri_key in (("wind", "wind", "HRCN_EALB"), ("river", "river", "RFLD_EALB"), ("coast", "coast", "CFLD_EALB")):
            e = 1000.0 * float(s[eng_key]) / tiv
            n = 1000.0 * float(r[nri_key] or 0) / bv
            ratio = (e / n) if n > 0 else float("nan")
            print(f"{name:12s} {lab:8s} {e:17.3f} {n:14.3f} {ratio:11.2f}")
    print("\nRead: a ratio between about 0.3 and 3 is the same order of magnitude; far outside means the engine peril is missing or mis-scaled.\n"
          "A 0.0 engine value against a positive NRI value is a miss (coast and river at these sites are known gaps, see docs/GOLDEN_SITES.md).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
