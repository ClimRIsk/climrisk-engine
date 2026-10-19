"""Compare the engine's insurer-view loss rates with FEMA's National Risk Index (county building expected annual loss).

NRI is the one free, US-wide, peril-split loss benchmark. It is a COUNTY average of all buildings and is itself a HAZUS-based model
(not claims), while the engine value is for one pin; expect scatter. The question is order of magnitude by peril, not agreement to a percent.

Usage:
  python tools/benchmark_fema_nri.py                       # live: reads the three counties from FEMA's public ArcGIS service
  python tools/benchmark_fema_nri.py counties.csv          # or from the NRI county CSV (https://hazards.fema.gov/nri/data-resources)
Fields: STCOFIPS, BUILDVALUE, HRCN_EALB (hurricane), IFLD_EALB (inland flooding, riverine and surface water), CFLD_EALB (coastal flooding).
"""
import csv
import json
import sys
import urllib.parse
import urllib.request

SITES = {"miami": "12086", "new_orleans": "22071", "houston": "48201"}      # Miami-Dade FL, Orleans Parish LA, Harris TX
SVC = "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/National_Risk_Index_Counties/FeatureServer/0/query"
FIELDS = "STCOFIPS,COUNTY,BUILDVALUE,HRCN_EALB,IFLD_EALB,CFLD_EALB"


def live():
    q = {"where": "STCOFIPS IN (%s)" % ",".join(SITES.values()), "outFields": FIELDS, "returnGeometry": "false", "f": "json"}
    with urllib.request.urlopen(SVC + "?" + urllib.parse.urlencode(q), timeout=60) as r:
        d = json.loads(r.read())
    return {f["attributes"]["STCOFIPS"]: f["attributes"] for f in d["features"]}


def from_csv(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["STCOFIPS"] in SITES.values():
                out[r["STCOFIPS"]] = r
    return out


def main(csv_path=None, snap_path="tests/golden/insurer_snapshot_2026-10-19.json"):
    snap = json.load(open(snap_path))["sites"]
    rows = from_csv(csv_path) if csv_path else live()
    print(f"{'site':12s} {'peril':8s} {'engine per mille':>17s} {'NRI per mille':>14s} {'engine/NRI':>11s}")
    for name, fips in SITES.items():
        if fips not in rows or name not in snap:
            print(f"{name:12s} missing in inputs"); continue
        r, s = rows[fips], snap[name]
        bv = float(r["BUILDVALUE"]) or float("nan")
        tiv = 10_000_000.0                                                   # the snapshot's TIV per site
        for lab, eng_key, nri_key in (("wind", "wind", "HRCN_EALB"), ("river", "river", "IFLD_EALB"), ("coast", "coast", "CFLD_EALB")):
            e = 1000.0 * float(s[eng_key]) / tiv
            n = 1000.0 * float(r[nri_key] or 0) / bv
            ratio = (e / n) if n > 0 else float("nan")
            print(f"{name:12s} {lab:8s} {e:17.3f} {n:14.3f} {ratio:11.2f}")
    print("\nRead: a ratio between about 0.3 and 3 is the same order of magnitude; far outside means the peril is missing or mis-scaled.\n"
          "'river' compares the engine's river-map loss with NRI inland flooding, which also includes surface-water flooding the engine does not model.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
