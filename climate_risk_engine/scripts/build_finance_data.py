"""
Build the offline data behind the Risk Analyst (physical → policy → transition → finance).

Stages (each writes compact runtime files into src/cri/data/; raw downloads are cached
under data/finance/raw/ and are not shipped)

  admin1    FLOPROS v1 (Scussolini et al. 2016, NHESS, CC BY 3.0) first-level admin units
            → 0.05° raster of unit ids + table (name, country, ISO3, river/coastal flood
            protection standard as a return period). One lookup gives both the policy
            jurisdiction (state/province carbon pricing) and flood defences.
  ngfs      NGFS Phase 5 scenarios (IIASA, anonymous pyam access): carbon price, coal/gas/oil
            power generation, primary energy, steel/cement production, electricity price for
            GCAM 6.0, MESSAGEix-GLOBIOM 2.0 and REMIND-MAgPIE 3.3, 2020–2050.
  deltares  Deltares Global Flood Maps (coastal, NASADEM 90 m; Planetary Computer) — compact
            chunk index for RP 2…250 at 2018 and 2050 sea level, so the engine can range-read
            the ~12 KB chunk around a site without the 12 MB reference files.
  jrc       JRC CEMS-GloFAS river flood hazard maps v2.1 (RP 10…500, 90 m) — tile index.

Usage
  python scripts/build_finance_data.py --stage admin1
  python scripts/build_finance_data.py --stage ngfs
  python scripts/build_finance_data.py --stage deltares     (repeat until "complete")
  python scripts/build_finance_data.py --stage jrc
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "finance" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "src" / "cri" / "data"
UA = {"User-Agent": "ClimRisk-Engine/1.0"}


def _get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


# ── admin1 + FLOPROS ──────────────────────────────────────────────────────────

FLOPROS_ZIP = "https://nhess.copernicus.org/articles/16/1049/2016/nhess-16-1049-2016-supplement.zip"
RES = 0.05


def stage_admin1() -> None:
    import shapefile
    from PIL import Image, ImageDraw

    z = RAW / "flopros.zip"
    if not z.exists():
        z.write_bytes(_get(FLOPROS_ZIP, 300))
    shp_dir = RAW / "flopros"
    if not shp_dir.exists():
        zipfile.ZipFile(z).extractall(shp_dir)
    base = next(shp_dir.rglob("FLOPROS_shp_V1.shp")).with_suffix("")
    r = shapefile.Reader(str(base), encoding="latin-1")
    names = [f[0] for f in r.fields[1:]]
    W, H = int(360 / RES), int(180 / RES)
    img = Image.new("I;16", (W, H), 0)
    draw = ImageDraw.Draw(img)
    table = [None]                                      # id 0 = none
    for i in range(len(r)):
        rec = dict(zip(names, r.record(i)))
        try:
            shp = r.shape(i)
        except Exception:                               # noqa: BLE001  (empty geometry)
            continue
        if not shp.points:
            continue
        uid = len(table)
        g = lambda k: float(rec.get(k) or 0.0)          # noqa: E731
        riv = g("MerL_Riv")
        coast = max(g("DL_Max_Co"), g("DL_Min_Co")) or max(g("PL_Max_Co"), g("PL_Min_Co"))
        table.append({"name": rec.get("name") or rec.get("geonunit"), "type": rec.get("type_en") or "",
                      "country": rec.get("geonunit"), "prot_river_rp": round(riv, 1),
                      "prot_coast_rp": round(coast, 1), "coast_from": "design/policy" if coast else "river merged layer"})
        parts = list(shp.parts) + [len(shp.points)]
        for a, b in zip(parts[:-1], parts[1:]):
            pts = [((x + 180) / RES, (90 - y) / RES) for x, y in shp.points[a:b]]
            if len(pts) >= 3:
                draw.polygon(pts, fill=uid)
    grid = np.array(img, dtype=np.uint16)

    # Fill sea cells within ~25 km of land with the nearest unit so coastal sites resolve.
    from scipy import ndimage
    empty = grid == 0
    idx = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
    dist = ndimage.distance_transform_edt(empty)
    filled = grid[idx[0], idx[1]]
    grid = np.where(empty & (dist <= 25 / (111 * RES)), filled, grid).astype(np.uint16)

    # ISO3 per country name: majority vote of register assets inside the country's units
    z2 = np.load(DATA / "asset_register.npz")
    lat, lon = z2["lat"], z2["lon"]
    iso = z2["countries"][z2["country"]]
    ii = np.clip(((90 - lat) / RES).astype(int), 0, H - 1)
    jj = np.clip(((lon + 180) / RES).astype(int), 0, W - 1)
    uid = grid[ii, jj]
    votes: dict = defaultdict(Counter)
    key = lambda u: table[u]["country"] or f"#unit{u}"      # noqa: E731  (some units lack a country)
    for u, c in zip(uid, iso):
        if u:
            votes[key(u)][str(c)] += 1
    iso_of = {k: v.most_common(1)[0][0] for k, v in votes.items() if v.most_common(1)[0][1] >= 2}
    for u, t in enumerate(table[1:], start=1):
        t["iso3"] = iso_of.get(key(u))
    for t in table[1:]:
        if t["prot_coast_rp"] == 0:
            t["prot_coast_rp"] = t["prot_river_rp"]
    np.savez_compressed(DATA / "admin1_grid.npz", grid=grid, res=np.array(RES))
    (DATA / "admin1_flopros.json").write_text(json.dumps(
        {"source": "FLOPROS v1 (Scussolini et al. 2016, NHESS 16:1049, CC BY 3.0) merged layer; "
                   "coastal = design/policy layer where given, else river merged value",
         "res_deg": RES, "units": table}, separators=(",", ":")))
    n_iso = sum(1 for t in table[1:] if t.get("iso3"))
    print(f"admin1: {len(table)-1} units, {n_iso} with ISO3; grid {grid.shape} → "
          f"{(DATA/'admin1_grid.npz').stat().st_size/1e6:.1f} MB")


# ── NGFS Phase 5 ──────────────────────────────────────────────────────────────

NGFS_MODELS = ["GCAM 6.0 NGFS", "MESSAGEix-GLOBIOM 2.0-M-R12-NGFS", "REMIND-MAgPIE 3.3-4.8"]
NGFS_VARS = ["Price|Carbon", "Secondary Energy|Electricity|Coal", "Secondary Energy|Electricity|Gas",
             "Secondary Energy|Electricity|Oil", "Primary Energy|Coal", "Primary Energy|Oil",
             "Primary Energy|Gas", "Production|Steel", "Production|Cement",
             "Price|Secondary Energy|Electricity"]
YEARS = list(range(2020, 2051, 5))

# GCAM 6.0 32-region definitions (ISO3). MESSAGE R12 and REMIND regions derived below.
GCAM = {
    "Africa_Eastern": "BDI COM DJI ERI ETH KEN MDG MUS REU RWA SDN SSD SOM SYC UGA",
    "Africa_Northern": "DZA EGY ESH LBY MAR TUN",
    "Africa_Southern": "AGO BWA LSO MOZ MWI NAM SWZ TZA ZMB ZWE",
    "Africa_Western": "BEN BFA CAF CIV CMR COD COG CPV GAB GHA GIN GMB GNB GNQ LBR MLI MRT NER NGA SEN SLE STP TCD TGO",
    "Argentina": "ARG", "Australia_NZ": "AUS NZL", "Brazil": "BRA", "Canada": "CAN",
    "Central America and Caribbean": "ABW AIA ATG BHS BLZ BMU BRB CRI CUB CYM DMA DOM GLP GRD GTM HND HTI JAM KNA LCA MSR MTQ NIC PAN PRI SLV TCA TTO VCT VGB VIR",
    "Central Asia": "ARM AZE GEO KAZ KGZ MNG TJK TKM UZB",
    "China": "CHN HKG MAC", "Colombia": "COL",
    "EU-12": "BGR CYP CZE EST HUN LTU LVA MLT POL ROU SVK SVN",
    "EU-15": "AUT BEL DEU DNK ESP FIN FRA GBR GRC IRL ITA LUX NLD PRT SWE",
    "Europe_Eastern": "BLR MDA UKR",
    "Europe_Non_EU": "ALB BIH HRV MKD MNE SRB TUR XKX",
    "European Free Trade Association": "CHE ISL LIE NOR",
    "India": "IND", "Indonesia": "IDN", "Japan": "JPN", "Mexico": "MEX",
    "Middle East": "ARE BHR IRN IRQ ISR JOR KWT LBN OMN PSE QAT SAU SYR YEM",
    "Pakistan": "PAK", "Russia": "RUS", "South Africa": "ZAF",
    "South America_Northern": "GUF GUY SUR VEN",
    "South America_Southern": "BOL CHL ECU PER PRY URY",
    "South Asia": "AFG BGD BTN LKA MDV NPL",
    "South Korea": "KOR",
    "Southeast Asia": "BRN FJI KHM LAO MMR MYS NCL PHL PNG SGP SLB THA TLS VNM VUT WSM PRK",
    "Taiwan": "TWN", "USA": "USA",
}
MSG = {  # GCAM region → MESSAGEix-GLOBIOM R12 region
    "Africa_Eastern": "Sub-saharan Africa", "Africa_Southern": "Sub-saharan Africa",
    "Africa_Western": "Sub-saharan Africa", "South Africa": "Sub-saharan Africa",
    "Africa_Northern": "Middle East and North Africa", "Middle East": "Middle East and North Africa",
    "Argentina": "Latin America and the Caribbean", "Brazil": "Latin America and the Caribbean",
    "Central America and Caribbean": "Latin America and the Caribbean", "Colombia": "Latin America and the Caribbean",
    "Mexico": "Latin America and the Caribbean", "South America_Northern": "Latin America and the Caribbean",
    "South America_Southern": "Latin America and the Caribbean",
    "Australia_NZ": "Pacific OECD", "Japan": "Pacific OECD",
    "Canada": "North America", "USA": "North America",
    "Central Asia": "Former Soviet Union", "Russia": "Former Soviet Union", "Europe_Eastern": "Former Soviet Union",
    "China": "China", "EU-12": "Eastern Europe", "Europe_Non_EU": "Eastern Europe",
    "EU-15": "Western Europe", "European Free Trade Association": "Western Europe",
    "India": "South Asia", "Pakistan": "South Asia", "South Asia": "South Asia",
    "Indonesia": "Other Pacific Asia", "South Korea": "Other Pacific Asia", "Southeast Asia": "Other Pacific Asia",
    "Taiwan": "Other Pacific Asia",
}
REM = {  # GCAM region → REMIND region
    "Africa_Eastern": "Sub-saharan Africa", "Africa_Southern": "Sub-saharan Africa",
    "Africa_Western": "Sub-saharan Africa", "South Africa": "Sub-saharan Africa",
    "Africa_Northern": "Middle East, North Africa, Central Asia", "Middle East": "Middle East, North Africa, Central Asia",
    "Central Asia": "Middle East, North Africa, Central Asia",
    "Argentina": "Latin America and the Caribbean", "Brazil": "Latin America and the Caribbean",
    "Central America and Caribbean": "Latin America and the Caribbean", "Colombia": "Latin America and the Caribbean",
    "Mexico": "Latin America and the Caribbean", "South America_Northern": "Latin America and the Caribbean",
    "South America_Southern": "Latin America and the Caribbean",
    "Australia_NZ": "Canada, NZ, Australia", "Canada": "Canada, NZ, Australia", "Japan": "Japan",
    "USA": "United States of America", "China": "China", "Taiwan": "China",
    "Russia": "Countries from the Reforming Economies of the Former Soviet Union",
    "Europe_Eastern": "Countries from the Reforming Economies of the Former Soviet Union",
    "EU-12": "EU 28", "EU-15": "EU 28", "Europe_Non_EU": "Non-EU28 Europe",
    "European Free Trade Association": "Non-EU28 Europe",
    "India": "India", "Pakistan": "Other Asia", "South Asia": "Other Asia", "Indonesia": "Other Asia",
    "South Korea": "Other Asia", "Southeast Asia": "Other Asia",
}
MSG_PREFIX = "MESSAGEix-GLOBIOM 2.0-R12|"
REM_PREFIX = "REMIND-MAgPIE 3.3-4.8|"
GCAM_PREFIX = "GCAM 6.0 NGFS|"


def stage_ngfs() -> None:
    import pandas as pd
    f = RAW / "ngfs5.csv"
    if not f.exists():
        import pyam
        df = pyam.read_iiasa("ngfs_phase_5", variable=NGFS_VARS, model=NGFS_MODELS)
        df.to_csv(f)
    d = pd.read_csv(f)
    d.columns = [c.lower() for c in d.columns]
    out: dict = {}
    units: dict = {}
    for _, row in d.iterrows():
        vals = [None if pd.isna(row.get(str(y))) else round(float(row[str(y)]), 3) for y in YEARS]
        if all(v is None for v in vals):
            continue
        # fill single gaps linearly (MESSAGE reports 2065/2075 gaps; within 2020–2050 rare)
        out.setdefault(row["model"], {}).setdefault(row["region"], {}).setdefault(row["variable"], {})[row["scenario"]] = vals
        units[row["variable"]] = row["unit"]
    country = {}
    for g, isos in GCAM.items():
        for iso in isos.split():
            country[iso] = {"GCAM 6.0 NGFS": GCAM_PREFIX + g,
                            "MESSAGEix-GLOBIOM 2.0-M-R12-NGFS": MSG_PREFIX + MSG[g],
                            "REMIND-MAgPIE 3.3-4.8": REM_PREFIX + REM[g]}
    # sanity: every mapped region exists for its model
    for iso, m in country.items():
        for mod, reg in m.items():
            assert reg in out[mod], (iso, mod, reg)
    (DATA / "ngfs_phase5.json").write_text(json.dumps({
        "source": "NGFS Phase 5 (Nov 2024) via IIASA Scenario Explorer (ngfs_phase_5), models GCAM 6.0, "
                  "MESSAGEix-GLOBIOM 2.0-M-R12, REMIND-MAgPIE 3.3-4.8",
        "years": YEARS, "units": units, "data": out, "country_region": country,
        "usd2010_to_usd2024": 1.438,   # US CPI-U annual avg 2024 / 2010 (313.689 / 218.056)
    }, separators=(",", ":")))
    print(f"ngfs: {sum(len(v) for v in out.values())} model-regions, {len(units)} variables, "
          f"{len(country)} countries mapped → {(DATA/'ngfs_phase5.json').stat().st_size/1e6:.2f} MB")


# ── Deltares coastal flood chunk index ───────────────────────────────────────

DELT_RPS = [2, 5, 10, 25, 50, 100, 250]
DELT_YEARS = [2018, 2050]
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/deltaresfloodssa/floods"
DELT_REF = "https://deltaresfloodssa.blob.core.windows.net/floods/references/NASADEM-90m-{y}-{rp:04d}.json"
DELT_NC = ("https://deltaresfloodssa.blob.core.windows.net/floods/v2021.06/global/NASADEM/90m/"
           "GFM_global_NASADEM90m_{y}slr_rp{rp:04d}_masked.nc")


def stage_deltares() -> None:
    tok = json.loads(_get(PC_TOKEN))["token"]
    def have(y, rp):
        f = RAW / f"delt_{y}_{rp}.npz"
        return f.exists() and "chunk" in np.load(f).files
    todo = [(y, rp) for y in DELT_YEARS for rp in DELT_RPS if not have(y, rp)]
    for y, rp in todo[:10]:
        t = time.time()
        ref = json.loads(_get(DELT_REF.format(y=y, rp=rp) + "?" + tok, 300))["refs"]
        za = json.loads(ref["inun/.zarray"]) if isinstance(ref["inun/.zarray"], str) else ref["inun/.zarray"]
        ch = za["chunks"][1]                            # most files 600×600; 2018 RP250 uses 4000×4000
        ny, nx = -(-za["shape"][1] // ch), -(-za["shape"][2] // ch)
        empty_max = 1440 if ch == 600 else 0
        cid, off, ln = [], [], []
        for i in range(ny):
            for j in range(nx):
                v = ref.get(f"inun/0.{i}.{j}")
                if v and v[2] > empty_max:                   # skip all-fill / all-zero chunks
                    cid.append(i * nx + j); off.append(v[1]); ln.append(v[2])
        np.savez_compressed(RAW / f"delt_{y}_{rp}.npz", cid=np.array(cid, np.int32),
                            off=np.array(off, np.int64), ln=np.array(ln, np.int32),
                            chunk=np.array(ch), nx=np.array(nx))
        print(f"deltares {y} rp{rp}: {len(cid)} non-empty chunks ({time.time()-t:.0f}s)", flush=True)
    if len([1 for y in DELT_YEARS for rp in DELT_RPS if have(y, rp)]) < len(DELT_RPS) * 2:
        print("deltares: more to do — run again")
        return
    arrs = {}
    for y in DELT_YEARS:
        for rp in DELT_RPS:
            z = np.load(RAW / f"delt_{y}_{rp}.npz")
            for k in ("cid", "off", "ln", "chunk", "nx"):
                arrs[f"{k}_{y}_{rp}"] = z[k]
    np.savez_compressed(DATA / "deltares_coastal_index.npz", rps=np.array(DELT_RPS), years=np.array(DELT_YEARS), **arrs)
    print(f"deltares complete → {(DATA/'deltares_coastal_index.npz').stat().st_size/1e6:.1f} MB")


# ── JRC river flood tiles ────────────────────────────────────────────────────

JRC_BASE = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-GLOFAS/flood_hazard/"


def stage_jrc() -> None:
    d = json.loads(_get(JRC_BASE + "tile_extents.geojson"))
    tiles = {f["properties"]["name"]: f["properties"]["id"] for f in d["features"]}
    (DATA / "jrc_flood_tiles.json").write_text(json.dumps({
        "source": "JRC CEMS-GloFAS global river flood hazard maps v2.1.2 (3 arc-sec), undefended",
        "base": JRC_BASE, "rps": [10, 20, 50, 75, 100, 200, 500], "tiles": tiles}, separators=(",", ":")))
    print(f"jrc: {len(tiles)} tiles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["admin1", "ngfs", "deltares", "jrc"], required=True)
    a = ap.parse_args()
    t0 = time.time()
    {"admin1": stage_admin1, "ngfs": stage_ngfs, "deltares": stage_deltares, "jrc": stage_jrc}[a.stage]()
    print(f"({time.time()-t0:.0f}s)")
