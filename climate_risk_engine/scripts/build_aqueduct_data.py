"""
Build the WRI Aqueduct 4.0 lookup shipped with the engine (src/cri/data/aqueduct40.npz).

Source: WRI Aqueduct 4.0 water risk data, Y2023M07D05 (CC BY 4.0)
        https://files.wri.org/aqueduct/aqueduct-4-0-water-risk-data.zip  (261 MB)
Citation: Kuzma, S. et al. 2023. "Aqueduct 4.0: Updated decision-relevant global water risk indicators."
          Technical Note. Washington, DC: World Resources Institute. doi:10.46830/writn.23.00061

What it produces
  * a 0.05° global grid of indices into the baseline_annual table (68,506 units = HydroBASINS L6 basin × GADM admin-1)
  * per unit: 13 baseline indicator scores (0–5) + raw values for the headline ones, overall water risk (default
    weighting) and the 9 industry-weighted overall scores (agriculture, chemicals, construction, electric power,
    food & beverage, mining, oil & gas, semiconductors, textiles)
  * per basin (pfaf_id): future water stress / depletion / interannual / seasonal variability scores for
    2030, 2050, 2080 under optimistic (SSP1 RCP2.6), business-as-usual (SSP3 RCP7.0), pessimistic (SSP5 RCP8.5)

Usage:  python scripts/build_aqueduct_data.py /path/to/Aqueduct40_waterrisk_download_Y2023M07D05
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyogrio
from rasterio import features
from rasterio.transform import from_origin

RES = 0.05
OUT = Path(__file__).resolve().parents[1] / "src" / "cri" / "data"

BASE_IND = ["bws", "bwd", "iav", "sev", "gtd", "rfr", "cfr", "drr", "ucw", "cep", "udw", "usa", "rri"]
RAW_KEEP = ["bws", "bwd", "gtd", "rfr", "cfr", "drr"]
INDUSTRY = ["def", "agr", "che", "con", "elp", "fnb", "min", "ong", "smc", "tex"]
FUT_IND = ["ws", "wd", "iv", "sv"]
FUT_SCEN = {"opt": "Optimistic (SSP1 RCP2.6)", "bau": "Business as usual (SSP3 RCP7.0)",
            "pes": "Pessimistic (SSP5 RCP8.5)"}
FUT_YEARS = {"30": 2030, "50": 2050, "80": 2080}


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if (not np.isfinite(v) or v <= -9990) else round(v, 3)


def main(src: str) -> None:
    gdb = str(Path(src) / "GDB" / "Aq40_Y2023D07M05.gdb")
    cols = ["pfaf_id", "gid_0", "name_0", "name_1"] + [f"{i}_score" for i in BASE_IND] + \
           [f"{i}_raw" for i in RAW_KEEP] + [f"{i}_label" for i in BASE_IND] + \
           [f"w_awr_{s}_tot_score" for s in INDUSTRY] + ["w_awr_def_tot_label"]
    print("reading baseline_annual …")
    df = pyogrio.read_dataframe(gdb, layer="baseline_annual", columns=cols) if hasattr(pyogrio, "read_dataframe") else None
    if df is None:
        raise SystemExit("pyogrio.read_dataframe needs geopandas")
    df = df.reset_index(drop=True)
    print(len(df), "units")

    W, H = int(360 / RES), int(180 / RES)
    transform = from_origin(-180, 90, RES, RES)
    shapes = ((geom, i + 1) for i, geom in enumerate(df.geometry) if geom is not None and not geom.is_empty)
    grid = features.rasterize(shapes, out_shape=(H, W), transform=transform, fill=0, dtype="uint32",
                              all_touched=False)
    # second pass with all_touched to fill slivers (coastal cells, small islands) without overwriting
    shapes2 = ((geom, i + 1) for i, geom in enumerate(df.geometry) if geom is not None and not geom.is_empty)
    touched = features.rasterize(shapes2, out_shape=(H, W), transform=transform, fill=0, dtype="uint32",
                                 all_touched=True)
    grid = np.where(grid == 0, touched, grid)
    print("cells filled:", int((grid > 0).sum()))

    units = []
    for r in df.itertuples(index=False):
        d = r._asdict()
        units.append({
            "pfaf": int(d["pfaf_id"]) if d["pfaf_id"] is not None else None,
            "iso3": d["gid_0"], "country": d["name_0"], "admin1": d["name_1"],
            "s": [_f(d[f"{i}_score"]) for i in BASE_IND],
            "raw": [_f(d[f"{i}_raw"]) for i in RAW_KEEP],
            "lab": [d[f"{i}_label"] for i in BASE_IND],
            "tot": [_f(d[f"w_awr_{s}_tot_score"]) for s in INDUSTRY],
            "tot_label": d["w_awr_def_tot_label"],
        })

    print("reading future_annual …")
    import csv
    fut = {}
    with open(Path(src) / "CVS" / "Aqueduct40_future_annual_y2023m07d05.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            pf = int(float(row["pfaf_id"]))
            rec = {}
            for sc in FUT_SCEN:
                for yy in FUT_YEARS:
                    rec[f"{sc}{yy}"] = [_f(row.get(f"{sc}{yy}_{i}_x_s")) for i in FUT_IND]
            fut[pf] = rec
    print(len(fut), "future basins")

    # compact label vocabulary
    labels = sorted({l for u in units for l in u["lab"] + [u["tot_label"]] if l})
    li = {l: k for k, l in enumerate(labels)}
    for u in units:
        u["lab"] = [li.get(l, -1) for l in u["lab"]]
        u["tot_label"] = li.get(u["tot_label"], -1)

    OUT.mkdir(parents=True, exist_ok=True)

    def q(v):  # score 0–5 → uint8 in 0.02 steps; 255 = no data
        return 255 if v is None else int(round(min(max(v, 0.0), 5.0) * 50))

    places = sorted({(u["country"] or "", u["admin1"] or "") for u in units})
    pi = {pl: k for k, pl in enumerate(places)}
    fkeys = sorted(fut)
    fut_arr = np.full((len(fkeys), len(FUT_SCEN), len(FUT_YEARS), len(FUT_IND)), 255, dtype=np.uint8)
    for a, pf in enumerate(fkeys):
        for b, sc in enumerate(FUT_SCEN):
            for c, yy in enumerate(FUT_YEARS):
                fut_arr[a, b, c] = [q(v) for v in fut[pf][f"{sc}{yy}"]]
    np.savez_compressed(
        OUT / "aqueduct40.npz", grid=grid, res=np.float32(RES),
        pfaf=np.array([u["pfaf"] or 0 for u in units], dtype=np.int64),
        iso3=np.array([u["iso3"] or "" for u in units], dtype="<U3"),
        place=np.array([pi[(u["country"] or "", u["admin1"] or "")] for u in units], dtype=np.int32),
        place_country=np.array([p[0] for p in places]), place_admin1=np.array([p[1] for p in places]),
        score=np.array([[q(v) for v in u["s"]] for u in units], dtype=np.uint8),
        raw=np.array([[np.nan if v is None else v for v in u["raw"]] for u in units], dtype=np.float32),
        label=np.array([u["lab"] for u in units], dtype=np.int16),
        total=np.array([[q(v) for v in u["tot"]] for u in units], dtype=np.uint8),
        total_label=np.array([u["tot_label"] for u in units], dtype=np.int16),
        labels=np.array(labels), fut_pfaf=np.array(fkeys, dtype=np.int64), future=fut_arr,
    )
    p = OUT / "aqueduct40.npz"
    print(p.name, round(p.stat().st_size / 1e6, 2), "MB")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/aq/Aqueduct40_waterrisk_download_Y2023M07D05")
