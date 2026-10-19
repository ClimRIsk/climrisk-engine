"""Does Rothermel rate of spread flag the days large fires are reported, as well as the Fire Weather Index and plain hot-day temperature do?

Same truth and weather as scripts/validate_wildfire.py (Canadian National Fire Database, fires >= 20 ha reported May-Sept 2000-2022 in 1-degree boxes; ERA5/MERRA-2 daily
T max, RH min, wind max, rain). Scores on the unseen years 2011-2022, nothing fitted (ROC area is rank-only). Rothermel rate of spread is computed per day for three fixed
fuel models (timber FM10, brush FM5, grass FM1) with 1-h dead moisture either from the same day's equilibrium moisture (ros_*) or from the FFMC, which remembers recent rain (ros_ffmc_*), so the test does not choose a favourable fuel per box; the
benchmark reports each, plus FWI and T max from the earlier validation.

  PYTHONPATH=.:src python scripts/fire_behaviour_benchmark.py     (needs data/backtest/raw/fire from validate_wildfire.py --stage fetch)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import validate_wildfire as V          # noqa: E402
from cri.finance import fire_behaviour as FB       # noqa: E402
from cri.modellab import fire           # noqa: E402

OUT = ROOT / "data" / "backtest" / "fire_behaviour_benchmark.json"


def main():
    rows = []
    for b in json.loads((V.RAW / "boxes.json").read_text()):
        t, T, RH, W, P, src = V._series(b["lat"], b["lon"])
        yr = np.array([int(x[:4]) for x in t]); mo = np.array([int(x[5:7]) for x in t])
        te = (yr >= V.SPLIT) & (yr <= V.Y1) & (mo >= 5) & (mo <= 9)
        y = np.array([x in set(b["days"]) for x in t])
        if y[te].sum() < 15:
            continue
        fwi = np.array(fire.run_series(T, RH, W, P, list(mo), tropical=False)["fwi"])
        row = {"lat": b["lat"], "lon": b["lon"], "large_fire_days_test": int(y[te].sum()), "auc_fwi": round(V._auc(fwi[te], y[te]), 3), "auc_tmax": round(V._auc(T[te], y[te]), 3)}
        idx = np.nonzero(te)[0]
        for m in ("FM10", "FM5", "FM1"):
            ros = FB.daily_spread(m, T[idx], RH[idx], W[idx])
            row[f"auc_ros_{m}"] = round(V._auc(ros, y[te]), 3)
            rf = FB.daily_spread_ffmc(m, np.array(fire.run_series(T, RH, W, P, list(mo), tropical=False)["ffmc"])[idx], W[idx])
            row[f"auc_ros_ffmc_{m}"] = round(V._auc(rf, y[te]), 3)
        rows.append(row)
        print(row, flush=True)
    summ = {k: round(float(np.median([r[k] for r in rows])), 3) for k in rows[0] if k.startswith("auc_")}
    summ["boxes"] = len(rows)
    OUT.write_text(json.dumps({"boxes": rows, "summary": summ}, indent=1))
    print(summ)


if __name__ == "__main__":
    main()
