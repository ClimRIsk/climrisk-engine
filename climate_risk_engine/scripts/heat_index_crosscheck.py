"""Cross-check of the engine's heat index against a physical shade bracket, at several climates.

The engine's extreme-heat study uses the Bureau of Meteorology simplified WBGT (sWBGT = 0.567 T + 0.393 e + 3.94) on NASA POWER daily maximum temperature with the
day's dew-point vapour pressure. Kong & Huber (2022, Earth's Future) found sWBGT greatly overestimates heat stress in hot-humid areas against the Liljegren model.
This script does NOT replace a Liljegren run (that needs hourly sun, wind and a validated implementation). It brackets the shade WBGT from first principles:
    WBGT_shade = 0.7 * Tnwb + 0.3 * Tg,  Tnwb in [Tw, Tw + 2],  Tg in [T, T + 2]   (Tw = psychrometric wet bulb, exact solution)
and reports how far sWBGT sits above the bracket and how many days a year each exceeds 32 (the study's "even light work is unsafe" line).
Usage: PYTHONPATH=src python scripts/heat_index_crosscheck.py [out.json]
"""
import json, sys
import numpy as np
from cri.modellab import data as D, heat

SITES = {"Virudhunagar IN (hot-humid, semi-arid)": (9.4713, 77.9107), "Dhaka BD (hot-humid)": (23.81, 90.41), "Kolkata IN": (22.57, 88.36),
         "Houston US (hot-humid)": (29.76, -95.37), "Phoenix US (hot-dry)": (33.45, -112.07), "Jacobabad PK (hot-dry)": (28.28, 68.44),
         "Riyadh SA (hot-dry)": (24.71, 46.68), "Oslo NO (cool)": (59.91, 10.75)}


def check(lat, lon, y0=1981, y1=2023):
    d = D.power_daily(lat, lon, y0, y1, "T2M_MAX,T2MDEW")
    ok = [i for i in range(len(d["time"])) if d["T2M_MAX"][i] is not None and d["T2MDEW"][i] is not None]
    t = np.array([d["T2M_MAX"][i] for i in ok]); td = np.array([d["T2MDEW"][i] for i in ok])
    e = heat.sat_vapour_pressure_hpa(td); s = 0.567 * t + 0.393 * e + 3.94
    rh = np.clip(100 * e / heat.sat_vapour_pressure_hpa(t), 1, 100)
    tw = np.array([heat.wet_bulb_exact(a, b) for a, b in zip(t, rh)])
    lo, hi = 0.7 * tw + 0.3 * t, 0.7 * (tw + 2) + 0.3 * (t + 2)
    k = int(np.argmax(s)); yrs = (y1 - y0 + 1)
    return {"tmax_c": round(float(t.max()), 1), "swbgt_max": round(float(s.max()), 1), "bracket_at_that_day": [round(float(lo[k]), 1), round(float(hi[k]), 1)],
            "swbgt_minus_bracket_top_c": round(float(s[k] - hi[k]), 1), "days_per_year_swbgt_gt32": round(float((s > 32).sum() / yrs)), "days_per_year_bracket_top_gt32": round(float((hi > 32).sum() / yrs))}


if __name__ == "__main__":
    out = {n: check(*c) for n, c in SITES.items()}
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1:
        json.dump(out, open(sys.argv[1], "w"), indent=1)
