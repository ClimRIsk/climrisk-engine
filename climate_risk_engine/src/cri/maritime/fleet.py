"""
Fleet baseline from the ship records themselves (EU MRV file or your own fleet CSV). Sums and ratios only: nothing is estimated.

  fleet fuel / CO2 / distance / hours    = sums over ships that report them (the count that did is shown; the rest are n/a, not zero)
  AER (g CO2 per dwt-nm)                 = CO2 x 1e6 / (dwt x distance): the IMO Annual Efficiency Ratio definition. EEOI needs cargo carried and is not computed.
  fuel per nm, average speed             = fuel / distance, distance / hours at sea
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from . import vessel as VS


def _s(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs), len(xs)) if xs else (None, 0)


def baseline(ships: list, by: str = "ship_type") -> dict:
    if not ships:
        raise ValueError("no ships")
    groups = defaultdict(list)
    for s in ships:
        groups[str(s.get(by) or "unknown")].append(s)
    rows = []
    for k, L in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        fuel, nf = _s([s.get("fuel_t") for s in L])
        co2, nc = _s([s.get("co2_t_annual") for s in L])
        dist, nd = _s([s.get("distance_nm") for s in L])
        hrs, nh = _s([s.get("hours_at_sea") for s in L])
        pairs = [(s["co2_t_annual"], s["dwt"], s["distance_nm"]) for s in L if s.get("co2_t_annual") and s.get("dwt") and s.get("distance_nm")]
        aer = (sum(p[0] for p in pairs) * 1e6 / sum(p[1] * p[2] for p in pairs)) if pairs else None
        rows.append({by: k, "ships": len(L), "fuel_t": fuel, "fuel_reported_by": nf, "co2_t": co2, "co2_reported_by": nc, "distance_nm": dist, "hours_at_sea": hrs,
                     "fuel_kg_per_nm": (fuel * 1000.0 / dist) if (fuel and dist and nf == nd) else None,
                     "avg_speed_kn": (dist / hrs) if (dist and hrs and nd == nh) else None,
                     "aer_g_co2_per_dwt_nm": aer, "aer_ships": len(pairs)})
    tf, _ = _s([s.get("fuel_t") for s in ships])
    tc, _ = _s([s.get("co2_t_annual") for s in ships])
    return {"by": by, "ships": len(ships), "total_fuel_t": tf, "total_co2_t": tc, "groups": rows,
            "notes": ["Totals sum only the ships that report the field; 'reported_by' counts them. A ship with no value is n/a, never zero.",
                      "Group fuel-per-nm and speed are shown only when every ship in the group reports the inputs.",
                      "AER pools ships that have CO2, dwt and distance (distance-weighted); EEOI needs cargo carried and is not computed."]}


def baseline_from_files(data_dir=None, by: str = "ship_type") -> dict:
    from pathlib import Path
    d = Path(data_dir) if data_dir else VS.DATA_DIR
    ships, files = [], []
    for f in sorted(d.glob("*")) if d.exists() else []:
        if f.suffix.lower() in (".csv", ".xlsx", ".xls"):
            # re-read with the raw columns the baseline needs
            import csv, io
            L = VS.load_fleet_file(f)
            ships += L["ships"]
            files.append(f.name)
    if not ships:
        raise ValueError("no fleet file in data/vessels/: add the EU MRV report or your own fleet CSV")
    out = baseline(ships, by)
    out["files"] = files
    return out
