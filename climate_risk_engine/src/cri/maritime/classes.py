"""
Class-median fallback when a ship has no record of its own.

Source: IMO MEPC 81/6/1 Annex Table 6 (2022 data, ships of 5,000 GT and above): the MEDIAN Annual Efficiency Ratio (AER, gCO2 per deadweight-tonne nautical mile) by Fourth IMO GHG Study
ship type and size bin. CO2 per nautical mile = AER x DWT / 1e6 tonnes. That median already contains the class's typical speed, load factor and ballast legs, so:
  - it is a CLASS median, not this ship: individual ships differ a lot;
  - fuel is expressed as HFO-equivalent (DCS mixes fuels): fuel = CO2 / HFO carbon factor;
  - the speed is the one you give (the median carries none), so the speed effect is applied only to what-if cases, relative to it.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ..lca import marine_fuels as MF

_P = Path(__file__).resolve().parents[1] / "data" / "imo_dcs_class_aer_2022.json"

TYPE_WORDS = [("container_ship", r"container|boxship|feeder"), ("bulk_carrier", r"bulk|bulker|capesize|panamax|supramax|handymax|handysize|ultramax"), ("oil_tanker", r"oil tanker|crude|vlcc|suezmax|aframax|product tanker|tanker"),
              ("chemical_tanker", r"chemical"), ("gas_carrier", r"lng|lpg|gas carrier|gas tanker|liquefied"), ("cruise", r"cruise"), ("ferry_ropax", r"ropax|ro-pax|ferry"),
              ("vehicle_carrier", r"vehicle|car carrier|pctc|pure car"), ("ro_ro", r"ro-?ro"), ("reefer", r"reefer|refrigerated"), ("general_cargo", r"general cargo|multipurpose|multi-purpose")]
UNIT_OF = {"bulk_carrier": "dwt", "chemical_tanker": "dwt", "container_ship": "teu", "cruise": "gt", "ferry_pax_only": "gt", "ferry_ropax": "gt", "general_cargo": "dwt", "gas_carrier": "cbm",
           "oil_tanker": "dwt", "other_liquids": "dwt", "reefer": "dwt", "ro_ro": "dwt", "vehicle_carrier": "gt"}


@lru_cache(maxsize=1)
def lib() -> dict:
    return json.loads(_P.read_text(encoding="utf-8"))


def classify(text: str) -> Optional[str]:
    t = (text or "").lower()
    if re.search(r"lng carrier|lpg carrier|lng tanker", t):
        return "gas_carrier"
    for k, rx in TYPE_WORDS:
        if re.search(rx, t):
            return k
    return None


def _bin(rows, value):
    for r in rows:
        if value >= r["lo"] and (r["hi"] is None or value <= r["hi"]):
            return r
    return None


def reference(ship_type: str, size: Optional[float] = None, size_unit: Optional[str] = None, dwt: Optional[float] = None) -> dict:
    """Class row for a ship. size is in the bin unit of the type (dwt / TEU / GT / CBM). For a type binned by TEU, GT or CBM, a DWT alone picks the bin with the nearest mean DWT (stated)."""
    rows = [r for r in lib()["rows"] if r["type"] == ship_type]
    if not rows:
        raise ValueError(f"no IMO class table for {ship_type!r}")
    unit = UNIT_OF[ship_type]
    notes = []
    row = None
    if size is not None and (size_unit in (None, unit)):
        row = _bin(rows, size)
    elif dwt is not None and unit == "dwt":
        row = _bin(rows, dwt)
    elif dwt is not None:
        row = min(rows, key=lambda r: abs(r["mean_dwt"] - dwt))
        notes.append(f"{ship_type.replace('_', ' ')} classes are binned by {unit.upper()}; your DWT was matched to the bin with the nearest mean DWT. Give the {unit.upper()} for a firm bin.")
    if row is None:
        raise ValueError(f"size needed: give the ship's {unit.upper()}" + (f" (the class tables start at {rows[0]['lo']:,.0f})" if size is not None else ""))
    use_dwt = dwt if dwt else row["mean_dwt"]
    if not dwt:
        notes.append(f"DWT not given: the class mean DWT ({row['mean_dwt']:,.0f}) is used.")
    cf = MF.wtw("hfo")
    cf_co2_per_t_fuel = cf["lcv_mj_per_g"] * cf["ttw"]                   # t CO2e per t fuel (TtW, includes CH4 and N2O)
    co2_per_nm_t = row["aer_median_2022"] * use_dwt / 1e6
    return {"row": row, "dwt_used": use_dwt, "co2e_t_per_nm": co2_per_nm_t, "fuel_kg_per_nm_hfo_eq": co2_per_nm_t / cf_co2_per_t_fuel * 1000.0, "notes": notes, "unit": unit}


def vessel_from_class(ship_type: str, speed_kn: float, size: Optional[float] = None, size_unit: Optional[str] = None, dwt: Optional[float] = None, name: Optional[str] = None) -> dict:
    r = reference(ship_type, size, size_unit, dwt)
    row = r["row"]
    return {"name": name, "imo": None, "ship_type": ship_type, "dwt": r["dwt_used"], "gt": None, "design_speed_kn": None,
            "reference_speed_kn": speed_kn, "fuel_kg_per_nm_ref": r["fuel_kg_per_nm_hfo_eq"], "class_proxy": True, "missing": [],
            "notes": [f"No record for this ship: IMO DCS 2022 class median used ({ship_type.replace('_', ' ')}, {r['unit'].upper()} {row['lo']:,.0f}-{'+' if row['hi'] is None else format(row['hi'], ',.0f')}, {row['ships']} ships, median AER {row['aer_median_2022']} gCO2/dwt-nm).",
                      "Fuel is HFO-equivalent; the median carries its own typical speed and load, so treat results as a class estimate, not this ship's figure."] + r["notes"]}
