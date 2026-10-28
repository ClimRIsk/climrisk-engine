"""
Ship particulars. Fuel burn is the one number the engine cannot invent: it comes from a record for that ship.

Sources, in order
  1. Fleet files placed in data/vessels/ (CSV or XLSX). The public EU MRV (THETIS-MRV) annual report is the intended source: it lists every ship of 5,000 GT and above that called at an EEA port,
     with fuel consumed, distance and hours at sea. Its column names were not verified here, so the loader matches headers by meaning and reports which columns it used and which it ignored.
  2. Values the user gives in the question (daily consumption at a speed, or fuel per nautical mile).
If neither exists the result lists exactly what is missing; nothing is guessed.

Derived: fuel per nautical mile at the reference speed = fuel / distance, and reference speed = distance / hours at sea (MRV annual averages include laden and ballast legs and port fuel, which is stated).
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "vessels"

ALIASES = {
    "imo": [r"^imo( number| no\.?)?$"],
    "name": [r"^(ship )?name$", r"^vessel( name)?$"],
    "ship_type": [r"^ship ?type$", r"^type$"],
    "gt": [r"^gross ?tonnage( \(gt\))?$", r"^gt$"],
    "dwt": [r"^dead ?weight( tonnage)?( \(dwt\))?$", r"^dwt$"],
    "fuel_t": [r"total fuel consumption.*(tonnes|m tonnes)", r"^fuel_t$", r"^annual fuel( t)?$"],
    "co2_t": [r"total co2 emissions.*(tonnes|m tonnes)", r"^co2_t$"],
    "hours_at_sea": [r"(total )?time spent at sea.*hours", r"^hours_at_sea$"],
    "distance_nm": [r"(total )?distance travel+ed.*(n mile|nm)", r"^distance_nm$"],
    "fuel_kg_per_nm": [r"fuel consumption per distance.*kg", r"^fuel_kg_per_nm$"],
    "service_speed_kn": [r"^service_speed_kn$", r"^service speed"],
    "daily_consumption_t": [r"^daily_consumption_t$", r"^consumption.*per day"],
    "design_speed_kn": [r"^design_speed_kn$", r"^design speed"],
}


def _norm(h: str) -> str:
    return re.sub(r"\s+", " ", str(h or "").strip().lower())


def _num(v) -> Optional[float]:
    if v is None or str(v).strip() in ("", "-", "N/A", "n/a", "nan"):
        return None
    try:
        return float(str(v).replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def _match(headers):
    out, used = {}, set()
    for field, pats in ALIASES.items():
        for h in headers:
            if h in used:
                continue
            if any(re.search(p, _norm(h)) for p in pats):
                out[field] = h
                used.add(h)
                break
    return out


def vessel_from_row(row: dict, colmap: dict) -> dict:
    g = lambda f: row.get(colmap[f]) if f in colmap else None
    v = {"imo": (re.sub(r"\D", "", str(g("imo") or "")) or None), "name": (str(g("name")).strip() if g("name") else None), "ship_type": g("ship_type"),
         "gt": _num(g("gt")), "dwt": _num(g("dwt")), "design_speed_kn": _num(g("design_speed_kn"))}
    fuel, hours, dist = _num(g("fuel_t")), _num(g("hours_at_sea")), _num(g("distance_nm"))
    kgnm, vref, daily = _num(g("fuel_kg_per_nm")), _num(g("service_speed_kn")), _num(g("daily_consumption_t"))
    notes = []
    if dist and hours:
        vref = vref or dist / hours
        notes.append("speed is the annual average (distance / hours at sea)")
    if kgnm is None and fuel and dist:
        kgnm = fuel * 1000.0 / dist
        notes.append("fuel per nm is annual fuel / annual distance: includes laden and ballast legs and port fuel")
    if kgnm is None and daily and vref:
        kgnm = daily * 1000.0 / (24.0 * vref)
    v.update({"reference_speed_kn": vref, "fuel_kg_per_nm_ref": kgnm, "co2_t_annual": _num(g("co2_t")), "notes": notes,
              "missing": [k for k, x in (("reference_speed_kn", vref), ("fuel_kg_per_nm_ref", kgnm)) if x is None]})
    return v


def load_fleet_csv(text: str) -> dict:
    rdr = csv.DictReader(io.StringIO(text))
    cm = _match(rdr.fieldnames or [])
    ships = [vessel_from_row(r, cm) for r in rdr]
    return {"ships": ships, "matched_columns": cm, "ignored_columns": [h for h in (rdr.fieldnames or []) if h not in cm.values()]}


def load_fleet_file(path: Path) -> dict:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(p, sheet_name=0, header=None)
        # THETIS-MRV style files carry a few title rows: use the first row that contains an IMO-like header
        hdr = next((i for i in range(min(15, len(df))) if any(re.search(r"imo", _norm(c)) for c in df.iloc[i].tolist())), 0)
        df.columns = [str(c) for c in df.iloc[hdr].tolist()]
        df = df.iloc[hdr + 1:]
        text = df.to_csv(index=False)
        return load_fleet_csv(text)
    return load_fleet_csv(p.read_text(encoding="utf-8-sig", errors="replace"))


def find(name_or_imo: str, data_dir: Optional[Path] = None, fleets: Optional[list] = None) -> dict:
    """Look a ship up by IMO number or name in the fleet files. Returns {'vessel': {...}|None, 'candidates': [...], 'files': [...], 'why': str}."""
    q = (name_or_imo or "").strip()
    imo = re.sub(r"\D", "", q) if re.fullmatch(r"(imo\s*)?\d{7}", q, re.I) else None
    files, ships = [], []
    if fleets is not None:
        ships = list(fleets)
    else:
        d = Path(data_dir) if data_dir else DATA_DIR
        for f in sorted(d.glob("*")) if d.exists() else []:
            if f.suffix.lower() in (".csv", ".xlsx", ".xls"):
                try:
                    ships += load_fleet_file(f)["ships"]
                    files.append(f.name)
                except Exception as e:                                  # noqa: BLE001
                    files.append(f"{f.name} (unreadable: {str(e)[:60]})")
    if not ships:
        return {"vessel": None, "candidates": [], "files": files, "why": "no fleet file found: put the EU MRV (THETIS-MRV) annual report, or your own fleet CSV, in data/vessels/"}
    hits = [s for s in ships if (imo and s.get("imo") == imo) or (not imo and s.get("name") and s["name"].strip().lower() == q.lower())]
    if not hits and not imo:
        hits = [s for s in ships if s.get("name") and q.lower() in s["name"].lower()][:6]
        if len(hits) > 1:
            return {"vessel": None, "candidates": [{"name": s["name"], "imo": s["imo"], "ship_type": s["ship_type"]} for s in hits], "files": files, "why": "several ships match: give the IMO number"}
    if not hits:
        return {"vessel": None, "candidates": [], "files": files, "why": f"{q!r} is not in the fleet files loaded ({len(ships)} ships)"}
    # more than one year of MRV rows for the same ship: keep the one with the most distance (the most complete record)
    best = max(hits, key=lambda s: (s.get("fuel_kg_per_nm_ref") is not None, s.get("reference_speed_kn") is not None))
    return {"vessel": best, "candidates": [], "files": files, "why": None}
