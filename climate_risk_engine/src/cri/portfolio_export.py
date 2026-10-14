"""
Consolidated portfolio run → one CSV row per asset.

Input columns (xlsx / csv; header names are case-insensitive):
  required  name, lat, lon
  optional  sector, type, capacity, emissions_t, value_usd, company
  earnings  revenue_usd + ebitda_usd (both, or neither) with optional da_usd, interest_usd, net_debt_usd, book_value_usd,
            tax_rate, electricity_mwh, fuel + fuel_gj, commodity, sustaining_capex_usd, covenant_max_leverage,
            covenant_min_icr — adds EBITDA, leverage, rating and covenant results per site (upload endpoint only)
Every asset goes through the Risk Analyst (physical hazards → policies in force → NGFS Phase 5 transition →
Monte Carlo loss distribution). Water stress is added from WRI Aqueduct 4.0.
"""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path
from typing import Callable, Iterable, Optional

REQUIRED = ("name", "lat", "lon")
OPTIONAL = ("sector", "type", "capacity", "emissions_t", "value_usd", "company")
FIN_NUM = ("commissioned_year", "retirement_year", "export_share_eu", "revenue_usd", "ebitda_usd", "da_usd", "interest_usd", "net_debt_usd", "book_value_usd", "tax_rate",
           "electricity_mwh", "fuel_gj", "price_realisation", "operating_leverage", "sustaining_capex_usd",
           "covenant_max_leverage", "covenant_min_icr", "lgd")
SCEN_SHORT = {"Net Zero 2050": "nz2050", "Below 2°C": "b2c", "Low demand": "lowdemand",
              "Delayed transition": "delayed", "NDCs": "ndcs", "Fragmented World": "fragmented",
              "Current Policies": "current"}


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(str(v).replace(",", "").strip())
    except ValueError:
        return None
    return None if math.isnan(f) else f


def read_portfolio(path: str | Path) -> list[dict]:
    import pandas as pd
    p = Path(path)
    df = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xlsm", ".xls") else pd.read_csv(p)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"latitude": "lat", "longitude": "lon", "lng": "lon", "asset": "name",
                            "asset_name": "name", "value": "value_usd"})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing column(s): {', '.join(missing)} (need name, lat, lon)")
    rows = []
    for i, r in df.iterrows():
        lat, lon = _num(r.get("lat")), _num(r.get("lon"))
        rec = {"row": int(i) + 2, "name": str(r.get("name") or f"asset {i + 1}"), "lat": lat, "lon": lon}
        for k in OPTIONAL:
            v = r.get(k)
            if k in ("capacity", "emissions_t", "value_usd"):
                rec[k] = _num(v)
            else:
                rec[k] = None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)
        for k in FIN_NUM:                       # optional earnings-layer columns (read by /analyst/sites/upload)
            if k in df.columns:
                rec[k] = _num(r.get(k))
        for k in ("fuel", "commodity"):
            v = r.get(k) if k in df.columns else None
            rec[k] = None if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip().lower()
        rec["error"] = None if (lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180) \
            else "invalid or missing coordinates"
        rows.append(rec)
    return rows


def analyse_rows(rows: Iterable[dict], n: int = 2000,
                 progress: Optional[Callable[[int, int, dict], None]] = None) -> list[dict]:
    from .finance.analyst import analyse_asset
    rows = list(rows)
    out = []
    for k, r in enumerate(rows):
        if r.get("error"):
            out.append({"input": r, "error": r["error"]})
        else:
            try:
                res = analyse_asset(r["lat"], r["lon"], name=r["name"], sector=r.get("sector"), atype=r.get("type"),
                                    capacity=r.get("capacity"), emissions_t=r.get("emissions_t"),
                                    value_usd=r.get("value_usd"), n=n)
                out.append({"input": r, "result": res})
            except Exception as exc:  # one bad asset must not stop the portfolio
                out.append({"input": r, "error": f"{type(exc).__name__}: {exc}"})
        if progress:
            progress(k + 1, len(rows), out[-1])
    return out


COLUMNS = (["row", "name", "company", "lat", "lon", "iso3", "admin1", "sector", "type", "value_usd", "value_basis",
            "cyclone", "river_flood", "coastal_flood", "physical_aal_pct", "physical_aal_usd", "physical_loss_1in100_usd",
            "carbon_price_in_force_usd_t", "policies_in_force", "water_stress_score", "water_stress_label",
            "water_stress_2050_bau"]
           + [f"pv_loss_{s}_{m}_usd" for s in SCEN_SHORT.values() for m in ("mean", "p95")]
           + ["headline", "caveats", "error"])


def to_csv(items: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader()
    for it in items:
        r = it["input"]
        row = {"row": r.get("row"), "name": r.get("name"), "company": r.get("company"), "lat": r.get("lat"),
               "lon": r.get("lon"), "sector": r.get("sector"), "type": r.get("type"), "error": it.get("error")}
        res = it.get("result")
        if res:
            a, ph, pol, fin = res["asset"], res["physical"], res["policy"], res["financial"]
            val = a.get("value") or {}
            water = ph.get("water") or {}
            ws = water.get("water_stress") or {}
            rp = fin["physical"].get("annual_loss_rp") or {}
            row.update({
                "iso3": a.get("iso3"), "admin1": a.get("admin1"), "sector": a.get("sector") or r.get("sector"),
                "type": a.get("type") or r.get("type"), "value_usd": val.get("mode"), "value_basis": val.get("basis"),
                "cyclone": ph["cyclone"].get("severity"),
                "river_flood": ph["river_flood"].get("severity") if ph["river_flood"].get("assessed", True) else "NOT ASSESSED",
                "coastal_flood": ph["coastal_flood"].get("severity") if ph["coastal_flood"].get("assessed", True) else "NOT ASSESSED",
                "physical_aal_pct": fin["physical"].get("aal_pct"), "physical_aal_usd": fin["physical"].get("aal_usd"),
                "physical_loss_1in100_usd": rp.get("100") if isinstance(rp, dict) else None,
                "carbon_price_in_force_usd_t": pol.get("effective_price_usd_t"),
                "policies_in_force": "; ".join(p["name"] for p in pol.get("in_force", [])),
                "water_stress_score": ws.get("score"), "water_stress_label": ws.get("label"),
                "water_stress_2050_bau": (water.get("water_stress_path_bau") or {}).get("2050"),
                "headline": res.get("headline"),
                "caveats": " | ".join(c for c in res.get("caveats", []) if c.startswith(("Transition", "No asset value"))),
            })
            for v in fin["scenarios"].values():
                s = SCEN_SHORT.get(v["label"])
                if s:
                    row[f"pv_loss_{s}_mean_usd"] = v["total_npv"].get("mean")
                    row[f"pv_loss_{s}_p95_usd"] = v["total_npv"].get("p95")
        for k, v in list(row.items()):
            if isinstance(v, float):
                row[k] = round(v, 4) if abs(v) < 1000 else round(v)
        w.writerow(row)
    return buf.getvalue()
