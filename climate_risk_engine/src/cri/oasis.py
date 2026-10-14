"""
cri/oasis.py — interoperability with the (re)insurance exposure and results standards (Oasis LMF Open Exposure Data, OED, and Open Results Data, ORD).

  read_oed(location_rows, account_rows)   OED location (and optional account) rows -> ClimRisk sites, with the Portfolio -> Account -> Location hierarchy kept as metadata,
                                          plus a list of everything that could not be carried over (nothing is dropped silently)
  ord_tables(result, meta)                an insurer run -> ORD-style tables: summary info, annual loss table (AAL), exceedance probability table, location-level AAL
  to_csv(rows)                            rows -> CSV text

What this is and is not
  * It reads the standard column names and writes tables shaped like ORD. It is NOT certified by Oasis; check the numeric codes below against the current OED/ORD
    documentation before exchanging files with a vendor model. The peril and type code tables are ASSUMED and are plain constants you can edit.
  * The engine models tropical-cyclone wind, river flood and coastal flood only. Rows covering other perils keep only those; rows with none are reported, not modelled.
  * Construction and occupancy codes are carried as metadata. They do NOT change the damage curves (the engine's curves are by hazard and sector, not by construction class);
    pass occupancy_map to set the sector from a code if you want the sector-level curve.
  * Policy-level terms (layers, account-level deductibles) are not applied; location terms are. Both are listed in the warnings so no one assumes otherwise.
"""
from __future__ import annotations

import csv
import io
from typing import Optional

# OED peril codes -> engine peril. ASSUMED; verify against the OED peril code list for your version.
WIND_CODES = {"WTC", "WEC", "WW1", "WW2", "WIND"}
FLOOD_CODES = {"OO1", "WSS", "FLOOD", "ORF"}
# OED deductible/limit type codes: 0 = flat amount, 2 = fraction of TIV. Others (e.g. fraction of loss) cannot be applied by this engine. ASSUMED; verify.
TYPE_MONEY, TYPE_PCT_TIV = 0, 2


def parse_csv(text: str) -> list:
    return [dict(r) for r in csv.DictReader(io.StringIO(text.strip()))]


def _f(v):
    if v in (None, ""):
        return None
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def _perils(raw, warn, ref):
    if raw in (None, ""):
        return ["wind", "flood"]
    out, other = [], []
    for c in str(raw).replace(",", ";").split(";"):
        c = c.strip().upper()
        if not c:
            continue
        if c in WIND_CODES:
            out.append("wind")
        elif c in FLOOD_CODES:
            out.append("flood")
        else:
            other.append(c)
    if other:
        warn.append({"ref": ref, "issue": "perils_not_modelled", "detail": f"{','.join(other)} are not modelled by this engine and were ignored"})
    return sorted(set(out))


def _terms(row, kind, tiv, warn, ref):
    """-> (money or None, pct or None) for deductible/limit 'All coverages'."""
    key = f"Loc{kind}6All"
    v = _f(row.get(key))
    if v is None:
        for k in row:
            if k.startswith(f"Loc{kind}") and k != key and _f(row.get(k)):
                warn.append({"ref": ref, "issue": "coverage_specific_terms", "detail": f"{k} is set but only {key} is applied; combine the coverages into {key}"})
                break
        return None, None
    ty = _f(row.get(f"Loc{kind}Type6All"))
    ty = TYPE_MONEY if ty is None else int(ty)
    if ty == TYPE_MONEY:
        return v, None
    if ty == TYPE_PCT_TIV:
        return None, 100.0 * v                                           # OED stores fractions; the engine takes percent
    warn.append({"ref": ref, "issue": "term_type_not_supported", "detail": f"{key} type {ty} cannot be applied by this engine; the term was ignored (deductible 0 / limit = TIV)"})
    return None, None


def read_oed(location_rows: list, account_rows: Optional[list] = None, occupancy_map: Optional[dict] = None, fx: Optional[dict] = None) -> dict:
    """fx: {'EUR': 1.08, ...} = USD per unit of currency. Rows in a currency with no rate (and not USD) are skipped and reported."""
    warn, sites, meta = [], [], []
    omap = {str(k): v for k, v in (occupancy_map or {}).items()}
    fx = {str(k).upper(): float(v) for k, v in (fx or {}).items()}
    fx.setdefault("USD", 1.0)
    accounts: dict = {}
    for a in account_rows or []:
        key = (str(a.get("PortNumber") or ""), str(a.get("AccNumber") or ""))
        accounts.setdefault(key, []).append(a)
        for col in ("LayerLimit", "LayerAttachment", "AccDed6All", "AccLimit6All", "PolDed6All", "PolLimit6All"):
            if _f(a.get(col)):
                warn.append({"ref": f"account {key[1]}", "issue": "policy_terms_not_applied", "detail": f"{col} is set; policy-level terms are not applied, only location terms"})
    seen = set()
    for i, r in enumerate(location_rows, 1):
        port, acc, loc = str(r.get("PortNumber") or ""), str(r.get("AccNumber") or ""), str(r.get("LocNumber") or i)
        ref = f"location {acc}/{loc}"
        if (port, acc, loc) in seen:
            warn.append({"ref": ref, "issue": "duplicate_location_id", "detail": "same PortNumber/AccNumber/LocNumber appears twice; the second row was skipped"})
            continue
        seen.add((port, acc, loc))
        lat, lon = _f(r.get("Latitude")), _f(r.get("Longitude"))
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
            warn.append({"ref": ref, "issue": "bad_coordinates", "detail": f"Latitude/Longitude missing or invalid ({r.get('Latitude')}, {r.get('Longitude')}); location skipped"})
            continue
        cov = {"building": _f(r.get("BuildingTIV")) or 0.0, "other": _f(r.get("OtherTIV")) or 0.0, "contents": _f(r.get("ContentsTIV")) or 0.0, "bi": _f(r.get("BITIV")) or 0.0}
        tiv = sum(cov.values())
        if tiv <= 0:
            warn.append({"ref": ref, "issue": "no_tiv", "detail": "BuildingTIV, OtherTIV, ContentsTIV and BITIV are all empty or zero; location skipped"})
            continue
        cur = str(r.get("LocCurrency") or "USD").upper()
        if cur not in fx:
            warn.append({"ref": ref, "issue": "currency_without_rate", "detail": f"currency {cur} has no rate in fx; location skipped (pass fx={{'{cur}': usd_per_unit}})"})
            continue
        rate = fx[cur]
        covers = _perils(r.get("LocPerilsCovered"), warn, ref)
        if not covers:
            warn.append({"ref": ref, "issue": "no_modelled_peril", "detail": "none of the covered perils is modelled (wind, flood); location skipped"})
            continue
        ded, ded_pct = _terms(r, "Ded", tiv, warn, ref)
        lim, lim_pct = _terms(r, "Limit", tiv, warn, ref)
        occ, con = str(r.get("OccupancyCode") or ""), str(r.get("ConstructionCode") or "")
        s = {"name": str(r.get("LocName") or f"{acc}-{loc}"), "lat": lat, "lon": lon, "tiv_usd": tiv * rate, "sector": omap.get(occ) or "other", "country": r.get("CountryCode") or None, "covers": covers}
        if ded is not None:
            s["deductible_usd"] = ded * rate
        elif ded_pct is not None:
            s["deductible_pct"] = ded_pct
        if lim is not None:
            s["limit_usd"] = lim * rate
        elif lim_pct is not None:
            s["limit_pct"] = lim_pct
        sites.append(s)
        meta.append({"PortNumber": port, "AccNumber": acc, "LocNumber": loc, "OccupancyCode": occ, "ConstructionCode": con, "currency": cur, "fx": rate,
                     "coverage_tiv": {k: v * rate for k, v in cov.items()}})
    if sites and not omap:
        warn.append({"ref": "portfolio", "issue": "construction_occupancy_not_used",
                     "detail": "OccupancyCode and ConstructionCode are carried as metadata; damage curves are by hazard and sector, not by construction class. Pass occupancy_map to set the sector."})
    n_bi = sum(1 for m in meta if m["coverage_tiv"]["bi"] > 0)
    if n_bi:
        warn.append({"ref": "portfolio", "issue": "bi_merged_into_physical_tiv",
                     "detail": f"{n_bi} location(s) have BITIV; it is added to TIV, so the physical damage curve is applied to the business-interruption value as a proxy. Interruption is a time-element loss (downtime, waiting period) that does not scale with physical damage; treat the interruption share of the result as indicative. The split is kept in meta[].coverage_tiv."})
    acc_n = len({(m["PortNumber"], m["AccNumber"]) for m in meta})
    return {"sites": sites, "meta": meta, "warnings": warn, "read": len(location_rows), "loaded": len(sites), "skipped": len(location_rows) - len(sites),
            "hierarchy": {"portfolios": len({m["PortNumber"] for m in meta}), "accounts": acc_n, "locations": len(meta), "account_rows_supplied": len(account_rows or [])}}


# ── ORD-style output ────────────────────────────────────────────────────────────────────────────────────────────────────────────

# ORD EPType codes as recalled: 1 OEP, 2 OEP TVaR, 3 AEP, 4 AEP TVaR. ASSUMED; the name column is the safe one to read.
EP_TYPES = {"oep": (1, "OEP"), "aep": (3, "AEP")}


def ord_tables(result: dict, meta: Optional[list] = None) -> dict:
    """Tables from an insurance.run() result. Summary levels: 1 = portfolio, 2xx = account, 3xxx = location. EP is portfolio-level only (the run keeps no per-account year losses)."""
    st = result["stats"]
    rows = result["sites"]
    meta = meta or [{} for _ in rows]
    info = [{"SummaryId": 1, "Level": "portfolio", "SummaryDescription": "All locations"}]
    alt = [{"SummaryId": 1, "Perspective": "gross of reinsurance, after location terms", "SampleType": 2, "SampleTypeName": "sampled", "MeanLoss": st["gross"]["aal"], "SDLoss": st["gross"]["std"]},
           {"SummaryId": 1, "Perspective": "ground up", "SampleType": 2, "SampleTypeName": "sampled", "MeanLoss": st["ground_up"]["aal"], "SDLoss": None}]
    if "net" in st:
        alt.append({"SummaryId": 1, "Perspective": "net of reinsurance", "SampleType": 2, "SampleTypeName": "sampled", "MeanLoss": st["net"]["aal"], "SDLoss": st["net"]["std"]})
    ept = []
    for persp, key in (("gross of reinsurance, after location terms", "gross"), ("net of reinsurance", "net")):
        if key not in st:
            continue
        for r in st[key]["rows"]:
            for k, (code, name) in EP_TYPES.items():
                ept.append({"SummaryId": 1, "Perspective": persp, "EPType": code, "EPTypeName": name, "ReturnPeriod": r["rp"], "Loss": r[k], "Reliable": r["reliable"]})
    accs: dict = {}
    loc_alt = []
    for j, (row, m) in enumerate(zip(rows, meta)):
        sid = 3000 + j + 1
        info.append({"SummaryId": sid, "Level": "location", "SummaryDescription": row["name"], "PortNumber": m.get("PortNumber"), "AccNumber": m.get("AccNumber"), "LocNumber": m.get("LocNumber")})
        loc_alt.append({"SummaryId": sid, "Perspective": "gross of reinsurance, after location terms", "SampleType": 2, "MeanLoss": row["aal"], "GroundUpMeanLoss": row["ground_up_aal"], "TIV": row["tiv"]})
        k = (m.get("PortNumber"), m.get("AccNumber"))
        if m.get("AccNumber") is not None:
            e = accs.setdefault(k, {"aal": 0.0, "gu": 0.0, "tiv": 0.0})
            e["aal"] += (row["aal"] or 0.0)          # a location with perils not assessed contributes nothing known; its MeanLoss stays blank in the location table
            e["gu"] += (row["ground_up_aal"] or 0.0)
            e["tiv"] += row["tiv"]
    for n, (k, e) in enumerate(sorted(accs.items(), key=lambda kv: str(kv[0])), 1):
        sid = 200 + n
        info.append({"SummaryId": sid, "Level": "account", "SummaryDescription": f"Account {k[1]}", "PortNumber": k[0], "AccNumber": k[1], "LocNumber": None})
        alt.append({"SummaryId": sid, "Perspective": "gross of reinsurance, after location terms", "SampleType": 2, "SampleTypeName": "sampled", "MeanLoss": e["aal"], "SDLoss": None})
    return {"summary_info": info, "alt": alt, "ept": ept, "location_alt": loc_alt,
            "notes": ["ORD-style, not certified: check numeric codes against the current ORD documentation before exchanging with a vendor model",
                      "account AAL is the sum of location AALs after location terms (AAL adds); account-level EP curves are not produced because per-account yearly losses are not kept",
                      "SDLoss is given for the portfolio only"]}


def to_csv(rows: list) -> str:
    if not rows:
        return ""
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return out.getvalue()
