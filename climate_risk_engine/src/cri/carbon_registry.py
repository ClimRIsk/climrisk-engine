"""
Legacy CDM CERs and registry-record import for the carbon-credit screen.

Two limits of carbon_markets.screen_credits are addressed here.

1. Registry records. There is no live registry connection: registries publish records and exports, and this module reads an export
   (CSV text or a list of dicts) from the CDM Registry, Verra, Gold Standard or any other source whose columns match a known alias.
   It reports which columns it matched and which it ignored, and it never fills a missing field with a default: a blank stays blank
   and any rule that needs it answers "cannot tell" and names the missing field.

2. Legacy CDM CERs (UNFCCC Clean Development Mechanism). Two pathways are screened against published text:
   * Article 6.4 transition for use towards a host country's first NDC. Conditions are paragraph 75(a)-(f) of the annex to decision
     3/CMA.3 as described in the UNFCCC secretariat's technical paper PA/A6.4/TP/4 (15 Sep 2022; a paper without formal status):
     (a) CDM activity registered on or after 1 Jan 2013; (b) CERs transferred to and held in the A6.4 mechanism registry, identified
     as pre-2021; (c) use towards the first NDC only; (d) no corresponding adjustment and no share of proceeds; (f) temporary CERs
     and long-term CERs excluded. Transfer deadlines and procedures change: confirm them with the UNFCCC before relying on them.
   * CORSIA. ICAO document "CORSIA Eligible Emissions Units" (Oct 2025 edition): CDM CERs are eligible only for the 2021-2023 pilot
     phase (first crediting period from 1 Jan 2016, reductions through 31 Dec 2020, excluding afforestation/reforestation including
     tCERs and lCERs). CDM is not listed for the 2024-2026 first phase or the 2027-2029 second phase, and the pilot phase has ended,
     so no open CORSIA phase accepts CDM CERs as listed in that edition.

This is a rules check on declared record fields. It does not verify the record against the registry, does not price units, and is not
legal advice.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Optional

SOURCES = {
    "a64": "UNFCCC secretariat technical paper PA/A6.4/TP/4 (15 Sep 2022) on decision 3/CMA.3 annex, paragraph 75",
    "corsia": "ICAO, CORSIA Eligible Emissions Units, October 2025 edition (13th)",
}
AR_TYPES = {"afforestation", "reforestation", "arr", "a/r", "afforestation_reforestation", "afforestation_and_reforestation"}

ALIASES = {
    "id": ["serial number", "serial", "unit serial number", "serial no", "id", "unit id", "block start"],
    "quantity_t": ["quantity", "quantity of units", "number of units", "amount", "volume", "units", "quantity issued", "quantity (tco2e)"],
    "vintage": ["vintage", "vintage year", "reduction year", "monitoring period end", "year of reduction", "emission reduction year"],
    "registered_year": ["registration date", "date of registration", "registered", "registration year", "project registration date"],
    "crediting_period_start_year": ["crediting period start", "crediting period start date", "start of crediting period"],
    "project_type": ["project type", "type", "activity type", "sectoral scope", "category"],
    "unit_type": ["unit type", "type of unit", "unit", "cer type"],
    "status": ["status", "unit status", "account status"],
    "retired_date": ["retirement date", "date retired", "cancellation date", "retired on", "cancelled on"],
    "country": ["host country", "host party", "country"],
    "project_id": ["project id", "project number", "project reference", "activity id"],
}


def _year(v) -> Optional[int]:
    if v is None or str(v).strip() == "":
        return None
    m = re.search(r"\b(19\d\d|20\d\d)\b", str(v))
    return int(m.group(1)) if m else None


def _num(v) -> Optional[float]:
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def import_rows(rows: list, programme: Optional[str] = None) -> dict:
    """Map registry-export rows to credit lots. Returns {lots, matched_columns, ignored_columns, incomplete}."""
    if not rows:
        return {"lots": [], "matched_columns": {}, "ignored_columns": [], "incomplete": [], "note": "no rows"}
    cols = list(rows[0].keys())
    low = {c: re.sub(r"\s+", " ", c.strip().lower()) for c in cols}
    matched: dict = {}
    for field, names in ALIASES.items():
        for c in cols:
            if low[c] in names and field not in matched:
                matched[field] = c
    ignored = [c for c in cols if c not in matched.values()]
    lots, incomplete = [], []
    for i, r in enumerate(rows):
        g = lambda f: r.get(matched[f]) if f in matched else None            # noqa: E731
        status = str(g("status") or "").strip().lower()
        retired = True if (status in ("retired", "cancelled", "canceled") or str(g("retired_date") or "").strip()) else (False if status else None)
        lot = {"id": str(g("id") or f"row{i + 1}"), "registry": programme or "", "project_type": str(g("project_type") or ""),
               "quantity_t": _num(g("quantity_t")), "vintage": _year(g("vintage")), "registered_year": _year(g("registered_year")),
               "crediting_period_start_year": _year(g("crediting_period_start_year")), "unit_type": str(g("unit_type") or "").strip() or None,
               "retired": retired, "country": g("country") or None, "project_id": g("project_id") or None}
        miss = [k for k in ("quantity_t", "vintage") if lot[k] is None]
        if miss:
            incomplete.append({"id": lot["id"], "missing": miss})
        lots.append(lot)
    return {"lots": lots, "matched_columns": matched, "ignored_columns": ignored, "incomplete": incomplete,
            "note": "Read from an export: nothing was checked against the registry. Blank fields stay blank."}


def import_csv(text: str, programme: Optional[str] = None) -> dict:
    return import_rows(list(csv.DictReader(io.StringIO(text))), programme)


def is_cdm_cer(lot: dict) -> bool:
    s = " ".join(str(lot.get(k) or "") for k in ("registry", "programme", "unit_type")).lower()
    return "cdm" in s or "clean development" in s or bool(re.search(r"\b(t|l)?cers?\b", s))


def _is_ar(lot: dict) -> bool:
    pt = (lot.get("project_type") or "").lower().replace(" ", "_").replace("-", "_")
    return pt in AR_TYPES or str(lot.get("unit_type") or "").strip().upper() in ("TCER", "LCER")


def assess_cer(lot: dict) -> dict:
    """Screen one CDM CER lot against the Article 6.4 first-NDC conditions and CORSIA. Every verdict names the fields it used."""
    v = lot.get("vintage")
    reg = lot.get("registered_year")
    cps = lot.get("crediting_period_start_year")
    ar = _is_ar(lot)
    out = []

    # Article 6.4: conditions 75(a), (b) pre-2021, (f)
    need = [k for k, x in (("vintage", v), ("registered_year", reg)) if x is None]
    if ar:
        out.append({"pathway": "Article 6.4 transition for use towards a first NDC", "status": "not_eligible", "reasons":
                    ["temporary and long-term CERs (and afforestation/reforestation units declared as such) are excluded, paragraph 75(f)"], "source": SOURCES["a64"]})
    elif need:
        out.append({"pathway": "Article 6.4 transition for use towards a first NDC", "status": "cannot_tell", "reasons":
                    ["missing field(s): " + ", ".join(need)], "source": SOURCES["a64"]})
    else:
        why, bad = [], []
        (why if reg >= 2013 else bad).append(f"CDM activity registered {reg} ({'on or after' if reg >= 2013 else 'before'} 1 Jan 2013), paragraph 75(a)")
        (why if v <= 2020 else bad).append(f"reductions from {v} ({'pre-2021' if v <= 2020 else 'post-2020: the CDM issues no CERs for reductions after 2020'}), paragraph 75(b)")
        out.append({"pathway": "Article 6.4 transition for use towards a first NDC", "status": "not_eligible" if bad else "may_be_eligible",
                    "reasons": bad + why + ([] if bad else ["still needs: transfer to and holding in the A6.4 mechanism registry, host country use only towards its first NDC "
                                                           "(paragraph 75(b)-(c)); no corresponding adjustment is required (75(d)). Confirm current transfer procedure and deadlines with the UNFCCC"]),
                    "source": SOURCES["a64"]})

    # CORSIA
    need = [k for k, x in (("vintage", v), ("crediting_period_start_year", cps)) if x is None]
    reasons = ["CDM is not among the programmes listed for the 2024-2026 first phase or the 2027-2029 second phase, and the 2021-2023 pilot phase has ended: "
               "no open CORSIA phase accepts CDM CERs in this edition"]
    if ar:
        reasons.append("afforestation/reforestation CERs (including tCERs and lCERs) were excluded even in the pilot phase")
    if not need and not ar:
        reasons.append(f"for the record: pilot-phase conditions were first crediting period from 1 Jan 2016 ({'met' if cps >= 2016 else 'not met'}: {cps}) "
                       f"and reductions through 31 Dec 2020 ({'met' if v <= 2020 else 'not met'}: {v})")
    out.append({"pathway": "CORSIA offsetting", "status": "not_eligible", "reasons": reasons, "source": SOURCES["corsia"]})
    return {"pathways": out, "limits": "Rules applied to declared fields; not checked against the registry; no price; not legal advice."}
