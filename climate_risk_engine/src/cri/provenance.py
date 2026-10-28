"""Provenance and freshness registry for every data file shipped in ``cri/data``.

Rule: a number the engine shows must be traceable to a named source and an edition. This module records, for each shipped file,
who published it, which edition/vintage it is, when we retrieved or built it (only where that is actually known), what licence applies,
how old it is today against a stated review interval, and what the engine uses it for.

Two kinds of fact are kept apart:
  * ``vintage``    the date/year of the underlying data (e.g. IBTrACS v04r01 as of the build). Age is computed from this.
  * ``retrieved``  the day we pulled it, only when a build script or the file itself records that. Never guessed.
Where a field is not known it is ``None`` and the file is counted as ``partial`` or ``undocumented``. Nothing is filled in to look complete.

``audit()`` also walks ``cri/data`` and reports any file that has no registry entry, so a new unlabelled data file fails the test suite.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

DATA_DIR = Path(__file__).resolve().parent / "data"

# review_days: how long this kind of data stays fit for use before someone should re-check the source.
ANNUAL, STATIC, LIVE = 400, 1830, 120

# status vocabulary
#   published      a named public/regulatory source, edition known
#   derived        computed by us from published sources (build script named)
#   fitted         a model we fitted on named data; heldout skill stated in the file
#   third_party    produced by someone else's model/curation we did not reproduce; treat as an indicator, not a measurement
#   code           code or lookup (no data claim)
REGISTRY: dict[str, dict[str, Any]] = {
    "admin1_flopros.json": dict(
        publisher="Scussolini et al. 2016, FLOPROS v1 (NHESS 16:1049)", licence="CC BY 3.0", vintage="2016", status="published",
        use="Flood protection standard by first-level admin unit", review_days=STATIC, build="scripts/build_finance_data.py --stage admin1"),
    "admin1_grid.npz": dict(
        publisher="Raster of admin-1 unit ids built from the FLOPROS v1 layer (0.05 deg)", licence="CC BY 3.0", vintage="2016", status="derived",
        use="Point -> state/province lookup", review_days=STATIC, build="scripts/build_finance_data.py --stage admin1"),
    "aqueduct40.npz": dict(
        publisher="WRI Aqueduct 4.0 water risk data, Y2023M07D05 (Kuzma et al. 2023, doi:10.46830/writn.23.00061)", licence="CC BY 4.0", vintage="2023-07-05",
        status="derived", use="Baseline and future water stress, 13 indicators, industry-weighted scores", review_days=STATIC, build="scripts/build_aqueduct_data.py"),
    "asset_register.npz": dict(
        publisher="Climate TRACE v6 API + EPA GHGRP 2023 facilities", licence="Climate TRACE: CC BY 4.0 (check current terms); EPA: public domain",
        vintage="2023", built="2026-09-29", status="derived", use="Company -> asset locations, capacity, emissions",
        review_days=ANNUAL, build="scripts/build_asset_register.py", note="Ownership attribution is as reported by the sources; no manual correction."),
    "asset_register_companies.json": dict(
        publisher="Same as asset_register.npz (owner names normalised by our build script)", licence="see asset_register.npz", vintage="2023", built="2026-09-29",
        status="derived", use="Company name/alias -> asset indices", review_days=ANNUAL, build="scripts/build_asset_register.py",
        note="Name matching is by normalised string; a wrong match is possible and is not scored."),
    "carbon_factors.json": dict(
        publisher="Compiled carbon factor library (per-entry source fields inside the file)", licence="per entry", vintage="2026-09-30", status="derived",
        use="Scope 1/2/3 emission factors", review_days=ANNUAL, build="scripts/build_carbon_factors.py", note="Each factor carries its own source; audit per entry with /carbon/factors."),
    "carbon_pricing_wb.json": dict(
        publisher="World Bank Carbon Pricing Dashboard, data release May 2026 (data updated 1 April 2026)", licence="CC BY 4.0",
        vintage="2026-04-01", status="published", use="Carbon price and coverage by jurisdiction", review_days=LIVE, url="https://carbonpricingdashboard.worldbank.org/",
        note="Annual snapshot of 1 April prices. EUA and other spot prices move daily; a user-supplied price overrides this."),
    "carbontlens_tickers.json": dict(
        publisher="HuggingFace dataset SubramaniMokkala/climate-credit-risk-analyzer (CarbonLens)", licence="NOT CHECKED", vintage=None, status="third_party",
        use="Per-ticker climate credit score, ESG level, high-carbon flag", review_days=ANNUAL, build=None,
        note="Output of a third-party XGBoost model we did not reproduce. An indicator, not a measurement. Licence and date not verified. Do not present as engine-derived."),
    "deltares_coastal_index.npz": dict(
        publisher="Deltares Global Flood Maps (coastal, NASADEM 90 m), via Planetary Computer", licence="CC BY 4.0 (check collection page)", vintage="2018 and 2050 sea level",
        status="derived", use="Coastal flood depth by return period; chunk index for range reads", review_days=STATIC, build="scripts/build_finance_data.py --stage deltares"),
    "emissions_calibration.json": dict(
        publisher="EPA GHGRP 2023 direct emitters + SEC EDGAR revenue (our fit)", licence="public domain inputs", vintage="2023", status="fitted",
        use="Sector emission intensities and residual spread for the estimator", review_days=ANNUAL, build="scripts/export_emissions_runtime.py",
        note="Held-out accuracy is stored in the file. Valid for the sectors listed there only."),
    "epa_ghg_hub_2025.json": dict(
        publisher="US EPA, GHG Emission Factors Hub (last modified 15 Jan 2025)", licence="US Government work", vintage="2025-01-15", retrieved="2026-10-28",
        status="published", use="Fuel, electricity, waste, F-gas GWP and blend factors", review_days=ANNUAL, url="https://www.epa.gov/system/files/documents/2025-01/ghg-emission-factors-hub-2025.pdf",
        note="Known printed discrepancy for R-424A recorded in known_discrepancies.",
        newer_edition="A 2026 edition (Cornerstone / Zenodo record dated 2026-05-26; eGRID2024 electricity, IPCC AR6 GWPs, updated refrigerant blends) was reported on 2026-10-28. "
                      "EPA's own page still listed the January 2025 file. Not ingested: not yet read or cross-checked. GWPs here are AR4/AR5 as printed in the 2025 file."),
    "flood_curve_nfip_us.json": dict(
        publisher="FEMA NFIP claims (empirical mean damage ratio by flood depth), residential US", licence="public domain (check OpenFEMA terms)", vintage=None,
        status="derived", use="US residential flood depth-damage curve with CI", review_days=STATIC, build="scripts/validate_flood_vulnerability.py",
        note="Vintage of the claims extract is not recorded in the file."),
    "flood_event_index.json": dict(
        publisher="Copernicus EMS Rapid Mapping and other catalogued flood extents 2023-2026 (see file)", licence="per source", vintage="2023-2026", status="derived",
        use="Bounding boxes of observed flood events near a site", review_days=LIVE, note="Bounding boxes only. Not a complete flood history of any place."),
    "flood_event_model.json": dict(
        publisher="Our logistic model fitted on Copernicus EMS flood polygons (72 events)", licence="n/a", vintage="2023-2026", status="fitted",
        use="Within-landscape flood ranking from HAND and relative elevation", review_days=ANNUAL, build="scripts/flood_event_benchmark.py --stage fit",
        note="Held-out mean AUC 0.76 on 24 events. Ranks places within a flooded landscape; not a return-period or depth model."),
    "ghcn_long.npz": dict(
        publisher="NOAA NCEI GHCN-Daily station index (long-record stations)", licence="public domain", vintage=None, status="derived",
        use="Nearest long-record station for the independent station check", review_days=STATIC, build=None,
        note="Station index; build script is not in the repo and the extract date is not recorded."),
    "glof_events_v4_2.csv.gz": dict(
        publisher="Luetzow & Veh, Glacial Lake Outburst Flood database v4.2", licence="NOT CHECKED", vintage="v4.2", status="published",
        use="Reported GLOFs near a site (history only)", review_days=STATIC, note="Record of what was reported, not a hazard model."),
    "ibtracs_since1980.npz": dict(
        publisher="NOAA NCEI IBTrACS v04r01 (since 1980, all basins)", licence="public domain", vintage="v04r01", status="derived",
        use="Tropical-cyclone tracks for the wind hazard model", review_days=ANNUAL, url="https://www.ncei.noaa.gov/products/international-best-track-archive",
        build="scripts/build_ibtracs_compact.py", note="WMO 10-min winds converted to 1-min by dividing by 0.88 where USA winds are absent."),
    "imo_dcs_class_aer_2022.json": dict(
        publisher="IMO MEPC 81/6/1, fleet carbon intensity report, Annex Table 6", licence="IMO publication", vintage="2022 reporting year", retrieved="2026-10-28",
        status="published", use="Ship-class median AER used as a fallback burn estimate", review_days=ANNUAL,
        note="Class medians, not per-ship burn. Not cross-checked against an EU MRV extract."),
    "iso2_iso3.json": dict(
        publisher="ISO 3166-1 alpha-2 to alpha-3 mapping", licence="n/a", vintage=None, status="code", use="Country code conversion", review_days=STATIC,
        note="Lookup table; not verified against the ISO register after transcription."),
    "jrc_flood_curves.json": dict(
        publisher="Huizinga, De Moel, Szewczyk 2017, JRC Global flood depth-damage functions (via CLIMADA-petals)", licence="JRC open data", vintage="2017",
        status="published", use="Depth-damage curves by continent", review_days=STATIC, note="Industrial buildings. Oceania uses the mean of other continents (flagged in file)."),
    "jrc_flood_tiles.json": dict(
        publisher="JRC CEMS-GloFAS river flood hazard maps v2.1.2 (3 arc-sec, undefended)", licence="JRC open data", vintage="v2.1.2", status="derived",
        use="Tile index for river flood depth by return period", review_days=STATIC, build="scripts/build_finance_data.py --stage jrc"),
    "lca_factors.json": dict(
        publisher="Compiled LCA factor library (per-entry source fields inside the file)", licence="per entry", vintage="2026-09-30", status="derived",
        use="Cradle-to-gate material and product factors", review_days=ANNUAL, build="scripts/build_carbon_factors.py"),
    "licence_public.pem": dict(
        publisher="Our licence-verification public key", licence="n/a", vintage=None, status="code", use="Licence signature check", review_days=STATIC),
    "marine_fuels_fueleu.json": dict(
        publisher="European Commission, FuelEU Maritime guidance Table 4.2 (Annex II of Regulation (EU) 2023/1805)", licence="EU publication", vintage="2023 regulation",
        status="published", use="WtT, TtW and GWP defaults by marine fuel", review_days=ANNUAL,
        note="TBM / N/A cells were filled by the guidance with the highest default; listed in the file notes."),
    "ndgain_extended.json": dict(
        publisher="ND-GAIN 2022 country scores for 59 countries, transcribed/curated (see connector header)", licence="ND-GAIN open data (check)", vintage="2022",
        status="third_party", use="Country adaptation / vulnerability / readiness", review_days=ANNUAL,
        note="Header says 'derived from the CarbonLens dataset + curated ND-GAIN 2022 values'. Not re-checked against ND-GAIN's own download."),
    "ngfs_phase5.json": dict(
        publisher="NGFS Phase 5 (Nov 2024) via IIASA Scenario Explorer; GCAM 6.0, MESSAGEix-GLOBIOM 2.0-M-R, REMIND-MAgPIE 3.3", licence="NGFS / IIASA terms (check for commercial use)",
        vintage="2024-11", status="published", use="Transition scenario pathways 2020-2050", review_days=ANNUAL, build="scripts/build_finance_data.py --stage ngfs",
        note="Newer NGFS phases may exist; check before relying on this for a regulatory submission."),
    "observed_emissions_us.json": dict(
        publisher="US EPA GHGRP 2023 facility data, parent-attributed by ownership share", licence="public domain", vintage="2023", status="derived",
        use="Observed Scope 1 by company (US facilities only)", review_days=ANNUAL, build="scripts/export_emissions_runtime.py",
        note="US reporting facilities above the GHGRP threshold only; not a company total."),
    "permafrost_obu2019_10km.npz": dict(
        publisher="Obu et al. 2019 permafrost probability map, resampled to 10 km", licence="CC BY 4.0 (check)", vintage="2019", status="derived",
        use="Permafrost probability at a site", review_days=STATIC, note="Published product; not tested here."),
    "resident_nlu.json": dict(
        publisher="Our intent classifier for the resident agent (756 training examples, 21 classes)", licence="n/a", vintage="2026-10-07", status="fitted",
        use="Routing of free-text questions to tools", review_days=ANNUAL, note="Held-out accuracy stored in the file meta. Routing only; carries no climate data."),
    "slr_ar6.npz": dict(
        publisher="IPCC AR6 relative sea-level projections, FACTS framework (Fox-Kemper 2021; Garner 2021; Kopp 2023), Zenodo 10.5281/zenodo.6382554", licence="CC BY 4.0",
        vintage="2021-2023", status="derived", use="Sea-level rise by site, SSP, quantile, decade", review_days=STATIC,
        note="Build script is not in the repo; compaction step is undocumented."),
    "spc_reports_1996_2024.npz": dict(
        publisher="NOAA Storm Prediction Center hail and tornado reports 1996-2024", licence="public domain", vintage="1996-2024", status="derived",
        use="Severe convective storm frequency", review_days=ANNUAL, build="scripts/build_spc_reports.py",
        note="Reports reflect where people were to see them; hail under 1.0 in dropped."),
    "subpart_w_table_w1.json": dict(
        publisher="US EPA 40 CFR Part 98 Subpart W Table W-1 (89 FR 42323, May 2024)", licence="US Government work", vintage="2024-05", retrieved="2026-10-28",
        status="published", use="Oil and gas default population emission factors", review_days=ANNUAL,
        url="https://www.law.cornell.edu/cfr/text/40/appendix-Table_W-1_to_subpart_W_of_part_98", note="Whole-gas volumes; CH4 fraction is a required user input."),
    "tc_validation.json": dict(
        publisher="Our tropical-cyclone wind validation results", licence="n/a", vintage=None, status="derived", use="Reported skill of the cyclone wind model",
        review_days=ANNUAL, build="scripts/validate_tc_wind.py", note="Observations are hourly snapshots and miss short peaks (file notes)."),
}

# Python modules that hold data tables are tracked separately because they are code, not data files.
CODE_TABLES = ("companies_climrisk.py", "companies_seed.py", "parameters.py", "schemas.py", "profile_builder.py", "asset_register.py")


def _age_days(vintage: Optional[str], today: date) -> Optional[int]:
    if not vintage:
        return None
    s = str(vintage).strip()
    try:
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            d = datetime.strptime(s[:10], "%Y-%m-%d").date()
        elif len(s) >= 7 and s[4] == "-":
            d = datetime.strptime(s[:7] + "-01", "%Y-%m-%d").date()
        else:
            # a bare year or a range such as 2023-2026: use the end of the latest year mentioned
            years = [int(t) for t in "".join(c if c.isdigit() else " " for c in s).split() if len(t) == 4]
            if not years:
                return None
            d = date(max(years), 12, 31)
    except ValueError:
        return None
    return (today - d).days


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _embedded(path: Path) -> dict:
    """Metadata the file declares about itself (JSON only). Used to check the registry has not drifted from the file."""
    if path.suffix != ".json":
        return {}
    try:
        d = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    return {k: d[k] for k in ("source", "url", "retrieved", "version", "built", "year") if k in d and isinstance(d[k], (str, int))}


def _grade(e: dict) -> str:
    """complete: publisher, vintage and licence known and not flagged unchecked. partial: something missing. undocumented: no registry entry."""
    if not e:
        return "undocumented"
    ok = e.get("publisher") and e.get("vintage") and e.get("licence") and "NOT CHECKED" not in str(e.get("licence"))
    return "complete" if ok and e.get("status") != "third_party" else "partial"


def entry(name: str, today: Optional[date] = None) -> dict:
    today = today or date.today()
    path = DATA_DIR / name
    e = dict(REGISTRY.get(name, {}))
    age = _age_days(e.get("vintage"), today)
    review = e.get("review_days")
    if not e:
        stale = None
    elif age is None:
        stale = "unknown"
    else:
        stale = "review_due" if (review and age > review) else "ok"
    out = dict(file=name, exists=path.exists(), size_bytes=path.stat().st_size if path.exists() else None,
               sha256_16=_sha(path) if path.exists() else None, documentation=_grade(e), age_days=age, freshness=stale, **e)
    emb = _embedded(path)
    drift = []
    if emb.get("retrieved") and e.get("retrieved") and emb["retrieved"] != e["retrieved"]:
        drift.append(f"retrieved: registry {e['retrieved']} vs file {emb['retrieved']}")
    if emb.get("version") and e.get("vintage") and str(e["vintage"]) not in str(emb["version"]) and str(emb["version"]) not in str(e["vintage"]):
        drift.append(f"version: registry {e['vintage']} vs file {emb['version']}")
    if drift:
        out["drift"] = drift
    return out


def audit(today: Optional[date] = None) -> dict:
    """Full registry. ``unregistered`` lists any data file in cri/data without an entry (a failure)."""
    today = today or date.today()
    files = sorted(p.name for p in DATA_DIR.iterdir() if p.is_file() and p.suffix in (".json", ".npz", ".gz", ".csv", ".pem"))
    rows = [entry(n, today) for n in files]
    unreg = [r["file"] for r in rows if r["documentation"] == "undocumented"]
    stale_lst = [r["file"] for r in rows if r["freshness"] == "review_due"]
    unknown = [r["file"] for r in rows if r["freshness"] == "unknown"]
    third = [r["file"] for r in rows if r.get("status") == "third_party"]
    missing = sorted(set(REGISTRY) - set(files))
    return dict(as_of=today.isoformat(), n_files=len(rows),
                summary=dict(complete=sum(r["documentation"] == "complete" for r in rows), partial=sum(r["documentation"] == "partial" for r in rows),
                             undocumented=len(unreg), review_due=len(stale_lst), age_unknown=len(unknown), third_party=len(third)),
                unregistered=unreg, registry_entries_without_file=missing, review_due=stale_lst, age_unknown=unknown, third_party=third,
                files=rows,
                note="age_days is the age of the data vintage, not of the download. 'review_due' means older than the stated review interval for that kind of data, "
                     "not that the data is wrong. 'third_party' files are someone else's model output and are indicators, not measurements.")


def stamp(*names: str, today: Optional[date] = None) -> list[dict]:
    """Short source stamps to attach to an answer: [{file, publisher, vintage, age_days, freshness, status}]."""
    out = []
    for n in names:
        e = entry(n, today)
        out.append({k: e.get(k) for k in ("file", "publisher", "vintage", "retrieved", "age_days", "freshness", "status", "licence", "newer_edition", "note") if e.get(k) is not None})
    return out
