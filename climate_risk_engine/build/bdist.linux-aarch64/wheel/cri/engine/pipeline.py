"""
CRI Full Pipeline — client data → open-source enrichment → financial results.

This is the main entry point for the production engine. It wires:
  1. Client data intake   (Excel/CSV → Company objects)
  2. Open-source enrichment  (WRI Aqueduct, NGFS, NASA, OWID)
  3. Hazard matrix           (asset-level physical risk)
  4. Scenario resolution     (NGFS → CRI Scenario)
  5. Engine run              (operations → financial → DCF)

Usage (Python):
    from cri.engine.pipeline import Pipeline
    pipeline = Pipeline()
    results = pipeline.run_file("client_data.xlsx", scenario_names=["Net Zero 2050"])

Usage (CLI):
    cri run-file client_data.xlsx --scenario "Net Zero 2050"
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..connectors.ngfs import NGFSConnector
from ..connectors.owid import OWIDConnector
from ..connectors.wri_aqueduct import WRIAqueductConnector
from ..climate.hazard_matrix import HazardMatrix
from ..data.schemas import (
    Asset, Company, HazardPath, HazardType, RunResults, Scenario,
)
from ..intake.parser import parse_excel
from ..intake.validate import validate_company
from .orchestrator import run as engine_run


# ── Scenario definitions (NGFS-aligned) ────────────────────────────────────

def _build_scenario_from_ngfs(ngfs_name: str) -> Scenario:
    """Build a CRI Scenario from an NGFS scenario name using live connector data."""
    from ..scenarios import NZE_2050, DELAYED_TRANSITION, CURRENT_POLICIES

    mapping = {
        "Net Zero 2050":       NZE_2050,
        "nze_2050":            NZE_2050,
        "Delayed Transition":  DELAYED_TRANSITION,
        "delayed_transition":  DELAYED_TRANSITION,
        "Current Policies":    CURRENT_POLICIES,
        "current_policies":    CURRENT_POLICIES,
    }
    key = ngfs_name.strip()
    if key in mapping:
        return mapping[key]
    raise ValueError(
        f"Unknown scenario '{ngfs_name}'. "
        f"Valid options: {list(mapping.keys())}"
    )


# ── Enrichment ──────────────────────────────────────────────────────────────

def _enrich_asset_hazards(
    asset: Asset,
    scenario_family: str,
    horizon: list[int],
    wri: WRIAqueductConnector,
    hm: HazardMatrix,
) -> Asset:
    """
    Enrich an asset with real hazard data.

    Runs HazardMatrix.assess() which delegates to:
      - PhysicalHazardEngine (25 hazards, SSP-scaled)
      - SpatialDownscaler (CMIP6 at asset lat/lon, live met, WRI point query)

    The asset object itself is returned unchanged — enrichment is captured
    in the HazardMatrix cache and applied when _build_enriched_hazard_paths()
    is called. This two-phase design lets the pipeline pre-warm the cache
    without duplicating the assessment.
    """
    # Pre-warm the downscaler cache for this asset's coordinates + scenario.
    # This is a no-op if lat/lon is None (falls through to region tables).
    try:
        hm.assess(asset, scenario_family, horizon)
    except Exception:
        pass   # non-fatal — pipeline continues with region-level data
    return asset


def _build_enriched_hazard_paths(
    asset: Asset,
    scenario_family: str,
    hm: HazardMatrix,
) -> list[HazardPath]:
    """
    Build HazardPath objects for an asset using the full enrichment stack.

    Data sources (in priority order):
      1. CMIP6 downscaling via Open-Meteo (when asset.lat/lon available)
      2. PhysicalHazardEngine 25-hazard model (always)
      3. WRI Aqueduct region tables (fallback)

    Only HazardTypes defined in the CRI schema are included; novel hazard
    names from the 25-hazard engine that don't map to a schema HazardType
    are silently dropped (forward-compatible).
    """
    years = list(range(2026, 2051))
    profile = hm.assess(asset, scenario_family, years)

    paths: list[HazardPath] = []
    for htype_str, probs_by_year in profile.hazard_probs.items():
        try:
            htype = HazardType(htype_str)
        except ValueError:
            continue
        paths.append(
            HazardPath(
                hazard=htype,
                region=asset.region,
                path={yr: probs_by_year.get(yr, 0.0) for yr in years},
            )
        )
    return paths


def _enrich_scenario_with_real_hazards(
    scenario: Scenario,
    company: Company,
    hm: HazardMatrix,
) -> Scenario:
    """
    Clone the scenario and replace its hazard paths with asset-level real data
    from WRI Aqueduct + NASA NEX-GDDP proxies.
    """
    new_hazards: list[HazardPath] = []

    for asset in company.assets:
        asset_paths = _build_enriched_hazard_paths(asset, scenario.family, hm)
        new_hazards.extend(asset_paths)

    # Also keep existing paths for any regions not covered by assets
    existing_regions = {p.region for p in new_hazards}
    for existing in scenario.hazards:
        if existing.region not in existing_regions:
            new_hazards.append(existing)

    return scenario.model_copy(update={"hazards": new_hazards})


# ── Pipeline result ─────────────────────────────────────────────────────────

@dataclass
class PipelineRunResult:
    company_id: str
    company_name: str
    scenario_id: str
    scenario_name: str
    results: RunResults
    enrichment_sources: dict[str, str]   # hazard_type → data source
    warnings: list[str]
    duration_s: float
    sector: str = "default"
    revenue_usd: float = 0.0
    # Real hazard profiles from CMIP6 + WRI spatial downscaling
    # Key: asset_id, Value: AssetHazardProfile (hazard_probs + sources)
    asset_hazard_profiles: dict = field(default_factory=dict)


@dataclass
class PipelineReport:
    """Full report for one client file across all scenarios."""
    source_file: str
    companies_processed: int
    scenarios_run: list[str]
    runs: list[PipelineRunResult]
    total_duration_s: float
    errors: list[str] = field(default_factory=list)

    def summary_table(self) -> list[dict[str, Any]]:
        """Return a flat list of dicts suitable for display or CSV export.

        Physical-risk NPV impact is derived from real CMIP6 + WRI Aqueduct
        hazard probabilities captured during enrichment (asset_hazard_profiles).
        Falls back to IPCC AR6 GMST scaling when no per-asset data is available.

        Both physical and combined (transition + physical) NPV impacts are
        returned for TCFD double-materiality disclosure.
        """
        from ..climate.physical_risk_financial import (
            compute_physical_npv_impact,
            combined_npv_impact_pct,
            _sector_key,
            SECTOR_EXPOSURE_RATIO,
        )
        from ..climate.ssp_scenarios import NGFS_TO_SSP, SSP_SCENARIOS

        _SC_TO_FAMILY: dict[str, str] = {
            "Net Zero 2050":       "nze_2050",
            "nze_2050":            "nze_2050",
            "Delayed Transition":  "delayed_transition",
            "delayed_transition":  "delayed_transition",
            "Current Policies":    "current_policies",
            "current_policies":    "current_policies",
            "Hot House World":     "hot_house",
            "hot_house":           "hot_house",
        }

        rows = []
        for r in self.runs:
            res      = r.results
            ev_usd   = res.enterprise_value * 1e6   # RunResults EV is in $M
            wacc     = res.wacc_used
            rev_usd  = r.revenue_usd
            ngfs_fam = _SC_TO_FAMILY.get(r.scenario_name, "delayed_transition")
            ssp_id   = NGFS_TO_SSP.get(ngfs_fam, "ssp245")
            ssp      = SSP_SCENARIOS[ssp_id]
            sec_key  = _sector_key(r.sector)
            phi      = SECTOR_EXPOSURE_RATIO.get(sec_key, 0.22)

            hazard_scores_2038: dict[str, float] = {}
            data_sources: list[str] = []

            if r.asset_hazard_profiles:
                # ── Primary path: real CMIP6 + WRI Aqueduct data ──────────
                # Aggregate ELF across assets and years using real hazard probs
                years = list(range(2026, 2051))
                yearly_total_drag = {yr: 0.0 for yr in years}
                yearly_elf: dict[int, float] = {}
                hazard_acc: dict[str, list[float]] = {}  # hazard → [prob per year]
                n_assets = len(r.asset_hazard_profiles)

                for asset_id, profile in r.asset_hazard_profiles.items():
                    data_sources.extend(profile.sources.values())
                    # Per year, compute joint ELF for this asset
                    asset_rev = rev_usd / max(n_assets, 1)
                    for yr in years:
                        probs_yr: dict[str, float] = {}
                        for haz, yr_probs in profile.hazard_probs.items():
                            p = yr_probs.get(yr, 0.0)
                            probs_yr[haz] = p
                            if yr == 2038:
                                hazard_acc.setdefault(haz, []).append(p)
                        # Joint survival across hazards for this asset/year
                        survival = 1.0
                        for p in probs_yr.values():
                            survival *= (1.0 - p)
                        elf = min(0.80, 1.0 - survival)
                        revenue_at_risk = asset_rev * phi * elf
                        discount = (1.0 + wacc) ** (yr - 2026)
                        yearly_total_drag[yr] += revenue_at_risk / discount

                total_drag   = sum(yearly_total_drag.values())
                ph_pct_raw   = -(total_drag / max(ev_usd, 1.0))

                # 2038 hazard scores = average across assets
                hazard_scores_2038 = {
                    h: round(sum(vs) / len(vs) * 100, 1)
                    for h, vs in hazard_acc.items()
                    if vs
                }
                data_source_str = "CMIP6 (Open-Meteo) + WRI Aqueduct + PhysicalHazardEngine"

            else:
                # ── Fallback: IPCC AR6 GMST scaling ────────────────────────
                phys = compute_physical_npv_impact(
                    ngfs_family=ngfs_fam,
                    sector=r.sector,
                    revenue_usd=rev_usd,
                    ev_base_usd=max(ev_usd, 1.0),
                    wacc=wacc,
                )
                ph_pct_raw       = phys.physical_npv_impact_pct
                hazard_scores_2038 = {k: round(v * 100, 1) for k, v in phys.hazard_probs_2038.items()}
                data_source_str  = "IPCC AR6 WG1/WG2 (GMST scaling fallback)"

            tr_pct   = res.npv_impact_pct or 0.0
            ph_pct   = ph_pct_raw
            comb_pct = combined_npv_impact_pct(tr_pct, ph_pct)
            # Unique sources, max 5 for readability
            unique_sources = list(dict.fromkeys(data_sources))[:5]

            rows.append({
                "company":                    r.company_name,
                "scenario":                   r.scenario_name,
                "ev_bn":                      round(res.enterprise_value / 1e3, 1),
                "equity_bn":                  round(res.equity_value / 1e3, 1),
                "share_price":                round(res.implied_share_price, 2),
                "wacc_pct":                   round(wacc * 100, 2),
                "npv_impact_pct":             round(tr_pct * 100, 1),
                "physical_npv_impact_pct":    round(ph_pct * 100, 1),
                "combined_npv_impact_pct":    round(comb_pct * 100, 1),
                "hazard_scores":              hazard_scores_2038,
                "ssp_id":                     ssp_id,
                "gmst_2050":                  ssp.gmst_2050,
                "data_sources":               unique_sources or [data_source_str],
                "sector":                     sec_key,
                "physical_exposure_ratio":    round(phi, 2),
                "warnings":                   len(r.warnings),
            })
        return rows

    def to_json(self) -> str:
        rows = self.summary_table()
        return json.dumps(rows, indent=2)


# ── Pipeline class ──────────────────────────────────────────────────────────

class Pipeline:
    """
    Main CRI pipeline. Accepts client data (Excel/Company objects) and runs
    the full climate-financial analysis with open-source data enrichment.
    """

    DEFAULT_SCENARIOS = ["Net Zero 2050", "Delayed Transition", "Current Policies"]

    def __init__(self, use_live_wri_api: bool = False):
        self.wri   = WRIAqueductConnector()
        self.ngfs  = NGFSConnector()
        self.owid  = OWIDConnector()
        self.hm    = HazardMatrix()
        self.use_live_wri = use_live_wri_api

    # ── Public API ─────────────────────────────────────────────────────────

    def run_file(
        self,
        path: str | Path,
        scenario_names: list[str] | None = None,
    ) -> PipelineReport:
        """
        Parse a client Excel file and run the full pipeline.

        Args:
            path: Path to client intake Excel (.xlsx)
            scenario_names: List of NGFS scenario names to run.
                            Defaults to all three canonical scenarios.

        Returns:
            PipelineReport with full results and metadata.
        """
        t0 = time.time()
        path = Path(path)
        scenario_names = scenario_names or self.DEFAULT_SCENARIOS
        errors: list[str] = []

        # 1. Parse client data
        companies = parse_excel(path)
        if not companies:
            return PipelineReport(
                source_file=str(path),
                companies_processed=0,
                scenarios_run=scenario_names,
                runs=[],
                total_duration_s=time.time() - t0,
                errors=["No companies found in intake file."],
            )

        # 2. Run all companies across all scenarios
        runs: list[PipelineRunResult] = []
        for company in companies:
            warnings = validate_company(company)
            for sc_name in scenario_names:
                try:
                    result = self.run_company(
                        company, sc_name, warnings=warnings
                    )
                    runs.append(result)
                except Exception as e:
                    errors.append(
                        f"{company.name} / {sc_name}: {type(e).__name__}: {e}"
                    )

        # 3. Compute baseline NPV (Current Policies) for % impact
        runs = self._attach_baseline_impacts(runs)

        return PipelineReport(
            source_file=str(path),
            companies_processed=len(companies),
            scenarios_run=scenario_names,
            runs=runs,
            total_duration_s=time.time() - t0,
            errors=errors,
        )

    def run_company(
        self,
        company: Company,
        scenario_name: str,
        warnings: list[str] | None = None,
    ) -> PipelineRunResult:
        """
        Run one company under one scenario with full open-source enrichment.
        """
        t0 = time.time()
        warnings = warnings or []

        # 1. Resolve scenario
        scenario = _build_scenario_from_ngfs(scenario_name)

        # 2. Enrich scenario with real hazard data for this company's assets
        enriched_scenario = _enrich_scenario_with_real_hazards(
            scenario, company, self.hm
        )

        # 3. Run the engine
        results = engine_run(company, enriched_scenario)

        # 4. Collect enrichment source attribution + real hazard profiles
        # Run hm.assess() over the full projection horizon so we get CMIP6-downscaled
        # per-year hazard probabilities for every asset. These feed directly into
        # the physical NPV calculation in summary_table().
        sources: dict[str, str] = {}
        asset_profiles: dict = {}
        for asset in company.assets:
            years = list(range(2026, 2051))
            profile = self.hm.assess(asset, scenario.family, years)
            sources.update(profile.sources)
            asset_profiles[asset.id] = profile

        # Revenue in USD: sum revenue_baseline across all assets
        revenue_usd = sum(
            getattr(a, "revenue_baseline", 0.0) or 0.0
            for a in company.assets
        )

        return PipelineRunResult(
            company_id=company.id,
            company_name=company.name,
            scenario_id=scenario.id,
            scenario_name=scenario_name,
            results=results,
            enrichment_sources=sources,
            warnings=warnings,
            duration_s=time.time() - t0,
            sector=getattr(company, "sector", "default"),
            revenue_usd=revenue_usd,
            asset_hazard_profiles=asset_profiles,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _attach_baseline_impacts(
        self, runs: list[PipelineRunResult]
    ) -> list[PipelineRunResult]:
        """Attach npv_impact_pct relative to Current Policies baseline."""
        # Group by company
        by_company: dict[str, dict[str, PipelineRunResult]] = {}
        for r in runs:
            by_company.setdefault(r.company_id, {})[r.scenario_name] = r

        updated: list[PipelineRunResult] = []
        for _, co_runs in by_company.items():
            baseline = co_runs.get("Current Policies")
            baseline_ev = baseline.results.enterprise_value if baseline else None

            for r in co_runs.values():
                if baseline_ev and baseline_ev != 0:
                    npv_pct = r.results.enterprise_value / baseline_ev - 1.0
                    r.results = r.results.model_copy(
                        update={"npv_impact_pct": npv_pct}
                    )
                updated.append(r)

        return updated
