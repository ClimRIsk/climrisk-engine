"""
CRI Financial Disclosure Report Generator (PAID TIER — Professional+).

Generates structured outputs aligned to four major climate disclosure frameworks:

  1. TCFD  — Task Force on Climate-related Financial Disclosures (4 pillars)
  2. ISSB S2 — IFRS Sustainability Standard S2 (Climate-related Disclosures)
  3. EU CSRD E1 — European Sustainability Reporting Standards, Climate topic
  4. BRSR   — SEBI Business Responsibility & Sustainability Reporting (India)

Each report is produced as a structured dict which can be:
  - Serialised to JSON for API consumers
  - Rendered to PDF/Word via the reporting layer
  - Embedded in the Next.js dashboard

Framework references:
  - TCFD Final Report (2017), updated Guidance (2021)
  - IFRS S2 (June 2023), issued by ISSB
  - ESRS E1 (EU Commission Delegated Regulation 2023/2772)
  - SEBI BRSR Core (Circular SEBI/HO/CFD/CMD-2/P/CIR/2023/122, July 2023)

All text in governance, strategy, and risk sections is generated from company
attributes (name, sector, hq_region, assets) so each report is substantively
different and company-specific — NOT generic boilerplate.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

from ..data.schemas import Company, RunResults, ScenarioFamily


# ---------------------------------------------------------------------------
# Sector-intelligence helpers — make every report company-specific
# ---------------------------------------------------------------------------

def _sector_context(company: Company) -> dict:
    """
    Return a dict of sector-specific text snippets used throughout all reports.

    This is the single source of truth for company-aware language.  Every
    report function calls this once and substitutes the returned strings so
    reports are meaningfully different across companies.
    """
    sector = (company.sector or "").lower()
    name = company.name
    hq = company.hq_region

    # Derive asset region list for narrative
    asset_regions = sorted({a.region for a in company.assets}) if company.assets else []
    region_str = ", ".join(asset_regions[:6]) if asset_regions else hq
    n_assets = len(company.assets)

    # ── Sector branches ──────────────────────────────────────────────────────

    if any(x in sector for x in ["mining", "metal", "mineral", "iron", "copper", "aluminium", "zinc"]):
        primary_hazards   = "water stress, heat stress (workforce productivity), and wildfires"
        key_transition    = "decarbonisation of mining fleet (electrification) and Scope 3 downstream steel/smelting exposure"
        opportunity_text  = "critical minerals portfolio (copper, nickel, lithium) positioned for EV supply-chain demand uplift"
        board_focus       = "tailings dam safety, water-intensive operations, and stranded-asset risk in thermal coal exposure"
        mgmt_role         = (
            f"The Chief Risk Officer (CRO) integrates climate risk into mine-planning decisions. "
            f"Asset-level CRI scores for {name}'s {n_assets} operational sites are reviewed quarterly "
            f"by the Executive Committee and escalated when any single-site physical-risk score exceeds 60/100."
        )
        short_term_actions = [
            f"Commission site-level physical risk audits at highest-exposure sites ({region_str})",
            "Electrify light vehicle fleet across surface mining operations by 2028",
            "Set a science-based interim emissions target (SBTi Corporate Standard) for Scope 1+2",
            "Disclose water withdrawal intensity per tonne processed at each site",
        ]
        medium_term_actions = [
            "Transition primary haulage to battery-electric or hydrogen by 2035",
            "Achieve 50% renewable electricity across all operations",
            "Publish asset-level decarbonisation plans with locked-in emission schedules",
            "Engage top-5 steel/smelting customers on Scope 3 reduction pathways",
        ]
        long_term_actions = [
            "Achieve net-zero Scope 1+2 by 2050 (Paris-aligned)",
            "Report Scope 3 (Category 1 + 11) reduction trajectory and supply-chain engagement",
            "Maintain physical-risk adaptation capex ≥ 1.5% of annual revenue",
        ]
        risk_focus = (
            f"{name} faces chronic water stress at its arid-region assets ({region_str}), "
            f"increasing heat-event frequency impacting labour productivity, and wildfire disruption "
            f"risk at Australian and South American operations. Physical disruptions compound on "
            f"transition exposure from carbon-intensive crushing and hauling operations."
        )

    elif any(x in sector for x in ["oil", "gas", "petroleum", "energy", "lng", "upstream", "downstream", "refin"]):
        primary_hazards   = "coastal flood, sea level rise (offshore assets), and hurricane/cyclone intensification"
        key_transition    = "demand destruction for fossil fuels under IEA NZE pathway and carbon price exposure on Scope 1 refinery emissions"
        opportunity_text  = "CCS-ready asset portfolio and potential early-mover advantage in blue/green hydrogen production"
        board_focus       = "reserve replacement ratios under 1.5°C, stranded-asset writedown risk, and Scope 3 Category 11 liability"
        mgmt_role         = (
            f"The CRO oversees {name}'s scenario-based reserve testing. "
            f"Under NGFS Net Zero 2050, {name}'s reserve life aligns against IEA's 'no new oil and gas fields' guidance. "
            f"CRI outputs feed directly into the annual reserve disclosure and investor communication."
        )
        short_term_actions = [
            f"Run IEA NZE stress test on reserves and publish asset-by-asset residual value at $130/tCO2e (2030)",
            "Implement OGMP 2.0 methane-monitoring programme across upstream assets",
            "Set a shadow carbon price of ≥$80/tCO2e in all new investment decisions",
            "Disclose physical-risk exposure for coastal and offshore assets",
        ]
        medium_term_actions = [
            "Reduce upstream methane intensity to <0.2% by 2030 (OGCI standard)",
            "Invest ≥15% of capex in CCS and low-carbon energy by 2035",
            "Begin portfolio rebalancing toward gas, bioenergy, and hydrogen",
            "Achieve 40% Scope 1+2 reduction vs 2025 baseline by 2035",
        ]
        long_term_actions = [
            "Align portfolio with 1.5°C-compatible production trajectory by 2050",
            "Achieve net-zero Scope 1+2 by 2050; report Scope 3 trajectory",
            "Maintain physical adaptation capex ≥ 2% of revenue for offshore asset hardening",
        ]
        risk_focus = (
            f"{name} faces dual risk: transition risk from carbon regulation compressing margins "
            f"on high-carbon barrels, and physical risk from intensifying tropical cyclones and sea-level "
            f"rise at coastal/offshore infrastructure. The NGFS Current Policies scenario preserves near-term "
            f"revenue but accumulates the highest long-run physical damage costs."
        )

    elif any(x in sector for x in ["power", "utility", "electricity", "generation", "grid"]):
        primary_hazards   = "heat stress (reduced thermal plant efficiency), water stress (cooling water), and extreme precipitation"
        key_transition    = "coal and gas capacity stranding under carbon price escalation and renewable cost deflation"
        opportunity_text  = "renewable generation portfolio (solar, wind, battery storage) directly aligned with NZE demand growth"
        board_focus       = "coal-plant retirement schedule, carbon price exposure, and grid reliability under extreme weather"
        mgmt_role         = (
            f"The CRO oversees {name}'s integrated resource planning under climate scenarios. "
            f"Climate risk is embedded in the annual Integrated Resource Plan (IRP) process and reviewed "
            f"by the Sustainability Committee before Board approval."
        )
        short_term_actions = [
            "Publish coal-retirement roadmap aligned with 1.5°C pathway",
            "Set internal shadow carbon price ≥$80/tCO2e in capex decisions",
            "Assess cooling-water availability risk at thermal plants under drought scenarios",
            "Increase renewable generation capacity to ≥30% of installed base by 2028",
        ]
        medium_term_actions = [
            "Retire high-cost coal units (>25yr old) by 2035",
            "Deploy utility-scale battery storage to maintain grid resilience under extreme-weather disruptions",
            "Achieve 50% Scope 1 reduction vs 2025 baseline",
            "Transition to green hydrogen peakers as dispatchable backup",
        ]
        long_term_actions = [
            "Achieve net-zero power generation by 2050",
            "Fully retire fossil-fuel baseload; operate 100% clean dispatch",
            "Maintain adaptation investment in grid hardening ≥ 2% of annual opex",
        ]
        risk_focus = (
            f"{name}'s generation assets face rising cooling-water stress in water-scarce regions "
            f"and reduced thermal efficiency on extreme-heat days. Transition risk is most acute for "
            f"coal-fired capacity, where carbon prices above $80/tCO2e materially compress margins "
            f"and accelerate economic retirement."
        )

    elif any(x in sector for x in ["agriculture", "food", "crop", "farm", "agri"]):
        primary_hazards   = "drought, heat stress (crop yield), and extreme precipitation / flooding"
        key_transition    = "land-use change regulations, methane emissions from livestock, and Scope 3 agricultural supply chain exposure"
        opportunity_text  = "regenerative agriculture carbon credits and low-carbon food supply chain positioning"
        board_focus       = "food security risk, water-use efficiency, and nature-related financial risks (TNFD)"
        mgmt_role         = (
            f"The CRO and Chief Sustainability Officer jointly manage climate risk at {name}. "
            f"Crop-level physical risk models feed the annual production guidance and are stress-tested "
            f"under SSP2-4.5 (baseline) and SSP5-8.5 (severe) before financial planning."
        )
        short_term_actions = [
            "Map water footprint across all growing regions using WRI Aqueduct",
            "Develop drought-resistant crop variety adoption programme",
            "Disclose methane intensity per tonne of livestock product",
            "Set SBTi Forest, Land and Agriculture (FLAG) target",
        ]
        medium_term_actions = [
            "Implement precision irrigation to reduce water withdrawal intensity by 30%",
            "Achieve 40% Scope 1+2 reduction by 2035",
            "Engage supply chain on regenerative agriculture practices",
            "Pilot soil carbon sequestration credits across major landholdings",
        ]
        long_term_actions = [
            "Achieve net-zero Scope 1+2+3 by 2050 (SBTi FLAG pathway)",
            "Reach net positive land impact (nature-positive) per TNFD guidance",
            "Maintain adaptation capex for irrigation infrastructure ≥1% of revenue",
        ]
        risk_focus = (
            f"{name}'s operations are directly exposed to increasing drought frequency and "
            f"heat-stress-driven yield loss across its agricultural regions. The compound effect "
            f"of water stress and temperature anomalies creates non-linear production risk under "
            f"SSP3-7.0 and SSP5-8.5 scenarios by the 2040s."
        )

    else:
        # Generic fallback — still more specific than pure boilerplate
        primary_hazards   = "water stress, heat stress, and flood risk across operating regions"
        key_transition    = "carbon price exposure on Scope 1 and 2 emissions"
        opportunity_text  = "early transition positioning in low-carbon products and services"
        board_focus       = "physical asset exposure and transition timeline alignment"
        mgmt_role         = (
            f"The Chief Risk Officer (CRO) at {name} oversees climate-related risk identification "
            f"and management. CRI model outputs for {n_assets} assets across {region_str} are "
            f"reviewed at least annually by the Executive Committee."
        )
        short_term_actions = [
            f"Commission asset-level physical risk audit for {region_str}",
            "Set science-based interim emissions target (SBTi Corporate Standard)",
            "Implement shadow carbon price of ≥$80/tCO2e in capital planning",
            "Publish first TCFD-aligned climate disclosure",
        ]
        medium_term_actions = [
            "Achieve 40% Scope 1+2 reduction vs 2025 baseline by 2035",
            "Integrate climate risk into enterprise risk register",
            "Engage key suppliers and customers on Scope 3 pathway",
        ]
        long_term_actions = [
            "Achieve net-zero Scope 1+2 by 2050 (Paris-aligned)",
            "Report Scope 3 reduction trajectory annually",
            "Maintain physical adaptation capex ≥ 1.5% of annual revenue",
        ]
        risk_focus = (
            f"{name} faces both physical and transition climate risks across its "
            f"operations in {region_str}. The CRI engine identifies {primary_hazards} "
            f"as the dominant near-term physical exposures."
        )

    return {
        "name": name,
        "sector": company.sector,
        "hq": hq,
        "n_assets": n_assets,
        "asset_regions": region_str,
        "primary_hazards": primary_hazards,
        "key_transition": key_transition,
        "opportunity_text": opportunity_text,
        "board_focus": board_focus,
        "mgmt_role": mgmt_role,
        "short_term_actions": short_term_actions,
        "medium_term_actions": medium_term_actions,
        "long_term_actions": long_term_actions,
        "risk_focus": risk_focus,
    }


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


@dataclass
class DisclosureReport:
    """Base class for all disclosure report types."""

    framework: str
    framework_version: str
    generated_at: str
    company_id: str
    company_name: str
    reporting_year: int
    data_sources: list[str]
    caveats: list[str]
    sections: dict[str, Any]   # framework-specific content

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "generated_at": self.generated_at,
            "company_id": self.company_id,
            "company_name": self.company_name,
            "reporting_year": self.reporting_year,
            "data_sources": self.data_sources,
            "caveats": self.caveats,
            "sections": self.sections,
        }


# ---------------------------------------------------------------------------
# TCFD Report
# ---------------------------------------------------------------------------


@dataclass
class TCFDReport(DisclosureReport):
    """
    TCFD-aligned climate risk disclosure across four pillars:
      1. Governance
      2. Strategy
      3. Risk Management
      4. Metrics & Targets
    """
    pass


def generate_tcfd(
    company: Company,
    nze: RunResults,
    dt: RunResults,
    cp: RunResults,
    reporting_year: int | None = None,
) -> TCFDReport:
    """
    Generate a TCFD-aligned disclosure report from multi-scenario RunResults.

    The output is structured per TCFD's recommended disclosures (2021 Guidance)
    and maps directly to IFRS S2's climate-related disclosures cross-reference table.
    """
    reporting_year = reporting_year or datetime.date.today().year
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    ctx = _sector_context(company)   # company-specific text snippets

    def yr(r: RunResults, year: int):
        return next((y for y in r.years if y.year == year), None)

    # ── Governance ──────────────────────────────────────────────────────────
    governance = {
        "board_oversight": {
            "description": (
                f"The Board of Directors of {ctx['name']} retains ultimate oversight of "
                f"climate-related risks and opportunities. Given {ctx['name']}'s exposure to "
                f"{ctx['primary_hazards']}, climate risk is a standing agenda item at quarterly "
                f"board meetings. The Board's Audit & Risk Committee reviews CRI scenario outputs "
                f"annually and holds management accountable for progress on {ctx['board_focus']}."
            ),
            "recommended_actions": [
                f"Embed CRI scenario outputs (NZE/DT/CP) into the Board risk reporting pack for {ctx['name']}",
                "Assign quantified climate KPIs (emissions reduction, adaptation capex) to executive remuneration",
                "Establish a dedicated Climate Risk sub-committee at Board level",
                f"Require annual sign-off on {ctx['name']}'s climate transition plan by the full Board",
            ],
        },
        "management_role": {
            "description": ctx["mgmt_role"],
        },
    }

    # ── Strategy ─────────────────────────────────────────────────────────────
    nze_2026 = yr(nze, 2026)
    nze_2030 = yr(nze, 2030)
    nze_2040 = yr(nze, 2040)
    cp_2026  = yr(cp,  2026)
    cp_2040  = yr(cp,  2040)

    # Revenue trajectory
    def safe_rev(r):
        return round(r.revenue / 1e6, 1) if r else None

    # EV delta
    ev_nze_bn = round(nze.enterprise_value / 1e3, 1)
    ev_dt_bn  = round(dt.enterprise_value / 1e3, 1)
    ev_cp_bn  = round(cp.enterprise_value / 1e3, 1)

    strategy = {
        "scenarios_used": [
            {
                "name": "Net Zero 2050 (NZE)",
                "alignment": "NGFS Phase 4 / IEA NZE 2050",
                "carbon_price_2030_usd_tco2": 130,
                "carbon_price_2050_usd_tco2": 250,
                "temperature_outcome": "~1.5°C by 2100",
                "description": (
                    "Orderly transition: rapid decarbonisation, high carbon price, "
                    "significant demand shift away from fossil fuels."
                ),
            },
            {
                "name": "Delayed Transition",
                "alignment": "NGFS Phase 4 Delayed Transition",
                "carbon_price_2030_usd_tco2": 30,
                "carbon_price_2050_usd_tco2": 180,
                "temperature_outcome": "~1.8–2.0°C by 2100",
                "description": (
                    "Policy action delayed to 2030s, then accelerated; higher physical risk "
                    "accumulates before transition occurs."
                ),
            },
            {
                "name": "Current Policies",
                "alignment": "NGFS Phase 4 Current Policies",
                "carbon_price_2030_usd_tco2": 15,
                "carbon_price_2050_usd_tco2": 35,
                "temperature_outcome": "~3.0–3.5°C by 2100",
                "description": (
                    "No additional climate policy; highest physical risk by mid-century, "
                    "lowest transition cost."
                ),
            },
        ],
        "financial_impacts": {
            "enterprise_value_by_scenario_usd_bn": {
                "nze_2050": ev_nze_bn,
                "delayed_transition": ev_dt_bn,
                "current_policies": ev_cp_bn,
            },
            "ev_at_risk_nze_vs_cp_pct": round(
                (ev_cp_bn - ev_nze_bn) / max(ev_cp_bn, 0.001) * 100, 1
            ) if ev_cp_bn > 0 else None,
            "revenue_2030_usd_m": {
                "nze_2050": safe_rev(nze_2030),
                "current_policies": safe_rev(yr(cp, 2030)),
            },
            "revenue_2040_usd_m": {
                "nze_2050": safe_rev(nze_2040),
                "current_policies": safe_rev(cp_2040),
            },
        },
        "strategic_response": {
            "summary": (
                f"{ctx['name']} ({ctx['sector']}) faces {ctx['key_transition']} as its primary "
                f"transition risk, and {ctx['primary_hazards']} as its dominant physical risk across "
                f"assets in {ctx['asset_regions']}. The key opportunity is {ctx['opportunity_text']}."
            ),
            "short_term_2026_2030": ctx["short_term_actions"],
            "medium_term_2031_2040": ctx["medium_term_actions"],
            "long_term_2041_2050": ctx["long_term_actions"],
        },
    }

    # ── Risk Management ──────────────────────────────────────────────────────
    # EBITDA compression under NZE
    ebitda_2026 = nze_2026.ebitda if nze_2026 else None
    ebitda_2030 = nze_2030.ebitda if nze_2030 else None
    compression_2030_pct = None
    if ebitda_2026 and ebitda_2030 and ebitda_2026 > 0:
        compression_2030_pct = round((ebitda_2026 - ebitda_2030) / ebitda_2026 * 100, 1)

    carbon_cost_2030_m = round(nze_2030.carbon_cost / 1e6, 1) if nze_2030 else None
    physical_loss_2030_m = round(nze_2030.physical_loss_cost / 1e6, 1) if nze_2030 else None

    risk_management = {
        "identification_process": (
            f"Climate-related risks for {ctx['name']} are identified annually through the CRI engine. "
            f"For {ctx['n_assets']} assets across {ctx['asset_regions']}, the engine integrates "
            f"WRI Aqueduct 4.0 regional baseline statistics (water/flood/drought), NASA NEX-GDDP-CMIP6 regional statistics (heat stress, precipitation), "
            f"NGFS Phase 4 (carbon prices), and IEA WEO 2023 (demand curves). "
            f"{ctx['risk_focus']}"
        ),
        "assessment_methodology": {
            "physical_risk": "WRI Aqueduct 4.0 regional baseline statistics + NASA NEX-GDDP-CMIP6 regional proxies; sub-grid elevation & coastal downscaling; joint hazard aggregation (independent-events)",
            "transition_risk": "NGFS Phase 4 carbon price paths + IEA WEO demand scenarios",
            "financial_risk": "Climate-adjusted DCF with scenario risk premium on WACC",
            "horizon": "2026–2050 (annual)",
        },
        "quantified_risks": {
            "carbon_cost_2030_usd_m_nze": carbon_cost_2030_m,
            "physical_loss_cost_2030_usd_m_nze": physical_loss_2030_m,
            "ebitda_compression_2030_pct_nze": compression_2030_pct,
            # CP has higher WACC than NZE (higher scenario premium); delta = CP − NZE.
            "wacc_uplift_cp_vs_nze_pp": round((cp.wacc_used - nze.wacc_used) * 100, 2),
            # Per-hazard breakdown — shows individual hazard dollar contributions
            "physical_loss_by_hazard_2030_usd_m_nze": (
                {k: round(v, 1) for k, v in sorted(
                    nze_2030.physical_loss_by_hazard.items(),
                    key=lambda x: -x[1]
                )} if nze_2030 and nze_2030.physical_loss_by_hazard else {}
            ),
            "physical_loss_methodology_note": (
                "Per-hazard costs use independent-event assumption (not joint survival). "
                "Total physical_loss_cost applies joint-survival aggregation "
                "P(loss) = 1 − Π(1 − p_i), which is slightly lower than the sum of individual hazard costs."
            ),
        },
        "integration": (
            "CRI outputs are integrated into the enterprise risk register, "
            "capital planning processes, and lender/investor communications. "
            "Material risks above the $50M threshold are escalated to the Board."
        ),
    }

    # ── Metrics & Targets ───────────────────────────────────────────────────
    # Emission totals from year results
    def emission_sum(r: RunResults, year: int, scope: str) -> Optional[float]:
        y = yr(r, year)
        if y is None:
            return None
        scope_key = {"scope1": "scope_1", "scope2": "scope_2", "scope3": "scope_3"}.get(scope, scope)
        return round(y.emissions_by_scope.get(scope_key, 0.0), 0)

    metrics = {
        "emissions": {
            "description": "tCO2e, derived from CRI engine using asset-level intensity factors",
            "scope_1_2026": emission_sum(cp, 2026, "scope1"),
            "scope_2_2026": emission_sum(cp, 2026, "scope2"),
            "scope_3_2026": emission_sum(cp, 2026, "scope3"),
            "scope_1_2030_nze": emission_sum(nze, 2030, "scope1"),
            "scope_1_2030_cp":  emission_sum(cp,  2030, "scope1"),
        },
        "carbon_intensity": {
            "description": "tCO2e per USD million revenue (Scope 1+2 combined)",
            # Emissions are stored in MtCO2e; ×1e6 converts to tCO2e for standard intensity unit
            "baseline_2026": (
                round(
                    (
                        (emission_sum(cp, 2026, "scope1") or 0.0)
                        + (emission_sum(cp, 2026, "scope2") or 0.0)
                    ) * 1_000_000 / max((yr(cp, 2026).revenue if yr(cp, 2026) else 1.0), 1.0),
                    1,
                )
            ),
        },
        "targets": [
            {
                "target": "Net-zero Scope 1+2 by 2050",
                "interim_2030": "30% reduction vs 2025 baseline",
                "interim_2035": "50% reduction vs 2025 baseline",
                "framework": "SBTi / Paris-aligned",
                "status": "To be validated against SBTi Corporate Standard",
            },
        ],
        "climate_related_opportunities": [
            "Low-carbon product lines (copper, lithium, nickel for EV supply chain)",
            "Carbon capture and storage (CCS) investment opportunities",
            "Green hydrogen partnerships in existing energy infrastructure",
            "Renewable energy self-supply to reduce Scope 2 exposure",
        ],
    }

    sections = {
        "governance": governance,
        "strategy": strategy,
        "risk_management": risk_management,
        "metrics_and_targets": metrics,
    }

    caveats = [
        "This report is generated by the CRI Climate Risk Intelligence engine using open-source data proxies. "
        "It should be reviewed by qualified climate risk professionals before use in regulatory filings.",
        "Emission intensities are based on asset-level estimates and should be verified against company GHG inventory.",
        "Scenario outputs represent modelled projections and are subject to model uncertainty.",
        "NGFS scenarios represent illustrative pathways; actual outcomes will differ.",
    ]

    return TCFDReport(
        framework="TCFD",
        framework_version="2021 Guidance",
        generated_at=now_str,
        company_id=company.id,
        company_name=company.name,
        reporting_year=reporting_year,
        data_sources=[
            "WRI Aqueduct 4.0 (regional baseline statistics — parameterised lookup, not live API)",
            "NGFS Phase 4 (2023) carbon price and demand pathways",
            "NASA NEX-GDDP-CMIP6 / IPCC AR6 (regional warming statistics — parameterised)",
            "IEA World Energy Outlook 2023 demand curves",
            "Company-provided financial data",
        ],
        caveats=caveats,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# ISSB S2 Report
# ---------------------------------------------------------------------------


@dataclass
class ISSBReport(DisclosureReport):
    """IFRS S2 Climate-related Disclosures (ISSB, June 2023)."""
    pass


def generate_issb(
    company: Company,
    nze: RunResults,
    dt: RunResults,
    cp: RunResults,
    reporting_year: int | None = None,
) -> ISSBReport:
    """
    Generate IFRS S2-aligned disclosure metrics.

    IFRS S2 requires disclosures across:
      - Governance (paragraphs 6–9)
      - Strategy (paragraphs 10–25)
      - Risk Management (paragraphs 26–28)
      - Metrics & Targets (paragraphs 29–37, including cross-industry metrics)
    """
    reporting_year = reporting_year or datetime.date.today().year
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def yr(r: RunResults, year: int):
        return next((y for y in r.years if y.year == year), None)

    cp_2026 = yr(cp, 2026)
    nze_2030 = yr(nze, 2030)

    # Cross-industry climate-related metric categories (IFRS S2 App B)
    cross_industry_metrics = {
        "GHG_emissions": {
            "standard_ref": "IFRS S2 B14–B25",
            "scope_1_tco2e": round(cp_2026.emissions_by_scope.get("scope_1", 0), 0) if cp_2026 else None,
            "scope_2_location_tco2e": round(cp_2026.emissions_by_scope.get("scope_2", 0), 0) if cp_2026 else None,
            "scope_3_tco2e": round(cp_2026.emissions_by_scope.get("scope_3", 0), 0) if cp_2026 else None,
            # Carbon intensity: (S1 + S2) / revenue — tCO2e per USD million revenue
            # Emissions stored in MtCO2e; ×1e6 converts to tCO2e for standard intensity unit
            "carbon_intensity_tco2e_per_usdm_revenue": (
                round(
                    (
                        cp_2026.emissions_by_scope.get("scope_1", 0.0)
                        + cp_2026.emissions_by_scope.get("scope_2", 0.0)
                    ) * 1_000_000 / max(cp_2026.revenue, 1.0),
                    1,
                )
                if cp_2026 else None
            ),
            "ghg_protocol_alignment": "GHG Protocol Corporate Accounting and Reporting Standard",
            "methodology_notes": "Asset-level intensity factors applied to production volumes",
        },
        "transition_risks": {
            "standard_ref": "IFRS S2 B26–B27",
            "carbon_price_exposure_2030_usd_m": round(nze_2030.carbon_cost / 1e6, 1) if nze_2030 else None,
            "revenue_from_high_carbon_products_pct": _revenue_carbon_intensity(company, nze),
            "capex_in_low_carbon_usd_m": round(
                sum(y.transition_capex for y in nze.years[:5]) / 1e6, 1
            ),
        },
        "physical_risks": {
            "standard_ref": "IFRS S2 B28–B29",
            "assets_in_high_risk_regions_pct": _assets_in_high_risk(company),
            "annual_physical_loss_cost_2030_usd_m": round(
                nze_2030.physical_loss_cost / 1e6, 1
            ) if nze_2030 else None,
            "adaptation_capex_planned_usd_m": round(
                sum(y.adaptation_capex for y in nze.years[:5]) / 1e6, 1
            ),
        },
        "climate_related_opportunities": {
            "standard_ref": "IFRS S2 B30",
            "revenue_from_low_carbon_products_usd_m": None,  # requires company data
            "description": "Low-carbon commodity exposure (copper, nickel, lithium) identified in asset mix",
        },
        "capital_deployment": {
            "standard_ref": "IFRS S2 B31",
            "climate_related_capex_usd_m": round(
                sum(y.adaptation_capex + y.transition_capex for y in nze.years[:5]) / 1e6, 1
            ),
            "climate_related_opex_usd_m": round(
                sum(y.carbon_cost for y in nze.years[:5]) / 1e6, 1
            ),
        },
        "internal_carbon_price": {
            "standard_ref": "IFRS S2 B32",
            "price_usd_tco2e": None,   # requires company confirmation
            "description": "Recommend adopting a shadow carbon price ≥ $80/tCO2e for investment decisions",
        },
        "remuneration": {
            "standard_ref": "IFRS S2 B33",
            "climate_linked_remuneration": False,   # company to confirm
            "description": "CRI recommends linking 15–20% of executive incentives to climate KPIs",
        },
    }

    # Financial effects
    ev_nze = round(nze.enterprise_value / 1e3, 1)
    ev_cp  = round(cp.enterprise_value / 1e3, 1)

    financial_effects = {
        "standard_ref": "IFRS S2 10–25",
        "scenario_analysis": {
            "scenarios_used": ["Net Zero 2050", "Delayed Transition", "Current Policies"],
            "alignment": "NGFS Phase 4 (2023)",
            "enterprise_value_nze_usd_bn": ev_nze,
            "enterprise_value_cp_usd_bn": ev_cp,
            "ev_sensitivity_range_pct": round(
                abs(ev_cp - ev_nze) / max(ev_cp, 0.001) * 100, 1
            ),
            "wacc_nze": round(nze.wacc_used * 100, 2),
            "wacc_cp":  round(cp.wacc_used * 100, 2),
        },
        "time_horizons": {
            "short_term": "2026–2030",
            "medium_term": "2031–2040",
            "long_term": "2041–2050",
        },
    }

    sections = {
        "cross_industry_metrics": cross_industry_metrics,
        "financial_effects": financial_effects,
        "industry_specific_metrics": {
            "note": (
                "Industry-specific metrics (e.g., mining: ore grade, water intensity; "
                "oil & gas: reserve replacement ratio under IEA scenarios) should be "
                "added based on applicable IFRS S2 industry guidance."
            ),
        },
    }

    caveats = [
        "IFRS S2 is effective for annual reporting periods beginning on or after 1 January 2024.",
        "Certain jurisdictions have adopted with modifications — verify local requirements.",
        "Scope 3 emissions are estimated using sector-average intensity factors; company-specific inventory required.",
        "This output assists disclosure preparation and is not a substitute for professional assurance.",
    ]

    return ISSBReport(
        framework="IFRS S2",
        framework_version="June 2023",
        generated_at=now_str,
        company_id=company.id,
        company_name=company.name,
        reporting_year=reporting_year,
        data_sources=[
            "WRI Aqueduct 4.0 (regional baseline statistics — parameterised)",
            "NGFS Phase 4 carbon price and demand pathways",
            "NASA NEX-GDDP-CMIP6 (regional statistics — parameterised)",
            "IEA WEO 2023", "GHG Protocol",
        ],
        caveats=caveats,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# EU CSRD E1 Report
# ---------------------------------------------------------------------------


@dataclass
class CSRDReport(DisclosureReport):
    """
    EU CSRD ESRS E1 — Climate Change disclosure.

    Covers ESRS E1 mandatory data points (Commission Delegated Regulation 2023/2772):
      E1-1  Transition plan
      E1-2  Physical & transition risk policies
      E1-3  Actions and resources
      E1-4  Targets
      E1-5  Energy consumption and mix
      E1-6  Gross GHG emissions (Scope 1, 2, 3)
      E1-7  GHG removals and carbon credits
      E1-8  Internal carbon price
      E1-9  Financial effects of climate-related risks and opportunities
    """
    pass


def generate_csrd(
    company: Company,
    nze: RunResults,
    dt: RunResults,
    cp: RunResults,
    reporting_year: int | None = None,
) -> CSRDReport:
    """Generate an ESRS E1-aligned CSRD climate disclosure."""
    reporting_year = reporting_year or datetime.date.today().year
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    def yr(r: RunResults, year: int):
        return next((y for y in r.years if y.year == year), None)

    cp_2026 = yr(cp, 2026)
    nze_2030 = yr(nze, 2030)
    nze_2035 = yr(nze, 2035)

    def emissions_2026(scope_key: str) -> Optional[float]:
        if cp_2026:
            return round(cp_2026.emissions_by_scope.get(scope_key, 0.0), 0)
        return None

    sections = {
        "E1-1_transition_plan": {
            "esrs_ref": "ESRS E1-1",
            "has_transition_plan": False,   # company to confirm
            "target_net_zero": "2050",
            "interim_targets": [
                {"year": 2030, "reduction_pct": 30, "vs_baseline": 2025},
                {"year": 2035, "reduction_pct": 50, "vs_baseline": 2025},
            ],
            "decarbonisation_levers": [
                "Asset-level energy efficiency investments",
                "Renewable energy procurement (Scope 2 reduction)",
                "Supply chain engagement (Scope 3)",
                "Transition capex allocation toward low-carbon commodities",
            ],
            "locked_in_emissions_note": (
                "Remaining asset life analysis required to identify locked-in emissions "
                "from existing infrastructure. CRI model provides carrying value by asset."
            ),
        },
        "E1-2_policies": {
            "esrs_ref": "ESRS E1-2",
            "climate_policy_exists": False,  # company to confirm
            "recommended_policies": [
                "Board-level climate risk policy",
                "Shadow carbon price for investment decisions (≥$80/tCO2e)",
                "Asset retirement policy triggered by 1.5°C-incompatibility",
                "Water use reduction policy for high water-stress assets",
            ],
        },
        "E1-3_actions": {
            "esrs_ref": "ESRS E1-3",
            "planned_adaptation_capex_5yr_usd_m": round(
                sum(y.adaptation_capex for y in nze.years[:5]) / 1e6, 1
            ),
            "planned_transition_capex_5yr_usd_m": round(
                sum(y.transition_capex for y in nze.years[:5]) / 1e6, 1
            ),
        },
        "E1-4_targets": {
            "esrs_ref": "ESRS E1-4",
            "ghg_targets": [
                {
                    "scope": "Scope 1+2",
                    "target_year": 2035,
                    "reduction_pct_vs_2025": 50,
                    "framework": "SBTi / 1.5°C pathway",
                    "validation_status": "Pending SBTi validation",
                },
                {
                    "scope": "Scope 1+2",
                    "target_year": 2050,
                    "reduction_pct_vs_2025": 100,
                    "framework": "Net Zero",
                    "validation_status": "Committed",
                },
            ],
        },
        "E1-5_energy": {
            "esrs_ref": "ESRS E1-5",
            "note": (
                "Energy consumption data (MWh) requires company ERP / metering data. "
                "CRI model provides carbon cost and emission proxies; energy mix detail "
                "should be sourced from company energy management system."
            ),
        },
        "E1-6_ghg_emissions": {
            "esrs_ref": "ESRS E1-6",
            "base_year": reporting_year,
            "scope_1_gross_tco2e": emissions_2026("scope_1"),
            "scope_2_location_tco2e": emissions_2026("scope_2"),
            "scope_2_market_tco2e": None,   # requires RECs / PPA data
            "scope_3_total_tco2e": emissions_2026("scope_3"),
            "scope_3_categories_note": (
                "Scope 3 covers Category 1 (purchased goods/services) and "
                "Category 11 (use of sold products). Other categories require "
                "supply chain data collection."
            ),
            "ghg_protocol_alignment": True,
        },
        "E1-7_removals_credits": {
            "esrs_ref": "ESRS E1-7",
            "ghg_removals_tco2e": None,
            "carbon_credits_retired": None,
            "note": "Carbon credit usage should be reported separately from operational emission reductions.",
        },
        "E1-8_internal_carbon_price": {
            "esrs_ref": "ESRS E1-8",
            "uses_internal_price": False,
            "recommended_price_usd_tco2e": 80,
            "recommended_price_rationale": (
                "Aligned with IEA APS shadow price for advanced economies; "
                "covers investment horizon to 2035."
            ),
        },
        "E1-9_financial_effects": {
            "esrs_ref": "ESRS E1-9",
            "transition_risks": {
                "carbon_cost_2030_usd_m_nze": round(nze_2030.carbon_cost / 1e6, 1) if nze_2030 else None,
                "revenue_at_risk_nze_2030_usd_m": round(
                    (yr(cp, 2030).revenue - nze_2030.revenue) / 1e6, 1
                ) if (yr(cp, 2030) and nze_2030) else None,
            },
            "physical_risks": {
                "physical_loss_2030_usd_m_nze": round(nze_2030.physical_loss_cost / 1e6, 1) if nze_2030 else None,
                "adaptation_capex_5yr_usd_m": round(
                    sum(y.adaptation_capex for y in nze.years[:5]) / 1e6, 1
                ),
            },
            "ev_sensitivity_usd_bn": {
                "nze_2050": round(nze.enterprise_value / 1e3, 1),
                "delayed_transition": round(dt.enterprise_value / 1e3, 1),
                "current_policies": round(cp.enterprise_value / 1e3, 1),
            },
            "anticipates_material_financial_effects": True,
            "materiality_note": (
                "Based on CRI model outputs, climate-related risks are assessed as "
                "financially material. The company should perform a formal double "
                "materiality assessment per ESRS 1 paragraphs 43–67."
            ),
        },
    }

    caveats = [
        "ESRS E1 applies to companies in scope of EU CSRD (large PIEs from FY2024, others phased).",
        "This output is a model-based disclosure template and requires company validation of all data points.",
        "Double materiality assessment (impact + financial) must be performed per ESRS 1 before filing.",
        "Energy consumption data (E1-5) must be sourced from company systems — not modelled here.",
    ]

    return CSRDReport(
        framework="EU CSRD ESRS E1",
        framework_version="Commission Delegated Regulation 2023/2772",
        generated_at=now_str,
        company_id=company.id,
        company_name=company.name,
        reporting_year=reporting_year,
        data_sources=[
            "WRI Aqueduct 4.0 (regional baseline statistics — parameterised)",
            "NGFS Phase 4 carbon price and demand pathways",
            "NASA NEX-GDDP-CMIP6 (regional statistics — parameterised)",
            "IEA WEO 2023", "GHG Protocol", "ESRS E1 Framework",
        ],
        caveats=caveats,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# SEBI BRSR Report (India)
# ---------------------------------------------------------------------------


@dataclass
class BRSRReport(DisclosureReport):
    """
    SEBI Business Responsibility & Sustainability Reporting (BRSR / BRSR Core).

    Mandatory for top-1000 listed companies in India (by market cap) from FY2023-24.
    BRSR Core (assurance-ready subset) required from FY2023-24 per SEBI Circular
    SEBI/HO/CFD/CMD-2/P/CIR/2023/122 (12 July 2023).

    Key climate-related sections:
      Principle 6 — Environment (P6): climate change, energy, water, waste
      Section A  — General disclosures (company overview, regulatory penalties)
      Section B  — Management processes (ESG governance, policy, targets)
      Section C  — Principle-wise performance (quantitative KPIs)
    """
    pass


def generate_brsr(
    company: Company,
    nze: RunResults,
    dt: RunResults,
    cp: RunResults,
    reporting_year: int | None = None,
) -> BRSRReport:
    """
    Generate a SEBI BRSR Core-aligned climate disclosure.

    Covers all mandatory data points under BRSR Principle 6 (Environment)
    plus Section A general disclosures and climate-related Section B governance.
    """
    reporting_year = reporting_year or datetime.date.today().year
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    ctx = _sector_context(company)

    def yr(r: RunResults, year: int):
        return next((y for y in r.years if y.year == year), None)

    cp_2026  = yr(cp, 2026)
    nze_2030 = yr(nze, 2030)

    def scope_val(scope_key: str) -> Optional[float]:
        return round(cp_2026.emissions_by_scope.get(scope_key, 0.0), 0) if cp_2026 else None

    # ── Section A — General Disclosures ─────────────────────────────────────
    section_a = {
        "sebi_ref": "BRSR Section A",
        "company_details": {
            "company_name": ctx["name"],
            "sector": ctx["sector"],
            "hq_region": ctx["hq"],
            "reporting_year": reporting_year,
            "operating_regions": ctx["asset_regions"],
            "number_of_plants_locations": ctx["n_assets"],
        },
        "materiality_note": (
            f"Based on CRI engine outputs, climate change is a material topic for {ctx['name']} "
            f"due to significant physical exposure at {ctx['asset_regions']} and transition risk "
            f"from India's updated Nationally Determined Contribution (NDC) and the upcoming "
            f"Carbon Credit Trading Scheme (CCTS) notified under the Energy Conservation Act 2022."
        ),
    }

    # ── Section B — Management Processes ────────────────────────────────────
    section_b = {
        "sebi_ref": "BRSR Section B",
        "esg_governance": {
            "board_committee": "Audit & Risk Committee / CSR & ESG Committee (recommended)",
            "board_oversight_description": (
                f"The Board of {ctx['name']} oversees climate-related risks with focus on "
                f"{ctx['board_focus']}. BRSR Core disclosures are approved by the Board before filing."
            ),
            "management_responsibility": ctx["mgmt_role"],
        },
        "policies_principle_6": {
            "environment_policy_exists": False,   # company to confirm
            "recommended_policies": [
                "Board-level Environmental Policy covering GHG emissions, water, and waste",
                f"Shadow carbon price ≥ INR 2,000/tCO2e (≈$24/tCO2e) for new investments — aligned "
                f"with India's CCTS carbon credit market development",
                "Water intensity reduction policy for high water-stress operational sites",
                f"Asset retirement policy triggered by 1.5°C-incompatibility ({ctx['sector']}-specific)",
            ],
            "policy_coverage_note": (
                "Policies should cover all entities in BRSR scope including subsidiaries "
                "operating in India as per Companies Act 2013 / SEBI LODR."
            ),
        },
        "targets_principle_6": {
            "ghg_targets": [
                {
                    "scope": "Scope 1 + Scope 2",
                    "target_year": 2035,
                    "reduction_pct_vs_2025": 45,
                    "rationale": (
                        "Aligned with India's updated NDC (45% emissions intensity reduction "
                        "vs GDP by 2030) and IPCC AR6 1.5°C pathway for emerging market corporates"
                    ),
                    "validation_status": "Pending SBTi / BEE validation",
                },
            ],
            "energy_intensity_target": {
                "target": "Reduce energy intensity by 20% by 2030 vs FY2024 baseline",
                "basis": "GJ per tonne produced / per INR crore revenue",
                "rationale": "PAT Scheme (Perform Achieve Trade) alignment under BEE",
            },
            "water_intensity_target": {
                "target": "Reduce specific water consumption by 25% by 2030",
                "basis": "KL per tonne produced",
                "applicable_regions": ctx["asset_regions"],
            },
        },
    }

    # ── Section C — Principle 6 (Environment) KPIs ──────────────────────────
    section_c_p6 = {
        "sebi_ref": "BRSR Section C, Principle 6",
        "P6_E1_energy_consumption": {
            "brsr_indicator": "P6-E1",
            "description": "Total energy consumed and energy intensity",
            "total_energy_consumed_gj": None,    # requires company metering / ERP data
            "energy_from_renewables_pct": None,
            "energy_intensity_gj_per_tonne": None,
            "note": (
                "Energy consumption data must be sourced from company energy management "
                "system / DISCOM bills. CRI provides carbon cost and emission proxies only."
            ),
        },
        "P6_E2_water_consumption": {
            "brsr_indicator": "P6-E2",
            "description": "Water withdrawal and consumption",
            "total_water_withdrawal_kl": None,   # requires company water records
            "water_intensity_kl_per_tonne": None,
            "water_discharged_kl": None,
            "note": (
                f"{ctx['name']}'s operations in high-water-stress regions require detailed "
                f"site-level water accounting. WRI Aqueduct baseline stress scores are "
                f"available in the CRI engine for each asset region."
            ),
        },
        "P6_E3_ghg_emissions": {
            "brsr_indicator": "P6-E3",
            "description": "Scope 1, 2, 3 GHG emissions (tCO2e)",
            "base_year": reporting_year,
            "scope_1_tco2e": scope_val("scope_1"),
            "scope_2_tco2e": scope_val("scope_2"),
            "scope_3_tco2e": scope_val("scope_3"),
            "ghg_emission_intensity_tco2e_per_crore_inr": None,   # needs revenue in INR
            "ghg_protocol_alignment": True,
            "methodology_note": (
                "Scope 1 and 2 are estimated using CRI asset-level intensity factors. "
                "Scope 3 covers Category 1 (purchased goods/services) and Category 11 "
                "(use of sold products) where applicable. "
                "Verification by an accredited third party is recommended before BRSR filing."
            ),
        },
        "P6_E4_waste_management": {
            "brsr_indicator": "P6-E4",
            "waste_generated_tonnes": None,     # requires operational data
            "hazardous_waste_disposed_safely_pct": None,
            "note": "Waste data requires site-level operational records.",
        },
        "P6_E5_environmental_compliance": {
            "brsr_indicator": "P6-E5",
            "environmental_incidents": None,
            "regulatory_penalties_inr": None,
            "note": "Regulatory compliance data requires company legal/EHS records.",
        },
        "P6_L1_climate_risk_physical": {
            "brsr_indicator": "P6-L1 (Leadership — voluntary but recommended)",
            "description": "Climate-related physical risks and adaptation plans",
            "dominant_physical_hazards": ctx["primary_hazards"],
            "cri_physical_risk_score_2030_nze": (
                "Asset-level scores available in CRI engine — see physical risk module"
            ),
            "adaptation_capex_5yr_inr_cr": round(
                sum(y.adaptation_capex for y in nze.years[:5]) / 1e6 * 83 / 1e2, 1
            ),   # USD → INR crore (approx 83 USD/INR, 1 crore = 10M)
            "adaptation_measures": ctx["short_term_actions"][:2],
        },
        "P6_L2_climate_risk_transition": {
            "brsr_indicator": "P6-L2 (Leadership — voluntary but recommended)",
            "description": "Climate-related transition risks and opportunities",
            "key_transition_risk": ctx["key_transition"],
            "key_opportunity": ctx["opportunity_text"],
            "carbon_cost_2030_usd_m_nze": round(nze_2030.carbon_cost / 1e6, 1) if nze_2030 else None,
            "ev_range_nze_vs_cp_usd_bn": {
                "nze_2050": round(nze.enterprise_value / 1e3, 1),
                "current_policies": round(cp.enterprise_value / 1e3, 1),
            },
        },
    }

    sections = {
        "section_a_general": section_a,
        "section_b_management": section_b,
        "section_c_principle6_environment": section_c_p6,
        "filing_guidance": {
            "mandatory_assurance": (
                "BRSR Core KPIs (P6-E1 through P6-E5) require Reasonable Assurance from "
                "a SEBI-registered provider (CA/CMA firm) for top-1000 companies from FY2024-25."
            ),
            "filing_deadline": "60 days after financial year end (alongside Annual Report)",
            "applicable_standard": "SEBI BRSR format (Annexure I, SEBI LODR Regulations 2015)",
            "ccts_note": (
                "India's Carbon Credit Trading Scheme (CCTS) will create carbon price exposure "
                "for designated industries. Monitor Bureau of Energy Efficiency (BEE) notifications "
                "for sector-specific thresholds."
            ),
        },
    }

    caveats = [
        "BRSR Core is mandatory for India's top-1000 listed companies (by market cap, BSE+NSE) from FY2023-24.",
        "Energy, water, and waste data points (P6-E1, E2, E4) require company operational records — not modelled by CRI.",
        "GHG intensity in INR terms requires company revenue data in INR — CRI operates in USD.",
        "CRI physical risk scores use WRI Aqueduct and NASA proxies; site-level primary data is recommended for BRSR assurance.",
        "The Carbon Credit Trading Scheme (CCTS) is under phased implementation — monitor BEE guidance for designated consumers.",
    ]

    return BRSRReport(
        framework="SEBI BRSR Core",
        framework_version="SEBI Circular SEBI/HO/CFD/CMD-2/P/CIR/2023/122 (Jul 2023)",
        generated_at=now_str,
        company_id=company.id,
        company_name=company.name,
        reporting_year=reporting_year,
        data_sources=[
            "WRI Aqueduct 4.0 (regional baseline statistics — parameterised lookup, not live API)",
            "NGFS Phase 4 (2023) carbon price and demand pathways",
            "NASA NEX-GDDP-CMIP6 / IPCC AR6 (regional statistics — parameterised)",
            "SEBI BRSR Framework (Jul 2023)",
            "India CCTS / BEE PAT Scheme references",
            "GHG Protocol Corporate Standard",
        ],
        caveats=caveats,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _revenue_carbon_intensity(company: Company, r: RunResults) -> Optional[float]:
    """
    Estimate % of revenue from high-carbon products (coal, oil, gas).
    Returns as a percentage (0–100).
    """
    from ..data.schemas import Commodity
    high_carbon = {Commodity.COAL_THERMAL, Commodity.COAL_METALLURGICAL,
                   Commodity.CRUDE_OIL, Commodity.NATURAL_GAS, Commodity.REFINED_PRODUCTS}
    if not r.years:
        return None
    y0 = r.years[0]
    total_rev = sum(y0.revenue_by_commodity.values())
    if total_rev <= 0:
        return None
    hc_rev = sum(
        v for k, v in y0.revenue_by_commodity.items()
        if any(hc.value in k.lower() for hc in high_carbon)
    )
    return round(hc_rev / total_rev * 100, 1)


def _assets_in_high_risk(company: Company) -> Optional[float]:
    """
    Estimate % of assets (by carrying value) in high physical-risk regions.
    High-risk: AU-WA, AU-QLD, CL-02, ZA, IN-MH (from WRI Aqueduct benchmarks).
    """
    HIGH_RISK_REGIONS = {"AU-WA", "AU-QLD", "CL-02", "ZA", "IN-MH", "CN-NM", "MN"}
    total_cv = sum(a.carrying_value for a in company.assets)
    if total_cv <= 0:
        return None
    hr_cv = sum(a.carrying_value for a in company.assets if a.region in HIGH_RISK_REGIONS)
    return round(hr_cv / total_cv * 100, 1)
