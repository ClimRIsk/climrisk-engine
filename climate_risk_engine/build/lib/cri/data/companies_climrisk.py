"""ClimRisk actual coverage universe — demo companies.

All companies are sourced from ClimRisk's live CRI report portfolio.
Financial data is sourced from company public filings and disclosures
(annual reports, CDP disclosures, stock exchange filings) as of FY2024.
All monetary values are in USD millions unless otherwise stated.

Companies:
  TATA_STEEL     — Tata Steel Limited (NS:TATASTEEL) — CRI Score 28 / Rating D
  DELTA_AIR      — Delta Air Lines Inc. (NYSE:DAL)
  CARNIVAL_CORP  — Carnival Corporation & plc (NYSE/LSE:CCL)
  ULTRATECH      — UltraTech Cement Limited (NS:ULTRACEMCO) — CRI Score C / +52 bps WACC

Sources:
  Tata Steel FY2024 Annual Report; CDP 2023 Disclosure; TCFD 2024 Supplement.
  Delta Air Lines 10-K FY2024; FAA Form 41 emissions data; SAF compliance plan.
  Carnival Corporation FY2024 Annual Report; IMO CII disclosures; EU ETS filings.
  UltraTech Cement FY2024 Annual Report; PAT scheme disclosures; GRI report.
"""

from __future__ import annotations

from .schemas import (
    Asset,
    Commodity,
    Company,
    EmissionsProfile,
    Financials,
    SegmentBaseline,
)


# ---------------------------------------------------------------------------
# Production calibration note
# ---------------------------------------------------------------------------
# The engine uses MANUFACTURING as the commodity proxy for steel, aviation, and
# cruise companies (no dedicated commodity curves exist for these sectors yet).
# MANUFACTURING price baseline = USD 200 / unit in 2026 (CP scenario).
# baseline_production is set so that volume × USD_200 ≈ company's actual revenue.
# scope1_intensity is calibrated from CDP/company Scope 1 disclosures.
# This produces correct relative scenario impacts; absolute EV is calibrated
# to the company's actual wacc_base and net_debt.
#
# For UltraTech, CEMENT commodity has no price curve in the scenarios, so
# MANUFACTURING is again used as the proxy.

_MFG_PRICE_2026 = 200.0   # USD / production unit (MANUFACTURING commodity, CP scenario 2026)


# ---------------------------------------------------------------------------
# Tata Steel Limited — Integrated Steel, CRI Score 28 / Rating D
# ---------------------------------------------------------------------------

TATA_STEEL = Company(
    id="tata_steel",
    name="Tata Steel Limited",
    sector="Steel",
    hq_region="IN-MH",   # Maharashtra (Mumbai HQ)
    financials=Financials(
        revenue=22_000.0,           # USD M — FY2024 consolidated
        ebitda=3_500.0,             # USD M — FY2025 normalised (post Port Talbot BF-BOF closure)
        capex=2_500.0,
        maintenance_capex_share=0.40,   # Normalised: FY2024 included ~$1.1B extraordinary
                                        # Port Talbot/green capex; ongoing sustaining ≈40%
        tax_rate=0.25,
        wacc_base=0.098,            # Steel sector median — CRI Engine calibrated
        net_debt=9_800.0,           # USD M — FY2024 net debt position
        shares_outstanding=12_200.0,  # M shares (as of FY2024)
        market_cap=23_500.0,
    ),
    assets=[
        # ── IJmuiden, Netherlands — BF-BOF integrated plant ──────────────────
        # Tata Steel Netherlands: ~6.5 Mt/yr crude steel, 1.85 tCO2/t (Scope 1)
        # Subject to EU ETS Phase 4; free allocation phasing out to 0% by 2034 (CBAM)
        Asset(
            id="ts_ijmuiden",
            name="Tata Steel IJmuiden (Netherlands)",
            commodity=Commodity.MANUFACTURING,
            region="NL-NH",
            baseline_production=30.0,      # M-units: revenue = 30 × $200 = $6,000M
            production_unit="rev_units",
            baseline_unit_cost=155.5,      # Calibrated: 110 units × $155.5 × 1.052 factor
                                           # ≈ $18,000M opex → EBITDA ~$3,500M (post-carbon)
            energy_cost_share=0.45,
            carrying_value=5_500.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=0.367,     # tCO2/unit — 11.0 MtCO2/yr Scope 1
                scope2_intensity=0.045,     # tCO2/unit — 1.35 MtCO2/yr Scope 2
                scope3_intensity=0.918,     # tCO2/unit — 27.5 MtCO2/yr Scope 3
                carbon_price_coverage=0.95,  # EU ETS covers ~95% of direct emissions
                free_allocation=0.20,        # ~20% free allocation remaining in 2026
            ),
            lat=52.46,  lon=4.59,          # IJmuiden, North Sea coast, North Holland
            equipment_type="steel_plant",
        ),
        # ── Port Talbot, Wales UK — BF-BOF transitioning to EAF ──────────────
        # Tata Steel UK: ~2.8 Mt/yr; transition to DRI-EAF underway (UK govt grant)
        # UK ETS (formerly EU-linked). Free allocation phasing out.
        Asset(
            id="ts_port_talbot",
            name="Tata Steel Port Talbot (UK)",
            commodity=Commodity.MANUFACTURING,
            region="GB-WLS",
            baseline_production=11.0,      # M-units: revenue = 11 × $200 = $2,200M
            production_unit="rev_units",
            baseline_unit_cost=155.5,
            energy_cost_share=0.42,
            carrying_value=1_200.0,
            remaining_life_years=15,       # transitioning → shorter life on BF-BOF basis
            emissions=EmissionsProfile(
                scope1_intensity=0.455,     # tCO2/unit — 5.0 MtCO2/yr Scope 1
                scope2_intensity=0.045,     # tCO2/unit
                scope3_intensity=1.138,     # tCO2/unit
                carbon_price_coverage=0.90,  # UK ETS — high coverage
                free_allocation=0.15,        # UK free allocation pool (more generous than EU)
            ),
            lat=51.59,  lon=-3.80,         # Port Talbot, Glamorgan, Wales
            equipment_type="steel_plant",
        ),
        # ── Jamshedpur, Jharkhand, India — integrated BF-BOF ─────────────────
        # Tata Steel's largest Indian plant: ~10 Mt/yr
        # India PAT (Perform Achieve Trade) scheme; BEE star rating scheme
        Asset(
            id="ts_jamshedpur",
            name="Tata Steel Jamshedpur (India)",
            commodity=Commodity.MANUFACTURING,
            region="IN-JH",
            baseline_production=41.0,      # M-units: revenue = 41 × $200 = $8,200M
            production_unit="rev_units",
            baseline_unit_cost=155.5,
            energy_cost_share=0.40,
            carrying_value=2_500.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=0.337,     # tCO2/unit — 13.8 MtCO2/yr Scope 1
                scope2_intensity=0.043,
                scope3_intensity=0.843,
                carbon_price_coverage=0.25,  # India PAT partial coverage
                free_allocation=0.0,
            ),
            lat=22.80,  lon=86.18,         # Jamshedpur, Jharkhand (Subarnarekha River basin)
            equipment_type="steel_plant",
        ),
        # ── Kalinganagar, Odisha, India — newer BOF plant ────────────────────
        # Tata Steel Kalinganagar: ~8 Mt/yr (Brahmani River basin)
        Asset(
            id="ts_kalinganagar",
            name="Tata Steel Kalinganagar (India)",
            commodity=Commodity.MANUFACTURING,
            region="IN-OD",
            baseline_production=28.0,      # M-units: revenue = 28 × $200 = $5,600M
            production_unit="rev_units",
            baseline_unit_cost=155.5,
            energy_cost_share=0.40,
            carrying_value=1_800.0,
            remaining_life_years=30,       # newer plant, longer life
            emissions=EmissionsProfile(
                scope1_intensity=0.250,     # tCO2/unit — 7.0 MtCO2/yr Scope 1
                scope2_intensity=0.040,
                scope3_intensity=0.625,
                carbon_price_coverage=0.25,
                free_allocation=0.0,
            ),
            lat=21.28,  lon=85.91,         # Kalinganagar, Odisha (Brahmani River basin)
            equipment_type="steel_plant",
        ),
    ],
    exposure_weight=0.70,     # High: 4 major sites across flood / water-stress zones
    transition_weight=0.90,   # High: BF-BOF stranded-asset risk, EU ETS cliff, CBAM
    data_quality="high",
)


# ---------------------------------------------------------------------------
# Delta Air Lines Inc. — Aviation, CRI analysis (NYSE: DAL)
# ---------------------------------------------------------------------------

DELTA_AIR = Company(
    id="delta_air",
    name="Delta Air Lines Inc.",
    sector="Aviation",
    hq_region="US-GA",   # Atlanta, Georgia
    financials=Financials(
        revenue=58_000.0,           # USD M — FY2024 consolidated
        ebitda=5_800.0,             # USD M — ~10% adjusted EBITDA margin
        capex=4_500.0,
        maintenance_capex_share=0.50,   # Fleet maintenance + fleet growth investments
        tax_rate=0.21,
        wacc_base=0.092,            # Aviation sector WACC (higher operational leverage)
        net_debt=14_000.0,          # USD M — FY2024 (incl. finance leases)
        shares_outstanding=645.0,   # M shares
        market_cap=25_000.0,
    ),
    assets=[
        # ── Atlanta Hartsfield-Jackson (ATL) — primary hub ───────────────────
        # World's busiest airport; Delta's core hub. Heat stress + storm surge risk.
        Asset(
            id="dal_atl",
            name="Delta Air Lines — Atlanta Hub (ATL)",
            commodity=Commodity.MANUFACTURING,
            region="US-GA",
            baseline_production=116.0,     # M-units: revenue = 116 × $200 = $23,200M
            production_unit="rev_units",
            baseline_unit_cost=169.0,      # Calibrated: 290 units × $169 × 1.0625 factor
                                           # ≈ $52,100M opex → EBITDA ~$5,800M (post-carbon)
            energy_cost_share=0.50,        # Aviation is energy-intensive
            carrying_value=8_000.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=0.0966,    # tCO2/unit — 11.2 MtCO2/yr Scope 1
                scope2_intensity=0.0020,
                scope3_intensity=0.121,
                carbon_price_coverage=0.15, # CORSIA offsetting (North Atlantic routes)
                free_allocation=0.0,
            ),
            lat=33.64,  lon=-84.43,        # Hartsfield-Jackson Atlanta International
            equipment_type="airport_terminal",
        ),
        # ── JFK / LaGuardia New York — coastal hub ────────────────────────────
        # Sea level + storm surge risk (Sandy precedent). EU ETS applies on transatlantic.
        Asset(
            id="dal_jfk",
            name="Delta Air Lines — New York Hub (JFK/LGA)",
            commodity=Commodity.MANUFACTURING,
            region="US-NY",
            baseline_production=43.5,      # M-units: revenue = 43.5 × $200 = $8,700M
            production_unit="rev_units",
            baseline_unit_cost=169.0,
            energy_cost_share=0.50,
            carrying_value=2_500.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=0.0966,
                scope2_intensity=0.0020,
                scope3_intensity=0.121,
                carbon_price_coverage=0.35,  # EU ETS applies on JFK-Europe flights
                free_allocation=0.0,
            ),
            lat=40.64,  lon=-73.78,        # JFK International, Queens, NY
            equipment_type="airport_terminal",
        ),
        # ── Los Angeles LAX — West Coast hub ─────────────────────────────────
        # Wildfire smoke operational risk; sea level; California carbon policy (CARB).
        Asset(
            id="dal_lax",
            name="Delta Air Lines — Los Angeles Hub (LAX)",
            commodity=Commodity.MANUFACTURING,
            region="US-CA",
            baseline_production=43.5,      # M-units: revenue = 43.5 × $200 = $8,700M
            production_unit="rev_units",
            baseline_unit_cost=169.0,
            energy_cost_share=0.50,
            carrying_value=2_000.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=0.0966,
                scope2_intensity=0.0020,
                scope3_intensity=0.121,
                carbon_price_coverage=0.20,  # CORSIA + CARB offset compliance
                free_allocation=0.0,
            ),
            lat=33.94,  lon=-118.41,       # Los Angeles International
            equipment_type="airport_terminal",
        ),
        # ── International + Other Domestic Routes ────────────────────────────
        # Minneapolis (MSP) hub + other routes; aggregate physical risk weighted
        Asset(
            id="dal_other",
            name="Delta Air Lines — MSP / Other Routes",
            commodity=Commodity.MANUFACTURING,
            region="US-MN",
            baseline_production=87.0,      # M-units: revenue = 87 × $200 = $17,400M
            production_unit="rev_units",
            baseline_unit_cost=169.0,
            energy_cost_share=0.50,
            carrying_value=5_000.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=0.0966,
                scope2_intensity=0.0020,
                scope3_intensity=0.121,
                carbon_price_coverage=0.10,
                free_allocation=0.0,
            ),
            lat=44.88,  lon=-93.22,        # Minneapolis-Saint Paul International
            equipment_type="airport_terminal",
        ),
    ],
    exposure_weight=0.50,     # Medium: hub disruption, extreme weather, coastal
    transition_weight=0.65,   # Medium-high: SAF mandate cost, aviation carbon pricing ramp
    data_quality="high",
)


# ---------------------------------------------------------------------------
# Carnival Corporation & plc — Cruise Lines
# ---------------------------------------------------------------------------

CARNIVAL_CORP = Company(
    id="carnival_corp",
    name="Carnival Corporation & plc",
    sector="Cruise / Hospitality",
    hq_region="US-FL",   # Miami, Florida HQ
    financials=Financials(
        revenue=21_600.0,           # USD M — FY2024 consolidated
        ebitda=4_800.0,             # USD M — ~22% EBITDA margin (recovering)
        capex=3_500.0,
        maintenance_capex_share=0.45,   # New ship orders dominate
        tax_rate=0.21,
        wacc_base=0.092,            # High leverage post-COVID
        net_debt=31_000.0,          # USD M — substantial post-COVID debt
        shares_outstanding=1_275.0,
        market_cap=18_000.0,
    ),
    assets=[
        # ── Miami homeport — Caribbean fleet operations ────────────────────────
        # Hurricane corridor; sea level rise; coral reef bleaching → destination risk
        Asset(
            id="ccl_miami",
            name="Carnival — Miami Homeport (Caribbean Fleet)",
            commodity=Commodity.MANUFACTURING,
            region="US-FL",
            baseline_production=43.2,      # M-units: revenue = 43.2 × $200 = $8,640M
            production_unit="rev_units",
            baseline_unit_cost=147.0,      # Calibrated: 108 units × $147 × 1.0475 factor
                                           # ≈ $16,600M opex → EBITDA ~$4,800M (post-carbon)
            energy_cost_share=0.38,
            carrying_value=12_000.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=0.102,     # tCO2/unit — 4.4 MtCO2/yr Scope 1
                scope2_intensity=0.003,
                scope3_intensity=0.130,
                carbon_price_coverage=0.20,  # CORSIA + IMO CII offsetting
                free_allocation=0.0,
            ),
            lat=25.78,  lon=-80.19,        # PortMiami, Miami, Florida
            equipment_type="port_terminal",
        ),
        # ── Southampton — European / Transatlantic operations ─────────────────
        # EU ETS maritime (Phase 1: 2024–2026, 40% of voyages covered)
        Asset(
            id="ccl_southampton",
            name="Carnival — Southampton Homeport (European Fleet)",
            commodity=Commodity.MANUFACTURING,
            region="GB-ENG",
            baseline_production=21.6,      # M-units: revenue = 21.6 × $200 = $4,320M
            production_unit="rev_units",
            baseline_unit_cost=147.0,
            energy_cost_share=0.38,
            carrying_value=8_000.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=0.102,
                scope2_intensity=0.003,
                scope3_intensity=0.130,
                carbon_price_coverage=0.55,  # EU ETS maritime applies to EU port voyages
                free_allocation=0.0,
            ),
            lat=50.90,  lon=-1.40,         # Port of Southampton, Hampshire, UK
            equipment_type="port_terminal",
        ),
        # ── Hamburg — Northern European operations ─────────────────────────────
        # Elbe storm surge; EU ETS maritime
        Asset(
            id="ccl_hamburg",
            name="Carnival — Hamburg Homeport (AIDA / Northern Europe)",
            commodity=Commodity.MANUFACTURING,
            region="DE-HH",
            baseline_production=16.2,      # M-units: revenue = 16.2 × $200 = $3,240M
            production_unit="rev_units",
            baseline_unit_cost=147.0,
            energy_cost_share=0.38,
            carrying_value=5_000.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=0.102,
                scope2_intensity=0.003,
                scope3_intensity=0.130,
                carbon_price_coverage=0.60,  # EU ETS — high coverage for Hamburg routes
                free_allocation=0.0,
            ),
            lat=53.55,  lon=9.99,          # Port of Hamburg, Germany
            equipment_type="port_terminal",
        ),
        # ── Barcelona — Mediterranean operations ──────────────────────────────
        # Heat stress; port congestion risk; EU ETS maritime
        Asset(
            id="ccl_barcelona",
            name="Carnival — Barcelona Homeport (Mediterranean Fleet)",
            commodity=Commodity.MANUFACTURING,
            region="ES-CT",
            baseline_production=27.0,      # M-units: revenue = 27 × $200 = $5,400M
            production_unit="rev_units",
            baseline_unit_cost=147.0,
            energy_cost_share=0.38,
            carrying_value=5_000.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=0.102,
                scope2_intensity=0.003,
                scope3_intensity=0.130,
                carbon_price_coverage=0.55,  # EU ETS maritime
                free_allocation=0.0,
            ),
            lat=41.38,  lon=2.17,          # Port of Barcelona, Catalonia, Spain
            equipment_type="port_terminal",
        ),
    ],
    exposure_weight=0.80,     # High: Hurricane belt, sea level rise, coastal operations
    transition_weight=0.55,   # Medium: EU ETS maritime (new from 2024), IMO 2030 target
    data_quality="high",
)


# ---------------------------------------------------------------------------
# UltraTech Cement Limited — India Cement, CRI Score C
# ---------------------------------------------------------------------------

ULTRATECH = Company(
    id="ultratech",
    name="UltraTech Cement Limited",
    sector="Cement",
    hq_region="IN-MH",   # Maharashtra (Mumbai)
    financials=Financials(
        revenue=8_500.0,            # USD M — FY2024 (Rs ~70,500 crore)
        ebitda=1_870.0,             # USD M — ~22% EBITDA margin (strong Indian cement)
        capex=1_000.0,
        maintenance_capex_share=0.60,
        tax_rate=0.25,
        wacc_base=0.105,            # India + cement sector premium — from CRI report
        net_debt=2_200.0,
        shares_outstanding=288.0,   # M shares
        market_cap=33_000.0,        # USD M (~Rs 2.7 lakh crore market cap)
    ),
    assets=[
        # ── Aditya Cement Works, Rajasthan ────────────────────────────────────
        # WRI Aqueduct: extreme water stress (5/5). PAT scheme.
        Asset(
            id="utc_aditya",
            name="UltraTech — Aditya Cement Works (Rajasthan)",
            commodity=Commodity.MANUFACTURING,
            region="IN-RJ",
            baseline_production=8.5,       # M-units: revenue = 8.5 × $200 = $1,700M
            production_unit="rev_units",
            baseline_unit_cost=136.0,      # Calibrated: 42.5 units × $136 × 1.0475 factor
                                           # ≈ $6,050M opex → EBITDA ~$1,870M (post-carbon)
            energy_cost_share=0.38,
            carrying_value=1_800.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=1.506,     # tCO2/unit — 12.8 MtCO2/yr Scope 1
                scope2_intensity=0.059,
                scope3_intensity=0.377,
                carbon_price_coverage=0.35,  # India PAT → ATS (carbon market transition)
                free_allocation=0.0,
            ),
            lat=24.80,  lon=74.60,         # Rajasthan (Aditya Cement Works)
            equipment_type="cement_plant",
        ),
        # ── Gujarat Cement Works ─────────────────────────────────────────────
        # Severe water scarcity + extreme heat stress (Rann of Kutch proximity)
        Asset(
            id="utc_gujarat",
            name="UltraTech — Gujarat Cement Works",
            commodity=Commodity.MANUFACTURING,
            region="IN-GJ",
            baseline_production=8.5,       # M-units: revenue = 8.5 × $200 = $1,700M
            production_unit="rev_units",
            baseline_unit_cost=136.0,
            energy_cost_share=0.38,
            carrying_value=1_800.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=1.506,
                scope2_intensity=0.059,
                scope3_intensity=0.377,
                carbon_price_coverage=0.35,
                free_allocation=0.0,
            ),
            lat=20.90,  lon=71.40,         # Gujarat (coastal / Saurashtra)
            equipment_type="cement_plant",
        ),
        # ── Andhra Pradesh Cement Works ──────────────────────────────────────
        # Cyclone exposure (Bay of Bengal); extreme heat; APSPDCL grid reliability
        Asset(
            id="utc_andhra",
            name="UltraTech — Andhra Pradesh Cement Works",
            commodity=Commodity.MANUFACTURING,
            region="IN-AP",
            baseline_production=8.5,       # M-units: revenue = 8.5 × $200 = $1,700M
            production_unit="rev_units",
            baseline_unit_cost=136.0,
            energy_cost_share=0.38,
            carrying_value=1_500.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=1.506,
                scope2_intensity=0.059,
                scope3_intensity=0.377,
                carbon_price_coverage=0.35,
                free_allocation=0.0,
            ),
            lat=15.00,  lon=78.00,         # Andhra Pradesh (Kurnool / Nandyal district)
            equipment_type="cement_plant",
        ),
        # ── Rajashree Cement Works, Karnataka ────────────────────────────────
        # Heat stress + drought risk (Deccan Plateau)
        Asset(
            id="utc_rajashree",
            name="UltraTech — Rajashree Cement Works (Karnataka)",
            commodity=Commodity.MANUFACTURING,
            region="IN-KA",
            baseline_production=8.5,       # M-units: revenue = 8.5 × $200 = $1,700M
            production_unit="rev_units",
            baseline_unit_cost=136.0,
            energy_cost_share=0.38,
            carrying_value=1_500.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=1.506,
                scope2_intensity=0.059,
                scope3_intensity=0.377,
                carbon_price_coverage=0.35,
                free_allocation=0.0,
            ),
            lat=17.20,  lon=77.20,         # Karnataka (Gulbarga / Kalaburagi district)
            equipment_type="cement_plant",
        ),
        # ── Bela Cement Works, Madhya Pradesh ────────────────────────────────
        # Inland flood risk (monsoon intensification); heat stress
        Asset(
            id="utc_bela",
            name="UltraTech — Bela Cement Works (Madhya Pradesh)",
            commodity=Commodity.MANUFACTURING,
            region="IN-MP",
            baseline_production=8.5,       # M-units: revenue = 8.5 × $200 = $1,700M
            production_unit="rev_units",
            baseline_unit_cost=136.0,
            energy_cost_share=0.38,
            carrying_value=1_400.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=1.506,
                scope2_intensity=0.059,
                scope3_intensity=0.377,
                carbon_price_coverage=0.35,
                free_allocation=0.0,
            ),
            lat=24.50,  lon=81.20,         # Madhya Pradesh (Rewa / Bela district)
            equipment_type="cement_plant",
        ),
    ],
    exposure_weight=0.80,     # High: all 5 plants in extreme heat/water-stress zones
    transition_weight=0.85,   # High: process CO2 from calcination unavoidable (0.55 tCO2/t)
    data_quality="high",
)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def all_climrisk() -> dict[str, Company]:
    return {c.id: c for c in [TATA_STEEL, DELTA_AIR, CARNIVAL_CORP, ULTRATECH]}


def get(company_id: str) -> Company:
    companies = all_climrisk()
    if company_id not in companies:
        raise KeyError(
            f"Unknown ClimRisk company {company_id!r}. "
            f"Available: {list(companies)}"
        )
    return companies[company_id]
