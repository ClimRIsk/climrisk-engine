"""Seed test companies used for Phase 1 validation.

CRI_TestCo is a fictional diversified miner — not a prediction about any
real company. Use it for tests, demos, and methodology experiments.

Shell, BHP, and Rio Tinto are real companies with baseline data sourced
from public annual reports, CDP disclosures, and industry databases (2025).
"""

from __future__ import annotations

from .schemas import (
    Asset,
    Commodity,
    Company,
    EmissionsProfile,
    Financials,
)


# ---------------------------------------------------------------------------
# Fictional test miner
# ---------------------------------------------------------------------------

CRI_TEST_CO = Company(
    id="cri_testco",
    name="CRI TestCo (fictional)",
    sector="Mining",
    hq_region="AU-WA",
    financials=Financials(
        revenue=55_000.0,         # USD millions
        ebitda=22_000.0,
        capex=9_000.0,
        maintenance_capex_share=0.65,
        tax_rate=0.30,
        wacc_base=0.08,
        net_debt=10_000.0,
        shares_outstanding=1_600.0,
        market_cap=140_000.0,
    ),
    assets=[
        Asset(
            id="tc_ironore_au",
            name="TestCo Pilbara Iron Ore complex",
            commodity=Commodity.IRON_ORE,
            region="AU-WA",
            baseline_production=250.0,     # Mtpa
            production_unit="Mtonnes",
            baseline_unit_cost=22.0,
            energy_cost_share=0.35,
            carrying_value=18_000.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=0.015,     # tCO2 per tonne
                scope2_intensity=0.010,
                scope3_intensity=1.9,       # customer steelmaking
                carbon_price_coverage=0.7,
                free_allocation=0.10,
            ),
            lat=-22.3, lon=118.6,          # Pilbara, WA — near Newman
            equipment_type="open_pit_mine",
        ),
        Asset(
            id="tc_copper_cl",
            name="TestCo Chilean Copper JV",
            commodity=Commodity.COPPER,
            region="CL-02",
            baseline_production=0.40,      # Mtpa Cu
            production_unit="Mtonnes",
            baseline_unit_cost=3_800.0,
            energy_cost_share=0.30,
            carrying_value=6_500.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=2.5,       # tCO2 per tonne Cu
                scope2_intensity=3.5,
                scope3_intensity=1.0,
                carbon_price_coverage=0.6,
                free_allocation=0.0,
            ),
            lat=-23.5, lon=-68.0,          # Atacama Desert, Region II, Chile
            equipment_type="open_pit_mine",
        ),
        Asset(
            id="tc_aluminium_ca",
            name="TestCo Aluminium Smelter (Canada)",
            commodity=Commodity.ALUMINIUM,
            region="CA-QC",
            baseline_production=1.0,       # Mtpa Al
            production_unit="Mtonnes",
            baseline_unit_cost=1_600.0,
            energy_cost_share=0.45,
            carrying_value=5_000.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=2.0,
                scope2_intensity=0.5,        # hydro-powered
                scope3_intensity=0.8,
                carbon_price_coverage=0.8,
                free_allocation=0.2,
            ),
            lat=48.4, lon=-71.1,           # Saguenay–Lac-Saint-Jean, Québec
            equipment_type="aluminium_smelter",
        ),
    ],
    exposure_weight=0.4,
    transition_weight=0.3,
    data_quality="high",
)


# ---------------------------------------------------------------------------
# Shell plc — Integrated Oil & Gas
# ---------------------------------------------------------------------------

SHELL = Company(
    id="shell",
    name="Shell plc",
    sector="Oil & Gas",
    hq_region="NL-NH",  # Netherlands (Hague)
    financials=Financials(
        revenue=380_000.0,         # USD millions, ~2024 baseline
        ebitda=65_000.0,
        capex=12_000.0,
        maintenance_capex_share=0.55,
        tax_rate=0.28,
        wacc_base=0.075,
        net_debt=8_000.0,
        shares_outstanding=1_670.0,  # millions
        market_cap=175_000.0,
    ),
    assets=[
        # Permian Basin (USA) - integrated crude oil & condensate production
        Asset(
            id="shell_permian",
            name="Shell Permian Basin (USA)",
            commodity=Commodity.CRUDE_OIL,
            region="US-TX",
            baseline_production=1500.0,    # Mtonnes crude oil equiv / year
            production_unit="Mtonnes",
            baseline_unit_cost=35.0,       # USD/tonne all-in cost
            energy_cost_share=0.25,
            carrying_value=18_000.0,
            remaining_life_years=20,
            emissions=EmissionsProfile(
                scope1_intensity=0.07,      # tCO2e/tonne
                scope2_intensity=0.02,
                scope3_intensity=0.45,      # combustion
                carbon_price_coverage=0.15,
                free_allocation=0.0,
            ),
            lat=31.8, lon=-102.5,          # Midland Basin, West Texas
            equipment_type="oil_well",
        ),
        # LNG Australia - Prelude FLNG + onshore projects
        Asset(
            id="shell_lng_au",
            name="Shell LNG Australia",
            commodity=Commodity.NATURAL_GAS,
            region="AU-WA",
            baseline_production=1000.0,    # Mtonnes LNG equiv
            production_unit="Mtonnes",
            baseline_unit_cost=2.0,        # USD/tonne (scaled for margin)
            energy_cost_share=0.30,
            carrying_value=25_000.0,
            remaining_life_years=22,
            emissions=EmissionsProfile(
                scope1_intensity=0.25,      # tCO2e/tonne
                scope2_intensity=0.05,
                scope3_intensity=2.1,       # combustion at customer
                carbon_price_coverage=0.10,
                free_allocation=0.0,
            ),
            lat=-14.3, lon=127.1,          # Browse Basin / Prelude FLNG, WA offshore
            equipment_type="lng_terminal",
        ),
    ],
    exposure_weight=0.35,
    transition_weight=0.40,
    data_quality="high",
)


# ---------------------------------------------------------------------------
# BHP Group — Diversified Mining
# ---------------------------------------------------------------------------

BHP = Company(
    id="bhp",
    name="BHP Group Limited",
    sector="Mining",
    hq_region="AU-WA",  # Western Australia (Perth)
    financials=Financials(
        revenue=62_500.0,          # USD millions, ~2024 baseline
        ebitda=28_000.0,
        capex=8_500.0,
        maintenance_capex_share=0.60,
        tax_rate=0.30,
        wacc_base=0.082,
        net_debt=5_000.0,
        shares_outstanding=2_400.0,  # millions
        market_cap=150_000.0,
    ),
    assets=[
        # Pilbara Iron Ore (Australia) - largest asset
        Asset(
            id="bhp_pilbara",
            name="BHP Pilbara Iron Ore",
            commodity=Commodity.IRON_ORE,
            region="AU-WA",
            baseline_production=287.0,     # Mtpa
            production_unit="Mtonnes",
            baseline_unit_cost=18.5,       # USD/tonne
            energy_cost_share=0.32,
            carrying_value=28_000.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=0.012,     # tCO2e/tonne (low-cost)
                scope2_intensity=0.008,
                scope3_intensity=1.95,      # steelmaking downstream
                carbon_price_coverage=0.65,
                free_allocation=0.08,
            ),
            lat=-22.3, lon=118.6,          # Mount Whaleback / Newman area, Pilbara WA
            equipment_type="open_pit_mine",
        ),
        # Olympic Dam (Australia) - copper & uranium
        Asset(
            id="bhp_olympic_dam",
            name="BHP Olympic Dam (Copper/Uranium)",
            commodity=Commodity.COPPER,
            region="AU-SA",
            baseline_production=0.20,      # Mtpa Cu (incl. uranium)
            production_unit="Mtonnes",
            baseline_unit_cost=2_200.0,    # USD/tonne Cu
            energy_cost_share=0.35,
            carrying_value=8_500.0,
            remaining_life_years=35,
            emissions=EmissionsProfile(
                scope1_intensity=2.1,       # tCO2e/tonne Cu
                scope2_intensity=2.8,       # high energy input
                scope3_intensity=0.8,
                carbon_price_coverage=0.60,
                free_allocation=0.05,
            ),
            lat=-30.44, lon=136.88,        # Olympic Dam, Roxby Downs, South Australia
            equipment_type="underground_mine",
        ),
        # Queensland Coal (Australia) - metallurgical + thermal
        Asset(
            id="bhp_qld_coal",
            name="BHP Queensland Coal",
            commodity=Commodity.COAL_METALLURGICAL,
            region="AU-QLD",
            baseline_production=50.0,      # Mtpa (coking coal primary)
            production_unit="Mtonnes",
            baseline_unit_cost=150.0,      # USD/tonne
            energy_cost_share=0.25,
            carrying_value=5_500.0,
            remaining_life_years=15,
            emissions=EmissionsProfile(
                scope1_intensity=0.045,     # tCO2e/tonne
                scope2_intensity=0.020,
                scope3_intensity=1.4,       # combustion
                carbon_price_coverage=0.40,
                free_allocation=0.0,
            ),
            lat=-22.0, lon=148.3,          # Bowen Basin, Queensland (Blackwater/Goonyella)
            equipment_type="open_pit_mine",
        ),
        # Nickel West (Australia) - emerging battery metals
        Asset(
            id="bhp_nickel",
            name="BHP Nickel West",
            commodity=Commodity.COPPER,     # proxy for battery metals
            region="AU-WA",
            baseline_production=0.085,     # Mtpa Ni
            production_unit="Mtonnes",
            baseline_unit_cost=8_500.0,    # USD/tonne
            energy_cost_share=0.38,
            carrying_value=3_200.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=5.5,       # tCO2e/tonne Ni
                scope2_intensity=2.2,
                scope3_intensity=0.5,
                carbon_price_coverage=0.60,
                free_allocation=0.06,
            ),
            lat=-27.9, lon=121.7,          # Kambalda / Kalgoorlie–Boulder, WA
            equipment_type="processing_plant",
        ),
    ],
    exposure_weight=0.30,
    transition_weight=0.25,
    data_quality="high",
)


# ---------------------------------------------------------------------------
# Rio Tinto Limited — Diversified Mining & Minerals
# ---------------------------------------------------------------------------

RIO_TINTO = Company(
    id="rio_tinto",
    name="Rio Tinto Limited",
    sector="Mining",
    hq_region="GB-ENG",  # London (UK)
    financials=Financials(
        revenue=53_800.0,          # USD millions, ~2024 baseline
        ebitda=24_000.0,
        capex=7_200.0,
        maintenance_capex_share=0.58,
        tax_rate=0.28,
        wacc_base=0.082,
        net_debt=2_500.0,
        shares_outstanding=1_980.0,  # millions
        market_cap=110_000.0,
    ),
    assets=[
        # Pilbara Iron Ore (Australia) - JV with BHP but Rio has significant stake
        Asset(
            id="rio_pilbara",
            name="Rio Tinto Pilbara Iron Ore",
            commodity=Commodity.IRON_ORE,
            region="AU-WA",
            baseline_production=96.0,      # Mtpa (Rio's share/attributable)
            production_unit="Mtonnes",
            baseline_unit_cost=17.0,       # USD/tonne (efficient operations)
            energy_cost_share=0.30,
            carrying_value=9_800.0,
            remaining_life_years=28,
            emissions=EmissionsProfile(
                scope1_intensity=0.013,     # tCO2e/tonne
                scope2_intensity=0.007,
                scope3_intensity=1.92,
                carbon_price_coverage=0.65,
                free_allocation=0.08,
            ),
            lat=-22.5, lon=117.8,          # Tom Price / Paraburdoo area, Pilbara WA
            equipment_type="open_pit_mine",
        ),
        # Oyu Tolgoi (Mongolia) - copper-gold mine
        Asset(
            id="rio_oyu_tolgoi",
            name="Rio Tinto Oyu Tolgoi (Copper-Gold)",
            commodity=Commodity.COPPER,
            region="MN-01",
            baseline_production=0.45,      # Mtpa Cu equiv
            production_unit="Mtonnes",
            baseline_unit_cost=2_100.0,    # USD/tonne Cu
            energy_cost_share=0.42,        # high energy for processing
            carrying_value=11_200.0,
            remaining_life_years=40,
            emissions=EmissionsProfile(
                scope1_intensity=2.3,       # tCO2e/tonne Cu
                scope2_intensity=3.1,
                scope3_intensity=0.7,
                carbon_price_coverage=0.25,  # limited carbon market coverage
                free_allocation=0.0,
            ),
            lat=43.0, lon=106.8,           # Omnogovi Province, southern Mongolia
            equipment_type="underground_mine",
        ),
        # Aluminium (Pacific) - smelters & refineries
        Asset(
            id="rio_aluminium",
            name="Rio Tinto Aluminium (Pacific)",
            commodity=Commodity.ALUMINIUM,
            region="AU-QLD",
            baseline_production=3.4,       # Mtpa Al (refining + smelting)
            production_unit="Mtonnes",
            baseline_unit_cost=1_850.0,    # USD/tonne
            energy_cost_share=0.48,        # energy-intensive smelting
            carrying_value=6_500.0,
            remaining_life_years=22,
            emissions=EmissionsProfile(
                scope1_intensity=1.2,       # tCO2e/tonne (improving via renewables)
                scope2_intensity=4.8,       # grid electricity share
                scope3_intensity=0.9,
                carbon_price_coverage=0.75,
                free_allocation=0.15,       # some renewable power PPAs
            ),
            lat=-23.9, lon=151.4,          # Boyne Island Smelter, Gladstone QLD
            equipment_type="aluminium_smelter",
        ),
        # Diamonds & Minerals (Argyle, Diavik, etc.)
        Asset(
            id="rio_diamonds",
            name="Rio Tinto Diamonds & Minerals",
            commodity=Commodity.IRON_ORE,  # proxy commodity (other minerals)
            region="AU-WA",
            baseline_production=12.0,      # Mtpa equivalent
            production_unit="Mtonnes",
            baseline_unit_cost=280.0,      # USD/tonne (high value, low volume)
            energy_cost_share=0.26,
            carrying_value=3_100.0,
            remaining_life_years=18,
            emissions=EmissionsProfile(
                scope1_intensity=0.25,      # tCO2e/tonne
                scope2_intensity=0.15,
                scope3_intensity=0.05,
                carbon_price_coverage=0.60,
                free_allocation=0.05,
            ),
            lat=-16.7, lon=128.4,          # Argyle Diamond Mine, East Kimberley WA
            equipment_type="open_pit_mine",
        ),
    ],
    exposure_weight=0.28,
    transition_weight=0.22,
    data_quality="high",
)


def all_seed() -> dict[str, Company]:
    return {c.id: c for c in [CRI_TEST_CO, SHELL, BHP, RIO_TINTO]}


def get(company_id: str) -> Company:
    seed = all_seed()
    if company_id not in seed:
        raise KeyError(f"Unknown company {company_id!r}. Available: {list(seed)}")
    return seed[company_id]
