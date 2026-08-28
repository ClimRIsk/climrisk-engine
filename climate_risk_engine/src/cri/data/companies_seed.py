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


# ---------------------------------------------------------------------------
# Heineken N.V. — Global Brewing / Beverages
# Sources: Heineken Annual Report 2024, CDP Climate 2024, company sustainability report
# ---------------------------------------------------------------------------

HEINEKEN = Company(
    id="heineken",
    name="Heineken N.V.",
    sector="Beverages",
    hq_region="NL-NH",  # Amsterdam, Netherlands
    financials=Financials(
        revenue=34_500.0,          # USD millions (EUR 31.7bn @ 1.09 FX, FY2024)
        ebitda=7_200.0,            # beia EBITDA margin ~20.9%
        capex=2_200.0,
        maintenance_capex_share=0.50,
        tax_rate=0.26,
        wacc_base=0.078,
        net_debt=9_800.0,
        shares_outstanding=576.0,  # millions
        market_cap=22_000.0,
    ),
    assets=[
        # Heineken Netherlands — flagship breweries (Zoeterwoude, 's-Hertogenbosch)
        # Production units calibrated so baseline_production × beverages_price ≈ asset revenue (USD M)
        # Beverages price base 2026 = 100 USD/unit. NL revenue target ~$8,000M → 80 units.
        Asset(
            id="heineken_nl_brewery",
            name="Heineken Netherlands Breweries",
            commodity=Commodity.BEVERAGES,
            region="NL-ZH",
            baseline_production=80.0,      # engine units (≈ 20M hl, scaled: 8,000 USD M / $100/unit)
            production_unit="engine_units",
            baseline_unit_cost=62.0,        # USD/unit (38% margin vs price=100)
            energy_cost_share=0.18,
            carrying_value=2_800.0,
            remaining_life_years=35,
            emissions=EmissionsProfile(
                scope1_intensity=0.006,     # tCO2e/hl (thermal, same scale as Shell)
                scope2_intensity=0.002,     # near-zero via renewable PPA
                scope3_intensity=0.048,     # packaging + agriculture upstream
                carbon_price_coverage=0.85, # EU ETS
                free_allocation=0.10,
            ),
            lat=52.0, lon=4.4,              # Zoeterwoude brewery complex
            equipment_type="brewery",
        ),
        # Heineken Vietnam — Tiger, Heineken brand; fast-growing market
        Asset(
            id="heineken_vn_brewery",
            name="Heineken Vietnam Breweries",
            commodity=Commodity.BEVERAGES,
            region="VN-SG",
            baseline_production=40.0,
            production_unit="engine_units",
            baseline_unit_cost=64.0,
            energy_cost_share=0.22,
            carrying_value=1_200.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=0.010,     # tCO2e/hl (thermal)
                scope2_intensity=0.025,     # coal-heavy grid
                scope3_intensity=0.055,
                carbon_price_coverage=0.05,
                free_allocation=0.0,
            ),
            lat=10.8, lon=106.6,            # Ho Chi Minh City region
            equipment_type="brewery",
        ),
        # Heineken Nigeria — Nigerian Breweries; high growth, high heat/water exposure
        Asset(
            id="heineken_ng_brewery",
            name="Heineken Nigeria (Nigerian Breweries)",
            commodity=Commodity.BEVERAGES,
            region="NG-LA",
            baseline_production=20.0,
            production_unit="engine_units",
            baseline_unit_cost=68.0,
            energy_cost_share=0.30,
            carrying_value=900.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=0.018,     # tCO2e/hl (diesel + LNG backup)
                scope2_intensity=0.022,
                scope3_intensity=0.060,
                carbon_price_coverage=0.0,
                free_allocation=0.0,
            ),
            lat=6.5, lon=3.4,               # Lagos / Ibadan corridor
            equipment_type="brewery",
        ),
        # Heineken Brasil — Devassa, Eisenbahn brands; water-intensive region
        Asset(
            id="heineken_br_brewery",
            name="Heineken Brasil Breweries",
            commodity=Commodity.BEVERAGES,
            region="BR-SP",
            baseline_production=55.0,
            production_unit="engine_units",
            baseline_unit_cost=60.0,
            energy_cost_share=0.16,
            carrying_value=1_600.0,
            remaining_life_years=28,
            emissions=EmissionsProfile(
                scope1_intensity=0.008,     # tCO2e/hl
                scope2_intensity=0.004,     # low-carbon hydro grid
                scope3_intensity=0.052,
                carbon_price_coverage=0.0,
                free_allocation=0.0,
            ),
            lat=-23.5, lon=-46.6,           # São Paulo metro, major hub
            equipment_type="brewery",
        ),
    ],
    exposure_weight=0.35,    # beverage sector — water & heat are dominant hazards
    transition_weight=0.20,  # lower direct carbon exposure vs. O&G/mining
    data_quality="high",
)


# ---------------------------------------------------------------------------
# Nouryon — Specialty Chemicals
# Sources: Nouryon Annual Report 2024, CDP 2024, company ESG data
# ---------------------------------------------------------------------------

NOURYON = Company(
    id="nouryon",
    name="Nouryon",
    sector="Chemicals",
    hq_region="NL-NH",  # Amsterdam, Netherlands
    financials=Financials(
        revenue=5_900.0,           # USD millions (FY2024 estimate; private company)
        ebitda=1_380.0,            # EBITDA margin ~23%
        capex=420.0,
        maintenance_capex_share=0.60,
        tax_rate=0.27,
        wacc_base=0.085,
        net_debt=2_800.0,
        shares_outstanding=0.0,    # private (Carlyle / PGGM ownership)
        market_cap=5_500.0,        # estimated EV
    ),
    assets=[
        # Delfzijl / Rotterdam — surfactants, salt electrolysis (chlorine)
        # Chemicals price base 2026 = 1200 USD/unit. NL revenue target ~$2,500M → 2.08 units.
        # baseline_unit_cost must be < 1200 to have positive margin.
        Asset(
            id="nouryon_nl_chemicals",
            name="Nouryon Netherlands Chemical Complexes",
            commodity=Commodity.CHEMICALS,
            region="NL-GR",
            baseline_production=2.08,      # engine units (≈ 1,200 ktonnes, scaled: $2,500M / $1,200/unit)
            production_unit="engine_units",
            baseline_unit_cost=820.0,      # USD/unit (~32% EBITDA margin vs price=1200)
            energy_cost_share=0.38,
            carrying_value=1_800.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=1.1,       # tCO2e/tonne product (electrolysis)
                scope2_intensity=0.9,
                scope3_intensity=0.4,
                carbon_price_coverage=0.90,
                free_allocation=0.20,
            ),
            lat=53.3, lon=6.9,
            equipment_type="chemical_plant",
        ),
        # Sweden (Stenungsund) — ethylene oxide, surfactants
        Asset(
            id="nouryon_se_chemicals",
            name="Nouryon Sweden (Stenungsund)",
            commodity=Commodity.CHEMICALS,
            region="SE-O",
            baseline_production=0.71,
            production_unit="engine_units",
            baseline_unit_cost=780.0,
            energy_cost_share=0.32,
            carrying_value=600.0,
            remaining_life_years=22,
            emissions=EmissionsProfile(
                scope1_intensity=0.8,
                scope2_intensity=0.1,       # very low-carbon Nordic grid
                scope3_intensity=0.35,
                carbon_price_coverage=0.85,
                free_allocation=0.15,
            ),
            lat=58.1, lon=11.8,
            equipment_type="chemical_plant",
        ),
        # USA (Deer Park, TX) — pulp & performance chemicals
        Asset(
            id="nouryon_us_chemicals",
            name="Nouryon USA (Deer Park / Marietta)",
            commodity=Commodity.CHEMICALS,
            region="US-TX",
            baseline_production=1.25,
            production_unit="engine_units",
            baseline_unit_cost=800.0,
            energy_cost_share=0.28,
            carrying_value=700.0,
            remaining_life_years=20,
            emissions=EmissionsProfile(
                scope1_intensity=1.3,
                scope2_intensity=0.4,
                scope3_intensity=0.5,
                carbon_price_coverage=0.0,
                free_allocation=0.0,
            ),
            lat=29.7, lon=-95.1,
            equipment_type="chemical_plant",
        ),
    ],
    exposure_weight=0.30,
    transition_weight=0.35,  # electrolysis → green hydrogen; EU ETS exposure
    data_quality="high",
)


# ---------------------------------------------------------------------------
# UltraTech Cement Limited — Cement (Aditya Birla Group)
# Sources: UltraTech Annual Report FY2024, CDP 2024
# ---------------------------------------------------------------------------

ULTRATECH = Company(
    id="ultratech",
    name="UltraTech Cement Limited",
    sector="Cement",
    hq_region="IN-MH",  # Mumbai, Maharashtra, India
    financials=Financials(
        revenue=8_800.0,           # USD millions (INR 73,800 Cr @ 84 INR/USD, FY2024)
        ebitda=2_100.0,            # EBITDA margin ~24%
        capex=1_200.0,
        maintenance_capex_share=0.55,
        tax_rate=0.25,
        wacc_base=0.092,           # higher WACC for India EM
        net_debt=2_200.0,
        shares_outstanding=288.0,  # millions
        market_cap=38_000.0,
    ),
    assets=[
        # Cement price base 2026 = 80 USD/unit (Mtonnes). Production calibrated:
        # target_asset_revenue_USD_M / 80 = production units.
        # Rajasthan / Gujarat — largest integrated cement plants
        # Revenue target ~$3,040M → 38 Mtonnes at $80/unit
        Asset(
            id="ultratech_raj_cement",
            name="UltraTech Rajasthan Integrated Plants",
            commodity=Commodity.CEMENT,
            region="IN-RJ",
            baseline_production=38.0,      # engine units (Mtonnes: $3,040M / $80/unit)
            production_unit="Mtonnes",
            baseline_unit_cost=52.0,       # USD/Mtonne; margin=35% vs price=80
            energy_cost_share=0.40,
            carrying_value=2_500.0,
            remaining_life_years=30,
            emissions=EmissionsProfile(
                scope1_intensity=0.52,      # tCO2e/tonne cement (clinker process)
                scope2_intensity=0.06,
                scope3_intensity=0.02,
                carbon_price_coverage=0.0,  # India no formal ETS yet
                free_allocation=0.0,
            ),
            lat=25.5, lon=73.0,
            equipment_type="cement_kiln",
        ),
        # Maharashtra / Karnataka — South & West India plants
        Asset(
            id="ultratech_mh_cement",
            name="UltraTech Maharashtra & Karnataka Plants",
            commodity=Commodity.CEMENT,
            region="IN-MH",
            baseline_production=28.0,
            production_unit="Mtonnes",
            baseline_unit_cost=54.0,
            energy_cost_share=0.38,
            carrying_value=1_800.0,
            remaining_life_years=28,
            emissions=EmissionsProfile(
                scope1_intensity=0.54,
                scope2_intensity=0.05,
                scope3_intensity=0.02,
                carbon_price_coverage=0.0,
                free_allocation=0.0,
            ),
            lat=17.3, lon=76.8,
            equipment_type="cement_kiln",
        ),
        # Madhya Pradesh & Chhattisgarh — coal-belt integrated plants
        Asset(
            id="ultratech_mp_cement",
            name="UltraTech MP & Chhattisgarh Plants",
            commodity=Commodity.CEMENT,
            region="IN-MP",
            baseline_production=22.0,
            production_unit="Mtonnes",
            baseline_unit_cost=50.0,
            energy_cost_share=0.42,
            carrying_value=1_200.0,
            remaining_life_years=25,
            emissions=EmissionsProfile(
                scope1_intensity=0.56,
                scope2_intensity=0.07,
                scope3_intensity=0.02,
                carbon_price_coverage=0.0,
                free_allocation=0.0,
            ),
            lat=22.7, lon=81.0,
            equipment_type="cement_kiln",
        ),
        # UAE / Bahrain — international operations
        Asset(
            id="ultratech_uae_cement",
            name="UltraTech International (UAE / Bahrain)",
            commodity=Commodity.CEMENT,
            region="AE-DU",
            baseline_production=5.0,
            production_unit="Mtonnes",
            baseline_unit_cost=60.0,
            energy_cost_share=0.35,
            carrying_value=500.0,
            remaining_life_years=20,
            emissions=EmissionsProfile(
                scope1_intensity=0.50,
                scope2_intensity=0.04,
                scope3_intensity=0.02,
                carbon_price_coverage=0.05,
                free_allocation=0.0,
            ),
            lat=24.5, lon=54.4,
            equipment_type="cement_kiln",
        ),
    ],
    exposure_weight=0.40,    # high physical exposure — India heat + water stress
    transition_weight=0.35,  # cement is ~8% of global CO2; stranded asset risk
    data_quality="high",
)


def all_seed() -> dict[str, Company]:
    return {c.id: c for c in [CRI_TEST_CO, SHELL, BHP, RIO_TINTO, HEINEKEN, NOURYON, ULTRATECH]}


def get(company_id: str) -> Company:
    seed = all_seed()
    if company_id not in seed:
        raise KeyError(f"Unknown company {company_id!r}. Available: {list(seed)}")
    return seed[company_id]
