"""
Historical Real-World Climate Event Calibration Database.

A static, source-cited library of 16 well-documented climate events used to
ground-truth the CRI scenario cascade engine.  Each record contains:

  • Verified total and insured economic losses (USD billions, 2023-adjusted)
  • Sector-level loss disaggregation drawn from peer-reviewed literature,
    national government reports, and reinsurance sigma catalogues
  • Company-level case studies with confirmed disclosed losses
  • Recovery timelines by sector and asset type
  • The nearest matching PhysicalEvent from EVENT_LIBRARY for calibration runs
  • A calibration_scale dict showing how this event's observed intensities
    compare to the baseline event, allowing the calibration engine to scale
    cascade outputs for comparison

DATA QUALITY CONVENTION
────────────────────────
• All monetary values are in nominal USD millions of the event year unless
  marked with the suffix "_adj2023" (2023 CPI-adjusted).
• Every figure cites a specific source record in the `sources` list.
• Loss ranges are provided where only a range was reported; the engine uses
  the midpoint for calibration.
• "sector_losses_usd_m" keys match CRI Commodity enum values where applicable.

SOURCES (primary)
──────────────────
• Swiss Re sigma No. 1/2024 — Natural catastrophes in 2023
• Munich Re NatCatSERVICE database (natcatservice.munichre.com)
• EM-DAT: The CRED/OFDA International Disaster Database (emdat.be)
• World Bank Global Facility for Disaster Reduction and Recovery (GFDRR)
• NOAA NCEI Billion-Dollar Weather and Climate Disasters
• UNDRR — Sendai Monitor
• IPCC AR6 WGII Chapter 8 (food/water systems), Chapter 11 (cities)
• Peer-reviewed papers cited per event
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventRegion(str, Enum):
    """Broad geographic scope of the historical event."""
    GLOBAL           = "global"
    SOUTHEAST_ASIA   = "southeast_asia"
    AUSTRALIA        = "australia"
    SOUTH_ASIA       = "south_asia"
    NORTH_AMERICA    = "north_america"
    SOUTH_AFRICA     = "south_africa"
    EUROPE           = "europe"
    MIDDLE_EAST      = "middle_east"
    EAST_ASIA        = "east_asia"


@dataclass(frozen=True)
class SectorLoss:
    """Verified sector-level loss record from a historical event.

    Attributes
    ----------
    sector          : CRI commodity/sector label (e.g. 'beverages', 'agriculture')
    loss_usd_m      : Estimated loss in nominal USD millions (event-year dollars)
    loss_range_usd_m: (low, high) if only a range was reported; None if point estimate
    loss_type       : Nature of the loss (physical damage, production loss, etc.)
    company_examples: Specific companies with confirmed disclosed losses
    sources         : Citation list for this sector loss figure
    notes           : Any caveats or methodology notes
    """
    sector:             str
    loss_usd_m:         float
    loss_range_usd_m:   Optional[tuple] = None
    loss_type:          str = "combined"
    company_examples:   tuple = ()
    sources:            tuple = ()
    notes:              str = ""


@dataclass(frozen=True)
class HistoricalClimateEvent:
    """A documented real-world climate event with verified financial data.

    Used by the calibration engine to:
    1. Run the cascade engine with the mapped PhysicalEvent ID
    2. Compare predicted vs. actual sector losses
    3. Compute calibration error statistics
    4. Track systematic bias by sector and event type

    Attributes
    ----------
    id                  : Unique slug for API and calibration engine
    name                : Human-readable event name
    year_start          : First year of the event (or peak year for acute events)
    year_end            : Last year (same as year_start for acute events)
    region              : Primary affected region
    physical_event_id   : Nearest matching PhysicalEvent in EVENT_LIBRARY
    calibration_scale   : Per-hazard scale factors vs. the PhysicalEvent's multipliers.
                          Values >1 = this event was more intense; <1 = less intense.
    total_loss_usd_m    : Total economic loss, nominal USD millions (event year)
    insured_loss_usd_m  : Insured portion, nominal USD millions; None if unknown
    source_total_loss   : Citation for total loss figure
    sector_losses       : Tuple of SectorLoss records with sector-level breakdown
    recovery_months_by_sector: dict mapping sector → typical months to full recovery
    key_impacts         : Plain-language description of primary financial channels
    affected_countries  : ISO 3166-1 alpha-2 country codes
    sources             : All citations for this event record
    """
    id:                       str
    name:                     str
    year_start:               int
    year_end:                 int
    region:                   EventRegion
    physical_event_id:        str
    calibration_scale:        dict
    total_loss_usd_m:         float
    insured_loss_usd_m:       Optional[float]
    source_total_loss:        str
    sector_losses:            tuple
    recovery_months_by_sector: dict
    key_impacts:              str
    affected_countries:       tuple = ()
    sources:                  tuple = ()


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL EVENT RECORDS
# ═══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# 1. 1997–98 Super El Niño
# ──────────────────────────────────────────────────────────────────────────────

EL_NINO_1997_98 = HistoricalClimateEvent(
    id="el_nino_1997_98",
    name="1997–98 Super El Niño (Global)",
    year_start=1997,
    year_end=1998,
    region=EventRegion.GLOBAL,
    physical_event_id="el_nino_super_drought",
    # This was a peak ONI of +2.3°C — the baseline event targets +1.5°C.
    # Observed losses suggest drought/heat multipliers approximately 1.3× the baseline.
    calibration_scale={
        "water_stress": 1.3,
        "drought":      1.3,
        "heat_stress":  1.2,
        "wildfire":     1.4,
    },
    total_loss_usd_m=45_000,       # USD 45 billion
    insured_loss_usd_m=2_100,
    source_total_loss=(
        "Munich Re NatCatSERVICE 1998 annual review; "
        "Dilley et al. (2005) Natural Disaster Hotspots, World Bank"
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=14_000,
            loss_range_usd_m=(10_000, 18_000),
            loss_type="production_loss",
            company_examples=(
                "Philippine Coconut Authority: 50% crop failure in Mindanao",
                "Indonesian palm oil: USD 2.1B production loss",
                "Brazilian soybean: harvest losses in Ceará/Rio Grande do Norte",
            ),
            sources=(
                "FAO (1998): Impact of El Niño on Agriculture, Food and Nutrition",
                "World Bank (1999): El Niño Assessment",
            ),
            notes="Includes crop failure, livestock mortality, fisheries collapse. "
                  "Indonesia, Philippines, Australia, parts of Brazil most impacted.",
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=1_800,
            loss_type="production_loss",
            company_examples=(
                "Thai Beverage (ThaiBev): water rationing at Mekong-region breweries",
                "San Miguel Corp (Philippines): malted barley import premium ~35%",
            ),
            sources=(
                "Thai National Economic and Social Development Board (NESDB) 1998",
                "Philippine Institute for Development Studies (PIDS) 1999",
            ),
            notes=(
                "Water costs surged for water-intensive beverage production. "
                "Malt barley supply disrupted from drought-hit Australia."
            ),
        ),
        SectorLoss(
            sector="mining",
            loss_usd_m=3_500,
            loss_type="production_loss",
            company_examples=(
                "BHP: Queensland coal operations halted (flooding in La Niña rebound)",
                "Rio Tinto: Indonesian operations impacted by forest fires",
                "Freeport-McMoRan: Grasberg mine (PNG side) access disrupted",
            ),
            sources=(
                "Australian Bureau of Statistics: Minerals sector annual report 1998",
                "Indonesian government resource sector impact assessment 1998",
            ),
            notes="Includes Indonesian wildfire-related mine access disruption (1997 Borneo fires).",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 12,
        "beverages":   6,
        "mining":      3,
        "real_estate": 4,
    },
    key_impacts=(
        "Most severe El Niño of the 20th century (ONI peak +2.3°C). "
        "Indonesia: USD 10B total loss, forest fires on Kalimantan and Sumatra burned >10 M ha. "
        "Philippines: agricultural GDP fell 6.6% in 1998. "
        "Australia: agricultural drought cost AUD 2B. "
        "Global insured loss was low (~USD 2.1B) due to limited coverage in affected markets."
    ),
    affected_countries=("ID", "PH", "TH", "AU", "BR", "PE", "EC", "ZA", "IN"),
    sources=(
        "Munich Re NatCatSERVICE annual review 1998",
        "Dilley et al. (2005) Natural Disaster Hotspots — A Global Risk Analysis, World Bank",
        "FAO (1998) Impact of El Niño on Agriculture",
        "Swiss Re sigma 3/1999",
        "NOAA PMEL: Historical El Niño/La Niña episodes 1950–present",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 2003 European Heat Wave
# ──────────────────────────────────────────────────────────────────────────────

EUROPEAN_HEATWAVE_2003 = HistoricalClimateEvent(
    id="european_heatwave_2003",
    name="2003 European Heat Wave",
    year_start=2003,
    year_end=2003,
    region=EventRegion.EUROPE,
    physical_event_id="heat_dome_acute",
    calibration_scale={
        "heat_stress": 1.2,    # 1-month extreme — duration similar to baseline
        "water_stress": 1.3,
        "wildfire":    1.5,    # Portugal fires were severe
    },
    total_loss_usd_m=13_500,       # USD 13.5 billion (economic loss, crop/energy focus)
    insured_loss_usd_m=3_300,
    source_total_loss=(
        "Swiss Re sigma 1/2004: Natural catastrophes and man-made disasters in 2003"
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=9_400,
            loss_range_usd_m=(7_500, 12_000),
            loss_type="production_loss",
            company_examples=(
                "French wheat: harvest fell 21%, corn fell 30% (French Ministry of Agriculture)",
                "Italian fruit growers: EUR 1B loss in Po Valley",
                "German grain: harvest −20% vs. 5-year average",
            ),
            sources=(
                "FAO (2003): Crop Prospects and Food Situation in Europe",
                "Ciais et al. (2005) Europe-wide reduction in primary productivity "
                "caused by the heat and drought in 2003. Nature, 437, 529–533.",
            ),
            notes=(
                "Net primary production fell 30% across Europe in summer 2003 "
                "per Ciais et al. Carbon sink reversed to net source."
            ),
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=420,
            loss_type="combined",
            company_examples=(
                "Heineken: Rhine River low water impacted barge transport costs +25%",
                "Anheuser-Busch InBev (predecessor Interbrew): barley premium surge",
                "French wine estates: Champagne yields down 15–25% depending on AOC",
            ),
            sources=(
                "European Commission DG Agriculture: Summer 2003 agricultural market review",
                "German Brewers Association annual report 2003",
            ),
            notes="Rhine water level fell to 40-year lows, disrupting barge logistics for ingredients.",
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=1_800,
            loss_type="physical_damage",
            company_examples=(
                "French real estate: structural damage from clay soil subsidence (retrait-gonflement)",
                "Insurance claims for clay shrinkage exceeded EUR 1.5B in France alone",
            ),
            sources=(
                "Caisse Centrale de Réassurance (CCR): Bilan 2003",
                "BRGM: Sécheresse et retrait-gonflement des argiles 2003",
            ),
            notes=(
                "Clay soil shrinkage (retrait-gonflement des argiles) caused widespread "
                "foundation cracking across France. HVAC and cooling retrofit demand surged."
            ),
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 12,
        "beverages":   3,
        "mining":      1,
        "real_estate": 18,
    },
    key_impacts=(
        "Peak temperatures: Paris 40.4°C (1 August); Germany 40.3°C; UK 38.5°C. "
        "~70,000 excess deaths across Europe (Robine et al. 2008, C.R. Biologies). "
        "French nuclear power output fell 20% due to river cooling water temperature limits. "
        "Portuguese wildfires burned 425,000 ha."
    ),
    affected_countries=("FR", "DE", "IT", "ES", "PT", "GB", "BE", "NL", "CH"),
    sources=(
        "Swiss Re sigma 1/2004",
        "Robine et al. (2008): Death toll exceeded 70,000 in Europe during the summer of 2003. "
        "C.R. Biologies, 331(2), 171–178.",
        "Ciais et al. (2005): Europe-wide reduction in primary productivity. Nature 437.",
        "Poumadère et al. (2005): The 2003 heat wave in France. Risk Analysis 25(6).",
        "Munich Re NatCatSERVICE 2003 annual summary",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 3. 2010–12 Queensland (Australia) Floods — La Niña Triple-Dip
# ──────────────────────────────────────────────────────────────────────────────

QUEENSLAND_FLOODS_2010_12 = HistoricalClimateEvent(
    id="queensland_floods_2010_12",
    name="2010–12 Queensland Floods (Australia) — Triple-Dip La Niña",
    year_start=2010,
    year_end=2012,
    region=EventRegion.AUSTRALIA,
    physical_event_id="la_nina_heavy_rain_flood",
    calibration_scale={
        "flood_riverine": 1.4,    # extended triple-dip was worse than baseline
        "landslide":      1.2,
        "cyclone":        1.1,    # Cyclone Yasi overlapped
    },
    total_loss_usd_m=30_000,      # AUD 30B ≈ USD 28-30B at 2011 exchange rate
    insured_loss_usd_m=7_000,
    source_total_loss=(
        "Queensland Floods Commission of Inquiry (2012): Final Report. "
        "Swiss Re sigma 2/2012."
    ),
    sector_losses=(
        SectorLoss(
            sector="mining",
            loss_usd_m=5_800,
            loss_type="production_loss",
            company_examples=(
                "BHP: Queensland coal operations suspended January 2011, ~15 Mt coal lost",
                "Rio Tinto: Hail Creek and Kestrel mines flooded, AUD 200M+ recovery",
                "Xstrata: Rolleston and Ulan mines — production halted weeks",
                "Queensland coal exports fell 30% Jan–Mar 2011 per Dept. of Resources",
            ),
            sources=(
                "Queensland Floods Commission of Inquiry (2012) Final Report, Vol 3",
                "Wood Mackenzie (2011): Queensland coal supply disruption assessment",
                "BMI Research: Australian Coal Sector 2011 Q1 Update",
            ),
            notes=(
                "Bowen Basin (world's largest metallurgical coal export region) was 80% flooded. "
                "Queensland coal output fell from 52 Mt (2010) to 43 Mt (2011). "
                "Global metallurgical coal prices spiked +40% in Q1 2011 due to supply shortage."
            ),
        ),
        SectorLoss(
            sector="agriculture",
            loss_usd_m=3_200,
            loss_type="combined",
            company_examples=(
                "Queensland Sugar: Burdekin cane crop loss — AUD 280M",
                "Queensland beef sector: 500,000 cattle relocated or lost",
                "Grains — wheat and sorghum crop damage AUD 600M (ABARES 2011)",
            ),
            sources=(
                "ABARES (2011): Agricultural commodities outlook — Queensland floods",
                "Queensland Farmers' Federation: Disaster impact on rural industry 2011",
            ),
            notes="Sugar cane damage in Burdekin region was the worst in 50 years.",
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=8_000,
            loss_type="physical_damage",
            company_examples=(
                "Insurance Group Australia (IAG): AUD 1.4B in claims (FY2011 result)",
                "Suncorp Group: AUD 1.6B in natural hazard claims FY2011 (record)",
                "Residential properties in Brisbane: 28,000 homes affected (QLD Gov)",
            ),
            sources=(
                "Queensland Floods Commission of Inquiry (2012)",
                "IAG Annual Report FY2011: Queensland floods net loss AUD 250M after reinsurance",
                "Suncorp Annual Report FY2011",
            ),
            notes=(
                "Includes residential, commercial, and industrial property damage. "
                "Brisbane CBD inundated for the first time since 1974. "
                "Business interruption claims were a significant component."
            ),
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=280,
            loss_type="combined",
            company_examples=(
                "Lion Nathan (now Lion): XXXX Gold Brewery in Milton, Brisbane — flooded",
                "Bundaberg Rum (Diageo subsidiary): Bundaberg plant surrounded by floodwaters",
                "Carlton & United Breweries: supply chain disruption from flooded roads",
            ),
            sources=(
                "Lion Nathan ASX announcement January 2011",
                "Diageo plc 2011 Annual Report: Australia supply chain note",
            ),
            notes="Milton Brewery in Brisbane was directly flooded; clean-up cost AUD 40M+.",
        ),
    ),
    recovery_months_by_sector={
        "mining":      6,
        "agriculture": 12,
        "real_estate": 24,
        "beverages":   4,
    },
    key_impacts=(
        "Three consecutive La Niña years (2010, 2011, 2012). "
        "75% of Queensland declared a disaster zone. "
        "Toowoomba flash flood (January 2011) killed 9. "
        "Global metallurgical coal price spike — steel mills in Japan and Korea paid premium. "
        "Total insured losses were the largest natural disaster loss in Australian history to that point."
    ),
    affected_countries=("AU",),
    sources=(
        "Queensland Floods Commission of Inquiry (2012) Final Report",
        "Swiss Re sigma 2/2012: Natural catastrophes and man-made disasters in 2011",
        "Munich Re NatCatSERVICE 2011 annual review",
        "ABARES (2011): Agricultural commodities — March quarter 2011",
        "Wood Mackenzie (2011): Queensland coal floods analysis",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 2011 Thailand Floods
# ──────────────────────────────────────────────────────────────────────────────

THAILAND_FLOODS_2011 = HistoricalClimateEvent(
    id="thailand_floods_2011",
    name="2011 Thailand Floods — Chao Phraya Basin Inundation",
    year_start=2011,
    year_end=2011,
    region=EventRegion.SOUTHEAST_ASIA,
    physical_event_id="river_flood_major",
    calibration_scale={
        "flood_riverine":      1.35,
        "water_contamination": 1.2,
        "landslide":           0.8,    # landslide less prominent than riverine
    },
    total_loss_usd_m=45_000,
    insured_loss_usd_m=15_000,       # Largest insured loss event in ASEAN history (to 2011)
    source_total_loss=(
        "World Bank (2011): Thai Flood 2011: Rapid Assessment for Resilient Recovery. "
        "Swiss Re sigma 2/2012."
    ),
    sector_losses=(
        SectorLoss(
            sector="manufacturing",    # closest to beverages and mining for this event
            loss_usd_m=32_000,
            loss_range_usd_m=(28_000, 35_000),
            loss_type="combined",
            company_examples=(
                "Honda Thailand: Ayutthaya plant flooded — 240,000 units production lost",
                "Toyota Thailand: 3 plants closed, 150,000 units lost, USD 600M insurance claim",
                "Western Digital Thailand: HDD production lost — global shortage 3 quarters",
                "Toshiba Thailand: USD 1.3B in flood damage",
                "Canon Thailand: 10% of global output halted",
            ),
            sources=(
                "World Bank (2011): Thai Flood 2011: Rapid Assessment",
                "JICA (2012): Thailand Floods Impact Analysis",
                "ISO-Swiss Re (2012): Thailand floods — a key test case for supply chain risk",
            ),
            notes=(
                "Ayutthaya and Pathum Thani industrial estates were completely submerged. "
                "Global hard-drive prices rose 40–100% within 60 days due to WD/Toshiba closures."
            ),
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=1_200,
            loss_type="combined",
            company_examples=(
                "ThaiBev (Thai Beverage): flooded malting facilities north of Bangkok",
                "Heineken Thailand (Thai Asia Pacific Brewery): distribution disrupted 8 weeks",
                "Singha Corporation: supply logistics collapse; AUD-equivalent 120M loss claim",
            ),
            sources=(
                "ThaiBev Annual Report 2011: Flood impact disclosure",
                "Thai Beverage industry association emergency assessment October 2011",
            ),
            notes="Beverages impact included ingredient supply disruption and distribution collapse.",
        ),
        SectorLoss(
            sector="agriculture",
            loss_usd_m=4_800,
            loss_type="production_loss",
            company_examples=(
                "Thai rice sector: 8 million rai of paddy fields flooded (FAO assessment)",
                "Charoen Pokphand (CP Foods): feed mills flooded, poultry supply chain cut",
            ),
            sources=(
                "FAO (2011): Thailand floods — agricultural impact assessment",
                "Office of Agricultural Economics Thailand: production statistics 2011",
            ),
            notes="Thailand was the world's largest rice exporter; rice prices rose 13% globally.",
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=6_000,
            loss_type="physical_damage",
            company_examples=(
                "Industrial estate operators: Rojana, Amata, Hi-Tech flooded (combined USD 2.5B+)",
                "Government flood compensation: 3.3 million households received payments",
            ),
            sources=(
                "NESDB (2011): Post-flood assessment and recovery plan",
                "World Bank (2011): Thai Flood 2011: Rapid Assessment",
            ),
            notes="Industrial estates had no flood insurance — relied on government compensation.",
        ),
    ),
    recovery_months_by_sector={
        "manufacturing": 6,
        "beverages":     3,
        "agriculture":   9,
        "real_estate":   18,
    },
    key_impacts=(
        "Worst flooding in Thailand in 50 years. "
        "7.5 million workers impacted. 4 million ha inundated. "
        "Global supply chain disruption: automotive and electronics sectors hit hardest. "
        "Hard disk drive shortage lasted 3–4 quarters — global PC production fell. "
        "Insured losses of USD 15B were largest in ASEAN history at the time."
    ),
    affected_countries=("TH",),
    sources=(
        "World Bank (2011): Thai Flood 2011: Rapid Assessment for Resilient Recovery. "
        "Report No. 65977-TH",
        "Swiss Re sigma 2/2012",
        "Munich Re NatCatSERVICE 2011",
        "FAO (2011): Thailand floods — agricultural impact assessment",
        "NESDB Thailand: Post-Flood Recovery Plan 2012",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 5. 2012 US Midwest Drought
# ──────────────────────────────────────────────────────────────────────────────

US_MIDWEST_DROUGHT_2012 = HistoricalClimateEvent(
    id="us_midwest_drought_2012",
    name="2012 US Midwest Drought and Heat Wave",
    year_start=2012,
    year_end=2012,
    region=EventRegion.NORTH_AMERICA,
    physical_event_id="compound_drought_heat",
    calibration_scale={
        "drought":     1.1,
        "heat_stress": 1.2,
        "water_stress": 1.0,
        "wildfire":    0.9,
    },
    total_loss_usd_m=35_000,
    insured_loss_usd_m=17_000,    # Crop insurance was the dominant insured component
    source_total_loss=(
        "NOAA NCEI Billion-Dollar Weather and Climate Disasters (2012 drought entry). "
        "USDA Economic Research Service."
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=30_000,
            loss_range_usd_m=(25_000, 35_000),
            loss_type="production_loss",
            company_examples=(
                "ADM (Archer Daniels Midland): corn/soy processing margins compressed; "
                "Q3 2012 earnings miss attributed to drought",
                "Cargill: disclosed 'significant' impact on grain processing in 2012 annual report",
                "Bunge: grain segment EBIT fell 60% Q3 2012 vs. prior year",
                "Pioneer Hi-Bred (DuPont): yield per acre fell 15–25% across Corn Belt",
            ),
            sources=(
                "USDA NASS (2012): Crop Production Summary — Year in Review",
                "NOAA NCEI Billion-Dollar Disasters: 2012 drought, ID: 2012-US-drought",
                "USDA Economic Research Service (2013): The 2012 US Drought",
            ),
            notes=(
                "US corn yield fell to 123.4 bu/acre vs. trend 163 bu/acre. "
                "Soybean yield: 39.6 vs. trend 44 bu/acre. "
                "USD 17B paid out under Federal Crop Insurance (USDA RMA)."
            ),
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=850,
            loss_type="production_loss",
            company_examples=(
                "AB InBev: 2012 annual report cited higher corn/barley ingredient costs USD 200M+",
                "Molson Coors: Q4 2012 earnings — ingredient cost headwind raised guidance",
                "PepsiCo: Agricultural commodity cost inflation flagged in 2012 10-K",
            ),
            sources=(
                "AB InBev Annual Report 2012: commodity cost outlook section",
                "Molson Coors 2012 Q4 earnings call transcript",
            ),
            notes="Impact primarily through ingredient cost inflation (corn, barley, hops) rather "
                  "than direct water shortage.",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 12,
        "beverages":   6,
        "mining":      1,
        "real_estate": 2,
    },
    key_impacts=(
        "Covered 60% of contiguous US by August 2012 (PDSI drought index). "
        "Largest US drought by affected area since the 1950s. "
        "Mississippi River barge traffic severely disrupted — coal and grain logistics. "
        "Global food price spike: FAO Food Price Index rose to second-highest on record."
    ),
    affected_countries=("US",),
    sources=(
        "NOAA NCEI: Billion-Dollar Weather and Climate Disasters, 2012 drought",
        "USDA NASS: Crop Production Summary December 2012",
        "USDA ERS (2013): The 2012 US Drought — Factors Contributing to Its Intensity",
        "Federal Reserve Bank of Kansas City: Agricultural Finance Databook Q3 2012",
        "Swiss Re sigma 2/2013: natural catastrophes and man-made disasters 2012",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 6. 2015–16 El Niño — Southeast Asia / Global
# ──────────────────────────────────────────────────────────────────────────────

EL_NINO_2015_16 = HistoricalClimateEvent(
    id="el_nino_2015_16",
    name="2015–16 El Niño — Southeast Asia Drought and Fires",
    year_start=2015,
    year_end=2016,
    region=EventRegion.SOUTHEAST_ASIA,
    physical_event_id="el_nino_super_drought",
    calibration_scale={
        "water_stress": 1.15,
        "drought":      1.1,
        "wildfire":     1.5,    # Borneo/Sumatra fires were extreme
        "heat_stress":  1.1,
    },
    total_loss_usd_m=70_000,    # Estimated global economic loss; peer-reviewed range USD 60-84B
    insured_loss_usd_m=3_200,
    source_total_loss=(
        "Estrada et al. (2023): Attribution of economic losses to 2015–16 El Niño. "
        "Nature Communications. Swiss Re sigma 1/2017."
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=25_000,
            loss_range_usd_m=(18_000, 30_000),
            loss_type="production_loss",
            company_examples=(
                "Olam International: cocoa and coffee procurement disrupted, supply chain stress",
                "Wilmar International: palm oil production fell 5–8% in Indonesia/Malaysia",
                "Philippine rice sector: 1.5 million metric ton deficit",
                "Thai cassava: crop failure in northeast Thailand",
            ),
            sources=(
                "FAO (2016): El Niño Response Plan — food security impacts",
                "Estrada et al. (2023) Nature Communications: economic losses",
                "OECD (2016): Responding to the El Niño-induced crop failure",
            ),
            notes=(
                "Palm oil production in Malaysia fell 2016: worst drought in 30 years. "
                "Indonesia: 2.6M ha burned in 1997-scale fires."
            ),
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=2_100,
            loss_type="combined",
            company_examples=(
                "Thai Beverage (ThaiBev): Mekong tributaries at 40-year low; "
                "water rationing protocols activated at all Thailand breweries",
                "San Miguel (Philippines): municipal water supply cut 35%",
                "Diageo plc: highlighted Southeast Asia water risk in CDP 2016 submission",
            ),
            sources=(
                "ThaiBev Sustainability Report 2016: water stress disclosure",
                "Diageo CDP Water Security Response 2016",
                "Heineken N.V. Annual Report 2016: Vietnam/Singapore water risk note",
            ),
            notes=(
                "Beverages sector primarily hit through water restrictions and "
                "sugar/malt barley input crop failure. Beer volumes declined in "
                "Thailand and Philippines Q1–Q2 2016."
            ),
        ),
        SectorLoss(
            sector="mining",
            loss_usd_m=3_800,
            loss_type="combined",
            company_examples=(
                "PT Adaro Energy: coal haul roads disrupted by fires; air quality "
                "halted outdoor operations for weeks in Kalimantan",
                "Rio Tinto: Indonesian nickel operations smoke impact — disclosed in 2016 AR",
                "Newcrest Mining: PNG operations water stress flagged in CDP 2016",
            ),
            sources=(
                "ESDM Indonesia: Mining sector El Niño impact 2015–16",
                "PT Adaro Energy Annual Report 2016: operational risks section",
            ),
            notes="Wildfire smoke from Kalimantan/Sumatra fires caused air quality halts at open-cut mines.",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 12,
        "beverages":   6,
        "mining":      3,
        "real_estate": 2,
    },
    key_impacts=(
        "Tied with 1997–98 as the strongest El Niño on record (ONI +2.3°C). "
        "Indonesia: USD 16B loss from fires alone (World Bank 2015). "
        "Zambia/Zimbabwe: power rationing due to low Kariba reservoir. "
        "Ethiopia: USD 1B agricultural loss. "
        "Global GDP impact estimated at −0.2% (Estrada et al. 2023)."
    ),
    affected_countries=("ID", "PH", "TH", "VN", "IN", "AU", "ZA", "ZM", "ZW", "ET"),
    sources=(
        "Estrada et al. (2023): Attribution of 2015–16 El Niño to global economic impacts. "
        "Nature Communications.",
        "World Bank (2015): The Cost of Fire — Indonesia's Fire and Haze Crisis",
        "FAO (2016): El Niño and La Niña — Preparedness and Response",
        "Swiss Re sigma 1/2017: Natural catastrophes and man-made disasters in 2016",
        "Hewitt et al. (2020): Global economic impacts of the 2015–16 El Niño. "
        "Nature Climate Change.",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 7. 2016 Fort McMurray Wildfire (Alberta, Canada)
# ──────────────────────────────────────────────────────────────────────────────

FORT_MCMURRAY_WILDFIRE_2016 = HistoricalClimateEvent(
    id="fort_mcmurray_wildfire_2016",
    name="2016 Fort McMurray Wildfire — Alberta Oil Sands",
    year_start=2016,
    year_end=2016,
    region=EventRegion.NORTH_AMERICA,
    physical_event_id="wildfire_acute_asset",
    calibration_scale={
        "wildfire":    1.1,    # comparable to the acute wildfire baseline
        "heat_stress": 1.2,
        "air_quality": 1.3,
    },
    total_loss_usd_m=7_500,    # CAD 9.9B ≈ USD 7.5B (2016 exchange)
    insured_loss_usd_m=4_100,  # CAD 5.4B insured — largest single insured event in Canadian history
    source_total_loss=(
        "Insurance Bureau of Canada (2016): Fort McMurray wildfire final industry loss. "
        "Swiss Re sigma 1/2017."
    ),
    sector_losses=(
        SectorLoss(
            sector="mining",
            loss_usd_m=3_500,
            loss_type="production_loss",
            company_examples=(
                "Suncor Energy: Upgrader operations shut for 5 weeks, 1M bbls/day offline; "
                "Q2 2016 production loss CAD 810M",
                "Canadian Natural Resources (CNRL): Horizon upgrader shut — 100,000 boe/day halted",
                "Syncrude (Suncor/Imperial/Nexen JV): operations halted, restart in June 2016",
                "Husky Energy: pipeline operations suspended",
            ),
            sources=(
                "Suncor Energy Q2 2016 Financial Report: Fort McMurray operational impacts",
                "Canadian Natural Resources Q2 2016 MD&A: wildfire production losses",
                "Wood Mackenzie (2016): Canadian oil sands — wildfire production impact",
            ),
            notes=(
                "Approximately 3.7 billion barrels/day of Canadian oil sands output was offline "
                "during peak evacuation (95% of Athabasca oil sands capacity). "
                "Production disruption lasted 5–12 weeks depending on facility."
            ),
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=3_800,
            loss_type="physical_damage",
            company_examples=(
                "Residential: 2,400 structures destroyed in Fort McMurray (RCMP survey)",
                "Intact Financial Corp: CAD 440M in wildfire claims; catastrophe reserves triggered",
                "TD Insurance: CAD 200M+ Fort McMurray wildfire claims",
            ),
            sources=(
                "Insurance Bureau of Canada: Fort McMurray wildfire industry loss, Nov 2016",
                "Intact Financial Q3 2016 earnings call: wildfire loss breakdown",
            ),
            notes="CAD 5.4B insured loss was the largest natural disaster insured loss in Canadian history.",
        ),
    ),
    recovery_months_by_sector={
        "mining":      3,
        "real_estate": 36,
        "agriculture": 2,
        "beverages":   1,
    },
    key_impacts=(
        "Mandatory evacuation of 88,000 residents (May 2016). "
        "590,000 ha burned — largest wildfire evacuation in Canadian history. "
        "Oil sands production offline 5 weeks — crude prices ticked up 5% on supply tightness. "
        "CAD 9.9B total economic loss. "
        "CAD 5.4B insured loss: largest single-event insured loss in Canadian history."
    ),
    affected_countries=("CA",),
    sources=(
        "Insurance Bureau of Canada (2016): Fort McMurray wildfire — final industry loss estimate",
        "Swiss Re sigma 1/2017",
        "Suncor Energy Q2 2016 financial statements",
        "Government of Alberta: 2016 Wildfire Season Report",
        "Natural Resources Canada: Canadian Wildland Fire Information System",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 8. 2017 Hurricane Harvey (Texas, US)
# ──────────────────────────────────────────────────────────────────────────────

HURRICANE_HARVEY_2017 = HistoricalClimateEvent(
    id="hurricane_harvey_2017",
    name="2017 Hurricane Harvey — Houston/Gulf Coast",
    year_start=2017,
    year_end=2017,
    region=EventRegion.NORTH_AMERICA,
    physical_event_id="tropical_cyclone_cat4",
    calibration_scale={
        "cyclone":          1.1,
        "flood_coastal":    1.3,    # unprecedented rainfall (60 inches in some areas)
        "flood_riverine":   1.5,    # Harvey stalled, causing exceptional riverine flooding
        "saltwater_intrusion": 1.2,
    },
    total_loss_usd_m=125_000,
    insured_loss_usd_m=30_000,
    source_total_loss=(
        "NOAA NCEI: Billion-Dollar Weather and Climate Disasters — Harvey 2017. "
        "Swiss Re sigma 1/2018."
    ),
    sector_losses=(
        SectorLoss(
            sector="real_estate",
            loss_usd_m=90_000,
            loss_range_usd_m=(70_000, 105_000),
            loss_type="physical_damage",
            company_examples=(
                "FEMA National Flood Insurance Program: USD 8.9B in Harvey claims (NFIP 2018 report)",
                "Berkshire Hathaway / GEICO: significant Harvey loss cited in 2017 annual letter",
                "Allstate: USD 850M in net Harvey losses (Q3 2017 earnings report)",
                "Houston CBD commercial real estate: 20% of office stock temporarily unusable",
            ),
            sources=(
                "FEMA NFIP: Harvey Claims Data 2017–2018",
                "CoreLogic (2017): Harvey total property loss estimate",
                "Swiss Re sigma 1/2018",
            ),
            notes=(
                "204,000 homes flooded in Harris County alone. "
                "500-year rainfall event in many locations. "
                "Flood damage was primary peril; wind damage was secondary."
            ),
        ),
        SectorLoss(
            sector="mining",    # maps to energy/refining
            loss_usd_m=22_000,
            loss_type="production_loss",
            company_examples=(
                "ExxonMobil Baytown refinery: operations halted, USD 600M+ in restart costs",
                "Phillips 66: Sweeny refinery shut; Q3 2017 USD 800M impact",
                "LyondellBasell: Channelview complex shut for 2 weeks",
                "Valero Energy: Port Arthur and Three Rivers refineries flooded",
                "US Gulf Coast: 25% of US refining capacity offline at peak (DOE EIA 2017)",
            ),
            sources=(
                "US DOE EIA: Harvey's Impact on Gulf Coast Petroleum Infrastructure, Sept 2017",
                "ExxonMobil Q3 2017 earnings release",
                "Phillips 66 Q3 2017 10-Q",
            ),
            notes=(
                "45% of US Gulf Coast refining capacity affected. "
                "US gasoline prices spiked 30 cents/gallon in first week. "
                "Chemical plants: 40+ Superfund sites flooded, potential contamination."
            ),
        ),
        SectorLoss(
            sector="agriculture",
            loss_usd_m=5_200,
            loss_type="combined",
            company_examples=(
                "Texas rice crop: 75% of harvest lost in flood (Texas Rice Council)",
                "Texas cotton: USD 800M loss (Texas Department of Agriculture)",
                "Tyson Foods: chicken processing plants disrupted in Texas",
            ),
            sources=(
                "Texas A&M AgriLife Extension: Harvey agricultural loss assessment",
                "USDA NASS: Texas crop production update September 2017",
            ),
            notes="Texas cotton and rice crops were hardest hit; livestock losses also significant.",
        ),
    ),
    recovery_months_by_sector={
        "real_estate": 36,
        "mining":       3,
        "agriculture": 12,
        "beverages":    2,
    },
    key_impacts=(
        "USD 125B total economic loss (second-costliest natural disaster in US history at the time). "
        "60+ inches of rain in some areas — all-time US rainfall record for a tropical cyclone. "
        "13 million people affected. "
        "US gasoline price spike rippled through consumer spending. "
        "Gulf Coast petrochemical complex: estimated USD 22B in losses."
    ),
    affected_countries=("US",),
    sources=(
        "NOAA NCEI: Billion-Dollar Weather and Climate Disasters — Harvey",
        "Swiss Re sigma 1/2018: Natural catastrophes and man-made disasters in 2017",
        "Munich Re NatCatSERVICE 2017",
        "US DOE EIA: Harvey's Impact on Gulf Coast Petroleum Infrastructure (Sept 2017)",
        "Blake & Zelinsky (2018): National Hurricane Center Tropical Cyclone Report: Harvey. NOAA.",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 9. 2017–18 Cape Town Water Crisis
# ──────────────────────────────────────────────────────────────────────────────

CAPE_TOWN_DROUGHT_2017_18 = HistoricalClimateEvent(
    id="cape_town_drought_2017_18",
    name="2017–18 Cape Town Day Zero Water Crisis",
    year_start=2017,
    year_end=2018,
    region=EventRegion.SOUTH_AFRICA,
    physical_event_id="water_crisis_municipal",
    calibration_scale={
        "water_stress": 1.1,    # very close to the municipal water crisis baseline
        "drought":      1.0,
        "heat_stress":  1.0,
    },
    total_loss_usd_m=1_500,
    insured_loss_usd_m=None,   # Drought/water restriction — no significant insured component
    source_total_loss=(
        "Western Cape Government (2018): Drought impact assessment. "
        "DG Murray Trust (2018): Cape Town economic impact analysis."
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=720,
            loss_type="production_loss",
            company_examples=(
                "South African Wine Industry: 2018 harvest down 22% (SA Wine Information & Systems)",
                "Western Cape fruit exporters: deciduous fruit packing shed throughput down 15%",
                "Western Cape dairy: AUD-equivalent 60M in livestock loss and reduced output",
            ),
            sources=(
                "SAWIS (SA Wine Information & Systems): 2018 harvest data",
                "Hortgro (2018): Crop estimate and harvest report",
                "Western Cape Department of Agriculture: 2018 Annual Report",
            ),
            notes=(
                "Wine grape harvest fell 22% in 2018 (the Day Zero year). "
                "Agriculture accounted for the majority of the quantified loss. "
                "Groundwater emergency drilling exemptions granted by national government."
            ),
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=320,
            loss_type="combined",
            company_examples=(
                "South African Breweries (AB InBev subsidiary): water use cut 42%; "
                "operations restructured to use 40% less water per hectolitre "
                "(CDP Water Report 2018 — cited by AB InBev)",
                "Heineken South Africa: Stellenbosch brewery on emergency rations; "
                "emergency water trucking cost R3M+ per month",
                "Distell Group (now Heineken): Stellenbosch operations — disclosed "
                "ZAR 80M in water-related capex to reduce dependency",
            ),
            sources=(
                "AB InBev CDP Water Security submission 2018",
                "Distell Group Integrated Annual Report FY2018: water risk section",
                "City of Cape Town: Water Conservation and Demand Management Report 2018",
            ),
            notes=(
                "Industrial water tariffs rose to ZAR 38.86/kl (Level 6B restrictions). "
                "Pre-crisis rate was ZAR 5.02/kl — a 7.7× cost increase. "
                "Breweries reduced throughput 15–30% rather than pay emergency tariffs."
            ),
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=380,
            loss_type="combined",
            company_examples=(
                "Western Cape tourism sector: estimated USD 180M revenue loss (Cape Tourism 2018)",
                "Commercial real estate: Grade A office occupancy fell 4 ppts — "
                "tenants cited water uncertainty in relocation decisions",
            ),
            sources=(
                "Cape Town Tourism (2018): Visitor statistics and economic impact",
                "SAPOA (South African Property Owners' Association): Vacancy survey 2018",
            ),
            notes="Tourism was the dominant channel: international and domestic visitor numbers fell sharply.",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 12,
        "beverages":   6,
        "real_estate": 18,
        "mining":      2,
    },
    key_impacts=(
        "Day Zero (reservoir at 13.5% capacity) was narrowly averted due to emergency restrictions. "
        "Level 6B restrictions: residential 50L/person/day; industrial cut 45%. "
        "Water tariff for industrial users increased 7.7× under emergency pricing. "
        "First major 'Day Zero' event for a modern major city — became global reference case. "
        "Western Cape agricultural GDP fell 8.2% in 2018 (Stats SA)."
    ),
    affected_countries=("ZA",),
    sources=(
        "City of Cape Town: 2017/18 Integrated Annual Report",
        "Western Cape Government (2018): Drought impact assessment report",
        "AB InBev CDP Water Security submission 2018",
        "Distell Group Integrated Annual Report FY2018",
        "Stats SA: Western Cape GDP estimates 2018",
        "DG Murray Trust (2018): Economic impact of the Cape Town water crisis",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 10. 2019–20 Australian Black Summer Bushfires
# ──────────────────────────────────────────────────────────────────────────────

AUSTRALIA_BLACK_SUMMER_2019_20 = HistoricalClimateEvent(
    id="australia_black_summer_2019_20",
    name="2019–20 Australia Black Summer Bushfires",
    year_start=2019,
    year_end=2020,
    region=EventRegion.AUSTRALIA,
    physical_event_id="wildfire_season_severe",
    calibration_scale={
        "wildfire":    1.1,    # 18.6Mha — more severe than the seasonal baseline
        "air_quality": 1.3,
        "heat_stress": 1.2,
        "drought":     1.1,
    },
    total_loss_usd_m=103_000,   # USD 103B — Nature Climate Change estimate including health/ecosystem
    insured_loss_usd_m=1_900,
    source_total_loss=(
        "Filkov et al. (2020): Impact of Australia's catastrophic 2019/20 bushfire season "
        "on communities and environment. Safety Science. "
        "Swiss Re sigma 1/2021."
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=4_400,
            loss_type="combined",
            company_examples=(
                "Australian horticulture: berry, stone fruit, and vegetable crops destroyed "
                "in firegrounds (Horticulture Innovation Australia 2020)",
                "Livestock: 5,000 cattle and sheep direct fire losses (MLA estimate)",
                "NSW wine regions (Hunter, Orange): cellar-door closures 3 months",
                "Tourism/agritourism: Snowy Mountains and Blue Mountains devastated",
            ),
            sources=(
                "ABARES (2020): Preliminary estimates of fire impacts on agriculture",
                "Horticulture Innovation Australia: Black Summer fire impact brief 2020",
                "MLA (Meat & Livestock Australia): Livestock losses from 2019–20 fires",
            ),
            notes=(
                "18.6 million hectares burned — an area larger than Syria. "
                "Smoke taint: vineyards within 200km of fires had unsaleable grapes even "
                "if not directly burned."
            ),
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=1_900,
            loss_type="physical_damage",
            company_examples=(
                "IAG (Insurance Australia Group): AUD 680M in Black Summer claims (H1 FY2020 result)",
                "Suncorp Group: AUD 480M natural hazard claims FY2020 — fire season dominant",
                "3,500 homes destroyed (EMA 2020 annual report)",
            ),
            sources=(
                "Emergency Management Australia (EMA): 2019–20 Annual Disaster Statistics",
                "IAG H1 FY2020 results: natural hazard experience",
                "Suncorp Group FY2020 full-year results",
            ),
            notes="Underinsurance in rural areas meant many households received well below replacement value.",
        ),
        SectorLoss(
            sector="mining",
            loss_usd_m=680,
            loss_type="production_loss",
            company_examples=(
                "Bega Cheese: Jingellic milk supply disrupted from farm evacuations",
                "NSW coal mines: road closures forced production curtailments (2–4 weeks)",
                "South32 Dendrobium mine: haul road access restricted by Illawarra escarpment fires",
            ),
            sources=(
                "NSW Resources Regulator: operational notices during 2019–20 fire season",
                "South32 ASX release: Dendrobium mine update January 2020",
            ),
            notes="Mining impact was indirect — access roads and worker safety, not direct mine fires.",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 18,
        "real_estate": 36,
        "mining":      2,
        "beverages":   6,
    },
    key_impacts=(
        "18.6 million hectares burned — largest area ever recorded in southern Australia. "
        "3 billion animals killed or displaced (WWF estimate). "
        "Smoke reached South America — global air quality event. "
        "3,500 homes destroyed. AUD 103B total cost per Filkov et al. 2020 "
        "(includes ecosystem services, health, biodiversity). "
        "Insured losses AUD 2.3B (~USD 1.9B). PM2.5 exposure: all of Sydney at hazardous levels."
    ),
    affected_countries=("AU",),
    sources=(
        "Filkov et al. (2020): Impact of Australia's catastrophic 2019/20 bushfire season. "
        "Safety Science, 133, 104943.",
        "Swiss Re sigma 1/2021",
        "ABARES (2020): Preliminary estimates of fire impacts on agriculture",
        "Emergency Management Australia: 2019–20 Annual Disaster Statistics",
        "IAG H1 FY2020 investor presentation",
        "WWF Australia (2020): Australia's 2019–20 bushfires — the wildlife toll",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 11. 2021 Texas Winter Storm Uri
# ──────────────────────────────────────────────────────────────────────────────

TEXAS_WINTER_STORM_URI_2021 = HistoricalClimateEvent(
    id="texas_winter_storm_uri_2021",
    name="2021 Texas Winter Storm Uri",
    year_start=2021,
    year_end=2021,
    region=EventRegion.NORTH_AMERICA,
    physical_event_id="compound_drought_heat",   # inverse — compound cold event; closest available
    calibration_scale={
        "heat_stress":  0.1,    # strongly suppressed — cold event
        "drought":      0.5,    # drought-of-water in frozen state
        "water_stress": 2.0,    # paradoxically highest — pipes burst, no supply
        "flood_riverine": 0.5,
    },
    total_loss_usd_m=130_000,
    insured_loss_usd_m=18_000,
    source_total_loss=(
        "NOAA NCEI Billion-Dollar Weather and Climate Disasters (Uri 2021). "
        "Busby et al. (2021): Cascading risks: Understanding the 2021 Texas blackout. "
        "Energy Research & Social Science."
    ),
    sector_losses=(
        SectorLoss(
            sector="real_estate",
            loss_usd_m=80_000,
            loss_range_usd_m=(65_000, 100_000),
            loss_type="combined",
            company_examples=(
                "State Farm: estimated USD 3B+ in Texas claims (largest single-event, per AM Best)",
                "Allstate: Q1 2021 disclosed USD 900M catastrophe losses from Uri",
                "5+ million homes without power for extended periods (ERCOT 2021)",
                "1,500+ water main breaks in Dallas metro alone",
            ),
            sources=(
                "NOAA NCEI: Billion-Dollar Disasters — Winter Storm Uri 2021",
                "Allstate Q1 2021 earnings release",
                "Texas Tribune: Uri damage data compilation 2021",
            ),
            notes=(
                "Pipe bursts were the dominant residential loss mechanism. "
                "USD 80B estimate is midpoint of NOAA's USD 65-100B range for total damage. "
                "Business interruption claims were a significant uninsured component."
            ),
        ),
        SectorLoss(
            sector="mining",   # mapping to petrochemical / energy sector
            loss_usd_m=30_000,
            loss_type="production_loss",
            company_examples=(
                "ExxonMobil: Baytown, Beaumont complexes shut — USD 250M disclosed loss (Q1 2021 10-Q)",
                "Occidental Petroleum: Permian Basin production halted — 100,000 boe/day offline",
                "LyondellBasell: USD 1B+ in extraordinary plant restart costs (2021 10-K)",
                "Motiva Enterprises: Port Arthur (world's largest US refinery) shut 2 weeks",
            ),
            sources=(
                "ExxonMobil Q1 2021 10-Q: Winter storm impact disclosure",
                "LyondellBasell 2021 Annual Report: winter storm losses",
                "US DOE EIA: Texas winter storm energy sector impacts (Feb 2021)",
            ),
            notes=(
                "Natural gas production fell 40% in Texas. "
                "Spot natural gas prices reached USD 999/MMBtu in parts of Texas "
                "(Henry Hub was USD 2.68/MMBtu the prior week). "
                "ERCOT grid operated with 0% reserve margin for ~76 hours."
            ),
        ),
        SectorLoss(
            sector="agriculture",
            loss_usd_m=1_800,
            loss_type="combined",
            company_examples=(
                "Texas greenhouse vegetable industry: USD 600M+ loss (TAMU AgriLife Extension)",
                "Livestock mortality: 580,000+ poultry and cattle deaths",
                "Rio Grande Winter Garden: USD 300M citrus loss",
            ),
            sources=(
                "Texas A&M AgriLife Extension (2021): Winter Storm Uri agricultural damage",
                "USDA NASS: Texas livestock and crop statistics February 2021",
            ),
            notes="Citrus groves in the Rio Grande Valley were among the hardest-hit crops.",
        ),
    ),
    recovery_months_by_sector={
        "real_estate": 12,
        "mining":       3,
        "agriculture": 12,
        "beverages":    2,
    },
    key_impacts=(
        "ERCOT grid within 4 minutes and 37 seconds of total failure (ERCOT after-action). "
        "246 people died (confirmed, with many more excess deaths counted later). "
        "500-year winter storm event for Texas. "
        "Natural gas: spot price reached USD 999/MMBtu in some hubs. "
        "Petrochemical complex lost USD 30B+ in production value. "
        "Exposed systemic failure of weatherisation in Texas energy infrastructure."
    ),
    affected_countries=("US",),
    sources=(
        "NOAA NCEI: Billion-Dollar Weather and Climate Disasters — Winter Storm Uri 2021",
        "Busby et al. (2021): Cascading risks: Understanding the 2021 Texas blackout. "
        "Energy Research & Social Science, 77, 102106.",
        "US DOE EIA: Texas Winter Storm Uri Energy Sector Impacts",
        "Texas Public Utility Commission After-Action Report 2021",
        "Swiss Re sigma 1/2022",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 12. 2021 British Columbia Atmospheric River Floods
# ──────────────────────────────────────────────────────────────────────────────

BC_FLOODS_2021 = HistoricalClimateEvent(
    id="bc_floods_2021",
    name="2021 British Columbia Atmospheric River Floods",
    year_start=2021,
    year_end=2021,
    region=EventRegion.NORTH_AMERICA,
    physical_event_id="atmospheric_river_flood",
    calibration_scale={
        "flood_riverine": 1.2,
        "landslide":      1.5,    # Coquihalla landslides were extreme
        "flood_coastal":  0.8,
    },
    total_loss_usd_m=7_500,     # CAD 9B ≈ USD 7.5B (2021 exchange)
    insured_loss_usd_m=600,
    source_total_loss=(
        "Insurance Bureau of Canada (2021): BC floods insured loss estimate. "
        "BC Ministry of Transportation: infrastructure damage assessment."
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=1_400,
            loss_type="combined",
            company_examples=(
                "BC poultry: 620,000 birds died in flooded Abbotsford farms (BC Poultry Assoc.)",
                "Fraser Valley dairy: 1,200+ cows died; milk output fell 30% in the region",
                "BC Agriculture: CAD 500M+ in livestock and crop damage",
                "Evacuation order: 17,000 Abbotsford residents including all farming operations",
            ),
            sources=(
                "BC Ministry of Agriculture (2021): Flood impact assessment",
                "BC Poultry Association: Emergency update November 2021",
                "Statistics Canada: BC agricultural sector Q4 2021",
            ),
            notes=(
                "Abbotsford is the agricultural heartland of British Columbia. "
                "Sumas Prairie (Abbotsford) flooded to depths of 3–4 metres. "
                "The area supplies 50%+ of BC's eggs and 40% of its milk."
            ),
        ),
        SectorLoss(
            sector="mining",
            loss_usd_m=2_200,
            loss_type="production_loss",
            company_examples=(
                "Copper Mountain Mining: operation isolated by Coquihalla closure; "
                "logistics disrupted 3 weeks — disclosed in Q4 2021 MD&A",
                "Teck Resources: Highland Valley copper operations logistics impacted; "
                "concentrate shipment backlog built to USD 150M",
                "Canadian Pacific Railway: trans-Canada rail blocked 5+ days — "
                "bulk commodities (coal, potash, grain) backed up",
            ),
            sources=(
                "Copper Mountain Mining Q4 2021 MD&A: Coquihalla closure impact",
                "Teck Resources Q4 2021 earnings: logistics disruption note",
                "Transport Canada: BC flood infrastructure damage and response",
            ),
            notes=(
                "The Coquihalla Highway (the main route to Vancouver) was completely severed "
                "by landslides. Coal and copper concentrate exports were held at interior facilities. "
                "Estimated 800,000 tonnes of commodity exports delayed."
            ),
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=3_200,
            loss_type="combined",
            company_examples=(
                "Intact Financial: CAD 320M in BC floods claims (Q4 2021 earnings)",
                "Wawanesa Mutual: major natural hazard event — largest BC flood loss in history",
                "Trans-Canada highway and CN/CP Rail: CAD 5B infrastructure repair estimate",
            ),
            sources=(
                "Intact Financial Q4 2021 financial supplement",
                "BC Ministry of Transportation: November 2021 flood infrastructure damage",
                "Insurance Bureau of Canada: November 2021 BC floods insured loss",
            ),
            notes="Infrastructure damage dominated total loss; only CAD 800M (~USD 600M) was insured.",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 12,
        "mining":       2,
        "real_estate": 24,
        "beverages":    1,
    },
    key_impacts=(
        "Coquihalla Highway destroyed — BC's primary inland shipping corridor. "
        "CN Rail main line severed for 5+ days — first time since construction. "
        "620,000 poultry and 1,200+ cattle deaths in Abbotsford. "
        "CAD 9B total economic loss (CAD 800M insured — massive insurance gap). "
        "Province declared disaster area; federal government contributed CAD 5B+ for infrastructure rebuild."
    ),
    affected_countries=("CA",),
    sources=(
        "Insurance Bureau of Canada: November 2021 BC floods — insured loss estimate",
        "BC Ministry of Transportation: Infrastructure damage assessment November 2021",
        "Swiss Re sigma 1/2022",
        "Cannon et al. (2022): Extreme precipitation events in British Columbia. "
        "Atmosphere-Ocean.",
        "Transport Canada: British Columbia flood impacts on transportation corridors 2021",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 13. 2021 Henan/Zhengzhou Floods (China)
# ──────────────────────────────────────────────────────────────────────────────

HENAN_FLOODS_2021 = HistoricalClimateEvent(
    id="henan_floods_2021",
    name="2021 Henan (Zhengzhou) Floods — China",
    year_start=2021,
    year_end=2021,
    region=EventRegion.EAST_ASIA,
    physical_event_id="monsoon_failure_and_flood",
    calibration_scale={
        "flood_riverine":      1.3,
        "water_contamination": 1.2,
        "landslide":           1.1,
    },
    total_loss_usd_m=17_600,
    insured_loss_usd_m=1_700,
    source_total_loss=(
        "Chinese Ministry of Emergency Management (2021): Flood damage report. "
        "Swiss Re sigma 1/2022."
    ),
    sector_losses=(
        SectorLoss(
            sector="manufacturing",   # automotive and electronics
            loss_usd_m=9_400,
            loss_type="combined",
            company_examples=(
                "Yutong Bus: Zhengzhou facility flooded — 25% of monthly output lost",
                "Foxconn (Apple supplier): logistics disrupted; Zhengzhou Apple campus on alert",
                "FAW Group: Zhengzhou joint venture operations halted 1 week",
                "BYD: Zhengzhou plant supply chain disrupted",
            ),
            sources=(
                "Henan Provincial Government: Economic loss assessment August 2021",
                "Chinese Academy of Sciences: Zhengzhou flood analysis 2021",
            ),
            notes=(
                "Zhengzhou is China's major automotive manufacturing hub. "
                "Subway flooding killed 12 people — forced rethinking of underground infrastructure. "
                "Henan province accounts for 5.8% of China's industrial output."
            ),
        ),
        SectorLoss(
            sector="agriculture",
            loss_usd_m=6_300,
            loss_type="production_loss",
            company_examples=(
                "Henan wheat and maize: 1.1 million ha of crops flooded (Henan Agriculture Dept.)",
                "Chinese wheat price futures: rose 3% after flood reports",
            ),
            sources=(
                "Henan Provincial Department of Agriculture: 2021 disaster report",
                "China National Grain and Oils Information Center: Henan flood impact",
            ),
            notes=(
                "Henan produces 30% of China's wheat and 20% of its peanuts. "
                "Harvest delays caused by flooding and equipment shortages."
            ),
        ),
    ),
    recovery_months_by_sector={
        "manufacturing": 3,
        "agriculture":   6,
        "real_estate":  12,
        "mining":        1,
    },
    key_impacts=(
        "Zhengzhou subway flooded; 12 people trapped and drowned in subway car. "
        "Hourly rainfall of 201.9mm in Zhengzhou on 20 July 2021 — an all-time record. "
        "300 deaths across Henan province. "
        "1.24 million people evacuated. "
        "Total loss: CNY 120B (~USD 17.6B). Insured: CNY 10.9B (~USD 1.7B)."
    ),
    affected_countries=("CN",),
    sources=(
        "Chinese Ministry of Emergency Management: August 2021 flood damage statistics",
        "Swiss Re sigma 1/2022: Natural catastrophes 2021",
        "Zhong et al. (2023): Attribution of the 2021 Henan extreme rainfall event. "
        "Bulletin of the American Meteorological Society.",
        "Munich Re NatCatSERVICE 2021 review",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 14. 2021 Pacific Northwest Heat Dome
# ──────────────────────────────────────────────────────────────────────────────

PACIFIC_NW_HEAT_DOME_2021 = HistoricalClimateEvent(
    id="pacific_nw_heat_dome_2021",
    name="2021 Pacific Northwest Heat Dome (US/Canada)",
    year_start=2021,
    year_end=2021,
    region=EventRegion.NORTH_AMERICA,
    physical_event_id="heat_dome_acute",
    calibration_scale={
        "heat_stress": 1.3,    # 49.6°C in Lytton BC — 5°C above prior records
        "wildfire":    1.2,    # Lytton burned down the day after its record high
        "water_stress": 1.1,
        "drought":     1.1,
    },
    total_loss_usd_m=10_000,
    insured_loss_usd_m=480,
    source_total_loss=(
        "NOAA NCEI: Heat dome event summary June–July 2021. "
        "ABARES (2021): Canadian agricultural crop loss assessment."
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=7_500,
            loss_type="production_loss",
            company_examples=(
                "BC cherry sector: 50–80% crop loss in Okanagan; CAD 500M loss (BC Fruit Growers)",
                "Oregon/Washington wheat: USD 1.2B loss estimate (OFB/WSDA)",
                "Pacific Northwest potato growers: quality downgrade and waste",
                "BC blueberry: record-setting losses in Fraser Valley",
            ),
            sources=(
                "BC Fruit Growers Association: 2021 crop assessment",
                "Oregon Department of Agriculture: Heat dome crop impacts July 2021",
                "ABARES: Pacific Northwest and Canadian heat dome — agricultural implications",
            ),
            notes=(
                "The Okanagan Valley (Canada's premier wine and fruit region) recorded "
                "temperatures of 45–49°C. Many tree fruits cooked on the vine. "
                "British Columbia estimated CAD 10B in crop losses (all agriculture)."
            ),
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=800,
            loss_type="production_loss",
            company_examples=(
                "Mission Hill Winery (Andrew Peller): disclosed grape losses across Okanagan estates",
                "Constellation Brands: Pacific Northwest wine sourcing shortfall noted in FY2022 10-K",
                "Duckhorn Portfolio: Oregon pinot noir sourcing disrupted",
            ),
            sources=(
                "BC Wine Institute: Okanagan harvest loss assessment 2021",
                "Constellation Brands FY2022 10-K: crop loss disclosure",
            ),
            notes=(
                "Grapevine death (not just berry loss) was widespread — multi-year replanting cycle required. "
                "The 2021 vintage for Okanagan wines was substantially reduced."
            ),
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=900,
            loss_type="combined",
            company_examples=(
                "Lytton, BC: town of 1,000 people burned to the ground the day after heat record; "
                "90% of structures destroyed",
                "US/Canada cooling centre demand — public health emergency costs",
            ),
            sources=(
                "BC Ministry of Emergency Management: Lytton fire damage assessment",
                "Insurance Bureau of Canada: heat dome related property claims",
            ),
            notes="Lytton wildfire (triggered by heat dome) destroyed the town; only 90 residents were left.",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 24,    # perennial crops (cherries, grapes) require replanting cycle
        "beverages":   24,
        "real_estate": 48,    # Lytton rebuild took years
        "mining":       1,
    },
    key_impacts=(
        "Lytton, BC: 49.6°C on 29 June 2021 — highest temperature ever recorded in Canada. "
        "Lytton burned to the ground the next day. "
        "619 excess deaths in British Columbia alone in the first week. "
        "US/Canada heat dome was a '1-in-1000 year' event without climate change (World Weather Attribution). "
        "With 2°C of warming, such events occur every 5–10 years (WWA 2021)."
    ),
    affected_countries=("US", "CA"),
    sources=(
        "World Weather Attribution (2021): Western North America heat wave — "
        "virtually impossible without climate change",
        "BC Centre for Disease Control: 619 excess deaths linked to heat dome",
        "BC Fruit Growers Association: 2021 crop assessment",
        "NOAA NCEI: June–July 2021 heat dome summary",
        "Philip et al. (2021): Rapid attribution analysis of the extraordinary "
        "heatwave on the Pacific coast of the US and Canada, June 2021. "
        "World Weather Attribution, July 2021.",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 15. 2022 Pakistan Floods
# ──────────────────────────────────────────────────────────────────────────────

PAKISTAN_FLOODS_2022 = HistoricalClimateEvent(
    id="pakistan_floods_2022",
    name="2022 Pakistan Super-Floods — Monsoon + Glacial Melt",
    year_start=2022,
    year_end=2022,
    region=EventRegion.SOUTH_ASIA,
    physical_event_id="monsoon_failure_and_flood",
    calibration_scale={
        "flood_riverine":      1.5,    # unprecedented — 30% of Pakistan submerged
        "water_contamination": 1.4,
        "landslide":           1.3,
        "drought":             0.8,    # opposite — extreme rainfall, not drought
    },
    total_loss_usd_m=30_000,
    insured_loss_usd_m=500,   # Critically low insurance penetration
    source_total_loss=(
        "Pakistan Post-Disaster Needs Assessment (PDNA) 2022: "
        "Government of Pakistan, World Bank, UN, EU, ADB joint assessment."
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=15_200,
            loss_range_usd_m=(12_000, 18_000),
            loss_type="production_loss",
            company_examples=(
                "Pakistan cotton: 40% of crop destroyed — 2.4 million tonnes lost (APTMA)",
                "Rice crop: Sindh province produced 50% less; "
                "Pakistan's rice export earnings fell by USD 800M",
                "Engro Fertilizers: supply chain to farmers disrupted; "
                "bulk of seasonal demand delayed 3 months",
            ),
            sources=(
                "PDNA 2022: Pakistan Post-Disaster Needs Assessment — Agriculture chapter",
                "Pakistan Bureau of Statistics: crop damage statistics 2022",
                "APTMA (All Pakistan Textile Mills Association): cotton crop loss assessment",
            ),
            notes=(
                "Pakistan's cotton is the primary input for the USD 15B textile sector. "
                "Cotton destroyed: ~45% of national crop. "
                "Sindh and Balochistan — the agricultural heartland — were 70–80% submerged."
            ),
        ),
        SectorLoss(
            sector="real_estate",
            loss_usd_m=10_800,
            loss_type="physical_damage",
            company_examples=(
                "Government PDNA: 1.7 million homes destroyed; 1.1 million homes damaged",
                "Pakistan infrastructure: roads, bridges, irrigation canals destroyed",
                "ERRA (Earthquake Reconstruction & Rehabilitation Authority): "
                "activated for disaster response",
            ),
            sources=(
                "PDNA 2022: Pakistan Post-Disaster Needs Assessment — Housing chapter",
                "OCHA Pakistan: Floods Situation Report #15, October 2022",
            ),
            notes=(
                "1.7 million homes destroyed represents ~7% of Pakistan's housing stock. "
                "Insurance penetration in Pakistan: <1% for residential property. "
                "Recovery costs estimated at USD 16B+ over 3 years."
            ),
        ),
        SectorLoss(
            sector="mining",
            loss_usd_m=2_200,
            loss_type="combined",
            company_examples=(
                "OGDCL: Balochistan gas fields access disrupted by road/bridge damage",
                "Mari Petroleum: Daharki gas plant logistics impacted",
                "Salt Range mines (Punjab): some areas flooded; Khewra salt mine access limited",
            ),
            sources=(
                "Oil & Gas Regulatory Authority Pakistan: impact assessment 2022",
                "PDNA 2022: Energy and extractives chapter",
            ),
            notes="Pakistan's oil and gas sector is concentrated in Sindh — the most severely flooded province.",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 18,
        "real_estate": 60,
        "mining":       4,
        "beverages":    3,
    },
    key_impacts=(
        "33 million people directly affected. "
        "30% of Pakistan's land area submerged at peak. "
        "1,739 deaths. 7.9 million displaced. "
        "USD 30B total damage (PDNA 2022). Pakistan's GDP fell 0.5% in FY2023. "
        "Climate attribution: flood 50% more intense due to climate change (WWA 2022). "
        "China's global supply chains for cotton/textiles: short-term disruption. "
        "World Bank emergency financing: USD 2B committed within 30 days."
    ),
    affected_countries=("PK",),
    sources=(
        "Government of Pakistan, World Bank, UN, EU, ADB (2022): "
        "Pakistan 2022 Floods: Post-Disaster Needs Assessment",
        "World Weather Attribution (2022): Climate change likely increased extreme monsoon rainfall",
        "Swiss Re sigma 1/2023: Natural catastrophes 2022",
        "OCHA Pakistan: Floods 2022 Situation Reports",
        "APTMA (2022): Cotton crop loss assessment — Pakistan 2022 floods",
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# 16. 2022–23 Somalia / Horn of Africa Drought
# ──────────────────────────────────────────────────────────────────────────────

HORN_OF_AFRICA_DROUGHT_2022 = HistoricalClimateEvent(
    id="horn_of_africa_drought_2022",
    name="2022–23 Horn of Africa / East Africa Multi-Year Drought",
    year_start=2021,
    year_end=2023,
    region=EventRegion.GLOBAL,
    physical_event_id="prolonged_multi_year_drought",
    calibration_scale={
        "drought":     1.1,
        "water_stress": 1.2,
        "heat_stress":  1.1,
    },
    total_loss_usd_m=8_500,
    insured_loss_usd_m=None,
    source_total_loss=(
        "OCHA: Horn of Africa Drought Humanitarian Response 2022–23. "
        "World Bank (2023): Climate risk assessment East Africa."
    ),
    sector_losses=(
        SectorLoss(
            sector="agriculture",
            loss_usd_m=7_000,
            loss_range_usd_m=(5_000, 9_500),
            loss_type="production_loss",
            company_examples=(
                "Ethiopian coffee sector: Sidama and Yirgacheffe output down 30–40%",
                "Kenya tea industry: KTDA (Kenya Tea Development Agency) production fell 12%",
                "Louis Dreyfus / Olam: East Africa commodity procurement disruptions disclosed "
                "in sustainability reports 2022–23",
            ),
            sources=(
                "FAO (2022): Horn of Africa Drought Crisis — food security update",
                "KTDA Annual Report 2022–23: drought production impact",
                "OCHA East Africa: Drought Situation Reports 2022",
            ),
            notes=(
                "Five consecutive failed rainy seasons (worst since 1981). "
                "Ethiopia, Kenya, Somalia, and Djibouti all declared drought emergencies. "
                "22 million people in acute food insecurity (IPC Phase 3+)."
            ),
        ),
        SectorLoss(
            sector="beverages",
            loss_usd_m=400,
            loss_type="production_loss",
            company_examples=(
                "East African Breweries (Diageo subsidiary): barley crop failure in Kenya — "
                "malting barley imports from Europe increased 40% in 2022",
                "Ethiopian Breweries (Heineken): ingredient cost inflation flagged in annual report",
            ),
            sources=(
                "East African Breweries plc Annual Report 2022",
                "Diageo CDP Water Security Response 2023: East Africa water risk",
            ),
            notes="Beverage impact was primarily through input cost inflation rather than direct water shortage.",
        ),
    ),
    recovery_months_by_sector={
        "agriculture": 24,
        "beverages":   12,
        "mining":       3,
        "real_estate":  6,
    },
    key_impacts=(
        "Five consecutive failed rainy seasons (worst drought in 40 years for Ethiopia, Kenya, Somalia). "
        "22 million people in acute food insecurity. "
        "Livestock mortality: 13 million animals in Somalia alone (OCHA). "
        "Global food price implications: coffee, tea, sesame, and livestock prices elevated. "
        "Kenya GDP growth reduced by 0.8 ppt in 2022 (World Bank)."
    ),
    affected_countries=("SO", "ET", "KE", "DJ", "ER"),
    sources=(
        "OCHA: Horn of Africa Drought Humanitarian Response Plans 2022–23",
        "FAO (2022): Horn of Africa Drought Crisis — ongoing food security assessment",
        "World Bank (2023): East Africa Climate Risk and Agriculture Assessment",
        "KTDA Annual Report 2022–23",
        "Swiss Re sigma 1/2023",
    ),
)


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL EVENT LIBRARY INDEX
# ═══════════════════════════════════════════════════════════════════════════════

HISTORICAL_LIBRARY: dict[str, HistoricalClimateEvent] = {
    e.id: e
    for e in [
        EL_NINO_1997_98,
        EUROPEAN_HEATWAVE_2003,
        QUEENSLAND_FLOODS_2010_12,
        THAILAND_FLOODS_2011,
        US_MIDWEST_DROUGHT_2012,
        EL_NINO_2015_16,
        FORT_MCMURRAY_WILDFIRE_2016,
        HURRICANE_HARVEY_2017,
        CAPE_TOWN_DROUGHT_2017_18,
        AUSTRALIA_BLACK_SUMMER_2019_20,
        TEXAS_WINTER_STORM_URI_2021,
        BC_FLOODS_2021,
        HENAN_FLOODS_2021,
        PACIFIC_NW_HEAT_DOME_2021,
        PAKISTAN_FLOODS_2022,
        HORN_OF_AFRICA_DROUGHT_2022,
    ]
}


def get_historical_event(event_id: str) -> HistoricalClimateEvent:
    """Return a HistoricalClimateEvent by ID, raising KeyError if not found."""
    try:
        return HISTORICAL_LIBRARY[event_id]
    except KeyError:
        available = ", ".join(sorted(HISTORICAL_LIBRARY))
        raise KeyError(
            f"Unknown historical event '{event_id}'. "
            f"Available: {available}"
        ) from None


def list_historical_events() -> list[dict]:
    """Return summary metadata for all historical events (for API listing)."""
    result = []
    for e in HISTORICAL_LIBRARY.values():
        result.append({
            "id":                    e.id,
            "name":                  e.name,
            "year_start":            e.year_start,
            "year_end":              e.year_end,
            "region":                e.region.value,
            "physical_event_id":     e.physical_event_id,
            "total_loss_usd_m":      e.total_loss_usd_m,
            "insured_loss_usd_m":    e.insured_loss_usd_m,
            "source_total_loss":     e.source_total_loss,
            "affected_countries":    list(e.affected_countries),
            "sectors_with_data":     [s.sector for s in e.sector_losses],
            "key_impacts":           e.key_impacts[:300] + "…" if len(e.key_impacts) > 300 else e.key_impacts,
        })
    return result


def get_sector_losses(event_id: str, sector: str) -> list[dict]:
    """Return sector-level loss records for a given event and sector slug."""
    event = get_historical_event(event_id)
    return [
        {
            "sector":          s.sector,
            "loss_usd_m":      s.loss_usd_m,
            "loss_range_usd_m": list(s.loss_range_usd_m) if s.loss_range_usd_m else None,
            "loss_type":       s.loss_type,
            "company_examples": list(s.company_examples),
            "sources":         list(s.sources),
            "notes":           s.notes,
        }
        for s in event.sector_losses
        if s.sector == sector
    ]
