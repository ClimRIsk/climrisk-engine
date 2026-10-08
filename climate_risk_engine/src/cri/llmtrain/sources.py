"""Open sources. train=True only for public-domain / permissively licensed text; everything else is retrieval-only."""
from __future__ import annotations

PD = "US government work (public domain)"
EUR = "EU legal text (reuse authorised, Decision 2011/833/EU, attribution)"
BSD = "BSD-3-Clause (Oasis LMF open source)"

# (url, title, licence, topic, train)
SOURCES = [
    # ── hazards: cyclone, surge, flood (NOAA / NWS / FEMA / USGS) ──
    ("https://www.nhc.noaa.gov/climo/", "NOAA NHC: Tropical cyclone climatology", PD, "hazard", True),
    ("https://www.nhc.noaa.gov/aboutsshws.php", "NOAA NHC: Saffir-Simpson Hurricane Wind Scale", PD, "hazard", True),
    ("https://www.nhc.noaa.gov/surge/", "NOAA NHC: Storm surge overview", PD, "hazard", True),
    ("https://www.nhc.noaa.gov/surge/slosh.php", "NOAA NHC: SLOSH storm surge model", PD, "hazard", True),
    ("https://www.nhc.noaa.gov/aboutgloss.shtml", "NOAA NHC: Glossary of terms", PD, "hazard", True),
    ("https://www.nhc.noaa.gov/prepare/ready.php", "NOAA NHC: Hurricane hazards", PD, "hazard", True),
    ("https://www.weather.gov/safety/flood", "NWS: Flood safety and flood types", PD, "hazard", True),
    ("https://www.weather.gov/safety/flood-states", "NWS: Flood watches and warnings", PD, "hazard", True),
    ("https://www.weather.gov/safety/heat", "NWS: Heat safety", PD, "hazard", True),
    ("https://www.weather.gov/safety/drought", "NWS: Drought", PD, "hazard", True),
    ("https://www.weather.gov/safety/wildfire", "NWS: Wildfire safety", PD, "hazard", True),
    ("https://www.usgs.gov/special-topics/water-science-school/science/100-year-flood", "USGS: The 100-year flood", PD, "hazard", True),
    ("https://www.usgs.gov/special-topics/water-science-school/science/floods-and-flooding", "USGS: Floods and flooding", PD, "hazard", True),
    ("https://www.usgs.gov/programs/landslide-hazards/landslides-101", "USGS: Landslides 101", PD, "hazard", True),
    ("https://www.drought.gov/what-is-drought", "NIDIS: What is drought", PD, "hazard", True),
    ("https://www.fema.gov/flood-maps/know-your-risk", "FEMA: Know your flood risk", PD, "hazard", True),
    ("https://www.fema.gov/glossary/flood-insurance-rate-map", "FEMA: Flood insurance rate map", PD, "insurance", True),
    ("https://www.fema.gov/glossary/base-flood-elevation", "FEMA: Base flood elevation", PD, "insurance", True),
    ("https://www.fema.gov/flood-insurance/risk/risk-rating", "FEMA: NFIP Risk Rating 2.0", PD, "insurance", True),
    ("https://www.fema.gov/emergency-managers/practitioners/hazus", "FEMA: Hazus loss estimation", PD, "insurance", True),
    # ── climate science and sea level (NOAA / NASA / EPA) ──
    ("https://oceanservice.noaa.gov/facts/sealevel.html", "NOAA: Is sea level rising", PD, "scenario", True),
    ("https://oceanservice.noaa.gov/hazards/sealevelrise/", "NOAA: Sea level rise", PD, "scenario", True),
    ("https://www.climate.gov/news-features/understanding-climate/climate-change-global-sea-level", "NOAA Climate.gov: Global sea level", PD, "scenario", True),
    ("https://www.climate.gov/news-features/understanding-climate/climate-change-global-temperature", "NOAA Climate.gov: Global temperature", PD, "scenario", True),
    ("https://www.climate.gov/maps-data/climate-data-primer/predicting-climate/climate-models", "NOAA Climate.gov: Climate models", PD, "scenario", True),
    ("https://www.climate.gov/news-features/understanding-climate/climate-change-atmospheric-carbon-dioxide", "NOAA Climate.gov: Atmospheric CO2", PD, "scenario", True),
    ("https://science.nasa.gov/climate-change/causes/", "NASA: Causes of climate change", PD, "scenario", True),
    ("https://science.nasa.gov/climate-change/evidence/", "NASA: Evidence of climate change", PD, "scenario", True),
    ("https://science.nasa.gov/climate-change/effects/", "NASA: Effects of climate change", PD, "scenario", True),
    ("https://science.nasa.gov/climate-change/adaptation-mitigation/", "NASA: Adaptation and mitigation", PD, "scenario", True),
    ("https://www.epa.gov/climatechange-science/causes-climate-change", "EPA: Causes of climate change", PD, "scenario", True),
    ("https://www.epa.gov/ghgemissions/overview-greenhouse-gases", "EPA: Overview of greenhouse gases", PD, "disclosure", True),
    ("https://www.epa.gov/ghgemissions/global-greenhouse-gas-overview", "EPA: Global greenhouse gas overview", PD, "disclosure", True),
    ("https://www.epa.gov/environmental-economics/scghg", "EPA: Social cost of greenhouse gases", PD, "disclosure", True),
    ("https://www.epa.gov/climateimpacts/climate-change-impacts-society", "EPA: Climate impacts on society", PD, "scenario", True),
    ("https://www.epa.gov/climateimpacts/climate-change-impacts-energy", "EPA: Climate impacts on energy", PD, "scenario", True),
    ("https://www.epa.gov/climateimpacts/climate-change-impacts-water-resources", "EPA: Climate impacts on water", PD, "scenario", True),
    # ── disclosure and transition (EU legal text) ──
    ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022L2464", "EU: CSRD (Directive 2022/2464)", EUR, "disclosure", True),
    ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32023R0956", "EU: CBAM Regulation 2023/956", EUR, "disclosure", True),
    ("https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32020R0852", "EU: Taxonomy Regulation 2020/852", EUR, "disclosure", True),
    ("https://taxation-customs.ec.europa.eu/carbon-border-adjustment-mechanism_en", "EC: CBAM overview", EUR, "disclosure", True),
    ("https://climate.ec.europa.eu/eu-action/eu-emissions-trading-system-eu-ets_en", "EC: EU ETS overview", EUR, "disclosure", True),
    ("https://finance.ec.europa.eu/capital-markets-union-and-financial-markets/company-reporting-and-auditing/company-reporting/corporate-sustainability-reporting_en", "EC: Corporate sustainability reporting", EUR, "disclosure", True),
    # ── insurance / cat modelling (open source docs) ──
    ("https://oasislmf.github.io/", "Oasis LMF: open source catastrophe modelling platform", BSD, "insurance", True),
    ("https://github.com/OasisLMF/OpenDataStandards", "Oasis: Open Data Standards (OED) repository", BSD, "insurance", True),
    # ── additional public-domain pages ──
    ("https://oceanservice.noaa.gov/facts/stormsurge-stormtide.html", "NOAA: Storm surge vs storm tide", PD, "hazard", True),
    ("https://oceanservice.noaa.gov/facts/nuisance-flooding.html", "NOAA: Nuisance flooding", PD, "hazard", True),
    ("https://oceanservice.noaa.gov/facts/tides.html", "NOAA: Tides and water levels", PD, "hazard", True),
    ("https://www.nhc.noaa.gov/prepare/hazards.php", "NOAA NHC: Hurricane hazards", PD, "hazard", True),
    ("https://www.weather.gov/safety/flood-turn-around", "NWS: Turn around, don't drown", PD, "hazard", True),
    ("https://www.weather.gov/safety/heat-index", "NWS: Heat index", PD, "hazard", True),
    ("https://www.weather.gov/safety/hurricane", "NWS: Hurricane safety", PD, "hazard", True),
    ("https://sealevel.nasa.gov/understanding-sea-level/overview", "NASA: Understanding sea level", PD, "scenario", True),
    ("https://sealevel.nasa.gov/understanding-sea-level/global-sea-level/overview", "NASA: Global sea level", PD, "scenario", True),
    ("https://science.nasa.gov/climate-change/extreme-weather/", "NASA: Extreme weather and climate change", PD, "scenario", True),
    ("https://science.nasa.gov/climate-change/what-is-climate-change/", "NASA: What is climate change", PD, "scenario", True),
    ("https://science.nasa.gov/climate-change/global-warming-vs-climate-change/", "NASA: Global warming vs climate change", PD, "scenario", True),
    ("https://science.nasa.gov/climate-change/faq/", "NASA: Climate change FAQ", PD, "scenario", True),
    ("https://www.climate.gov/news-features/understanding-climate/climate-change-ocean-heat-content", "NOAA Climate.gov: Ocean heat content", PD, "scenario", True),
    ("https://www.climate.gov/news-features/understanding-climate/climate-change-annual-greenhouse-gas-index", "NOAA Climate.gov: Greenhouse gas index", PD, "scenario", True),
    ("https://www.climate.gov/enso", "NOAA Climate.gov: ENSO", PD, "scenario", True),
    ("https://www.epa.gov/climate-indicators/climate-change-indicators-sea-level", "EPA indicator: Sea level", PD, "scenario", True),
    ("https://www.epa.gov/climate-indicators/climate-change-indicators-coastal-flooding", "EPA indicator: Coastal flooding", PD, "hazard", True),
    ("https://www.epa.gov/climate-indicators/climate-change-indicators-heavy-precipitation", "EPA indicator: Heavy precipitation", PD, "hazard", True),
    ("https://www.epa.gov/climate-indicators/climate-change-indicators-river-flooding", "EPA indicator: River flooding", PD, "hazard", True),
    ("https://www.epa.gov/climate-indicators/climate-change-indicators-drought", "EPA indicator: Drought", PD, "hazard", True),
    ("https://www.epa.gov/climate-indicators/climate-change-indicators-wildfires", "EPA indicator: Wildfires", PD, "hazard", True),
    ("https://www.epa.gov/climate-indicators/climate-change-indicators-high-and-low-temperatures", "EPA indicator: High and low temperatures", PD, "hazard", True),
    ("https://www.epa.gov/climate-indicators/climate-change-indicators-tropical-cyclone-activity", "EPA indicator: Tropical cyclone activity", PD, "hazard", True),
    ("https://www.epa.gov/ghgemissions/sources-greenhouse-gas-emissions", "EPA: Sources of greenhouse gas emissions", PD, "disclosure", True),
    ("https://www.epa.gov/ghgemissions/understanding-global-warming-potentials", "EPA: Global warming potentials", PD, "disclosure", True),
    ("https://www.epa.gov/ghgemissions/inventory-us-greenhouse-gas-emissions-and-sinks", "EPA: US greenhouse gas inventory", PD, "disclosure", True),
    # ── retrieval-only (Wikipedia is CC BY-SA: kept out of the weights, cited at answer time) ──
]

# Wikipedia titles for the retrieval index only (share-alike licence: not used to train weights).
WIKI_EXTRA = [
    "Catastrophe modeling", "Reinsurance", "Quota share", "Excess of loss reinsurance", "Exceedance probability", "Probable maximum loss",
    "Deductible", "Insurance policy", "Actuarial science", "Value at risk", "Expected shortfall", "Combined ratio", "Insurance-linked securities",
    "Catastrophe bond", "Parametric insurance", "Solvency II", "Natural disaster risk reduction", "Shared Socioeconomic Pathways",
    "Representative Concentration Pathway", "IPCC Sixth Assessment Report", "Coupled Model Intercomparison Project", "Equilibrium climate sensitivity",
    "Sea level rise", "Extreme value theory", "Generalized extreme value distribution", "Copula (probability theory)", "Gaussian copula",
    "Damage function", "Hazus", "Flood insurance", "National Flood Insurance Program", "Climate change adaptation", "Loss and damage",
]
