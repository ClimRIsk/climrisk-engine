# Gap register

What the scenario suite and event checks found missing or wrong, ranked. Evidence: `docs/SCENARIO_RESEARCH.md`, `docs/SCENARIO_REPORT.md`, `data/backtest/scenario_events_validation.json`. Data sources named under "to close" are candidates that have not been evaluated for coverage, licence or fit.

## A. Found by running real events and sites (the model was wrong or blind)

| # | Finding | Evidence | Status |
|---|---|---|---|
| A1 | River flood map is blind to small rivers and surface water. At Ahrweiler (Ahr valley flood, July 2021, 134 deaths in the valley) the 1-in-100-year depth is 0 m; it does show 1.6 m at the Ayutthaya cluster (2011) and 2.3 m on Buffalo Bayou, Houston (Harvey). | scenario event check | Flood-covered sites now carry a "surface-water and small-river" not-assessed line and the premium is flagged a lower bound; river_flood caveat names the Ahr result. **Not closed**: no surface-water or small-river model |
| A2 | Rainfall extreme for Dubai April 2024 not recognised on fallback data (1-in-14 years against 142 mm at a station, 254.8 mm at another). | event check, NASA POWER | Limit text added to the rain study. **Open**: ERA5 and gauge or satellite-rain check; arid downpours need a rain product that resolves convection |
| A3 | Fire weather index on the Fort McMurray day reached only the 94th percentile (fallback data). Palisades day was at the 99.9th. | event check | Limit text added. **Open**: ERA5 re-run; the index is no better than temperature in the earlier Canadian test |
| A4 | Freeze (Uri) had no hazard. Built `cold_extremes`: 1-in-29 years at Houston and 1-in-35 at Dallas for Uri on fallback data. Design-minimum exceedance given when a plant's design temperature is supplied. | event check | **Closed at hazard level**; no freeze-failure or loss model |
| A5 | Offshore wind in a typhoon basin (Taiwan) was priced with the North Sea storm path. | scenario test | Fixed: such sites are withheld with the reason |
| A6 | Dubai Marina shows a 1.35 m river-flood depth in a desert marina (map artefact or wadi flow; the JRC spurious-depth layer is not used). | event check | **Open** |
| A7 | Houston Ship Channel refinery area shows 0 m river and 0 m coastal depth. Plausible at that point, unverified; terrain cross-check could not be completed (tile download too slow in the sandbox). | event check | **Open**: run the terrain model and check against Harvey and Ike surge records |
| A8 | Earthquake returned null at Houston (too little data to fit). Honest but unhelpful in low-seismicity places. | site profile | **Open**: a national hazard map would give a value |

## B. Peril coverage (ranked by scenarios affected, 30 scenarios)

| Rank | Peril | Scenarios | Engine now | To close | Anchor events for validation |
|---|---|---|---|---|---|
| 1 | Severe convective storm and hail | 7 | none | hail reports (ESSL European Severe Weather Database, NOAA storm reports) plus ERA5 convective-parameter proxy; module hail vulnerability by glass thickness and roof type | Midway 2019; Calgary 2020; N. Italy 2023 |
| 2 | Supply chain and contingent BI | 7 | none | needs the client's supplier network and dependency data; cannot be built from public hazard data alone. A BI layer (indemnity period, waiting time) for first-party BI is a smaller first step | Thailand 2011; Tohoku 2011; Uri |
| 3 | Drought and water stress to loss | 6 | hazard only | sector water-dependency (cooling water, process water, hydropower inflow) from the client | none public found |
| 4 | Extratropical windstorm onshore | 5 | hazard only (ERA5 gusts) | storm-footprint catalogue (for example the Extreme Wind Storms catalogue) and an excess-wind damage function calibrated to PERILS market losses | Lothar, Kyrill, Friederike (PERILS) |
| 5 | Pluvial and flash flood | 5 | hazard only | convection-resolving rain, terrain flow paths, urban drainage; loss path | Dubai 2024; Harvey; Ahr 2021 |
| 6 | Wildfire to loss | 5 | hazard only | burnt-area and ignition data, structure vulnerability | Fort McMurray 2016; Palisades 2025 |
| 7 | Earthquake vulnerability and low-seismic hazard | 4 | hazard only, null in low-seismic places | national or global seismic hazard maps; fragility by structure and plant type | Tohoku 2011 |
| 8 | Freeze and ice storm to loss | 4 | hazard only | freeze-failure model (instrument lines, pumps, power loss); freezing-rain hazard | Uri; 1998 ice storm |
| 9 | Landslide and debris flow | 4 | screen (slope ranking tested, AUC 0.70) | rainfall-threshold hazard with inventory validation | NASA catalogue |
| 10 | Heat on equipment (derating) | 3 | none | hourly temperature exceedance against an equipment rating the client supplies | none sourced |
| 11 | Lightning | 3 | none | flash-density climatology; claims experience | DNV, loss-adjuster statements |
| 12 | Grid outage dependency | 2 | none | client dependency data | Uri |
| 13 | Permafrost thaw | 1 (more in the north) | permafrost probability from the published Obu et al. map (10 km), built 22 Oct 2026; no thaw settlement | borehole ground temperatures and a ground-ice map | Arctic infrastructure studies |
| 14 | Tsunami, dust and sand, snow load, subsidence, volcanic, glacial lake outburst | not in the 30 | glacial lake outburst: record of reported events only (22 Oct 2026); avalanche terrain screen built, weak (AUC 0.59); others none | separate hazard modules; dust and snow load are the cheapest | none sourced |

## B2. Built 22 October 2026: cold-region site models

Freeze-thaw (B), avalanche release terrain (C, weak), permafrost (B, as published) and a glacial lake outburst record. Results, limits and what was not built: `docs/CRYOSPHERE_BENCHMARK.md`. Sources probed for the other hazards: `docs/HAZARD_SOURCE_PROBE.md` (hail, tornado and marine heatwave were then built).

## B3. Built 22 October 2026: hail, tornado, marine heatwave

Hail (B) and tornado (B) from US storm reports with a temporal hold-out test; marine heatwave (B) on NOAA sea-surface temperature, matching the published reference code and finding 7 of 9 documented events. Results, limits and what was not tested: `docs/CONVECTIVE_MARINE_BENCHMARK.md`. Hail and tornado cover the United States only; marine heatwave is opt-in in the report because a new site takes minutes to read.

## C. Structure of the model (not peril-specific)

| # | Gap | Effect |
|---|---|---|
| C1 | **Flood closed 28 Oct 2026; wind not.** The insurer now uses the JRC depth-damage table for each site's asset class (residential, commercial, industrial, transport, infrastructure, agriculture) and continent, from `data/jrc_flood_curves.json`, and each site's result states class, why that class, and which table. Where JRC publishes no table for a continent (for example infrastructure outside Asia and Europe) the mean of the published ones is used and said so. Values were transcribed from the CLIMADA-petals implementation and **not yet compared with the JRC report's own tables**. The class comes from a keyword match on sector or an `asset_class` the user gives; a classification, not a measurement. JRC has no chemical or energy class. Wind still uses one sigmoid per cyclone basin for every asset. The old `climate/damage_curves.py` parameters (d50, k, max damage) could not be traced to JRC and were removed. | Flood loss for non-industrial assets is on a published curve, but no loss validation exists (see A/B tables and FEMA benchmark) |
| C2 | No first-party BI model. Premium covers property damage only; indemnity period and waiting time are not modelled. | BI dominated Uri, Thailand and offshore wind repair delays |
| C3 | Hazard data depend on a free service with a non-commercial licence and a daily quota; the fallback is smoother and fails two of the event checks. | Commercial use needs a licensed source (Copernicus CDS or a paid plan) |
| C4 | Sector and region peril tables are public-source or labelled judgement; hydropower, agriculture and pipeline lists are judgement only. | Needs review by an engineer or underwriter per sector |
| C5 | Offshore: gust and wave return levels not checked against buoys; damage curve uncalibrated; typhoon turbine vulnerability absent; claims layer tested on invented claims only. | The premium for a farm is only as good as the claims the insurer supplies |
| C6 | Resident AI: gold answers for offshore, scope and cold are written but the model has not been retrained; the chat context does not yet carry per-site peril scope. | The local model cannot be relied on for premium numbers |
| C7 | Event checks use one event per peril. They detect blindness; they do not validate tails. | Do not quote them as validation of return levels |


## B4. Competitor check, 23 October 2026 (Prometeia and others)

Compared against Prometeia's public product sheet (full list: `docs/COMPETITIVE_LANDSCAPE.md`). What it showed, corrected after reading the code:

| Their capability | Where we are |
|---|---|
| Damage carried into PD, LGD, expected credit loss and capital | **Exists, not validated.** `climate/credit_risk.py` computes PD, LGD, IFRS 9 ECL and stages, and RWA with a logistic PD uplift whose coefficients are described as calibrated to the ECB 2022 climate stress test. It has no test against observed defaults or losses and no row in the ledger. `creditbridge.py` (entity, facility, collateral, loan-to-value after a 1-in-N loss) deliberately does not give PD or LGD. |
| Company ID to asset location | Partly. `data_acquisition/entity_resolver.py` resolves a name to a legal entity via GLEIF; EDGAR, CDP, EU ETS and geocoding clients exist. Facility locations still come from the client or a register, not from a tax ID. |
| Regulatory climate stress test | Not packaged. `stresstest.py` is an operations and supply-chain scenario, not an ECB or EBA template. NGFS carbon prices are in `connectors/ngfs.py`. |
| Transition engine with projected balance sheets | Partly. Transition plan, carbon and ESRS gap modules exist; no projected company financials. |
| Portfolio optimisation, economic capital, climate-adjusted capital ratio | Not built. |
| 200+ clients, regulator relationships | Not a code gap. |

## Carbon market side (added 28 Oct 2026)

| Item | Where we are |
|---|---|
| Registry records for the credit screen | **Connector-based, not a scrape.** `carbon_connectors.py` and `POST /carbon-markets/sync-registry` pull from a folder/URL that receives registry exports, or a JSON API the operator is licensed to call (token from an env var, field map), and stamp each result with source, fetch time and hash. Run it on a schedule and nobody uploads by hand. **What is not built:** a direct feed from the CDM Registry, Verra or Gold Standard. We found no open per-unit API for any of them (CDM publishes aggregate PDF statistics; Gold Standard offers a web CSV export; Verra's terms govern access and we saw no bulk provision; CarbonPlan OffsetsDB has paused updates and omits the CDM). Automated reading of their web pages could breach their terms, so we do not do it. Real-time per-unit data needs a data licence or a commercial provider, then `HttpJson` plus a field map. Tested with mocked sources only; untested on any real registry delivery. |
| Legacy CDM CERs | **Built as a rules check** against UNFCCC PA/A6.4/TP/4 and ICAO's CORSIA list (Oct 2025). Transfer deadlines and procedures are not encoded and change; confirm with the UNFCCC. |
| Credit price | Not built, on purpose. No price source we can cite. |
| Validation | Carbon modules graded "rules" (fidelity to published text), not A-D skill. Rules in the older credit screen and ETS/CBAM factors were not re-verified in this pass. |

