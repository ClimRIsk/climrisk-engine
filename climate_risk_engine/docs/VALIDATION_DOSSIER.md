# ClimRisk validation dossier

Status: screening-grade prototype. Prepared for external review (Mr Charles Hett). Date: 14 October 2026. Engine commit a928977 or later.

Every number below comes from a file in the repository and can be regenerated with the command shown. Where the evidence is weak or absent this document says so.

## 1. What the engine does for one location

Given a coordinate (and optionally sector, value and policy terms) it returns:

1. **Hazards at the site.** Tropical-cyclone wind (NOAA IBTrACS 1980–2025, Holland wind model), river flood depths by return period (JRC CEMS-GloFAS), coastal flood depths with sea-level rise to 2050 (Deltares), flood-defence standards (FLOPROS), terrain and satellite/radar water history, water stress and drought (WRI Aqueduct 4.0, basin scale), and a land-cover urban-heat screen.
2. **Loss.** A Monte Carlo of direct physical damage to 2050: average annual loss, loss at 10/50/100/250-year return periods, capped at asset value.
3. **Insurer view** (`cri/insurance.py`). A site list with policy terms (deductible, limit, per peril) gives AAL, occurrence and aggregate exceedance curves, probable maximum loss, TVaR, a spatially correlated event simulation, quota-share and excess-of-loss reinsurance, a like-for-like climate-pathway rerun, and an indicative technical premium.
4. **Transition and financial layer** for company assets (carbon price scenarios, abatement, stranding, earnings, covenants). Not the subject of this dossier's validation.

Hazards in the dollar loss: cyclone wind, river flood, coastal flood. Heat, drought, water stress, wildfire, landslide and earthquake are shown as evidence or context and are **not** in the dollar loss. Business interruption and supply-chain loss are not modelled.

## 2. Evidence table

| Component | Test | Result | Reproduce |
|---|---|---|---|
| Cyclone wind hazard | 148 station-storm pairs, 26 storms, measured NOAA ISD winds, leave-one-storm-out calibration | Pearson r 0.525; median model/observed 1.006; 63 % within ±30 %; gale hit rate 79 %, false alarm 26 %. Weak in West Pacific (r 0.31) and South Pacific (median ratio 0.57) | `scripts/validate_tc_wind.py` |
| Cyclone damage function | Emanuel (2011) form with regional v_half from Eberenz et al. (2021), calibrated to EM-DAT | Taken from the literature; **not independently re-tested here** | n/a |
| Flood terrain skill on real events | Copernicus EMS observed flood extents, 72 events 2023–2026 (125 windows, 19 events outside Europe). Ranking of flooded vs non-flooded cells, AUC, held-out events | Old rule (built-up land < 3 m / 6 m): 0.55 (0.52–0.59). Event-trained terrain model: 0.76 (0.72–0.80). Fitted on Europe, scored outside Europe: 0.71 | `scripts/flood_event_benchmark.py` stages `catalog…evaluate`; `docs/FLOOD_EVENT_BENCHMARK.md` |
| Same, historic | England EA recorded flood outlines (Open Government Licence), 216 fluvial windows, 24 per decade 1940s–2020s, model fitted only on the modern events | AUC 0.81 (0.79–0.82) vs old rule 0.57. No decline with event age (Spearman −0.01, p 0.87) | stages `uk_fetch…history_eval` |
| Flood score vs insured claims by area | FEMA NFIP claims per built-up km², 180 census tracts, 3 US coastal-plain counties | Elevation of built-up land carries signal (held-out AUC 0.64–0.94). Terrain routing, drainage height, optical water, distance to water showed no skill (AUC about 0.5) | `scripts/backtest_flood_area.py`, `scripts/backtest_flood_nfip.py` |
| Flood depth-damage curve vs real damage | 279,517 NFIP single-family claims with measured exterior flood depth, 917 events, damage ÷ replacement cost | Engine curve (JRC industrial, North America) vs observed mean: 0.043 vs 0.096 below 15 cm; 0.25 vs 0.22 at 30–50 cm; 0.37 vs 0.24 at 50–75 cm; 0.57 vs 0.31 at 1–1.5 m; 0.78 vs 0.33 at 2–3 m. **Engine is 1.5 to 2.4 times the observed residential damage above 0.5 m** | `scripts/validate_flood_vulnerability.py` |
| Water stress | WRI Aqueduct 4.0 offline | Published dataset; WRI states it is modelled and not directly validated; basin scale | n/a |
| Heat | Cross-check of the engine's simplified WBGT against a shade-bracket WBGT at 8 sites | Simplified WBGT is above the bracket top by 3.5–5.2 °C at all hot sites; it counts 107–239 days a year above 32 °C at humid sites where the bracket counts 8–44. Heat is therefore **not** used for loss and is labelled screening | `scripts/heat_index_crosscheck.py` |

## 3. What changed because of this testing

1. The flood finding now includes an event-trained terrain-position component (relative, at most +2 points, cannot reach HIGH on its own).
2. Missing is no longer shown as low. A flood or other layer that was not read shows NOT ASSESSED; portfolio runs report how many assets had flood assessed and flag physical loss as a **lower bound** (flood is fetched live for the top N sites only; the rest carry cyclone loss only).
3. The asset result carries a coverage table: modelled, proxy, basin screening, missing, not assessed.
4. Water stress and drought from Aqueduct are in the asset finding (CRITICAL at the example site in Tamil Nadu, baseline stress 4.92 of 5, basin scale).
5. An optional empirical US residential flood curve fitted to NFIP claims is available per policy line (`vulnerability: "nfip_residential_us"`). The default industrial curve is unchanged because the claims data cannot validate it.

## 4. Known gaps, stated plainly

- **No industrial damage validation.** No open claims dataset exists for industrial plant. The residential comparison suggests the default curve is conservative at depth, but that does not transfer.
- **No comparison of simulated insured loss with a named event** (for example a specific New Zealand or US event with known insured loss). This is the validation most worth adding.
- **Event benchmark is flood only, and Europe-heavy.** 2023–2026 events from the public Copernicus dashboard; older Copernicus events are not available through it. Historic coverage is England only. The ranking skill (AUC about 0.8) is within flood-affected landscapes, not a probability that a random place floods. Modern terrain is used for old floods, so levees and channel changes since the event are not represented.
- **Terrain model adds little over one simple variable.** "Share of the 8 km surroundings lying below the site" scores as well as the fitted model (0.81 vs 0.81 on England).
- **Point misses occur.** A pin on the Carlisle riverside, a town that has flooded repeatedly, rated NEGLIGIBLE for flood on terrain evidence alone.
- **Portfolio physical loss is a lower bound** unless every asset's flood maps were read.
- **Flood hazard frequencies are historical with no climate uplift** (coastal depth includes sea-level rise to 2050).
- **Damage and correlation assumptions in the insurer layer are not calibrated to claims**: correlation decay lengths (wind 300 km, river 100 km, coast 60 km) are assumed; terms are applied per site per peril per year.
- **Heat, drought, wildfire, landslide, earthquake** have no event benchmark and are outside the dollar loss.
- **No external reviewer has yet checked the model.**

## 5. Questions for the reviewer

1. Is a correlated annual simulation with per-peril terms acceptable for screening, and what would you add first?
2. Which single event with a known insured loss would you trust as a test, and what data would you need?
3. Is the 1.5–2.4 times gap to residential claims at depth a reasonable reason to keep the industrial curve as is, or should the default be lowered for non-industrial lines?
4. Is "NOT ASSESSED plus lower-bound flag" an acceptable way to present partial hazard coverage in a portfolio?
5. What minimum validation would you want before the output is used in underwriting, as opposed to prioritising assets for survey?

## 6. Reproduction

All scripts are in `scripts/`; raw caches are under `data/backtest/raw/` (about 3 GB, not committed). Tests: `pytest tests/` (flood-event, coverage, integrity, insurance, physical report, portfolio). Data sources and licences: Copernicus EMS (open, attribution), England EA Recorded Flood Outlines (Open Government Licence), FEMA OpenFEMA NFIP (public), NOAA IBTrACS and ISD (public), WRI Aqueduct 4.0 (CC BY 4.0), JRC CEMS-GloFAS, Deltares, FLOPROS.

## Heat stress: outdoor WBGT validated against weather stations (added)

The engine's extreme-heat study now computes hourly outdoor WBGT with the Liljegren et al. (2008) heat-balance method (port checked against the reference implementation: max differences 0.016 °C globe, 0.034 °C wet bulb over 4000 random inputs). It replaces the earlier shade-only shortcut.

Held-out test (scripts/validate_wbgt.py, 20 airport stations, 2018 to 2022, calibration and held-out sites alternate): the same solver is driven once by ERA5 temperature and humidity and once by the station's measured temperature and dew point (ERA5 wind, pressure and sunshine in both). The comparison tests the reanalysis inputs; it does not test Liljegren against black-globe measurements, because no open archive of measured WBGT exists.

| Site | Split | ERA5 minus station, mean °C | hottest 5 % of hours °C | RMSE °C |
|---|---|---|---|---|
| Virudhunagar/Madurai IN | calibration | -1.08 | -1.16 | 1.42 |
| Dhaka BD | held-out | -0.82 | -0.55 | 1.74 |
| Kolkata IN | calibration | -0.69 | -1.69 | 1.50 |
| Houston US | held-out | -0.32 | -0.56 | 1.04 |
| Phoenix US | calibration | -0.48 | -0.15 | 1.00 |
| Jacobabad PK | held-out | -1.52 | -2.28 | 2.42 |
| Riyadh SA | calibration | -1.01 | -1.60 | 1.46 |
| Oslo NO | held-out | +0.02 | +0.15 | 1.17 |
| Bangkok TH | calibration | -0.69 | -0.67 | 1.18 |
| Dubai AE | held-out | -0.69 | -1.11 | 1.33 |
| Delhi IN | calibration | -0.88 | -1.11 | 1.66 |
| Chennai IN | held-out | -1.04 | -1.47 | 1.40 |
| Lagos NG | calibration | -0.90 | -1.50 | 1.31 |
| Miami US | held-out | -0.74 | -0.62 | 1.15 |
| Karachi PK | calibration | -0.56 | -1.40 | 1.29 |
| Brisbane AU | held-out | +0.26 | -0.28 | 0.84 |
| Manaus BR | calibration | +0.33 | -0.56 | 1.07 |
| Singapore SG | held-out | -0.36 | -0.93 | 0.81 |
| Basra IQ | calibration | -0.37 | -1.11 | 1.42 |
| Lahore PK | held-out | -0.68 | -0.90 | 1.81 |

A single additive correction fitted on the calibration sites (+1.1 °C) cuts the held-out hot-hour mean absolute bias from 0.89 to 0.55 °C. The engine reports uncorrected values and states that peaks are understated by about 1 °C.
Failures and limits: ERA5 runs cool at every hot site tested except Oslo, Brisbane and Manaus (near zero); Jacobabad is worst (about -1.5 °C mean, -2.3 °C on hot hours). Station wind is missing for up to 75 % of hours and is measured at sheltered masts, so it is excluded from the primary test; using it would shift WBGT by up to about 2 °C where winds are near calm. The hot-hour column is selected on the station chain and so is biased low for the reanalysis; selecting on the reanalysis chain gives the opposite bias (see the JSON). Radiation error is not tested.

## Air quality: regional PM2.5 record validated against 65 ground monitors (added)

Test (scripts/validate_air.py, year 2023): the CAMS PM2.5 record the engine reads (Open-Meteo air-quality API) at the coordinates of 45 US EPA AQS monitors (spaced 250 km apart) and 20 OpenAQ monitors on other continents, compared with each monitor's daily means. Sites alternate calibration / held-out by longitude.

- Daily correlation, median over sites: 0.58.
- Held-out sites, uncorrected: annual mean within 30 % of the monitor at 66 % of sites; median absolute relative error 0.19; mean error in days above 15 µg/m3 38.8 days a year.
- A single global correction factor (1.09) fitted on the calibration sites made the held-out sites worse (mean absolute bias 3.63 to 4.35 µg/m3), so the engine applies none.
- Where monitors read under 20 µg/m3 (58 sites) CAMS reads high at about half of them (median ratio 1.17). Where monitors read 20 or more (7 sites) CAMS reads low in most (ratios 0.26, 0.41, 0.48, 0.66, 0.98, 1.15, 1.44; Dhaka 0.48, Karachi 0.66).

Limits and failures: monitors are in towns, CAMS is a 10 to 40 km analysis, only one year was tested, and the open archive holds almost no South Asian monitors (two found: Dhaka, Colombo; Kathmandu lacked a full year). The region the engine is built for is its least-tested one. The source plume screen (Gaussian, Briggs urban) is NOT validated against measurements: it is checked only for physical behaviour. The engine states this in every air-quality result.

## Drought: engine soil-water model validated against observed drought (added)

Until now the drought study was scored against ERA5-Land soil moisture, which is itself a model. It is now also scored against the US Drought Monitor (USDM), the operational record of when a county was in drought, which blends rain gauges, streamflow, soil probes, satellite and local expert reports.

Test (scripts/validate_drought.py): 26 US counties across humid, semi-arid, desert and irrigated climates. The engine's own water balance (structure chosen by AIC, always the seasonal one) is fitted on 2000 to 2010 only and scored on the weekly USDM maps of 2011 to 2023. A drought week is one with 50 % or more of the county in D1 or worse (or 25 % or more in D2 or worse).

| Measure | D1+ (50 % of county) | D2+ (25 % of county) |
|---|---|---|
| Sites scored (at least 10 drought weeks) | 26 | 23 |
| Median ROC area, engine model (0.5 = no skill) | 0.702 | 0.695 |
| Median ROC area, ERA5-Land soil moisture | 0.681 | 0.672 |
| Median ROC area, 30-day rain deficit (no physics) | 0.639 | 0.628 |
| Sites where the engine beats the rain deficit | 21 of 26 | 18 of 23 |
| Drought weeks caught at the engine's default threshold (median), false-alarm share | 20 %, 43 % | 22 %, 66 % |

Failures and limits: skill is modest. The default flag (driest 10 % of days) is far narrower than the share of weeks the Monitor calls drought, so it misses most of them; it is a ranking tool, not a drought declaration. Irrigated places are poor (ROC area 0.56 to 0.61 at Fresno, Yakima, Salt Lake City), as are Dauphin PA and Greene MO. USDM is an expert-blended product, not a gauge. The test is US-only; monsoon, tropical and snow-fed systems, including the engine's home region, are untested. The basin water-stress layer (WRI Aqueduct) is a published indicator looked up by location; the engine does not validate it and says so.

## Wildfire: fire-weather index validated against observed large fires (added)

The fire study's Canadian Fire Weather Index was previously hindcast against ERA5 itself. It is now scored against where and when fires were actually reported: the Canadian National Fire Database, 2000 to 2022.

Test (scripts/validate_wildfire.py): 7 one-degree boxes in the most fire-prone parts of western and central Canada (spaced 300 km apart; an eighth had too few events). Truth is a day on which a fire of 20 ha or more was reported in the box, May to September. The index is computed untuned from daily weather (ERA5 where Open-Meteo's allowance allowed, NASA POWER otherwise, labelled per box). A logistic link is fitted on 2000 to 2010 and scored on 2011 to 2022.

| Measure | Result |
|---|---|
| Median ROC area, engine FWI (0.5 = no skill) | 0.752 |
| Median ROC area, daily maximum temperature alone | 0.804 |
| Median ROC area, season calendar alone | 0.726 |
| Boxes where FWI beats maximum temperature | 0 of 7 |
| Boxes with Brier skill above the season base rate | 4 of 7 (median 0.007) |
| Large-fire rate on days with FWI 30 or more, relative to an average day (median, boxes where reached) | 4.7 times |

Failures and limits: the index is no better than temperature alone in any box, and its probabilities add essentially nothing over the season calendar; its value is the high-end signal (days at FWI 30 or more carry about five times the average large-fire risk). Daily maxima stand in for the noon weather the official index uses, which probably costs skill; a noon-hour version is the first improvement to test. Canada's boreal forest only: Mediterranean, tropical and savanna regimes, where fuel and ignition behave differently, are untested. Large-fire days cluster, so the ROC areas are optimistic.

## Thermal stress (UTCI, heat and cold): built for any place, validated at 51 stations (added)

Published UTCI archives cover a limited period and are not available for an arbitrary coordinate, so the engine builds the index itself: ERA5 hourly temperature, humidity, wind and sunshine (1981 to within days of today) give the mean radiant temperature of a standing person (solar geometry and direct/diffuse split after Liljegren 2008; six-direction radiation budget after Thorsson 2007), then the UTCI polynomial of Broede et al. (2012). The polynomial implementation matches the reference package to 0.05 °C over its valid range (tests/test_utci.py). It reports hours a year in each heat and cold stress category, annual extremes and a Theil-Sen trend.

Held-out test (scripts/validate_utci.py): at 51 airport stations (16 Europe, 11 US, 8 Canada, 16 industrial regions in South and Southeast Asia, the Gulf, Africa and South America; 2018 to 2022), UTCI built from reanalysis temperature and humidity was compared with UTCI built from the station's measured temperature and dew point, with wind, pressure and sunshine from ERA5 in both. Sites alternate calibration / held-out within each region.

| Region | Sites | Mean bias °C | RMSE °C | Hottest 5 % of hours, bias °C | Coldest 5 % of hours, bias °C |
|---|---|---|---|---|---|
| Canada | 8 | -0.05 | 2.02 | -0.37 | +0.30 |
| Europe | 16 | -0.13 | 1.47 | -0.26 | +0.42 |
| Industrial regions elsewhere | 16 | -1.02 | 2.19 | -1.73 | -0.53 |
| US | 11 | -0.79 | 1.87 | -0.84 | -0.77 |

Region-specific correction of the hot tail, fitted on calibration sites and scored on held-out sites (mean absolute bias on the hottest hours, before to after): industrial regions elsewhere 1.66 to 0.52 °C, US 0.98 to 0.33, Canada 0.32 to 0.12, Europe 0.50 to 0.44. The engine reports the adjusted count as the central value and the unadjusted count as the lower bound. A cold-tail correction did not help held-out sites (0.89 to 0.88 °C) and is not applied.

Future: a sensitivity, not a forecast. The recent hourly weather is warmed by the CMIP6 ensemble change in TXx (heat categories) or annual mean temperature (cold categories), radiant excess is kept, and humidity is held two ways (relative humidity constant; vapour pressure constant) to bracket the result. It has not been backtested.

Failures and limits: Europe and Canada are close to unbiased; the US and especially the industrial regions run cool (hot tail about -0.8 and -1.7 °C), so unadjusted counts understate them. Wind and sunshine errors are not tested because both chains share them. No archive of measured UTCI or radiant temperature exists, so the absolute level rests on the published method; the ERA5-HEAT product could not be compared because it needs a Copernicus account. Airports are open and flat; city centres and industrial yards run hotter. The hourly source is the free Open-Meteo tier, which is non-commercial and rate-limited; commercial release needs the Copernicus data store or a paid plan.


## Offshore wind (North Sea): scope, hazard and what is NOT validated (added)

Scope (background research, not only wind). Public sources agree that most offshore wind insurance loss is not weather: subsea cable damage was 53 % of claims by value in Allianz Commercial's German and Central/Eastern European data (2014 to 2020), turbine failure 20 %; GCube reports cabling about 55 % and foundations 35 % over a 12-month window; DNV cites a loss adjuster putting cable failures near 80 % of financial losses. Different datasets, not comparable. Lightning is the most common insurance claim for wind farm owners per a DNV statement relayed in news reports, and a loss adjuster puts it at 60 % of operational blade losses (onshore-dominated, vendor-sourced). Icing is rarer offshore because salt lowers the freezing point of spray; sea ice affects foundations and vessel access in the Baltic and in cold winters in shallow seas; freeze-thaw acts on concrete and grout in the tidal and splash zone as gradual deterioration. The search found no public offshore claims for icing or freeze-thaw, no Danish or Wadden Sea ice statistics, and no peril list for scour or marine growth.

Built: ERA5 gust and wave return levels (Gumbel, L-moments), wind-wave dependence, vessel access windows, a PLACEHOLDER storm damage curve shown as a benign, central and severe bracket, a cold-season screen (freezing degree-days, icing-prone days, air freeze-thaw days, no loss effect), and an experience-rating layer (Gamma-Poisson credibility) on the insurer's claims. With no claims the premium is withheld and only the weather component is shown.

Not validated: the gust and wave return levels have not yet been scored against buoy or platform observations; the damage curve is uncalibrated; the cold screen is air-temperature only; the claims layer has been exercised on invented claims in tests, not on real ones. NASA POWER fallback winds read low. The resident AI has gold answers for the framework but has not been retrained on them.
