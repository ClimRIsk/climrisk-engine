# How sure are we? Validation ledger (19 Oct 2026)

One row per hazard or model. "Tested against" means independent observations the engine did not use to fit. Grades: **rules** = a deterministic rule or calculation checked against the published text it implements, not against observations (so no skill claim, and not legal advice); **A** = quantitative skill shown against observations in the region of use; **B** = skill shown but modest or only in some regions; **C** = built and sanity-checked, not tested against observations; **D** = known blind or wrong in a tested case. The grades are mine, not an expert's. Detail and reproduction commands are in `VALIDATION_DOSSIER.md`.

| Model | Tested against | Result | Grade |
|---|---|---|---|
| Flood ranking (terrain, event-trained) | Copernicus EMS flood extents (72 events), EA recorded outlines (216 windows) | AUC 0.76 / 0.81 (old rule 0.55 / 0.57). Ranks places within flood landscapes; not a depth or a probability | B |
| Flood ranking with open LiDAR (England; new, `LIDAR_FLOOD_BENCHMARK.md`) | EA Recorded Flood Outlines, 12 flooded towns, same points scored on both terrains | HAND AUC median 0.96 (LiDAR) vs 0.76 (global tiles); better in all 12. Not independent of how thresholds were set; England only | B+ (England), not available elsewhere |
| River flood depth maps (JRC GloFAS, third-party) | Event spot checks only | Right at Ayutthaya 2011 and Buffalo Bayou 2017; 0 m at the Ahr 2021; dry at several large-river city pins | D (small and mid rivers) |
| Coastal flood depth (Deltares, third-party) | Spot checks | Dry on built-up and island ground; engine adds an elevation floor, itself unvalidated. Dutch 0.5 m LiDAR now supplies ground height there (Rotterdam: 0.5 m against 6.4 m from the global tiles) | C |
| Flood depth-damage curve by asset class (new, `data/jrc_flood_curves.json`) | Unit tests read published values back; not compared with the JRC report tables, not tested against losses | Tables for six asset classes, transcribed from the CLIMADA-petals implementation of Huizinga et al. 2017. Missing continent tables use the mean of published ones (stated per site). Class assignment is a keyword judgement | rules |
| Flood depth-damage curve | 279,517 NFIP claims | Engine 1.5-2.4x observed residential damage above 0.5 m; no industrial data exists | B (residential), C (industrial) |
| Cyclone wind hazard | 148 station-storm pairs | r 0.53, 63 % within +-30 %, weak in the West Pacific | B |
| Cyclone to loss (learned) | 61 US storms, leave-one-out | Median error factor 2.3 vs 4.0 for guessing; rain/surge storms under-predicted ~10x | B |
| Insurer simulation vs FEMA NRI (new, `FEMA_NRI_BENCHMARK.md`) | NRI county building loss, 3 counties | Wind 1.2-3.6x NRI; inland flood 0-0.6x; coastal 0 | C/D |
| Heat (observed temperature days) | ERA5 vs station chain, 20-51 stations | Reanalysis runs 0.4-1.7 C cool in hot places; day counts therefore lower bounds | B |
| WBGT / UTCI | Station inputs through the same solver | Method matches reference code; no measured WBGT archive exists | B |
| Drought (soil-water model) | US Drought Monitor, 26 counties | ROC 0.70 vs 0.64 for rain deficit; irrigated and non-US untested | B |
| Fire-weather index | Canadian large-fire database | ROC 0.75, no better than temperature alone; FWI >= 30 days carry ~4.7x risk. Arithmetic matches the NRCan reference record exactly. Daily extremes as inputs inflate day counts at Mediterranean sites | B (Canada only) |
| Earthquake (new) | Not tested here | US: USGS National Seismic Hazard Model 2018 curves (published, peer-reviewed model; PGA on rock). Elsewhere: recorded-seismicity count only, which misses unruptured faults and gives no ground motion | B (US, as published), C (elsewhere) |
| Landslide susceptibility | NASA landslide catalogue, 40 events | AUC 0.57-0.85, mean 0.71; slope ranking only | B |
| Sea-level rise | AR6 projections, backtest to tide gauge | Method checks pass; regional multiplier default without coordinates | B |
| Air quality | 65 monitors | Daily r 0.58; plume screen not validated | B / C |
| Water stress (WRI Aqueduct) | Not re-tested | Published indicator, basin scale | C |
| Insurer correlation, terms, reinsurance | None | Assumed decay lengths; no claims | C |
| Offshore wind return levels, damage curve | None | Not checked against buoys; placeholder curve | C |
| Freeze-thaw, frost and ice days (new, `CRYOSPHERE_BENCHMARK.md`) | GHCN-Daily, 44 stations, 2004-2023 | Rank correlation 0.80 (freeze-thaw days), mean error 11 days a year, band within one at 98 %; under-counts by ~11 days in the freeze-thaw-richest climates. Air temperature, not ground or damage; sample is mostly cold Eurasia | B |
| Avalanche release terrain (new) | SLF Swiss avalanche accidents, 300 events against nearby controls | AUC 0.59 (95 % range 0.52-0.66) against elevation-matched terrain; release slope within 60 m of only 37 % of start zones. Terrain only, Swiss Alps only; settings chosen on the same events | C |
| Permafrost probability (new) | 10 km resample checked against the published 1 km Obu et al. 2019 map; map itself not re-tested | 698 points: mean difference 0.023, same zone class 95 % | B (as published) |
| Glacial lake outburst floods (new) | Not tested: a record of 6,263 reported events (Luetzow and Veh v4.2) | Lists reported events within 50 km; no unburst lakes, no probability | record |
| Severe hail (new, `CONVECTIVE_MARINE_BENCHMARK.md`) | NOAA SPC storm reports; 1996-2009 against 2010-2024 at 600 random US points | Rank correlation 0.93, AUC 0.94, median error 0.28 days a year. Reports, not measurements; US only (not within ~33 km of the border) | B |
| Tornado (new) | Same hold-out | Rank correlation 0.77-0.83, AUC 0.90, median error 0.07-0.14 days a year. Band on EF1+; counts of past tornadoes, not a probability for a building | B |
| Marine heatwave (new) | Published reference code on 8 cells; 9 documented events | Same events as the reference on 892 of 895; 7 of 9 events found against a 24 % chance rate in other years. Surface temperature on a 25 km cell; intensity not independently tested; report zone opt-in | B |
| Climate-adjusted credit risk: PD, LGD, ECL, RWA (`climate/credit_risk.py`) | None. Coefficients are described as calibrated to the ECB 2022 stress-test outcomes; no test against observed defaults or losses | Illustrative uplift, not a validated credit model; do not present as a bank-grade PD or LGD | not validated |
| Legacy CDM CER screen (`carbon_registry.py`, new) | Text only: UNFCCC technical paper PA/A6.4/TP/4 on decision 3/CMA.3 annex para 75, and ICAO CORSIA Eligible Emissions Units (Oct 2025) | 11 unit tests encode each published condition (registration on or after 2013, pre-2021 reductions, no tCER/lCER or A/R, CDM absent from open CORSIA phases) and the "cannot tell" cases. Not checked against registry records or reviewed by a lawyer or the UNFCCC. Transfer deadlines are not encoded | rules |
| Registry export import (new) | Synthetic export only | Maps columns by alias, reports matched and ignored columns, keeps blanks blank. Not tested on real CDM, Verra or Gold Standard exports; no scrape of registry sites; scheduled pickup via connectors (folder/URL drop or licensed JSON API), tested with mocked sources only | rules |
| Carbon-credit quality screen (ICVCM label, removals, vintage, SBTi, EU 2024/825) | Unit tests of the rules as coded | Rules were written from the standards' text and not re-verified against them in this review; the ICVCM label is taken as declared | rules |
| EU ETS compliance balance and CBAM free-allocation phase-out | Arithmetic unit tests | Balance arithmetic tested; the CBAM factors are from Regulation (EU) 2023/956 as coded and not re-checked in this review; the allowance price is an input | rules |
| Lightning, blade icing, dust, subsidence, BI, supply chain, earthquake loss (damage) | None | Not modelled at site level (see `HAZARD_SOURCE_PROBE.md`; earthquake shaking hazard is in, damage is not) | not built |

## What this says
- The engine is a **ranking and triage tool**. Where it has been tested, it separates hazardous from non-hazardous places better than a naive rule (AUC 0.7-0.8, correlation 0.5-0.6), and for the hazards tested it is far from random. None of that is the accuracy of a calibrated site model.
- A specialist model for one hazard (a 2D hydraulic model calibrated to gauges on surveyed LiDAR with local defences, a basin-specific wildfire spread model, an event-based CAT model fitted to claims) will beat the engine on depth, extent, timing and tail loss at the site it was built for. The engine cannot claim parity, and does not use those inputs.
- The engine's advantage is breadth and honesty: eight perils in one report, each labelled with what it cannot see, and a "not assessed" instead of a clean zero. It is the stage before the specialist study, and the Oasis import/export is the route to plug in a specialist model's output.
- Open LiDAR is now used wherever it exists for free (England 1 m, Netherlands 0.5 m). It improved flood ranking sharply in England (see above), and is the template for other countries: add a source to `connectors/lidar_dem.py` and re-run `scripts/lidar_flood_benchmark.py`-style tests for that country before trusting it.
- The accuracy gap to close first, by commercial importance: (1) insured-loss validation against a named event or claims, (2) sector-specific vulnerability, (3) surface-water flood, (4) small and mid-size river flood.


## Surface-water (rain-on-grid) model
- Evidence: `docs/PLUVIAL_BENCHMARK.md` — AUC 0.64–0.71 on LiDAR in 3 English towns (0.46–0.63 on global tiles); a plain height-above-local-low score beat it at 2 of 3 towns. Grade: **screening, modest skill; can only raise a rating**.

## Small/mid river depth (HAND + rating curve)
- Evidence: `docs/RIVER_DEPTH_BENCHMARK.md` — pooled CSI 0.37 vs 0.76 for the plain HAND<5 m band on 11 English towns. Grade: **not validated as an improvement; not shipped in the report**. GloFAS point flows misassign river size.

## Wildfire surface-fire behaviour (Rothermel)
- Evidence: `docs/FIRE_BEHAVIOUR_BENCHMARK.md` — spread formula within ±40 % of published values; as a day-ranker 0.53–0.76 ROC area vs 0.80 for hot-day temperature and 0.71 for FWI (Canada, 8 boxes). Grade: **physical illustration, not a validated risk score; fact only, never changes the rating**.

## Named-event hindcast (building level)
- Evidence: `docs/HINDCAST_STUDY.md` — Carlisle (Desmond 2015, Jan 2005), Cockermouth 2009, York (2015, 2020), 70k footprints vs EA recorded outlines. LiDAR HAND ROC area 0.81-0.92 for the five major events (global tiles 0.50-0.87); top-10 % flagged finds 52-60 % of flooded buildings. Fixed bands are susceptibility (HIGH band hit rate 4-64 %). Grade: **supported as a ranking for England river floods; depth and loss untested**.

## CSRD / ESRS E1 gap assessment
- Evidence: `tests/test_esrs_gap.py` (16 tests: null is not zero, format checks, conditional items, engine estimates never counted as provided, coverage arithmetic, and a regression that the CSRD template no longer invents company facts). Grade: **mechanical checklist, verified by tests; not a compliance opinion**.
- What it is: a requirement-level checklist for ESRS E1-1 to E1-9 plus GOV-3, IRO-1 and SBM-3 (41 items), reporting provided / proxy-only / missing / invalid per item and per requirement. `POST /reports/csrd/gap`, `GET /reports/csrd/checklist`.
- What it is not: paragraph-level or EFRAG data-point coverage, other topical standards (E2-E5, S1-S4, G1), applicability or phase-in relief, content quality, or digital tagging. The checklist is my reading of the standard and has not been reviewed by a sustainability-reporting specialist.
- Fixed in the same change: the CSRD template (`/reports/csrd`) used to hard-code targets, a "Committed" net-zero status, "Pending SBTi validation" and a true material-effects flag. It now echoes only supplied inputs and labels engine values as modelled.
