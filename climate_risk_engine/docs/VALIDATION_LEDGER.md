# How sure are we? Validation ledger (19 Oct 2026)

One row per hazard or model. "Tested against" means independent observations the engine did not use to fit. Grades: **A** = quantitative skill shown against observations in the region of use; **B** = skill shown but modest or only in some regions; **C** = built and sanity-checked, not tested against observations; **D** = known blind or wrong in a tested case. The grades are mine, not an expert's. Detail and reproduction commands are in `VALIDATION_DOSSIER.md`.

| Model | Tested against | Result | Grade |
|---|---|---|---|
| Flood ranking (terrain, event-trained) | Copernicus EMS flood extents (72 events), EA recorded outlines (216 windows) | AUC 0.76 / 0.81 (old rule 0.55 / 0.57). Ranks places within flood landscapes; not a depth or a probability | B |
| River flood depth maps (JRC GloFAS, third-party) | Event spot checks only | Right at Ayutthaya 2011 and Buffalo Bayou 2017; 0 m at the Ahr 2021; dry at several large-river city pins | D (small and mid rivers) |
| Coastal flood depth (Deltares, third-party) | Spot checks | Dry on built-up and island ground; engine adds an elevation floor, itself unvalidated | C |
| Flood depth-damage curve | 279,517 NFIP claims | Engine 1.5-2.4x observed residential damage above 0.5 m; no industrial data exists | B (residential), C (industrial) |
| Cyclone wind hazard | 148 station-storm pairs | r 0.53, 63 % within +-30 %, weak in the West Pacific | B |
| Cyclone to loss (learned) | 61 US storms, leave-one-out | Median error factor 2.3 vs 4.0 for guessing; rain/surge storms under-predicted ~10x | B |
| Insurer simulation vs FEMA NRI (new, `FEMA_NRI_BENCHMARK.md`) | NRI county building loss, 3 counties | Wind 1.2-3.6x NRI; inland flood 0-0.6x; coastal 0 | C/D |
| Heat (observed temperature days) | ERA5 vs station chain, 20-51 stations | Reanalysis runs 0.4-1.7 C cool in hot places; day counts therefore lower bounds | B |
| WBGT / UTCI | Station inputs through the same solver | Method matches reference code; no measured WBGT archive exists | B |
| Drought (soil-water model) | US Drought Monitor, 26 counties | ROC 0.70 vs 0.64 for rain deficit; irrigated and non-US untested | B |
| Fire-weather index | Canadian large-fire database | ROC 0.75, no better than temperature alone; FWI >= 30 days carry ~4.7x risk. Arithmetic matches the NRCan reference record exactly. Daily extremes as inputs inflate day counts at Mediterranean sites | B (Canada only) |
| Landslide susceptibility | NASA landslide catalogue, 40 events | AUC 0.57-0.85, mean 0.71; slope ranking only | B |
| Sea-level rise | AR6 projections, backtest to tide gauge | Method checks pass; regional multiplier default without coordinates | B |
| Air quality | 65 monitors | Daily r 0.58; plume screen not validated | B / C |
| Water stress (WRI Aqueduct) | Not re-tested | Published indicator, basin scale | C |
| Insurer correlation, terms, reinsurance | None | Assumed decay lengths; no claims | C |
| Offshore wind return levels, damage curve | None | Not checked against buoys; placeholder curve | C |
| Hail, convective storm, BI, supply chain, surface-water flood, earthquake loss | None | Not modelled | not built |

## What this says
- The engine is a **ranking and triage tool**. Where it has been tested, it separates hazardous from non-hazardous places better than a naive rule (AUC 0.7-0.8, correlation 0.5-0.6), and for the hazards tested it is far from random. None of that is the accuracy of a calibrated site model.
- A specialist model for one hazard (a 2D hydraulic model calibrated to gauges on surveyed LiDAR with local defences, a basin-specific wildfire spread model, an event-based CAT model fitted to claims) will beat the engine on depth, extent, timing and tail loss at the site it was built for. The engine cannot claim parity, and does not use those inputs.
- The engine's advantage is breadth and honesty: eight perils in one report, each labelled with what it cannot see, and a "not assessed" instead of a clean zero. It is the stage before the specialist study, and the Oasis import/export is the route to plug in a specialist model's output.
- The accuracy gap to close first, by commercial importance: (1) insured-loss validation against a named event or claims, (2) sector-specific vulnerability, (3) surface-water flood, (4) small and mid-size river flood.
