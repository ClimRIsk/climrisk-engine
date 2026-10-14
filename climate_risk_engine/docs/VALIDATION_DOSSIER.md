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
