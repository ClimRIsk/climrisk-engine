# Uncertainty ledger (12 Nov 2026)

`cri/uncertainty.py`, `POST /uncertainty/ledger`. Puts every doubt behind a site's number side by side: type (internal variability, scenario, structural, parameter, sampling), status, and a low-to-high range where the engine can measure one.

## What is quantified from data

| Entry | What it measures | Source in the engine |
|---|---|---|
| Design rain: which data product | Spread of the 1-in-100 daily rain across ERA5, NASA POWER and the nearest gauge | `finance/rain_design.design` |
| Future change in extreme rain | p10 / median / p90 across CMIP6 models, one pathway | `modellab/scenarios` (World Bank CCKP) |
| Return level: length of record | 5 to 95 % bootstrap interval of the fitted level | `modellab/evt.bootstrap_levels` |
| Damage curve | Damage fraction at a depth from the site's JRC curve against the range across published continents | `finance/vulnerability` (Huizinga et al. 2017) |
| Flood: do independent sources agree | Severity steps between the JRC river map, the terrain model and the rain run | site profile entries |

## Stated assumption, not measured

Surface-water depth against curve number, roughness, drainage, rain and return period (`finance/pluvial.surface_water_sensitivity`). The ranges tried are chosen, and the entry says so.

## Not assessed (and why)

Internal climate variability (needs initial-condition ensembles), the choice of emissions pathway (run each), the JRC model chain's own form (no second global flood model is held), the cyclone event set (no independent stochastic set such as IRIS is held), and the asset values.

## Limits, stated

The ranges are of different quantities (mm, a ratio, a damage fraction), so the ranking is "which doubt is relatively widest", not a combined loss range. Producing the combined range needs the loss model run across the cases; that is not done here. The damage-curve entry is a transfer spread across continents, not the published confidence band of the curve.


## Do independent sources agree, for every peril (`peril_concordance`, `hazard_concordance`; 12 Nov 2026)

A single source is not enough to trust a hazard number. `uncertainty.hazard_concordance(profile)` reads a site profile and, for each peril, compares every source the engine holds for it. `POST /uncertainty/concordance` does the same over HTTP, and `site_ledger(..., profile=...)` / `POST /uncertainty/ledger` adds one entry per peril that ran, replacing hand-typed ratings.

| Peril | Sources the engine holds (profile keys) | What can be cross-checked |
|---|---|---|
| Flood | JRC river map (`river_flood`), rain-on-grid (`surface_water`), terrain + observed water (`flood_terrain_radar`), Deltares coast (`coastal_flood`), plus lab river/flood/surge/sea-level studies | Yes, several mechanisms; also recorded flood footprints |
| Wind and cyclone | cyclone track record (`cyclone`), all-cause extreme wind (`wind`) | Yes, two |
| Heat | site screening (`thermal_stress`), WBGT study (`heat`), land-cover urban heat | Yes, if they return ratings |
| Wildfire | fire-weather index (`fire`), fuel and land cover (`wildfire_fuel`) | Yes, two mechanisms |
| Water stress, drought | WRI Aqueduct (`water_stress`), soil water balance (`drought`) | Yes, but Aqueduct is basin scale |
| Landslide, debris flow | slope screening, infinite-slope physics, debris-flow screening | Yes, but all share one terrain model |
| Air quality | local screening (`air_local`), PM2.5 study (`air`) | Yes, may share a reanalysis |
| Cold | `cold_extreme` | **No: one source** |
| Earthquake | earthquake shaking study | **No: one source** |

Rules, in order of importance:
- The verdict is the worst credible reading, never an average. A site can be safe from a river and still flood from rain; a quiet source that cannot see a mechanism does not cancel a loud one that can.
- Agreement is stated separately: `agree` (within one severity step), `disagree` (two or more steps), `single_source`, `unrated` or `none`. If the engine holds only one source for a peril, the answer says it cannot be cross-checked.
- A source that gave no answer is listed under `not_assessed` with its reason; it is never counted as safe. A study that ran and returned a value but no severity rating is listed under `unrated` with its value, and is not compared, because turning it into a severity would need thresholds I would be inventing.
- Each peril names the inputs its sources share, so agreement is not mistaken for independence.
- For flood, recorded flood events with a mapped box over the site are shown as a separate check, with a warning if models say NEGLIGIBLE or LOW anyway. A box is the mapped area, not proof the site was wet.

Real run, Velachery pin (a subset of hazards, not the full profile): flood river HIGH vs coast NEGLIGIBLE (`disagree` by 3 steps, expected since they see different mechanisms); wind, water stress and cold each `single_source`; heat and earthquake did not return an answer in that run, so they are not assessed.

Known gaps this exposes, stated plainly:
- The WBGT heat, extreme-wind and drought lab studies return values without a severity rating, so they cannot yet join the comparison. Severity thresholds for them need a source, not a guess.
- No second global river model, no second seismic hazard source and no second cold-extreme source is held.
- Agreement among sources is not validation against observation. Validation is separate (`FLOOD_EVENT_BENCHMARK.md`, `RIVER_DEPTH_BENCHMARK.md`, `VALIDATION_LEDGER.md`).
