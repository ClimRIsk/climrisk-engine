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


## Flood: do independent sources agree (`flood_concordance`, added 12 Nov 2026)

`uncertainty.flood_concordance(profile)` reads a site profile and compares every flood source in it, by mechanism: river (JRC CEMS-GloFAS map), rain on the ground (the catchment-scale rain-on-grid run), terrain position and observed water (HAND and radar history), and coast and sea level (Deltares). Pass `profile` to `site_ledger` or to `POST /uncertainty/ledger` and it replaces the hand-typed ratings.

Rules, in order of importance:
- Verdict is the worst credible reading, never an average. A site can be safe from a river and still flood from rain; a quiet source that cannot see a mechanism does not cancel a loud one that can.
- Agreement is stated separately: `agree` (within one severity step), `disagree` (two or more steps), `single_source` (one source only, so flood has not been cross-checked) or `none`.
- A source that gave no answer is listed under `not_assessed` with its reason. It is never counted as dry.
- Recorded flood events with a mapped box over the site are shown as a separate check, and a warning is added if models rate the site NEGLIGIBLE or LOW anyway. A box is the mapped area, not proof the site was wet.

Real run, Velachery pin (river and coastal sources only; the rain run was not part of that call): river HIGH, coast NEGLIGIBLE, `disagree` by 3 steps. That is the expected pattern, since different mechanisms are being seen.

Limits: the sources are not fully independent (the river map and the terrain model share elevation data), no second global river model is held, and agreement among sources is not a validation against observed depths. Validation against recorded floods is separate (`FLOOD_EVENT_BENCHMARK.md`: terrain model AUC about 0.76 on held-out events; `RIVER_DEPTH_BENCHMARK.md`).
