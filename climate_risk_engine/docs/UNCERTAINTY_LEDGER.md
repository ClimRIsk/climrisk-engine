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
