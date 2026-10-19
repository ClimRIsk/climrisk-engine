# Surface-water (rain-on-grid) model — benchmark

**Model:** `finance/pluvial.py` → `site_pluvial`. Local-inertial shallow-water solver (`modellab/flood.simulate`, mass-conserving, checked in `modellab.flood.run_benchmarks`) on the best open terrain (1 m EA LiDAR in England, resampled to ~13 m; global tiles elsewhere), driven by a design storm: ERA5 annual-maximum daily rain → Gumbel return level → share of the daily total falling in the storm (1 h: 0.35, 3 h: 0.50, 6 h: 0.65, an assumption) → pervious ground sheds SCS-CN excess (CN 75), built-up ground (ESA WorldCover share) sheds everything above a 12 mm/h sewer allowance. Wired into the report's surface-water zone as a **second opinion that can only raise the rating**; switch off with `CRI_PLUVIAL=0`.

**Test:** `scripts/pluvial_benchmark.py`, scored on Environment Agency Recorded Flood Outlines with cause "local drainage/surface water" (small polygons only). Positives are points inside an outline; negatives are points 100 m to 1 km outside any recorded outline, so both classes sit in the same neighbourhood. ROC AUC, nothing fitted, 1-in-50 design storm, 150 + 150 points per town.

| Town | Model on LiDAR | Same model, global tiles | Pit (blue-spot) depth | Height above local low (500 m) |
|---|---|---|---|---|
| Oxford | 0.66 | 0.46 | 0.65 | **0.97** |
| Reading | 0.64 | 0.54 | 0.72 | **0.97** |
| Bristol | 0.71 | 0.63 | 0.78 | 0.71 |

Five further towns had no usable small surface-water outline in the box and were skipped.

## What this says (read it straight)

- **LiDAR helps the model in every town** (+0.08 to +0.20 AUC over the global tiles). On the global tiles at Oxford the model has no skill (0.46).
- **The model has modest skill (0.64–0.71) and is beaten by a plain terrain score.** Height above the lowest nearby ground scored 0.97 at Oxford and Reading, and the blue-spot depth beat the model at Reading and Bristol. At Oxford and Reading the recorded "surface-water" outlines sit on low ground next to the river, so that score is partly picking up fluvial floodplain that the outline cause field does not separate; do not read 0.97 as surface-water skill. Three towns is a small sample.
- **Depths are low.** ERA5 daily rain is smooth and the 12 mm/h sewer allowance removes most light storms, so a 1-in-100 storm often gives a few cm. A shallow or dry result is not a clean bill; the report says so. Short convective bursts, sewers, culverts and buildings are not resolved.
- Nothing was tuned on these three towns (storm share, CN and allowance are the values written before the run).

## Use in the product

Screening only. It raises, never lowers, the surface-water rating; every result lists terrain source, design rain, storm share and sewer allowance. It is not a substitute for a drainage study or the national surface-water maps where those exist.
