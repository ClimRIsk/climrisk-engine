# Landslide surface: check against NASA's Global Landslide Catalog

**Verdict: slope ranking beats chance at 5 of 5 sites. It is not a validation and not a calibration.**

Method (`tools/landslide_validate.py`): NASA Global Landslide Catalog (Kirschbaum et al. 2010, 2015; 11,033 events, 2007-2016 export). Only events with `location_accuracy = exact`, rain-triggered, landslide/mudslide/debris-flow/rock-fall categories. Greedy clustering within 5 km; sites with at least 5 events. Terrain: AWS terrain tiles, about 9.6 m cells at these sites. Score: AUC of the engine's stability ranking, event cells versus all other land cells (0.5 = no skill).

| Site (lat, lon) | Events | AUC, event cell only | AUC, event within ~95 m |
|---|---|---|---|
| 12.35, -84.81 (Nicaragua) | 14 | 0.67 | 0.77 |
| 45.52, -122.70 (Portland, OR) | 9 | 0.60 | 0.71 |
| 47.72, -122.28 (Seattle area) | 7 | 0.95 | 0.85 |
| -22.96, -43.21 (Rio de Janeiro) | 5 | 0.66 | 0.66 |
| 47.55, -122.24 (Seattle area) | 5 | 0.60 | 0.57 |
| Mean | 40 | 0.70 | 0.71 |

What this does and does not show
- At fixed soil strength the factor of safety is monotone in slope, so this AUC measures the **slope ranking only**. The rainfall and pore-pressure term is not tested by it.
- 40 events at 5 sites is a small sample. Per-site AUCs range 0.57-0.95; no confidence interval is claimed.
- The catalogue is media-reported, so it is biased to where people live. Events near settlements sit on terrain unlike the surrounding landscape. The bias can push the score either way.
- Four of five sites are in the US and Brazil. Nothing here says the surface works in other settings, soils or climates.
- The class edges (FS 1.0 / 1.25 / 1.5) and generic soil strength are not calibrated to any inventory.
- Raw results: `docs/landslide_validation_glc.json`. Reproduce: `python tools/landslide_validate.py <GLC csv> out.json` (needs network for terrain tiles).
