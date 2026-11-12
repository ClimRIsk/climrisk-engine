# Rain-driven flooding: why Chennai 2015 was missed, and what changed (11 Nov 2026)

## What was wrong

For Velachery (12.9815, 80.2180), which stood in 1 to 2 m of water on 1 to 2 December 2015, the engine said:

| Layer | Before | Why it missed |
|---|---|---|
| Surface-water screen (`site_pluvial`) | **0.07 m, LOW** | 600 m window with free edges: water from kilometres upslope never arrived. 1 to 3 h burst, 12 mm/h sewer allowance applied as if it lasted a day. Design rain came from one smooth reanalysis. |
| Design rain | ERA5: 2015 annual maximum **101 mm**; 1-in-100 daily **202 mm** (365 mm over 3 days). NASA POWER: 191 mm / 350 mm | Grids average over tens of km. The Minambakkam gauge (4 km away) recorded **344.9 mm on 2 Dec 2015**; 436.8 mm over 3 days. |
| 3D twin | River 1-in-100 only (1.7 m from the JRC map); **no rain scenario** | The twin imposes a map water level; it has no rainfall at all. |
| Terrain second opinion | Used only when the JRC map was unavailable | Never raised a low river or surface-water rating. |

## What changed

1. **Gauge-anchored design rain** (`finance/rain_design.py`). ERA5, NASA POWER and the nearest long-record NOAA GHCN-Daily gauge (public domain, up to 60 km, at least 25 years). Annual maxima of 1-day and 3-day totals, Gumbel fit, design value = largest source, the others and the distance stated. At Minambakkam the 1-in-100 daily value is **325 mm** (3-day **535 mm**) against 202 / 365 mm gridded; the gauge's own 2015 value (345 mm) sits near its 1-in-100 level, which matches how the event was described.
2. **Catchment-scale run** (`finance/pluvial.site_surface_water`). Same local-inertial solver, about 12 km window at 67 m cells, 24 h storm, two bounds: drains working (12 mm/h removed from built-up ground) and drains overwhelmed (none). The rating uses the overwhelmed bound. Results are cached on disk.
3. **Rain scenario in the 3D twin** (`hydraulics.rain_field`, `simulate_rain`, `asset_twin.rain_scenario`). The coarse run gives a water-surface elevation; depth on the site is that surface minus the site's own ground. No velocity or arrival time is claimed. A perimeter wall is stated not to keep rain out.
4. **Site profile and report**: new `surface_water` entry with the range, the rain source and the gridded comparison; the river entry takes the terrain-position second opinion (rank at least 0.8 raises NEGLIGIBLE or LOW to MODERATE, never above, never lowers a mapped depth). The report's surface-water zone uses the new run.
5. **Limit texts** now state what was measured instead of a blanket warning.

## Result at Velachery and neighbours (1-in-100, drains overwhelmed)

**Correction (12 Nov 2026).** The first version of this page reported Velachery as 0.97 m CRITICAL. That figure was the deepest cell in a 3x3 window around the pin, and at 67 m cells the window is about 200 m wide. The pin cell itself is 0.01 m. The headline is now the **mean over that window**, and the pin cell and deepest cell are reported next to it.

| Site | Before (600 m model) | Window mean (rating) | Pin cell | Deepest cell within ~100 m |
|---|---|---|---|---|
| Velachery (flooded 2015) | 0.07 m LOW | **0.26 m MODERATE** (drains working 0.25 m) | 0.01 m | 0.97 m |
| Anna Nagar (flooded 2015, less deep) | 1.14 m | **0.40 m HIGH** | 0.83 m | 0.98 m |
| T Nagar (higher ground) | 0.67 m | 0.02 m NEGLIGIBLE | 0.04 m | 0.04 m |
| Mudichur (flooded 2015) | 0.02 m | **0.02 m NEGLIGIBLE: a miss** | 0.01 m | 0.08 m |

Mudichur was flooded in December 2015 by the Adyar river and the reservoir release upstream. Rain-on-grid cannot see water that arrives from a river or a release, and this layer does not, so it reads dry. That is a limit of the method, not a tuning problem.

Control sites, rerun on 12 Nov 2026 with the window-mean metric (1-in-100, drains overwhelmed; rain from the nearest gauge, global terrain tiles):

| Site | Window mean (rating) | Pin cell | Deepest cell nearby | Design rain (gauge / best grid) |
|---|---|---|---|---|
| Shimla (hillside) | 0.00 m NEGLIGIBLE | 0.00 m | 0.00 m | 209 / 185 mm |
| Denver | 0.02 m NEGLIGIBLE | 0.02 m | 0.02 m | 87 / 72 mm |
| Houston (downtown) | 0.01 m NEGLIGIBLE | 0.01 m | 0.02 m | 279 / 207 mm |
| Amsterdam (polder, pumps not modelled) | 0.09 m LOW | 0.14 m | 0.14 m | 69 / 47 mm |
| Mumbai Kurla | 0.08 m LOW | 0.04 m | 0.25 m | 486 / 296 mm |

The controls that should be dry are dry. Mumbai Kurla reads LOW even though the 2005 floods hit that area: one pin in one 6 km window, with the Mithi river doing much of the damage (a river mechanism, the same limit as Mudichur). Houston's own record is mostly bayou and reservoir water too. These are screening results, not a skill score: five negatives and mixed positives do not make an AUC.

## Sensitivity at Velachery (one input at a time, window-mean depth)

| Input varied | Range tried (a stated assumption) | Depth |
|---|---|---|
| Design rain | gauge 325 mm; gridded only 202 mm; gauge -25 % / +25 % | 0.25 to 0.27 m |
| Curve number | 65 to 85 | 0.26 m (both) |
| Manning roughness | 0.03 to 0.08 | 0.25 to 0.28 m |
| Drains | working to overwhelmed | 0.25 to 0.26 m |

Nothing moves it. The response to storm size shows why: with the same terrain, the deepest cell near the pin is 0.19 m at 32 mm of rain, 0.39 m at 65 mm, 0.89 m at 130 mm, and 0.97 m at 325 mm. The nearby low ground fills to its spill height by about 130 mm, and above that the water level is set by the terrain, not by the storm. **At sites like this the number is a ponding flag, not a return-period depth**, and a 1-in-100 label on it overstates what the storm size contributes. `finance/pluvial.surface_water_sensitivity` now reports the rain-response curve and the rain at which depth saturates, so the output says this itself.

## What is and is not shown (read it straight)

- This is a case check on one event plus controls, with one miss (Mudichur) and a metric that I first reported too high and then corrected. It is **not** a skill score. The earlier 600 m configuration scored AUC 0.64 to 0.71 on English recorded outlines and was beaten by a plain height-above-low-ground score (0.97 at Oxford and Reading, `docs/PLUVIAL_BENCHMARK.md`). The new configuration has not been rescored against those outlines; that is the next test and it needs the EA LiDAR tiles.
- Terrain position ranks flooded ground above dry ground with **AUC about 0.69 on 23 independent satellite-observed floods** (Global Flood Database, MODIS 250 m, CC BY-NC, measurement only). Those maps cannot see dense-urban flooding, so the result says nothing about cities.
- Sewers, culverts, pumps and buildings are not resolved; 67 m cells smooth small depressions; the gauge is a point measurement 4 km from the site; the Gumbel level underestimates the largest event at Mumbai. Every output carries these.
- Run time is about 20 s per bound (40 s cold), then cached.
