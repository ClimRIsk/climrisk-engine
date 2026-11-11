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

## Result at Velachery and neighbours (1-in-100, drains overwhelmed / working)

| Site | Before | Now |
|---|---|---|
| Velachery (flooded 2015) | 0.07 m LOW | **0.97 m / 0.93 m CRITICAL** |
| Anna Nagar (flooded, less deep) | 1.14 m | 0.99 m CRITICAL (one 6 km run) |
| T Nagar (high ground) | 0.67 m CRITICAL | 0.04 m NEGLIGIBLE (one 6 km run) |

Control sites, same configuration, live rain: Shimla hillside 0.003 m, Denver 0.02 m, Houston downtown 0.02 m, Amsterdam polder 0.13 m (LOW; pumps not modelled), Mumbai Kurla 0.25 m MODERATE (gauge 1-in-100 daily 482 mm; the 2005 record at that gauge was 944 mm, so the Gumbel level is still below the largest event on record).

## What is and is not shown (read it straight)

- This is a case check on one event plus controls. It is **not** a skill score. The earlier 600 m configuration scored AUC 0.64 to 0.71 on English recorded outlines and was beaten by a plain height-above-low-ground score (0.97 at Oxford and Reading, `docs/PLUVIAL_BENCHMARK.md`). The new configuration has not been rescored against those outlines; that is the next test and it needs the EA LiDAR tiles.
- Terrain position ranks flooded ground above dry ground with **AUC about 0.69 on 23 independent satellite-observed floods** (Global Flood Database, MODIS 250 m, CC BY-NC, measurement only). Those maps cannot see dense-urban flooding, so the result says nothing about cities.
- Sewers, culverts, pumps and buildings are not resolved; 67 m cells smooth small depressions; the gauge is a point measurement 4 km from the site; the Gumbel level underestimates the largest event at Mumbai. Every output carries these.
- Run time is about 20 s per bound (40 s cold), then cached.
