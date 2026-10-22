# Cold-region hazards: what was built and how it was tested

Built 22 October 2026 in `src/cri/cryosphere.py`; tests in `tests/test_cryosphere.py`; benchmark code in `scripts/cryosphere_benchmark.py`. Replaces the regional parametric scores for these perils with models that read a time series, a terrain model, a published map or a record at the site. All four appear in the physical report as **context** zones: they carry their evidence grade, and they never change the overall rating, the issue list or the measures.

| Hazard | What it does | Data | Tested against | Result | Grade |
|---|---|---|---|---|---|
| Freeze-thaw, frost days, ice days | Counts per year from 20 years of daily minimum and maximum air temperature | ERA5 via Open-Meteo archive, 2004-2023 | GHCN-Daily, 44 stations | See below. Ranks sites well, mild under-count in the freeze-thaw-richest climates | B |
| Avalanche release terrain | Share of terrain within 150 m at 28-50 degrees (dry-slab release range), steepest slope, dominant aspect | AWS terrain tiles, zoom 14 (about 10 m) | SLF Swiss avalanche accidents, 300 events | AUC 0.59 against nearby matched terrain (95 % range 0.52-0.66). Weak skill | C |
| Permafrost | Probability at the site from the published map | Obu et al. 2019 (PANGAEA 10.1594/PANGAEA.888600, CC BY 3.0), resampled from 1 km to 10 km | Resampling check against the 1 km map; the map itself is not re-tested here | 10 km grid reproduces the 1 km map (below) | B as published |
| Glacial lake outburst floods | Lists reported events within 50 km | Lützow and Veh global GLOF database v4.2 (Zenodo 7330344, CC BY 4.0), 6,263 located events | Not applicable: a record, not a model | Reported events only; no unburst lakes, no probability | record |

## Freeze-thaw (44 stations)

Method: GHCN-Daily stations with 20-year daily minimum and maximum temperature (2004-2023), blank quality flag only, years with at least 350 complete days, at least 10 such years. 90 stations were drawn at random from one-per-4x5-degree-cell candidates with complete records listed in the inventory; 46 failed the completeness rule, leaving 44 (latitude -47 to 72, elevation 5-1,184 m, median 150 m; 19 in Russia, 6 China, 6 Canada, 3 US, the rest scattered, so **cold continental Eurasia dominates the sample**). Observed counts from the station; modelled counts from the ERA5 cell at the station coordinates, same years. Freeze-thaw day = minimum below 0 C and maximum above 0 C.

| Measure | Mean observed | Mean bias (model minus observed) | Mean absolute error | Rank correlation | Within 10 days |
|---|---|---|---|---|---|
| Frost days (min < 0) | 168.6 | +0.6 | 14.0 | 0.965 | 61 % |
| Ice days (max < 0) | 106.1 | +5.0 | 6.2 | 0.996 | 89 % |
| Freeze-thaw days | 61.6 | -4.2 | 10.7 | 0.801 | 66 % |

Bands (low under 10, moderate under 40, high under 80, very high above, days a year): the modelled band matches the station's band at 77 % of stations and is within one band at 98 %. Bias by observed band: very high (9 stations) -10.9 days, high (29) -3.6, moderate (4) +4.1, low (2) 0.0. Worst cases: Calico Rock, Arkansas (observed 86, modelled 43), Marion Island (48, 6), Carmacks, Yukon (109, 79), Graz (67, 98), Dalatangi, Iceland (72, 101). Station-minus-grid elevation offset explains little of the error (correlation -0.14).

What this does not show: freeze-thaw of ground, pavement or building surfaces (air temperature only), damage (the band edges are screening judgement and no durability data was used), the tropics and the Southern Hemisphere (2 stations south of the equator), or the 1-in-50 cold extreme. Per-station results: `docs/cryosphere_freeze_thaw_stations.json`.

## Avalanche release terrain (300 accidents)

Data: SLF avalanche accidents in Switzerland since 1970/71 (WSL Institute for Snow and Avalanche Research, EnviDat 10.16904/envidat.411, WSL data policy; raw coordinates are not redistributed here). Kept: backcountry (`tour`) and off-piste accidents with start-zone coordinates accurate to 100 m or better, inside the Swiss Alps; 300 drawn at random. Controls: one point per event, 2-8 km from it in a random direction. Score: AUC of the release-terrain share (event versus control), 1 = every event above every control, 0.5 = chance.

| Statistic | All controls (300) | Controls within 200 m of the event's elevation (77) |
|---|---|---|
| Share of 28-50 degree cells within 150 m | 0.617 | 0.589 (bootstrap 95 % range 0.519-0.655) |
| Same, 60 m radius | 0.562 | 0.535 |
| Same, 300 m radius | 0.612 | 0.573 |
| Steepest slope within 150 m | 0.659 | 0.646 |

A release-range cell exists within 60 m of 37 % of the recorded start zones and of 24 % of controls. The first design tried (500 m radius, 19 m cells, controls scattered across Switzerland) gave AUC 0.69 against all controls but 0.48 against elevation-matched ones, which is no skill; the settings above were chosen after that, using the same events, so the 0.59 is not an out-of-sample figure.

Limits: accident locations are where people ski, not where snow slides, and the controls include terrain nobody visits; start-zone coordinates carry up to 100 m of error against a 10 m model; the test is Swiss Alps only; and it covers release terrain, not snowpack, forest, runout or frequency. The report shows the zone only where frost days reach 60 a year and release terrain exists, titles it "weak test", and says it is a pointer, not a rating.

## Permafrost (resampling check)

The published 1 km probability map (1.9 GB GeoTIFF) is block-averaged to 10 km and stored as 212 KB (`src/cri/data/permafrost_obu2019_10km.npz`, uint8 percent). Check: 698 random land points between 30 N and 80 N, 10 km value against the 1 km value at the same point: mean absolute difference 0.023, correlation 0.984, same zone class (isolated or none, sporadic, discontinuous, continuous) at 94.7 % and within one class at 99.6 %. The lookup returns "not covered" south of 25 N, in the Southern Hemisphere and over water or glaciers. It does not return mean annual ground temperature (the separate temperature raster was not downloaded). The map is a model (TTOP) calibrated to boreholes by its authors; nothing here re-tests it. Mountain sites sit inside 10 km cells that mix valley and summit.

## Glacial lake outburst floods

6,325 rows in the v4.2 database; 6,263 have coordinates. Regions with the most: Alaska 2,239, Central Asia 924, Iceland 794, southern Andes 610, Central Europe 449. Many Alaska and Iceland rows are repeated subglacial jökulhlaups from the same lake. `glof_history(lat, lon, radius_km)` returns count, lakes, years, reported fatalities and the nearest events with distance to the lake. Dates are missing for some rows. Remote events are under-reported, unburst lakes are absent, and a flood travels far further than the radius.

## Not built, and why

Blade icing: no open measurement of ice accretion or turbine icing downtime to test against; a meteorological definition alone would be a rating with no evidence. Avalanche runout, snowpack and permafrost thaw settlement: need data that were not found openly. Lightning: the NASA LIS/OTD flash-rate archive refused the connection from the build sandbox.

Reproduce (needs network): `python scripts/cryosphere_benchmark.py freeze_thaw <stations.json> out.json`; `... avalanche2 <slf.csv> out.json 300`. Station selection and the permafrost resample are described above.
