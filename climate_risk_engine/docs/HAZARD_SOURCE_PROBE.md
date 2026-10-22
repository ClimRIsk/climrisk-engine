# Open data for the hazards that still have no site model (probe of 22 October 2026)

From the build sandbox: which sources answer, and what could be built and tested from them. "Reached" means an HTTP request returned data; it does not check licence fit for paid use.

| Hazard | Source reached | Can it be tested? | Status |
|---|---|---|---|
| Hail and convective storm | NOAA SPC storm reports 1955-2024 (hail 7 MB, wind 10 MB, tornado 8 MB CSV, public domain) | United States only: ERA5 convective ingredients (CAPE, freezing level; Open-Meteo hourly) against report density. Europe: ESSL's ESWD site answers but its data need registration | Built: CONVECTIVE_MARINE_BENCHMARK.md (hold-out test, US only) |
| Tornado | SPC tornado CSV 1950-2024, same source | United States only; rate map against reports | Built: CONVECTIVE_MARINE_BENCHMARK.md |
| Lightning | NASA LIS/OTD flash-rate archive refused the connection | Not reached | No source |
| Extreme cold | GHCN-Daily (same stations as the freeze-thaw test) | Yes, as freeze-thaw | Buildable quickly; the UTCI cold-stress module already covers hours of cold stress |
| Marine heatwave | NOAA OISST via ERDDAP (daily sea-surface temperature since 1981) | Yes against named events (Hobday et al. definition) | Built: CONVECTIVE_MARINE_BENCHMARK.md |
| Extratropical windstorm | ERA5 gusts via Open-Meteo (reached). PERILS market losses are not open | Footprint check against named storms only; no loss test | Hazard exists; no loss model |
| Subsidence | EGMS (Copernicus ground motion) answers but needs a login for tiles | Europe only, with registration | Not reached |
| Dust storm | No dataset probed | - | No source |
| Blade icing | No ice accretion measurements | No | Not built |
| Flash and compound flood | Surface-water model and flood outlines already in the engine | Partly (PLUVIAL_BENCHMARK.md) | Existing |
| Avalanche, permafrost, GLOF, freeze-thaw | SLF accidents (EnviDat), Obu map (PANGAEA), Lützow-Veh database (Zenodo), GHCN | Yes | Built: CRYOSPHERE_BENCHMARK.md |
