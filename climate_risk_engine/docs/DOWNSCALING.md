# Downscaling: what ClimRisk actually does, what it does not, and the open data around it

13 Nov 2026. Written after an interview question about the downscaling method. The short truth: **ClimRisk does not run its own downscaling model.** It inherits statistical downscaling done upstream, applies climate change as a change factor, and has a separate, physical step that downscales water levels to a site's own ground. Reports said more than that in places; those claims were corrected today (see "Claims corrected").

## 1. Climate projections: inherited statistical downscaling plus delta-change

- **Source.** World Bank Climate Change Knowledge Portal CMIP6 collection at 0.25 degrees, which CCKP states is statistically downscaled and bias-corrected from NASA NEX-GDDP-CMIP6 (Thrasher et al.). NEX-GDDP uses bias-correction and spatial disaggregation (BCSD): the global model's output is bias-corrected against an observational dataset and then disaggregated to a finer grid. We do not run that step; we read its result (`modellab/scenarios.py`).
- **What we take from it.** Ensemble change in a hazard index per scenario and 20-year window, as p10 / median / p90 across the models: annual maximum one-day rainfall (Rx1day) as a ratio, annual maximum daily temperature (TXx) as a shift in degrees C.
- **Delta-change (change-factor) method.** The change factor is applied to a present-day baseline from observation or reanalysis (the rain gauge and ERA5 for rain; see `finance/rain_design.py`). So the future number is "today's observed level times the model's relative change".

**What this assumes, and where it can be wrong.**
1. The relative change from a 25 km model also holds for point extremes. Short, intense, convective rain is poorly resolved at 25 km, so this is the weakest link for extreme rain.
2. The baseline's temporal structure (how storms arrive) does not change. Delta-change cannot represent a change in storm type, only in size.
3. Ensemble p10 to p90 spread is model spread, not a probability of the future.
4. We use one downscaled product. A dynamical regional model (CORDEX) or a different statistical method is not held.

## 2. Second model pipeline for future extreme rain (new)

`modellab/projection_check.py` compares the CCKP number with seven HighResMIP models (20 to 51 km) read through Open-Meteo, which bias-corrects each to ERA5-Land with a linear monthly correction and interpolates to about 10 km. The metric is the change in the 20-year mean of annual maximum daily rain, 2031-2050 over 1995-2014. The CCKP windows (2020-2039, 2040-2059) are interpolated linearly to 2031-2050 and the output says so. It appears in the uncertainty ledger as `projection_sources` when passed to `site_ledger(..., projection=...)`.

Chennai pin, run today: HighResMIP per-model ratios 0.77, 1.05, 0.98, 1.56, 1.51 (two more models were rate-limited and are reported as failed, not dropped silently); CCKP SSP5-8.5 interpolated p10 / median / p90 = 0.84 / 1.05 / 1.32. The HighResMIP median (1.05) is inside the CCKP band, but the HighResMIP range is wider (0.77 to 1.56). That spread is the honest message: at one location the sign of the change is not settled by the models.

Limits: different experiment (HighResMIP is not ScenarioMIP, forcing is close to SSP5-8.5), 20-year Rx1day samples are noisy, the free Open-Meteo tier is non-commercial, and agreement is a robustness check, not validation.

## 3. Downscaling water to the site's own ground (physical, separate)

`hydraulics.rain_field`: the catchment-scale rain-on-grid run (6 km radius, about 67 m cells, 24 h storm) gives a water-surface elevation; the site's own finer terrain is subtracted from it, so depth is the water surface minus the local ground. That is how a coarse flood result reaches the pin. Limits: no culverts or drains, no velocity or arrival time, no recorded surface-water outline to validate against in most places (`SURFACE_WATER_CHENNAI.md`).

## 4. Point versus grid rain (bias correction, not downscaling)

Gridded rain underestimates point extremes (Chennai 2015: gauge about 345 mm against about 101 mm in ERA5). The engine now anchors design rain to the nearest NOAA GHCN-Daily gauge when there is one with 25 or more annual maxima, and reports all sources side by side. This is an observational correction at the baseline, not a downscaling model.

## 5. Elevation heuristics in the older report path (heuristics, now labelled)

`climate/hazard_layers.py` has piecewise "sub-grid elevation" factors for flood, heat and water stress. They were attributed to LISFLOOD, WRI and FAO documents; they are not derived from those and are not validated. They are now labelled as heuristics in the code, in the methodology text and in the disclosure text. Only the lapse rate (about 6.5 K per km) is established physics. Where the engine has real terrain (HAND, the site DEM, the rain-on-grid run) the site profile and the asset twin use that instead.

## Claims corrected today

- PDF methodology said asset baselines were "delta-downscaled ... D_eff = max(0, W_downscaled - E_asset)". That formula is not in the code. Replaced with the real description.
- Disclosure text said "sub-grid elevation & coastal downscaling". Now says heuristic factors, not calibrated.
- The Open-Meteo connector docstring called its model "CMIP6, about 0.25 degrees, consistent with NEX-GDDP". It is a HighResMIP model with a different downscaling. Corrected.

## Open data: what is used, what is not

Found in the code and used: ERA5 and ERA5-Land (through Open-Meteo), NASA POWER, NOAA GHCN-Daily gauges, Sentinel-1 and Sentinel-2 imagery, ESA WorldCover, Copernicus / NASADEM / SRTM / Terrarium elevation tiles, JRC Global Surface Water, NASA FIRMS (MODIS/VIIRS fire), GloFAS river flow, Deltares coastal flood maps, WRI Aqueduct, OpenStreetMap, IBTrACS cyclone tracks, Copernicus EMS flood extents, the World Bank CCKP CMIP6 ensemble.

Open and relevant, not found in the code when this was written, so not used: GPM IMERG directly (NASA POWER's precipitation is IMERG-calibrated, so satellite rainfall was partly there). CHIRPS v3 has since been added for a one-storm check (`connectors/chirps.py`, see `INDIA_FLOOD_ATLAS_CHECK.md`); GHSL built-up and population layers for exposure; Dynamic World land cover; SMAP soil moisture; CORDEX regional climate models; NEX-GDDP-CMIP6 read directly at daily resolution (we only see CCKP's summary statistics, so no daily bias-corrected series for our own extreme-value fit).

The most valuable next additions, in order: satellite rainfall (CHIRPS or IMERG) as a third rain source; daily NEX-GDDP series so extremes are fitted on projected data rather than scaled by a ratio; a dynamical regional model for one region as a structural check.
