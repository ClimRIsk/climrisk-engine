# 3D hazard viewer

One screen (`/hazard3d`, sidebar "3D Hazard Viewer") that drapes a hazard model's output on real 3D terrain, with the study area outlined. Screening grade.

## What is on the map today

| Hazard | Surface | How it is made | Validation status |
|---|---|---|---|
| Landslide | Factor of safety after a design storm, 4 classes | Infinite-slope stability + rainfall pore pressure on the 10-30 m AWS terrain (`modellab.landslide`) | Verified against hand calculations; not validated against real landslides unless an inventory is supplied |
| Flood (terrain) | Height above nearest drainage, 4 classes | D8 flow routing (`terrain.hand`) | Area-level backtest vs FEMA NFIP claims; cell values not validated |
| Sea-level rise | Depth of connected still water at an AR6 or user water level | Bathtub with connectivity to open sea | Verified only; AR6 rise backtested at tide gauges |
| Release plume | Ground concentration decades, for one declared release | Gaussian plume (`release_plume`) | Closed-form benchmarks; not validated on a measured release |
| Slope | Slope classes | Finite differences | Exact on a plane |

## What deliberately has no map

Heat, drought, earthquake ground motion, wildfire weather, design cyclone wind, air quality, basin water stress: the engine models these as one value per site (or a regional record), not as a surface. Asking for a map returns `available: false` with the reason, never an invented raster. River-flood depth for a return period exists in the asset twin's solver and is not exposed in this viewer yet.

Null is not zero: cells a model did not evaluate (sea cells for landslide, the near field of a plume) are transparent and counted as "not evaluated", not as "low".

## Base map

- **Google Photorealistic 3D Tiles** (Apple-Maps-style 3D meshes) when the user stores their own Google Maps Platform key (Map Tiles API enabled, billing on). The key is kept in `~/.climrisk/maps_key.json` (owner-only), returned only to loopback callers, and sent only to Google. Google bills per session and sets terms and attribution; check them. **This path could not be tested without a key**: the code follows Google's documented CesiumJS usage, but nothing has rendered from Google's servers yet.
- **Fallback (no key)**: OpenStreetMap imagery on real terrain relief built from the same open elevation tiles the models use. No 3D buildings.

Apple Maps' 3D (Flyover) is not usable: MapKit JS exposes no way to overlay custom raster hazard layers on it that I could confirm.

## Study area

KML, KMZ, GeoJSON upload or click-to-draw. Shares of each class are reported for the whole box and inside the polygon. Holes are ignored (and the response says so).

## Files

`src/cri/hazardsurface.py`, `src/cri/geo_io.py`, `src/cri/api/hazard3d_api.py`, `src/cri/api/hazard3d.html`, `tests/test_hazardsurface.py`.
