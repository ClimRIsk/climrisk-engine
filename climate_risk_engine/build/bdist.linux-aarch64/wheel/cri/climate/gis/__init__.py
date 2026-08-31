"""GIS resolution sub-package for the CRI engine.

Converts raw lat/lon coordinates on an Asset into the spatial attributes
consumed by PhysicalHazardEngine:
  - Elevation (metres above sea level)
  - Coastal distance (km, straight-line haversine to nearest coastline point)
  - Köppen–Geiger climate zone
  - Permafrost presence flag
  - Cyclone belt presence flag
  - Arid / dryland flag (for dust-storm, drought)
  - Flood plain proximity flag
  - Mean winter temperature (°C, used for blade-icing / freeze-thaw)

Public API
----------
from cri.climate.gis import resolve

attrs = resolve(lat=-22.3, lon=118.6)
# attrs.elevation_m, attrs.coastal_km, attrs.koppen_zone, ...
"""

from .resolver import AssetGISAttributes, resolve  # noqa: F401
