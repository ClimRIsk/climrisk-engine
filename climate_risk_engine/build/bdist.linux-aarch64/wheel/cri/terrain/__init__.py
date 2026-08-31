"""
CRI Terrain Hazard Analysis Module
====================================

Site-specific terrain morphology assessment using 30m SRTM DEM data.

Unlike regional hazard layers (which classify broad zones), this module
analyses the actual terrain surface around a specific asset to identify
physically realistic mechanisms by which hazards can reach it.

Key concepts
------------
Terrain position (TPI)
    Classifies the asset's position in the landscape: ridge, upper slope,
    mid slope, footslope, or valley bottom. A negative TPI indicates the
    asset sits in a depression relative to its neighbourhood — a debris
    accumulation zone.

Upslope catchment
    The upstream drainage area feeding the asset's cell via D8 flow paths.
    Large catchment + steep upslope gradient = high debris flow potential.

Debris flow susceptibility
    Empirical thresholds (Lateltin 1997, Hungr et al. 2014, Jakob 2005):
    - Slope >25° in source zone
    - Catchment area >0.5 km² draining toward asset
    - Terrain position = footslope or valley (negative TPI)

Landslide susceptibility
    Asset-level slope + soil susceptibility inferred from DEM morphology:
    - Slope >15°: potential instability
    - Slope >30°: high susceptibility
    - Convex-to-concave profile transitions: failure planes

Entry point
-----------
    from cri.terrain import TerrainHazardAnalyzer, TerrainHazardResult

    analyzer = TerrainHazardAnalyzer()
    result   = analyzer.assess(lat=22.8046, lon=86.2029, asset_name="Jamshedpur Steel Plant")
    print(result.summary)
"""

from .analyzer import TerrainHazardAnalyzer, TerrainHazardResult

__all__ = ["TerrainHazardAnalyzer", "TerrainHazardResult"]
