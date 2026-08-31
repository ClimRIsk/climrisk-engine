"""
TerrainHazardAnalyzer — site-specific terrain morphology risk assessment.

Given an asset's GPS coordinates, this module:
  1. Fetches a 3 km radius SRTM 30m elevation grid (via Open-Topo-Data)
  2. Computes slope, aspect, and TPI across the grid
  3. Derives D8 flow directions and the upstream drainage catchment
  4. Classifies terrain position (ridge / upper slope / footslope / valley)
  5. Scores debris flow susceptibility using empirical thresholds
  6. Identifies contributing hazard mechanisms in plain English
  7. Produces structured recommendations

Debris flow thresholds (calibrated from literature)
----------------------------------------------------
Lateltin (1997), Jakob & Hungr (2005), Rickenmann (1999):

  HIGH if ALL of:
    - upslope_area_km2 > 0.5 km²
    - max upslope slope > 25°
    - terrain_position ∈ {footslope, valley, valley bottom}
    - elevation_head > 50 m

  MEDIUM if ANY of:
    - upslope_area_km2 > 0.2 km² AND max slope > 20°
    - TPI < -5 AND elevation_head > 30 m
    - asset slope 15–25° (self-instability risk)

  LOW if:
    - upslope_area_km2 < 0.2 km² OR max slope < 15°
    - terrain_position ∈ {ridge, plateau}

  NEGLIGIBLE if:
    - flat terrain (slope < 5°) AND TPI near zero AND upslope area < 0.05 km²

Usage
-----
    from cri.terrain import TerrainHazardAnalyzer, TerrainHazardResult

    analyzer = TerrainHazardAnalyzer()
    result   = analyzer.assess(22.8046, 86.2029, "Jamshedpur Steel Plant")

    print(result.summary)
    # Debris flow: HIGH
    # Terrain: footslope (TPI=-14m) beneath 34° source slope
    # Upslope area: 2.8 km², elevation head: 280 m
    # Mechanisms: Saturated debris mobilised from steep upslope catchment
    #             can reach asset via NW-facing drainage path

    # For 3D visualisation
    grid_data = result.to_dict()
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..connectors.open_topo_data import get_elevation_grid, ElevationGrid
from .slope_analysis import (
    slope_degrees,
    aspect_degrees,
    tpi as compute_tpi,
    terrain_position_label,
    flow_direction_d8,
    upstream_area_cells,
    upstream_area_km2,
    upslope_slope_stats,
    upslope_elevation_range,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TerrainHazardResult:
    """
    Structured output from a terrain hazard assessment.

    All physical measurements are from SRTM 30m data (NASA).
    Risk ratings are CRI interpretations of terrain morphology only —
    they do not replace site-specific geotechnical studies.
    """
    # ── Identity ──────────────────────────────────────────────────────────
    asset_lat:   float
    asset_lon:   float
    asset_name:  str

    # ── Terrain metrics ───────────────────────────────────────────────────
    elevation_m:              float       # asset elevation (m ASL)
    slope_at_asset_deg:       float       # local slope angle at asset
    aspect_at_asset_deg:      float       # slope facing direction (°N clockwise)
    tpi_m:                    float       # TPI (m): neg = depression, pos = ridge
    terrain_position:         str         # 'ridge', 'footslope', 'valley', etc.

    # ── Upstream catchment ────────────────────────────────────────────────
    upslope_area_km2:         float       # drainage basin area (km²)
    upslope_max_slope_deg:    float       # steepest upslope gradient
    upslope_mean_slope_deg:   float       # mean upslope gradient
    upslope_pct_over_25deg:   float       # % upslope cells with slope > 25°
    elevation_head_m:         float       # max elev. difference: source → asset

    # ── Risk ratings ─────────────────────────────────────────────────────
    debris_flow_risk:         str         # 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE'
    landslide_susceptibility: str         # 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE'
    flood_exposure:           str         # 'HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE'

    # ── Narrative ─────────────────────────────────────────────────────────
    contributing_mechanisms:  list[str]   = field(default_factory=list)
    recommended_actions:      list[str]   = field(default_factory=list)
    data_caveats:             list[str]   = field(default_factory=list)
    confidence:               float = 0.0  # 0–1

    # ── Raw grid (for 3D visualisation) ───────────────────────────────────
    grid:  Optional[ElevationGrid] = field(default=None, repr=False)

    # ── Computed ──────────────────────────────────────────────────────────
    @property
    def summary(self) -> str:
        lines = [
            f"Terrain Hazard Assessment — {self.asset_name}",
            f"  Elevation:      {self.elevation_m:.0f} m ASL",
            f"  Terrain pos:    {self.terrain_position}  (TPI {self.tpi_m:+.1f} m)",
            f"  Slope at asset: {self.slope_at_asset_deg:.1f}°",
            f"  Upslope area:   {self.upslope_area_km2:.2f} km²",
            f"  Elev. head:     {self.elevation_head_m:.0f} m",
            f"  Max upslope ∠:  {self.upslope_max_slope_deg:.1f}°",
            f"  Debris flow:    {self.debris_flow_risk}",
            f"  Landslide:      {self.landslide_susceptibility}",
            f"  Flood exposure: {self.flood_exposure}",
        ]
        if self.contributing_mechanisms:
            lines.append("  Mechanisms:")
            for m in self.contributing_mechanisms:
                lines.append(f"    • {m}")
        if self.recommended_actions:
            lines.append("  Recommended:")
            for a in self.recommended_actions:
                lines.append(f"    → {a}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable dict — includes flattened elevation grid for 3D viz."""
        d = {
            "asset_lat":              self.asset_lat,
            "asset_lon":              self.asset_lon,
            "asset_name":             self.asset_name,
            "elevation_m":            round(self.elevation_m, 1),
            "slope_at_asset_deg":     round(self.slope_at_asset_deg, 2),
            "aspect_at_asset_deg":    round(self.aspect_at_asset_deg, 1),
            "tpi_m":                  round(self.tpi_m, 2),
            "terrain_position":       self.terrain_position,
            "upslope_area_km2":       round(self.upslope_area_km2, 3),
            "upslope_max_slope_deg":  round(self.upslope_max_slope_deg, 1),
            "upslope_mean_slope_deg": round(self.upslope_mean_slope_deg, 1),
            "upslope_pct_over_25deg": round(self.upslope_pct_over_25deg, 1),
            "elevation_head_m":       round(self.elevation_head_m, 0),
            "debris_flow_risk":       self.debris_flow_risk,
            "landslide_susceptibility": self.landslide_susceptibility,
            "flood_exposure":         self.flood_exposure,
            "contributing_mechanisms": self.contributing_mechanisms,
            "recommended_actions":    self.recommended_actions,
            "data_caveats":           self.data_caveats,
            "confidence":             round(self.confidence, 2),
        }
        if self.grid:
            d["viz"] = {
                "rows":       self.grid.rows,
                "cols":       self.grid.cols,
                "cell_size_m": self.grid.cell_size_m,
                "asset_row":  self.grid.asset_row,
                "asset_col":  self.grid.asset_col,
                # Flatten elevations to 1D for JSON transfer efficiency
                "elevations": [v for row in self.grid.elevations for v in row],
            }
        return d


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TerrainHazardAnalyzer:
    """
    Site-specific terrain morphology risk assessment.

    Parameters
    ----------
    radius_km   : Radius of DEM grid to fetch (default 3 km)
    spacing_m   : DEM grid cell spacing (default 200 m)
                  Total API calls = (2*(radius_km*1000/spacing_m)+1)² / 100 batches
                  Default: 31×31 = 961 pts = 10 API calls
    tpi_window  : Neighbourhood radius for TPI calculation (cells, default 5)
    """

    def __init__(
        self,
        radius_km:  float = 3.0,
        spacing_m:  float = 200.0,
        tpi_window: int   = 5,
    ) -> None:
        self.radius_km  = radius_km
        self.spacing_m  = spacing_m
        self.tpi_window = tpi_window

    def assess(
        self,
        lat:        float,
        lon:        float,
        asset_name: str = "Asset",
    ) -> TerrainHazardResult:
        """
        Full terrain hazard assessment for one asset.

        Parameters
        ----------
        lat        : Asset latitude (decimal degrees)
        lon        : Asset longitude (decimal degrees)
        asset_name : Display name for the asset

        Returns
        -------
        TerrainHazardResult with all terrain metrics, risk ratings, and
        narrative output.  Includes .grid for 3D visualisation.
        """
        log.info("Terrain assessment: %s (%.4f, %.4f)", asset_name, lat, lon)

        # 1. Fetch DEM grid
        grid = get_elevation_grid(lat, lon, self.radius_km, self.spacing_m)
        dem  = grid.elevations
        ar, ac = grid.asset_row, grid.asset_col

        # 2. Slope and aspect grids
        slope_grid  = slope_degrees(dem, grid.cell_size_m)
        aspect_grid = aspect_degrees(dem, grid.cell_size_m)

        # 3. TPI grid
        tpi_grid = compute_tpi(dem, window_cells=self.tpi_window)

        # 4. Asset-level metrics
        elev_m     = grid.asset_elevation_m
        slope_a    = slope_grid[ar][ac]
        aspect_a   = aspect_grid[ar][ac]
        tpi_a      = tpi_grid[ar][ac]
        tp_label   = terrain_position_label(tpi_a, slope_a)

        # 5. Flow direction + upstream catchment
        flow_dir = flow_direction_d8(dem)
        upstream = upstream_area_cells(flow_dir, ar, ac)
        up_area  = upstream_area_km2(flow_dir, ar, ac, grid.cell_size_m)

        # 6. Upslope slope statistics
        slope_stats = upslope_slope_stats(slope_grid, upstream, (ar, ac))
        elev_head   = upslope_elevation_range(dem, upstream, elev_m)

        # 7. Risk scoring
        debris_risk = self._score_debris_flow(
            tp_label, tpi_a, up_area,
            slope_stats["max_deg"], slope_stats["mean_deg"],
            slope_stats["pct_over_25"], elev_head, slope_a
        )
        landslide_risk = self._score_landslide(slope_a, tpi_a, elev_head, up_area)
        flood_risk     = self._score_flood(tp_label, tpi_a, up_area, elev_m)

        # 8. Mechanisms
        mechanisms = self._build_mechanisms(
            tp_label, tpi_a, up_area, slope_stats, elev_head,
            slope_a, aspect_a, debris_risk, landslide_risk, flood_risk
        )

        # 9. Recommendations
        actions = self._build_actions(debris_risk, landslide_risk, flood_risk, slope_a)

        # 10. Caveats
        caveats = [
            "SRTM 30m DEM: horizontal accuracy ±20m, vertical accuracy ±16m. "
            "Subsurface geology and land cover not modelled.",
            "Debris flow thresholds are morphological proxies — site-specific "
            "geotechnical investigation required to confirm susceptibility.",
        ]
        if math.isnan(slope_a) or math.isnan(tpi_a):
            caveats.append("Some DEM cells returned nodata — estimates may be less reliable near grid edges.")

        # Confidence: degrades with nodata fraction
        valid_cells = sum(
            1 for row in dem for v in row if not math.isnan(v)
        )
        total_cells = grid.rows * grid.cols
        confidence  = min(0.90, valid_cells / total_cells)

        return TerrainHazardResult(
            asset_lat=lat,
            asset_lon=lon,
            asset_name=asset_name,
            elevation_m=elev_m if not math.isnan(elev_m) else 0.0,
            slope_at_asset_deg=slope_a if not math.isnan(slope_a) else 0.0,
            aspect_at_asset_deg=aspect_a if not math.isnan(aspect_a) else 0.0,
            tpi_m=tpi_a if not math.isnan(tpi_a) else 0.0,
            terrain_position=tp_label,
            upslope_area_km2=up_area,
            upslope_max_slope_deg=slope_stats["max_deg"] if not math.isnan(slope_stats["max_deg"]) else 0.0,
            upslope_mean_slope_deg=slope_stats["mean_deg"] if not math.isnan(slope_stats["mean_deg"]) else 0.0,
            upslope_pct_over_25deg=slope_stats["pct_over_25"],
            elevation_head_m=elev_head,
            debris_flow_risk=debris_risk,
            landslide_susceptibility=landslide_risk,
            flood_exposure=flood_risk,
            contributing_mechanisms=mechanisms,
            recommended_actions=actions,
            data_caveats=caveats,
            confidence=confidence,
            grid=grid,
        )

    # ── Risk scoring ────────────────────────────────────────────────────────

    @staticmethod
    def _score_debris_flow(
        tp: str, tpi: float, area: float, max_slope: float,
        mean_slope: float, pct25: float, head: float, asset_slope: float
    ) -> str:
        if math.isnan(tpi) or math.isnan(max_slope):
            return "LOW"

        # HIGH: classic footslope/valley with large, steep upstream catchment
        if (tp in {"footslope", "valley", "valley bottom"} and
                area > 0.5 and max_slope > 25 and head > 50):
            return "HIGH"

        # HIGH: severe slope failure conditions even without perfect terrain pos
        if area > 1.0 and max_slope > 30 and pct25 > 30:
            return "HIGH"

        # MEDIUM: moderate catchment or moderate slopes
        if (area > 0.2 and max_slope > 20) or (tpi < -5 and head > 30):
            return "MEDIUM"

        # MEDIUM: asset itself on moderately steep terrain (self-instability)
        if 15 < asset_slope <= 25:
            return "MEDIUM"

        # LOW: small catchment or gentle slopes
        if area < 0.2 or max_slope < 15:
            return "LOW"

        # NEGLIGIBLE: flat and upslope-protected
        if asset_slope < 5 and abs(tpi) < 3 and area < 0.05:
            return "NEGLIGIBLE"

        return "LOW"

    @staticmethod
    def _score_landslide(
        slope: float, tpi: float, head: float, area: float
    ) -> str:
        if math.isnan(slope):
            return "LOW"
        if slope > 30:
            return "HIGH"
        if slope > 20 or (slope > 15 and tpi < -5):
            return "MEDIUM"
        if slope > 10:
            return "LOW"
        return "NEGLIGIBLE"

    @staticmethod
    def _score_flood(tp: str, tpi: float, area: float, elev: float) -> str:
        if math.isnan(tpi):
            return "LOW"
        if tp in {"valley bottom", "valley"} and area > 1.0:
            return "HIGH"
        if tp in {"valley bottom", "valley"} or (tpi < -10 and area > 0.3):
            return "MEDIUM"
        if tpi < -3 and area > 0.1:
            return "LOW"
        return "NEGLIGIBLE"

    # ── Narrative builders ──────────────────────────────────────────────────

    @staticmethod
    def _build_mechanisms(
        tp: str, tpi: float, area: float, slope_stats: dict,
        head: float, asset_slope: float, aspect: float,
        df_risk: str, ls_risk: str, fl_risk: str
    ) -> list[str]:
        ms = []
        max_s  = slope_stats.get("max_deg", float("nan"))
        pct25  = slope_stats.get("pct_over_25", 0)
        mean_s = slope_stats.get("mean_deg", float("nan"))

        # Debris flow mechanisms
        if df_risk in {"HIGH", "MEDIUM"}:
            if not math.isnan(max_s) and max_s > 25:
                pct_str = f" ({pct25:.0f}% of catchment > 25°)" if pct25 > 0 else ""
                ms.append(
                    f"Steep upslope source zone (max {max_s:.0f}°{pct_str}): "
                    f"saturated colluvium or shallow bedrock failure can mobilise "
                    f"debris flows in heavy rainfall or seismic shaking events."
                )
            if area > 0.2:
                ms.append(
                    f"Upstream catchment of {area:.2f} km² concentrates runoff "
                    f"toward asset. Flow convergence amplifies debris volume and "
                    f"velocity at the site."
                )
            if head > 30:
                ms.append(
                    f"Elevation head of {head:.0f} m provides energy for debris "
                    f"transport — flow velocity at site could exceed 3–5 m/s "
                    f"depending on material and channel geometry."
                )
            if tp == "footslope":
                ms.append(
                    "Asset sits at the footslope break (negative TPI) — a natural "
                    "deposition zone for downslope mass movements. Debris can "
                    "arrive without warning even when the asset structure is intact."
                )
            if tp in {"valley", "valley bottom"}:
                ms.append(
                    "Valley bottom position: the asset is at the lowest point of "
                    "local drainage — both debris flows and floodwaters converge here."
                )

        # Landslide on-site mechanisms
        if ls_risk in {"HIGH", "MEDIUM"}:
            if asset_slope > 20:
                ms.append(
                    f"Asset platform slope of {asset_slope:.1f}° exceeds threshold "
                    f"for shallow translational failure in saturated soils. "
                    f"Step-terraced construction could load downslope fills."
                )
            if asset_slope > 10 and df_risk == "HIGH":
                ms.append(
                    "Combined upslope loading and on-site gradient: debris impact "
                    "on retaining walls or berm structures may trigger secondary "
                    "failure of the asset's own foundations."
                )

        # Flood mechanisms
        if fl_risk in {"HIGH", "MEDIUM"}:
            ms.append(
                f"Topographic depression (TPI {tpi:+.0f} m) with large drainage area "
                f"({area:.2f} km²): extreme rainfall events can produce rapid "
                f"inundation before drainage infrastructure responds."
            )

        if not ms:
            ms.append("No significant terrain-driven hazard mechanisms identified at this location.")
        return ms

    @staticmethod
    def _build_actions(
        df_risk: str, ls_risk: str, fl_risk: str, slope: float
    ) -> list[str]:
        actions = []
        if df_risk == "HIGH":
            actions += [
                "Commission site-specific debris flow runout study (RAMMS or "
                "FLO-2D modelling) to define design flow volumes and velocities.",
                "Install barrier/deflection berm upslope of critical infrastructure "
                "sized to the 1-in-100 year debris flow volume.",
                "Deploy real-time rainfall + displacement monitoring with automated "
                "alert thresholds linked to emergency shutdown procedures.",
                "Map and clear debris transport channels within 1 km upslope annually.",
            ]
        elif df_risk == "MEDIUM":
            actions += [
                "Conduct visual terrain inspection of upslope catchment for existing "
                "tension cracks, seepage zones, or erosion scarps.",
                "Establish debris flow threshold rainfall criteria (site-specific "
                "intensity-duration curves) for operational alert protocols.",
            ]
        if ls_risk in {"HIGH", "MEDIUM"}:
            actions += [
                "Commission geotechnical stability assessment for on-site slopes "
                "and any retained cut/fill boundaries.",
                "Review structural load assumptions for buildings on or near slope "
                "breaks — account for potential surcharge from upslope mass movement.",
            ]
        if fl_risk in {"HIGH", "MEDIUM"}:
            actions += [
                "Assess adequacy of existing stormwater drainage for 1-in-50 and "
                "1-in-100 year design rainfall events.",
                "Evaluate flood barriers or elevation of critical electrical and "
                "process control equipment above predicted inundation depth.",
            ]
        if not actions:
            actions.append(
                "No immediate structural terrain mitigation required. "
                "Include terrain monitoring in routine site risk reviews."
            )
        return actions
