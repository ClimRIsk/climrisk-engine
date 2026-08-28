"""
Production-grade terrain hazard analyzer using pysheds + numpy + scipy.

Data source
-----------
AWS Mapzen terrain tiles (terrarium PNG format) at zoom-14 (~9.6 m/pixel).
This replaces Open-Topo-Data point-sampling and gives ~100× more cells per km²
for proper D8 flow-direction and accumulation computation.

Upstream area algorithm
-----------------------
grd.catchment() only works for pour-point / outlet cells.  For industrial assets
that sit on an intermediate position in the landscape (footslope, coastal plain,
river terrace), the D8 routing passes THROUGH the asset cell toward the actual
basin outlet. We therefore use two complementary metrics:

    direct_upstream_km²  = acc[ar, ac] × cell_area
        → cells whose D8 path routes *through* the asset cell
          (meaningful when asset IS at a local flow convergence)

    nearby_channel_km²   = max(acc within 500 m of asset) × cell_area
        → catchment of the nearest major channel within striking distance
          (meaningful when the asset is adjacent to a channel that could
           carry debris or floodwater toward it)

    effective_upstream_km² = max(direct, nearby)

This gives physically correct results for both valley-bottom assets (large
direct_upstream) and footslope / coastal-plain assets (large nearby_channel).

Stack
-----
terrain_tiles  — AWS terrarium tile downloader (~9.6 m/pixel, free, no key)
pysheds 0.5    — fill_pits, fill_depressions, resolve_flats, flowdir, accumulation
numpy          — vectorised slope (gradient), elevation statistics
scipy.ndimage  — TPI via uniform_filter (O(n))
affine         — Affine geotransform

Usage
-----
    from cri.terrain.pysheds_analyzer import TerrainEngine

    engine = TerrainEngine()
    result = engine.assess(lat=17.6868, lon=83.2185, asset_name="Vizag Steel Plant")
    print(result.summary)
    d = result.to_dict()   # JSON-serialisable, includes viz payload for 3D renderer
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from affine import Affine
from pysheds.grid import Grid
from pysheds.view import Raster, ViewFinder
from scipy import ndimage

from ..connectors.terrain_tiles import get_elevation_grid, ElevationGrid

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NODATA       = -9999.0
_DEFAULT_ZOOM = 14          # ~9.6 m/pixel at equator
_TPI_WINDOW   = 15          # cells — 15 × 9.6 m ≈ 144 m neighbourhood
_NEARBY_M     = 500         # metres — search radius for nearby channel


# ---------------------------------------------------------------------------
# Geodetic helpers
# ---------------------------------------------------------------------------

def _m_per_deg_lat(lat: float) -> float:
    return 111_132.0 - 559.8 * math.cos(2 * math.radians(lat))


def _m_per_deg_lon(lat: float) -> float:
    return 111_412.0 * math.cos(math.radians(lat))


def _deg_per_m_lat(lat: float) -> float:
    return 1.0 / _m_per_deg_lat(lat)


def _deg_per_m_lon(lat: float) -> float:
    return 1.0 / _m_per_deg_lon(lat)


# ---------------------------------------------------------------------------
# Terrain computations (vectorised)
# ---------------------------------------------------------------------------

def compute_slope_deg(dem: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    Slope angle (degrees) from DEM via numpy.gradient (Horn-equivalent).

    Parameters
    ----------
    dem         : 2-D float64 array, rows N→S, cols W→E.  May contain NaN.
    cell_size_m : pixel spacing in metres

    Returns
    -------
    slope_deg : same shape as dem
    """
    drow, dcol = np.gradient(dem, cell_size_m, cell_size_m)
    return np.degrees(np.arctan(np.sqrt(drow ** 2 + dcol ** 2)))


def compute_tpi(dem: np.ndarray, window_cells: int = _TPI_WINDOW) -> np.ndarray:
    """
    Topographic Position Index = dem − neighbourhood_mean.

    Positive TPI → ridge/hill.  Negative TPI → valley/depression.
    scipy.ndimage.uniform_filter is O(n) regardless of window size.
    """
    size = 2 * window_cells + 1
    nbr_mean = ndimage.uniform_filter(dem.astype(np.float64), size=size, mode="reflect")
    return dem - nbr_mean


def terrain_position(tpi: float, slope: float) -> str:
    """Classify terrain position from TPI (m) and slope (°) — Weiss (2001)."""
    if math.isnan(tpi) or math.isnan(slope):
        return "unknown"
    if tpi > 10 and slope < 6:  return "plateau/ridge"
    if tpi > 10:                 return "ridge"
    if tpi > 3:                  return "upper slope"
    if tpi < -10 and slope < 6: return "valley bottom"
    if tpi < -10:                return "valley"
    if tpi < -3:                 return "footslope"
    if slope < 6:                return "flat"
    return "mid slope"


# ---------------------------------------------------------------------------
# pysheds grid builder
# ---------------------------------------------------------------------------

def _build_pysheds_grid(
    dem_np:    np.ndarray,
    top_lat:   float,          # latitude of the northern edge (row 0)
    left_lon:  float,          # longitude of the western edge (col 0)
    cell_lat:  float,          # degrees per row (positive south)
    cell_lon:  float,          # degrees per column (positive east)
    nodata:    float = _NODATA,
) -> Tuple[Grid, Raster]:
    """
    Build a pysheds Grid + Raster from a numpy DEM and its geotransform.

    The Affine transform: col_delta=cell_lon, row_delta=-cell_lat (Mercator convention).
    """
    rows, cols = dem_np.shape
    transform = Affine(cell_lon, 0.0, left_lon, 0.0, -cell_lat, top_lat)
    vf  = ViewFinder(shape=(rows, cols), affine=transform, nodata=nodata, crs="epsg:4326")
    grd = Grid(viewfinder=vf)
    dem_clean = np.where(np.isnan(dem_np), nodata, dem_np).astype(np.float64)
    raster = Raster(dem_clean, viewfinder=vf, metadata={"nodata": nodata})
    return grd, raster


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TerrainHazardResult:
    """
    Structured output from TerrainEngine.assess().

    Physical metrics are derived from AWS terrain tiles (~9.6 m/pixel).
    Risk ratings are CRI morphological proxies — they do not replace
    site-specific geotechnical investigation.
    """
    asset_lat:   float
    asset_lon:   float
    asset_name:  str

    # ── Terrain metrics ──────────────────────────────────────────────────────
    elevation_m:              float
    slope_at_asset_deg:       float
    tpi_m:                    float
    terrain_position:         str

    # ── Upstream flow analysis ───────────────────────────────────────────────
    upslope_area_km2:         float   # effective = max(direct, nearby_channel)
    direct_upstream_km2:      float   # cells draining *through* the asset cell
    nearby_channel_km2:       float   # largest channel within 500 m
    upslope_max_slope_deg:    float   # max slope in cells above asset elevation
    upslope_mean_slope_deg:   float
    upslope_pct_over_25deg:   float
    elevation_head_m:         float   # max elev above asset within grid

    # ── Risk ratings ─────────────────────────────────────────────────────────
    debris_flow_risk:         str
    landslide_susceptibility: str
    flood_exposure:           str

    # ── Narrative ────────────────────────────────────────────────────────────
    contributing_mechanisms: list[str] = field(default_factory=list)
    recommended_actions:     list[str] = field(default_factory=list)
    data_caveats:            list[str] = field(default_factory=list)
    confidence:              float = 0.0

    # ── Viz payload (for HTML 3D widget — pre-computed server-side) ──────────
    viz: Optional[dict] = field(default=None, repr=False)

    @property
    def summary(self) -> str:
        lines = [
            f"Terrain Hazard — {self.asset_name}",
            f"  Elevation:          {self.elevation_m:.0f} m ASL",
            f"  Terrain position:   {self.terrain_position}  (TPI {self.tpi_m:+.1f} m)",
            f"  Slope @ asset:      {self.slope_at_asset_deg:.1f}°",
            f"  Direct upstream:    {self.direct_upstream_km2:.4f} km²",
            f"  Nearby channel:     {self.nearby_channel_km2:.3f} km²  (within 500 m)",
            f"  Effective upstream: {self.upslope_area_km2:.3f} km²",
            f"  Elevation head:     {self.elevation_head_m:.0f} m",
            f"  Max upslope slope:  {self.upslope_max_slope_deg:.1f}°",
            f"  Debris flow:        {self.debris_flow_risk}",
            f"  Landslide:          {self.landslide_susceptibility}",
            f"  Flood exposure:     {self.flood_exposure}",
        ]
        for m in self.contributing_mechanisms:
            lines.append(f"  • {m}")
        for a in self.recommended_actions:
            lines.append(f"  → {a}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "asset_lat":               self.asset_lat,
            "asset_lon":               self.asset_lon,
            "asset_name":              self.asset_name,
            "elevation_m":             round(self.elevation_m, 1),
            "slope_at_asset_deg":      round(self.slope_at_asset_deg, 2),
            "tpi_m":                   round(self.tpi_m, 2),
            "terrain_position":        self.terrain_position,
            "upslope_area_km2":        round(self.upslope_area_km2, 4),
            "direct_upstream_km2":     round(self.direct_upstream_km2, 4),
            "nearby_channel_km2":      round(self.nearby_channel_km2, 4),
            "upslope_max_slope_deg":   round(self.upslope_max_slope_deg, 1),
            "upslope_mean_slope_deg":  round(self.upslope_mean_slope_deg, 1),
            "upslope_pct_over_25deg":  round(self.upslope_pct_over_25deg, 1),
            "elevation_head_m":        round(self.elevation_head_m, 1),
            "debris_flow_risk":        self.debris_flow_risk,
            "landslide_susceptibility":self.landslide_susceptibility,
            "flood_exposure":          self.flood_exposure,
            "contributing_mechanisms": self.contributing_mechanisms,
            "recommended_actions":     self.recommended_actions,
            "data_caveats":            self.data_caveats,
            "confidence":              round(self.confidence, 2),
            "viz":                     self.viz,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TerrainEngine:
    """
    Production terrain hazard engine using AWS terrain tiles at ~9.6 m/pixel.

    Parameters
    ----------
    radius_km   : half-width of the DEM grid to fetch (default 2.5 km).
    zoom        : tile zoom level (default 14 → ~9.6 m/px).
    """

    def __init__(self, radius_km: float = 2.5, zoom: int = _DEFAULT_ZOOM) -> None:
        self.radius_km = radius_km
        self.zoom      = zoom

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(
        self,
        lat:        float,
        lon:        float,
        asset_name: str = "Asset",
    ) -> TerrainHazardResult:
        """
        Full terrain hazard assessment at (lat, lon).

        Pipeline
        --------
        1. Download AWS terrain tiles  → ElevationGrid (numpy, ~9.6 m/px)
        2. numpy.gradient              → slope_deg
        3. scipy.uniform_filter        → TPI, terrain position
        4. pysheds D8 pipeline         → flowdir, accumulation
        5. acc[ar,ac] + nearby max     → effective upstream area
        6. Elevation mask              → upslope slope statistics, head
        7. CRI risk scoring            → debris flow / landslide / flood
        8. Build TerrainHazardResult   → narrative + viz payload
        """
        t0 = time.time()
        log.info("TerrainEngine.assess: %s (%.4f, %.4f)", asset_name, lat, lon)

        # ── 1. Fetch DEM ──────────────────────────────────────────────────────
        eg: ElevationGrid = get_elevation_grid(lat, lon,
                                               radius_km=self.radius_km,
                                               zoom=self.zoom)
        dem  = eg.elevations.astype(np.float64)   # (rows, cols) float64
        ar   = eg.asset_row
        ac   = eg.asset_col
        sp_m = eg.cell_size_m                     # metres per pixel (lat direction)

        valid_mask = ~np.isnan(dem)

        # ── 2. Slope ──────────────────────────────────────────────────────────
        dem_safe = np.where(valid_mask, dem, np.nan)
        slope    = compute_slope_deg(dem_safe, sp_m)

        # ── 3. TPI + terrain position ─────────────────────────────────────────
        tpi_grid  = compute_tpi(dem_safe, window_cells=_TPI_WINDOW)
        asset_elev  = float(dem[ar, ac])   if valid_mask[ar, ac] else 0.0
        asset_slope = float(slope[ar, ac]) if not np.isnan(slope[ar, ac]) else 0.0
        asset_tpi   = float(tpi_grid[ar, ac]) if not np.isnan(tpi_grid[ar, ac]) else 0.0
        tp_label    = terrain_position(asset_tpi, asset_slope)

        # ── 4. pysheds D8 flow accumulation ──────────────────────────────────
        # Reconstruct the exact bounding-box corners from the ElevationGrid.
        # terrain_tiles stores: top-left = (crop_north, crop_west), step = (lat_per_px, lon_per_px).
        # We recover these via the deg_per_m conversions.
        dpl  = _deg_per_m_lat(lat)
        dplo = _deg_per_m_lon(lat)
        cell_lat_deg = sp_m * dpl    # degrees per row (south)
        cell_lon_deg = sp_m * dplo   # degrees per col (east)

        # Top-left corner: asset at (ar, ac) maps to (lat, lon)
        top_lat  = lat + ar * cell_lat_deg     # latitude of row-0 (northern edge)
        left_lon = lon - ac * cell_lon_deg     # longitude of col-0 (western edge)

        nodata = _NODATA
        dem_filled = np.where(valid_mask, dem, nodata)

        acc_np   = np.zeros_like(dem, dtype=np.float64)
        fdir_np  = None
        pysheds_ok = False
        try:
            grd, raster = _build_pysheds_grid(dem_filled, top_lat, left_lon,
                                               cell_lat_deg, cell_lon_deg, nodata)
            pit    = grd.fill_pits(raster)
            flood  = grd.fill_depressions(pit)
            inf    = grd.resolve_flats(flood)
            fdir   = grd.flowdir(inf)
            acc    = grd.accumulation(fdir)
            acc_np  = np.asarray(acc, dtype=np.float64)
            fdir_np = np.asarray(fdir)
            pysheds_ok = True
        except Exception as exc:
            log.warning("pysheds failed for %s: %s — acc metrics will be 0", asset_name, exc)

        # ── 5. Upstream area (two-metric approach) ────────────────────────────
        cell_km = sp_m / 1_000.0

        # 5a. Direct: cells whose D8 path passes through the asset cell
        direct_cells   = float(acc_np[ar, ac]) if pysheds_ok else 0.0
        direct_km2     = direct_cells * cell_km ** 2

        # 5b. Nearby channel: max accumulation within 500 m of asset
        r_cells = max(1, int(_NEARBY_M / sp_m))
        r0 = max(0, ar - r_cells);  r1 = min(eg.rows, ar + r_cells + 1)
        c0 = max(0, ac - r_cells);  c1 = min(eg.cols, ac + r_cells + 1)
        nearby_max     = float(acc_np[r0:r1, c0:c1].max()) if pysheds_ok else 0.0
        nearby_km2     = nearby_max * cell_km ** 2

        # Effective upstream area for risk scoring
        effective_km2  = max(direct_km2, nearby_km2)

        # ── 6. Upslope statistics (elevation-mask approach) ───────────────────
        # "Upslope" = cells within the grid that are ABOVE the asset elevation.
        # This correctly captures the potential energy source for debris / floods
        # even when the asset is an intermediate cell rather than a basin outlet.
        above_mask     = valid_mask & (dem > asset_elev)
        upslope_slopes = slope[above_mask]
        upslope_elevs  = dem[above_mask]

        if upslope_slopes.size > 0:
            max_slope   = float(np.nanmax(upslope_slopes))
            mean_slope  = float(np.nanmean(upslope_slopes))
            pct_over_25 = 100.0 * float(np.sum(upslope_slopes > 25)) / upslope_slopes.size
        else:
            max_slope = mean_slope = pct_over_25 = 0.0

        elev_head = float(upslope_elevs.max() - asset_elev) if upslope_elevs.size > 0 else 0.0

        # ── 7. Risk scoring ───────────────────────────────────────────────────
        df_risk = self._score_debris_flow(
            tp_label, asset_tpi, effective_km2, max_slope, elev_head, asset_slope, pct_over_25
        )
        ls_risk = self._score_landslide(asset_slope, asset_tpi)
        fl_risk = self._score_flood(tp_label, asset_tpi, effective_km2)

        # ── 8. Narrative ───────────────────────────────────────────────────────
        mechanisms = self._build_mechanisms(
            tp_label, asset_tpi, effective_km2, nearby_km2,
            max_slope, mean_slope, pct_over_25, elev_head, asset_slope,
            df_risk, ls_risk, fl_risk
        )
        actions  = self._build_actions(df_risk, ls_risk, fl_risk, asset_slope)
        caveats  = ["AWS terrain tiles (terrarium): ~9.6 m/pixel, vertical accuracy ±1–3 m (Copernicus DSM basis)."]
        if not pysheds_ok:
            caveats.append("Flow accumulation unavailable — risk scoring based on elevation mask and TPI only.")

        confidence  = min(0.92, float(valid_mask.sum()) / valid_mask.size)
        elapsed     = time.time() - t0
        log.info("TerrainEngine.assess done in %.1f s (pysheds=%s)", elapsed, pysheds_ok)

        # ── 9. Viz payload (for HTML 3D renderer) ─────────────────────────────
        viz = self._build_viz(dem, slope, tpi_grid, acc_np, ar, ac, sp_m, eg)

        return TerrainHazardResult(
            asset_lat=lat, asset_lon=lon, asset_name=asset_name,
            elevation_m=asset_elev,
            slope_at_asset_deg=asset_slope,
            tpi_m=asset_tpi,
            terrain_position=tp_label,
            upslope_area_km2=round(effective_km2, 4),
            direct_upstream_km2=round(direct_km2, 4),
            nearby_channel_km2=round(nearby_km2, 4),
            upslope_max_slope_deg=max_slope,
            upslope_mean_slope_deg=mean_slope,
            upslope_pct_over_25deg=pct_over_25,
            elevation_head_m=max(0.0, elev_head),
            debris_flow_risk=df_risk,
            landslide_susceptibility=ls_risk,
            flood_exposure=fl_risk,
            contributing_mechanisms=mechanisms,
            recommended_actions=actions,
            data_caveats=caveats,
            confidence=confidence,
            viz=viz,
        )

    # ── Risk scoring (Lateltin 1997, Jakob & Hungr 2005) ─────────────────────

    @staticmethod
    def _score_debris_flow(tp, tpi, area, max_slope, head, asset_slope, pct_over_25=0.0) -> str:
        """
        Debris flow risk — terrain-position aware.

        Ridge and upper-slope assets are SOURCE zones, not receivers.
        Debris accumulates at footslope/valley/flat positions only.
        Reference thresholds: Lateltin (1997), Jakob & Hungr (2005).
        """
        # Ridges and upper slopes shed material — no deposition risk
        if tp in {"ridge", "plateau/ridge", "upper slope"}:
            return "LOW"

        # HIGH: classic convergence zone (Lateltin 1997)
        if tp in {"footslope", "valley", "valley bottom"} and area > 0.5 and max_slope > 25 and head > 50:
            return "HIGH"
        # HIGH: large effective catchment with meaningful steep fraction
        # pct_over_25 > 3 guards against single-cell gorge-wall outliers
        if area > 1.5 and max_slope > 30 and pct_over_25 > 3:
            return "HIGH"
        # MEDIUM: moderate catchment + steep upslope with meaningful steep fraction
        # pct_over_25 > 1 separates real debris source zones from single gorge-wall cells
        # or large flat rivers (which belong in flood scoring, not debris flow)
        if (area > 0.2 and max_slope > 20 and pct_over_25 > 1) or (tpi < -5 and head > 30):
            return "MEDIUM"
        # MEDIUM: asset itself on significant slope (on-site instability)
        if 15 < asset_slope <= 30:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _score_landslide(slope, tpi) -> str:
        if slope > 30:   return "HIGH"
        if slope > 20:   return "MEDIUM"
        if slope > 10:   return "LOW"
        return "NEGLIGIBLE"

    @staticmethod
    def _score_flood(tp, tpi, area) -> str:
        """
        Flood exposure from channel proximity and topographic position.

        Large flow-accumulation channels within 500 m of a flat asset represent
        genuine overbank / fluvial flood risk (distinct from debris flow).
        This correctly captures Jamshedpur (Subarnarekha), Port Talbot (Afan),
        and other riparian industrial assets.
        """
        # HIGH: true valley bottom with large upstream drainage
        if tp in {"valley bottom", "valley"} and area > 1.0:  return "HIGH"
        # HIGH: strongly negative TPI (deep depression) with meaningful drainage
        if tpi < -10 and area > 0.5:                          return "HIGH"
        # MEDIUM: any low-lying flat terrain with significant nearby channel
        if tp in {"flat", "footslope", "valley", "valley bottom"} and area > 0.8:
            return "MEDIUM"
        if tp in {"valley bottom", "valley"} or tpi < -10:    return "MEDIUM"
        if tpi < -3 and area > 0.1:                           return "LOW"
        return "NEGLIGIBLE"

    # ── Narrative builders ────────────────────────────────────────────────────

    @staticmethod
    def _build_mechanisms(tp, tpi, area, nearby_km2, max_slope, mean_slope,
                          pct25, head, asset_slope, df, ls, fl) -> list[str]:
        ms = []
        if df in {"HIGH", "MEDIUM"}:
            if max_slope > 25:
                ms.append(
                    f"Steep source zone within grid ({max_slope:.0f}° max, {mean_slope:.0f}° mean, "
                    f"{pct25:.0f}% of upslope area >25°): saturated colluvium or shallow-bedrock "
                    f"failure can mobilise debris in heavy rainfall or seismic shaking."
                )
            if area > 0.2:
                ms.append(
                    f"Effective upstream catchment of {area:.2f} km² concentrates runoff "
                    f"({'direct D8 path' if area > nearby_km2 else 'channel within 500 m'}). "
                    f"Flow convergence amplifies debris volume and velocity at the site."
                )
            if head > 30:
                v_est = min(15, 0.18 * math.sqrt(head))   # ~Manning/energy estimate
                ms.append(
                    f"Elevation head {head:.0f} m — estimated debris velocity at site "
                    f"≈ {v_est:.0f}–{v_est*1.5:.0f} m/s depending on channel geometry and material type."
                )
            if tp == "footslope":
                ms.append(
                    "Asset sits at the footslope (negative TPI) — a natural deposition zone for "
                    "downslope mass movements. Debris can arrive without warning even when the "
                    "asset's own structure is stable."
                )
            if tp in {"valley", "valley bottom"}:
                ms.append(
                    "Valley-bottom position: debris flows and floodwaters both converge here. "
                    "Step-terraced infrastructure above the asset could redirect debris pathways "
                    "toward the site during extreme events."
                )
        if ls in {"HIGH", "MEDIUM"} and asset_slope > 15:
            ms.append(
                f"On-site slope {asset_slope:.1f}° may exceed threshold for shallow translational "
                f"failure in saturated soils. Step-terraced construction loads downslope fills and "
                f"retaining walls — geotechnical assessment required."
            )
        if fl in {"HIGH", "MEDIUM"}:
            ms.append(
                f"Topographic depression (TPI {tpi:+.0f} m) with effective drainage area "
                f"{area:.2f} km² can accumulate floodwater rapidly during intense rainfall "
                f"or upstream channel blockage."
            )
        if not ms:
            ms.append(
                "No significant terrain-driven hazard mechanisms identified at this location from "
                f"9.6 m resolution DEM. Asset is on {tp} terrain (TPI {tpi:+.1f} m, slope {asset_slope:.1f}°)."
            )
        return ms

    @staticmethod
    def _build_actions(df, ls, fl, slope) -> list[str]:
        acts = []
        if df == "HIGH":
            acts += [
                "Commission site-specific debris flow runout study (RAMMS or FLO-2D) to define "
                "design flow volume, velocity, and runout distance at the asset boundary.",
                "Design retention/deflection berm upstream of critical structures, sized to the "
                "1-in-100yr event volume.",
                "Deploy real-time rainfall + slope displacement monitoring with automated "
                "alert-to-shutdown protocol.",
                "Clear and maintain debris transport channels within 1 km upslope annually.",
            ]
        elif df == "MEDIUM":
            acts += [
                "Conduct visual terrain inspection of upslope catchment for tension cracks, "
                "seepage zones, or active erosion scarps.",
                "Establish site-specific debris flow threshold rainfall criteria "
                "(intensity-duration-frequency curves from nearest gauge).",
            ]
        if ls in {"HIGH", "MEDIUM"}:
            acts += [
                "Geotechnical stability assessment for on-site slopes and retained cut/fill boundaries.",
                "Review structural load assumptions for buildings near slope breaks or terrace edges.",
            ]
        if fl in {"HIGH", "MEDIUM"}:
            acts += [
                "Assess stormwater drainage adequacy for 1-in-50 and 1-in-100yr design storms.",
                "Elevate critical electrical, process control, and chemical storage above "
                "predicted 1-in-100yr inundation depth.",
            ]
        if not acts:
            acts.append(
                "No immediate terrain mitigation required. Include terrain condition in routine "
                "site risk reviews (annual visual inspection)."
            )
        return acts

    # ── Viz payload builder ───────────────────────────────────────────────────

    @staticmethod
    def _build_viz(
        dem:   np.ndarray,
        slope: np.ndarray,
        tpi:   np.ndarray,
        acc:   np.ndarray,
        ar:    int,
        ac:    int,
        sp_m:  float,
        eg:    ElevationGrid,
    ) -> dict:
        """
        Downsample all grids to ≤ 50×50 and return a flat-list payload
        suitable for JSON transfer to the HTML 3D renderer (no JS recomputation).
        """
        def _downsample(arr: np.ndarray, max_dim: int = 50) -> np.ndarray:
            r, c = arr.shape
            rstep = max(1, r // max_dim)
            cstep = max(1, c // max_dim)
            return arr[::rstep, ::cstep]

        def _flat(arr: np.ndarray) -> list:
            return [round(float(v), 2) if not np.isnan(v) else None
                    for v in arr.flatten()]

        dem_ds   = _downsample(dem)
        slope_ds = _downsample(slope)
        tpi_ds   = _downsample(tpi)
        acc_log  = np.log1p(acc)
        acc_ds   = _downsample(acc_log)

        ds_rows, ds_cols = dem_ds.shape
        ds_ar = int(ar * ds_rows / eg.rows)
        ds_ac = int(ac * ds_cols / eg.cols)

        return {
            "rows":        ds_rows,
            "cols":        ds_cols,
            "cell_size_m": round(sp_m * max(1, eg.rows // 50), 1),
            "asset_row":   ds_ar,
            "asset_col":   ds_ac,
            "elevations":  _flat(dem_ds),
            "slope":       _flat(slope_ds),
            "tpi":         _flat(tpi_ds),
            "flow_acc":    _flat(acc_ds),
        }
