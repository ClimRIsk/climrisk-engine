"""
JRC Global Flood Hazard Map — point depth lookup.

Queries the JRC LISFLOOD-FP pre-computed flood depth rasters via HTTP range
requests using rasterio's VSICURL driver for Cloud Optimized GeoTIFF access.
Only the relevant tile is fetched — no full file download (~2 HTTP requests
per point per return period).

Data
----
JRC Global Flood Hazard Maps (Copernicus EMS, open licence)
  Return periods : 10, 20, 50, 100, 200, 500 years
  Resolution     : 90 m (3 arc-seconds, EPSG:4326)
  Value          : inundation depth in metres
  Dry land       : 0.0 (or nodata)

References
----------
Dottori, F. et al. (2016). Development and evaluation of a framework for global
  flood hazard mapping. Advances in Water Resources, 94, 87–102.
  https://doi.org/10.1016/j.advwatres.2016.05.002

Baugh, C. et al. (2024). A global flood hazard map. Nature Communications.
  https://doi.org/10.1038/s41467-024-53002-4
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── JRC data URLs ─────────────────────────────────────────────────────────────
# Primary: JRC FTP mirror (Copernicus open data)
_JRC_BASE_PRIMARY = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps"
)
# Secondary mirror: cidportal (same data, alternative endpoint)
_JRC_BASE_SECONDARY = (
    "https://cidportal.jrc.ec.europa.eu/ftp/jrc-opendata/FLOODS/GlobalMaps"
)

RETURN_PERIODS: tuple[int, ...] = (10, 20, 50, 100, 200, 500)

_JRC_FILENAMES: dict[int, str] = {
    rp: f"floodMapGL_rp{rp}y.tif"
    for rp in RETURN_PERIODS
}

# ── Cache ─────────────────────────────────────────────────────────────────────
# Key: (lat_3dp, lon_3dp, return_period) → depth_m
# 3 decimal places ≈ 111 m precision, small enough for asset-level accuracy.
_DEPTH_CACHE: dict[tuple[float, float, int], float] = {}

# ── Sentinel values ───────────────────────────────────────────────────────────
DEPTH_DRY_LAND: float = 0.0       # confirmed not in flood zone
DEPTH_FETCH_FAILED: float = -1.0  # HTTP/rasterio error — caller should fall back

# ── Rasterio availability ─────────────────────────────────────────────────────
_RASTERIO_AVAILABLE: bool = False
try:
    import rasterio  # noqa: F401
    _RASTERIO_AVAILABLE = True
    logger.debug("rasterio available — JRC COG flood depth enabled")
except ImportError:
    logger.warning(
        "rasterio not installed — JRC hydraulic flood depth unavailable. "
        "Physical risk will use WRI Aqueduct proxy score instead. "
        "Install: pip install 'cri[gis]'"
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _nearest_rp(return_period: int) -> int:
    """Snap a return period to the nearest available JRC value."""
    return min(RETURN_PERIODS, key=lambda x: abs(x - return_period))


def _round_coord(v: float, decimals: int = 3) -> float:
    return round(v, decimals)


def _fetch_depth(lat: float, lon: float, return_period: int) -> float:
    """
    Fetch one point from a JRC COG via VSICURL.
    Tries primary URL, falls back to secondary mirror.
    Returns depth_m >= 0 on success, DEPTH_FETCH_FAILED on error.
    """
    import rasterio

    fname = _JRC_FILENAMES[return_period]
    urls = [
        f"/vsicurl/{_JRC_BASE_PRIMARY}/{fname}",
        f"/vsicurl/{_JRC_BASE_SECONDARY}/{fname}",
    ]

    for vsicurl_path in urls:
        try:
            with rasterio.open(vsicurl_path) as src:
                # src.sample expects (lon, lat) order — EPSG:4326 x=lon y=lat
                samples = list(src.sample([(lon, lat)], indexes=1))
                if not samples:
                    continue
                val = float(samples[0])
                nd = src.nodata
                # JRC nodata is -9999; dry land is stored as 0
                if nd is not None and abs(val - nd) < 1.0:
                    return DEPTH_DRY_LAND
                if val < 0.0:
                    return DEPTH_DRY_LAND
                return val
        except Exception as exc:
            logger.debug(
                "JRC COG fetch error [%s] RP%dy (%.4f,%.4f): %s",
                vsicurl_path, return_period, lat, lon, exc,
            )
            continue

    return DEPTH_FETCH_FAILED


# ── Public API ────────────────────────────────────────────────────────────────

def get_flood_depth(
    lat: float,
    lon: float,
    return_period: int = 100,
) -> float:
    """
    Return JRC flood inundation depth (metres) at (lat, lon) for a given return
    period.

    Parameters
    ----------
    lat, lon : float
        WGS84 decimal degrees.
    return_period : int
        Target annual return period in years. Snapped to nearest of
        10, 20, 50, 100, 200, 500.

    Returns
    -------
    float
        Depth in metres.  0.0 = dry land.  -1.0 = fetch failed (use fallback).

    Notes
    -----
    - Results are cached in memory for the process lifetime (keyed to 3 dp).
    - Two HTTP range requests are made per unique (lat, lon, rp) combination
      (COG header + relevant tile).  Subsequent calls hit the in-process cache.
    """
    if not _RASTERIO_AVAILABLE:
        return DEPTH_FETCH_FAILED

    rp = _nearest_rp(return_period)
    clat = _round_coord(lat)
    clon = _round_coord(lon)
    key = (clat, clon, rp)

    if key in _DEPTH_CACHE:
        return _DEPTH_CACHE[key]

    depth = _fetch_depth(clat, clon, rp)
    # Cache both dry-land zeros and valid depths; do NOT cache fetch failures
    # so a transient network error doesn't permanently poison the cache.
    if depth >= DEPTH_DRY_LAND:
        _DEPTH_CACHE[key] = depth

    return depth


def get_flood_depth_curve(
    lat: float,
    lon: float,
    return_periods: tuple[int, ...] = RETURN_PERIODS,
) -> dict[int, float]:
    """
    Return the full flood depth exceedance curve at (lat, lon).

    Returns
    -------
    dict[int, float]
        Maps return_period (years) → depth (m).
        -1.0 values indicate fetch failures for that return period.
    """
    return {rp: get_flood_depth(lat, lon, rp) for rp in return_periods}


def compute_flood_eal(
    lat: float,
    lon: float,
    asset_rcv_usd: float,
    sector_key: str,
    gmst_delta: float = 0.0,
    return_periods: tuple[int, ...] = RETURN_PERIODS,
) -> tuple[float, dict]:
    """
    Compute Expected Annual Loss (EAL) in USD from flood using the JRC depth
    curve and JRC Huizinga depth-damage functions.

    Climate signal: flood depths are scaled by a climate amplification factor
    derived from IPCC AR6 WG1 Ch11 — approximately +10% depth per +1°C of
    GMST warming, consistent with IPCC's intensification projections for
    extreme precipitation and river discharge.

    Parameters
    ----------
    lat, lon : float
        Asset coordinates (WGS84).
    asset_rcv_usd : float
        Replacement cost value of the asset in USD.
    sector_key : str
        CRI sector key (e.g. "chemicals", "industrials").  Used to select the
        JRC damage curve.
    gmst_delta : float
        GMST anomaly above pre-industrial (°C) for the target year. Default 0
        gives the baseline (current climate) EAL.
    return_periods : tuple[int, ...]
        Return periods to include in the integration.

    Returns
    -------
    (eal_usd, audit)
        eal_usd  : float — expected annual loss in USD
        audit    : dict  — per-return-period breakdown for transparency

    Notes
    -----
    Integration method: trapezoidal rule on the annual exceedance probability
    axis.  Annual exceedance probability P = 1 / return_period.

    If all JRC fetches fail, returns (0.0, {"note": "jrc_unavailable"}).
    """
    from .damage_curves import get_damage_curve

    curve = get_damage_curve(sector_key)

    # Climate amplification: +10% depth per +1°C (IPCC AR6 WG1 Ch11)
    # Capped at 2× to avoid extrapolation beyond dataset validity
    climate_amp = min(2.0, 1.0 + 0.10 * max(0.0, gmst_delta))

    # Build exceedance probability → loss pairs
    ep_loss: list[tuple[float, float]] = []   # (exceedance_prob, loss_usd)
    audit_rp: list[dict] = []
    any_valid = False

    for rp in sorted(return_periods, reverse=True):   # high RP = low probability first
        depth_base = get_flood_depth(lat, lon, rp)
        if depth_base < 0.0:
            # Fetch failed for this RP — skip (trapezoidal integration handles gaps)
            continue
        any_valid = True
        depth_climate = depth_base * climate_amp
        frac = curve.damage_fraction(depth_climate)
        loss = frac * asset_rcv_usd
        ep = 1.0 / rp
        ep_loss.append((ep, loss))
        audit_rp.append({
            "return_period_yr": rp,
            "annual_exceedance_prob": round(ep, 5),
            "depth_baseline_m": round(depth_base, 3),
            "depth_climate_m": round(depth_climate, 3),
            "climate_amp": round(climate_amp, 3),
            "damage_fraction": round(frac, 4),
            "loss_usd": round(loss, 0),
        })

    if not any_valid:
        return 0.0, {"note": "jrc_unavailable", "fallback": "phi_elf_model"}

    # Sort by ascending exceedance probability for trapz integration
    ep_loss.sort(key=lambda x: x[0])

    # Trapezoidal integration: EAL = ∫ Loss(p) dp  over p in [0, max_p]
    # p = 0 corresponds to the 500-yr event (or longest available)
    # We add a (0, 0) anchor at p=0 (no loss at zero probability)
    eps  = [0.0] + [x[0] for x in ep_loss]
    loss = [0.0] + [x[1] for x in ep_loss]
    eal = 0.0
    for i in range(1, len(eps)):
        eal += 0.5 * (loss[i - 1] + loss[i]) * (eps[i] - eps[i - 1])

    audit = {
        "method":        "JRC LISFLOOD depth-damage trapz integration",
        "asset_class":   curve.asset_class,
        "d50_m":         curve.d50,
        "k":             curve.k,
        "asset_rcv_usd": asset_rcv_usd,
        "gmst_delta":    gmst_delta,
        "climate_amp":   round(climate_amp, 3),
        "return_periods": audit_rp,
        "eal_usd":       round(eal, 0),
        "data_source":   "JRC Global Flood Hazard Maps (LISFLOOD-FP, Dottori 2016)",
    }
    return eal, audit
