"""
geocoder_client.py — Address → lat/lon via OpenStreetMap Nominatim.

Used when GLEIF returns a registered address but we need coordinates
for physical hazard calculations (flood maps, SLR models etc.).

API: https://nominatim.openstreetmap.org/search (free, no auth)
Policy: max 1 req/s; include User-Agent with project name + email.

Returns WGS84 lat/lon with confidence tier ESTIMATED (geocoding is an
approximation of the registered address, not the actual asset location).

For more precise physical risk, the practitioner should provide:
  1. The actual asset coordinates (specific plant, office, port, etc.)
  2. Or upload an asset list with coordinates
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .provenance import ConfidenceTier, DataGap, ProvenanceField, today_iso

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_USER_AGENT    = "ClimRiskEngine/1.0 contact@climrisk.io"
_TIMEOUT       = 10.0

# Country centroid fallbacks (when address geocoding fails)
_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "US": (37.09, -95.71), "GB": (55.38, -3.44),  "DE": (51.17, 10.45),
    "FR": (46.23,   2.21), "JP": (36.20, 138.25), "CN": (35.86, 104.20),
    "AU": (-25.27, 133.78),"BR": (-14.24, -51.93), "IN": (20.59, 78.96),
    "NL": (52.13,   5.29), "SE": (60.13, 18.64),  "NO": (60.47,  8.47),
    "DK": (56.26,  9.50),  "CH": (46.82,  8.23),  "SG": ( 1.35, 103.82),
    "ZA": (-30.56, 22.94), "NG": (9.08,   8.68),   "EG": (26.82, 30.80),
    "SA": (23.89, 45.08),  "AE": (23.42, 53.85),
}


@dataclass
class GeocodeResult:
    lat:           Optional[float]           = None
    lon:           Optional[float]           = None
    display_name:  Optional[str]             = None
    provenance:    Optional[ProvenanceField] = None
    data_gap:      Optional[DataGap]         = None

    def to_dict(self) -> dict:
        return {
            "lat":          self.lat,
            "lon":          self.lon,
            "display_name": self.display_name,
            "provenance":   self.provenance.to_dict() if self.provenance else None,
            "data_gap":     (
                {"field": self.data_gap.field_name,
                 "reason": self.data_gap.reason,
                 "action": self.data_gap.suggested_action}
                if self.data_gap else None
            ),
        }


async def geocode_address(
    address: str,
    country_code: Optional[str] = None,
) -> GeocodeResult:
    """
    Geocode a free-text address to WGS84 lat/lon.

    Falls back to country centroid if Nominatim can't parse the address.
    """
    params = {
        "q":              address,
        "format":         "json",
        "limit":          1,
        "addressdetails": 0,
    }
    if country_code:
        params["countrycodes"] = country_code.lower()

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                _NOMINATIM_URL,
                params=params,
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
            results = resp.json()

        if results:
            top = results[0]
            lat = float(top["lat"])
            lon = float(top["lon"])
            return GeocodeResult(
                lat=lat,
                lon=lon,
                display_name=top.get("display_name", ""),
                provenance=ProvenanceField(
                    value=f"{lat:.4f}, {lon:.4f}",
                    source="OpenStreetMap Nominatim",
                    url="https://nominatim.openstreetmap.org",
                    retrieval_date=today_iso(),
                    confidence_tier=ConfidenceTier.ESTIMATED,
                    notes=(
                        "Geocoded from registered address — approximation only. "
                        "For precise physical risk, provide actual asset coordinates."
                    ),
                ),
            )

        # No results → fall back to country centroid
        if country_code and country_code.upper() in _COUNTRY_CENTROIDS:
            lat, lon = _COUNTRY_CENTROIDS[country_code.upper()]
            logger.info("Nominatim found nothing for '%s'; using %s centroid", address, country_code)
            return GeocodeResult(
                lat=lat,
                lon=lon,
                display_name=f"{country_code.upper()} country centroid (fallback)",
                provenance=ProvenanceField(
                    value=f"{lat:.4f}, {lon:.4f}",
                    source=f"Country centroid fallback ({country_code.upper()})",
                    url=None,
                    retrieval_date=today_iso(),
                    confidence_tier=ConfidenceTier.ESTIMATED,
                    notes=(
                        "Address geocoding failed — using country centroid. "
                        "Physical risk accuracy is significantly reduced. "
                        "Provide actual asset lat/lon to improve results."
                    ),
                ),
            )

        return GeocodeResult(
            data_gap=DataGap(
                field_name="coordinates",
                reason=f"Nominatim returned no results for: {address!r}",
                impact=(
                    "Physical hazard (flood, SLR, heat) cannot be calculated "
                    "without coordinates"
                ),
                suggested_action=(
                    "Provide latitude and longitude for the primary asset / HQ location"
                ),
            )
        )

    except httpx.TimeoutException:
        logger.warning("Nominatim geocoding timed out for '%s'", address)
        if country_code and country_code.upper() in _COUNTRY_CENTROIDS:
            lat, lon = _COUNTRY_CENTROIDS[country_code.upper()]
            return GeocodeResult(
                lat=lat,
                lon=lon,
                display_name=f"{country_code.upper()} centroid (timeout fallback)",
                provenance=ProvenanceField(
                    value=f"{lat:.4f}, {lon:.4f}",
                    source="Country centroid (geocoding timeout)",
                    url=None,
                    retrieval_date=today_iso(),
                    confidence_tier=ConfidenceTier.ESTIMATED,
                    notes="Geocoding timed out; using country centroid.",
                ),
            )
        return GeocodeResult(
            data_gap=DataGap(
                field_name="coordinates",
                reason="Geocoding API timeout",
                impact="Physical hazard calculations will use country-level defaults",
                suggested_action="Provide lat/lon directly or re-run assessment",
            )
        )

    except Exception as exc:
        logger.warning("Geocoding failed for '%s': %s", address, exc)
        return GeocodeResult(
            data_gap=DataGap(
                field_name="coordinates",
                reason=str(exc),
                impact="Physical hazard calculations unavailable",
                suggested_action="Provide latitude and longitude directly",
            )
        )
