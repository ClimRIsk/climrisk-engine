"""
PredictiveEventEngine — main orchestrator.

Accepts a company definition (assets with GPS coordinates + revenue), queries
all live forecast and disaster connectors, runs event detection, applies the
financial impact matrix, and returns a ranked list of EventAlerts.

Usage
-----
    from cri.predictive import PredictiveEventEngine

    engine = PredictiveEventEngine()

    alerts = engine.assess_company(
        company_id   = "carnival_corp",
        company_name = "Carnival Corporation",
        assets = [
            {
                "id":     "miami_port",
                "name":   "PortMiami Cruise Terminal",
                "lat":    25.7741,
                "lon":   -80.1842,
                "sector": "cruise_terminal",
                "annual_revenue_usd": 7_200_000_000,   # asset-level revenue attribution
            },
            {
                "id":     "southampton",
                "name":   "Southampton Cruise Terminal",
                "lat":    50.9097,
                "lon":    -1.4044,
                "sector": "cruise_terminal",
                "annual_revenue_usd": 3_600_000_000,
            },
        ],
        annual_revenue_usd = 21_600_000_000,  # total company revenue (for capex)
    )

    for alert in alerts:
        print(alert.summary_line)

    # Or serialise to JSON
    import json
    print(json.dumps([a.to_dict() for a in alerts], indent=2))
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Optional

from .alerts import EventAlert, build_alert
from .events import (
    EventDetection,
    detect_from_forecast,
    detect_from_nhc_storms,
    detect_from_gdacs,
)
from .impact_matrix import lookup

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset definition
# ---------------------------------------------------------------------------

@dataclass
class AssetSpec:
    """One asset within a company's portfolio."""
    id:                  str
    name:                str
    lat:                 float
    lon:                 float
    sector:              str          # must match impact_matrix sector codes
    annual_revenue_usd:  float        # revenue attributable to this asset


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PredictiveEventEngine:
    """
    Short-to-medium horizon climate event predictor and financial alert generator.

    Connector availability
    ----------------------
    The engine gracefully degrades: if a connector fails or returns no data,
    that source is silently skipped and remaining sources still contribute.

    Parallel fetching
    -----------------
    Per-asset forecast fetches run concurrently (ThreadPoolExecutor) to keep
    total latency low even for large asset portfolios.
    """

    def __init__(
        self,
        max_workers: int = 8,
        nhc_buffer_km: float = 150.0,   # storm cone buffer for asset threats
        min_severity: int = 2,           # filter: only alerts ≥ MODERATE (2)
        min_confidence: float = 0.35,    # filter: only include confident detections
    ) -> None:
        self.max_workers   = max_workers
        self.nhc_buffer_km = nhc_buffer_km
        self.min_severity  = min_severity
        self.min_confidence = min_confidence

    # ── Public API ──────────────────────────────────────────────────────────

    def assess_company(
        self,
        company_id:         str,
        company_name:       str,
        assets:             list[dict],    # list of asset dicts (see AssetSpec)
        annual_revenue_usd: float,
    ) -> list[EventAlert]:
        """
        Run the full predictive assessment for all assets of one company.

        Parameters
        ----------
        company_id          : e.g. "carnival_corp"
        company_name        : e.g. "Carnival Corporation"
        assets              : list of dicts with keys: id, name, lat, lon,
                              sector, annual_revenue_usd
        annual_revenue_usd  : total company annual revenue (for capex sizing)

        Returns
        -------
        List of EventAlert objects, sorted by urgency_score descending.
        """
        specs = [self._parse_asset(a) for a in assets]

        # 1. Fetch NHC active storms once (company-level, not per-asset)
        nhc_storms = self._fetch_nhc_storms()

        # 2. Per-asset assessment in parallel
        all_alerts: list[EventAlert] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._assess_asset,
                    spec,
                    company_id,
                    company_name,
                    annual_revenue_usd,
                    nhc_storms,
                ): spec
                for spec in specs
            }
            for fut in concurrent.futures.as_completed(futures):
                spec = futures[fut]
                try:
                    alerts = fut.result()
                    all_alerts.extend(alerts)
                except Exception as exc:
                    log.warning("Asset %s assessment failed: %s", spec.id, exc)

        # 3. Filter and rank
        filtered = [
            a for a in all_alerts
            if a.severity.value >= self.min_severity
            and a.confidence >= self.min_confidence
        ]
        return sorted(filtered, key=lambda a: a.urgency_score, reverse=True)

    def assess_asset(
        self,
        company_id:         str,
        company_name:       str,
        asset:              dict,
        annual_revenue_usd: float,
    ) -> list[EventAlert]:
        """Assess a single asset (convenience method)."""
        spec = self._parse_asset(asset)
        nhc  = self._fetch_nhc_storms()
        return self._assess_asset(spec, company_id, company_name, annual_revenue_usd, nhc)

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_asset(d: dict) -> AssetSpec:
        return AssetSpec(
            id                 = d["id"],
            name               = d.get("name", d["id"]),
            lat                = float(d["lat"]),
            lon                = float(d["lon"]),
            sector             = d.get("sector", "default"),
            annual_revenue_usd = float(d.get("annual_revenue_usd", 0)),
        )

    def _assess_asset(
        self,
        spec:               AssetSpec,
        company_id:         str,
        company_name:       str,
        annual_revenue_usd: float,
        nhc_storms:         list,
    ) -> list[EventAlert]:
        """Full pipeline for a single asset."""
        detections: list[EventDetection] = []

        # ── NWP + seasonal forecasts ───────────────────────────────────────
        try:
            from ..connectors.open_meteo_forecast import get_forecast, get_seasonal
            forecast = get_forecast(spec.lat, spec.lon)
            seasonal = get_seasonal(spec.lat, spec.lon)
            detections += detect_from_forecast(
                spec.id, spec.lat, spec.lon, forecast, seasonal
            )
        except Exception as e:
            log.debug("Open-Meteo forecast failed for %s: %s", spec.id, e)

        # ── NHC storm track ────────────────────────────────────────────────
        try:
            detections += detect_from_nhc_storms(
                spec.id, spec.lat, spec.lon, nhc_storms, self.nhc_buffer_km
            )
        except Exception as e:
            log.debug("NHC detection failed for %s: %s", spec.id, e)

        # ── GDACS live alerts ──────────────────────────────────────────────
        try:
            from ..connectors.gdacs import get_active_disasters
            gdacs_obs = get_active_disasters(spec.lat, spec.lon)
            detections += detect_from_gdacs(spec.id, spec.lat, spec.lon, gdacs_obs)
        except Exception as e:
            log.debug("GDACS detection failed for %s: %s", spec.id, e)

        # ── Build alerts ───────────────────────────────────────────────────
        # Sectors that are not meaningfully water-sensitive — skip drought alerts
        _NON_WATER_SECTORS = {"airport", "cruise_terminal", "port_terminal", "office"}

        alerts: list[EventAlert] = []
        for det in detections:
            # Drought is only meaningful for water-intensive operations
            if det.event_type.value == "drought" and spec.sector in _NON_WATER_SECTORS:
                continue
            try:
                impact = lookup(det.event_type, det.severity, spec.sector)
                alert  = build_alert(
                    detection                  = det,
                    impact                     = impact,
                    company_id                 = company_id,
                    company_name               = company_name,
                    asset_name                 = spec.name,
                    asset_sector               = spec.sector,
                    annual_revenue_usd         = spec.annual_revenue_usd,
                    annual_revenue_usd_company = annual_revenue_usd,
                )
                alerts.append(alert)
            except Exception as e:
                log.debug("Alert build failed for detection %s/%s: %s",
                          spec.id, det.event_type, e)

        return alerts

    @staticmethod
    def _fetch_nhc_storms() -> list:
        try:
            from ..connectors.noaa_nhc import get_active_storms
            return get_active_storms()
        except Exception:
            return []
