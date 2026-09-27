"""HuggingFace Climate Risk Monitoring MCP connector.
══════════════════════════════════════════════════════════════════════════════
Source: towardsinnovationlab/Climate_Risk_Monitoring (HuggingFace Spaces)
  https://huggingface.co/spaces/towardsinnovationlab/Climate_Risk_Monitoring

What it provides
────────────────
  • Historical weather records for any lat/lon
  • Satellite image analysis
  • Short-term climate forecasts
  • Structured climate risk report per location

This connector calls the Space's Gradio REST API (available on all running
HF Spaces via /api/predict).  When the Space is sleeping it auto-wakes on
the first request (may take ~30 s).

Usage
─────
    from cri.connectors.hf_climate_monitor import ClimateMonitorClient

    client = ClimateMonitorClient()

    # Get a risk report for a lat/lon
    report = client.get_risk_report(lat=51.5, lon=-0.1, location="London")
    # → {'risk_level': 'Low', 'temperature_anomaly': +1.2,
    #    'precipitation_anomaly': -0.08, 'summary': '...', 'raw': {...}}

    # Quick hazard score 0-100 from the monitoring Space
    score = client.hazard_score(lat=13.0, lon=80.0, location="Chennai")
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# HuggingFace Spaces Gradio REST endpoint
_HF_SPACE_URL = "https://towardsinnovationlab-climate-risk-monitoring.hf.space"
_API_ENDPOINT  = f"{_HF_SPACE_URL}/api/predict"
_TIMEOUT_S     = 60   # Spaces may cold-start up to 60 s

# Retry config for sleeping spaces
_MAX_RETRIES    = 3
_RETRY_DELAY_S  = 20


class ClimateMonitorClient:
    """Thin HTTP client for the towardsinnovationlab Climate_Risk_Monitoring Space."""

    def __init__(self, base_url: str = _HF_SPACE_URL, timeout: int = _TIMEOUT_S):
        self._base     = base_url.rstrip("/")
        self._endpoint = f"{self._base}/api/predict"
        self._timeout  = timeout

    # ── Public API ────────────────────────────────────────────────────────────

    def get_risk_report(
        self,
        lat: float,
        lon: float,
        location: str = "",
    ) -> Dict[str, Any]:
        """Fetch a climate risk report for a coordinate.

        Returns a dict with at least:
            risk_level          str   Low / Moderate / High / Severe
            temperature_anomaly float °C above baseline
            precipitation_anomaly float fraction change vs baseline
            summary             str   plain-English assessment
            source              str   'hf_climate_monitor'
            lat, lon            float echoed back
        Returns a fallback dict on error (never raises).
        """
        payload = {"data": [lat, lon, location or f"{lat:.2f},{lon:.2f}"]}
        raw = self._call(payload)
        if raw is None:
            return self._fallback(lat, lon, "request_failed")
        return self._parse_report(raw, lat, lon)

    def hazard_score(
        self,
        lat: float,
        lon: float,
        location: str = "",
    ) -> float:
        """Return a 0-100 hazard score derived from the monitoring Space.

        Useful as a quick cross-check against the engine's own hazard estimates.
        Returns 50.0 on error (neutral / unknown).
        """
        report = self.get_risk_report(lat=lat, lon=lon, location=location)
        tier_scores = {"Low": 20.0, "Moderate": 40.0, "High": 65.0, "Severe": 85.0}
        tier  = report.get("risk_level", "Moderate")
        base  = tier_scores.get(tier, 40.0)
        # Adjust ±10 pts by temperature anomaly (clamped)
        delta = report.get("temperature_anomaly", 0.0)
        return round(min(100.0, max(0.0, base + delta * 5.0)), 1)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call(self, payload: Dict) -> Optional[Dict]:
        """POST to the HF Space API with retries on 503 (sleeping space)."""
        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            self._endpoint,
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (503, 502) and attempt < _MAX_RETRIES:
                    log.info(
                        "HF Climate Monitor sleeping (HTTP %s), waiting %ss …",
                        e.code, _RETRY_DELAY_S,
                    )
                    time.sleep(_RETRY_DELAY_S)
                else:
                    log.warning("HF Climate Monitor HTTP %s after %s tries.", e.code, attempt)
                    return None
            except Exception as exc:
                log.warning("HF Climate Monitor request failed: %s", exc)
                return None
        return None

    @staticmethod
    def _parse_report(raw: Dict, lat: float, lon: float) -> Dict[str, Any]:
        """Extract structured fields from the Space's raw API response."""
        try:
            # Gradio /api/predict wraps output in {"data": [...]}
            data = raw.get("data", [raw])
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, str):
                # Try parsing embedded JSON string
                try:
                    data = json.loads(data)
                except Exception:
                    pass
            if isinstance(data, dict):
                return {
                    "risk_level":              data.get("risk_level", "Moderate"),
                    "temperature_anomaly":     float(data.get("temperature_anomaly", 0.0)),
                    "precipitation_anomaly":   float(data.get("precipitation_anomaly", 0.0)),
                    "summary":                 data.get("summary", ""),
                    "source":                  "hf_climate_monitor",
                    "lat":                     lat,
                    "lon":                     lon,
                    "raw":                     data,
                }
        except Exception as exc:
            log.debug("Could not parse HF Climate Monitor response: %s", exc)

        # Graceful degradation — return neutral record
        return {
            "risk_level":             "Moderate",
            "temperature_anomaly":    0.0,
            "precipitation_anomaly":  0.0,
            "summary":                "Data unavailable — using engine defaults.",
            "source":                 "hf_climate_monitor_fallback",
            "lat": lat, "lon": lon,
        }

    @staticmethod
    def _fallback(lat: float, lon: float, reason: str) -> Dict[str, Any]:
        return {
            "risk_level":             "Moderate",
            "temperature_anomaly":    0.0,
            "precipitation_anomaly":  0.0,
            "summary":                f"Climate Monitor unavailable ({reason}).",
            "source":                 "hf_climate_monitor_fallback",
            "lat": lat, "lon": lon,
        }


# ── Module-level singleton (lazy) ─────────────────────────────────────────────
_client: Optional[ClimateMonitorClient] = None


def get_client() -> ClimateMonitorClient:
    """Return the shared ClimateMonitorClient instance."""
    global _client
    if _client is None:
        _client = ClimateMonitorClient()
    return _client
