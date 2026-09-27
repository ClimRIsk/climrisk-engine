"""HuggingFace CarbonLens + ND-GAIN enricher.
══════════════════════════════════════════════════════════════════════════════
Source 1 – CarbonLens (SubramaniMokkala/climate-credit-risk-analyzer)
  • 421 S&P 500 tickers → XGBoost climate-credit risk scores, ESG levels,
    high-carbon flag.

Source 2 – ND-GAIN Extended (Notre Dame Global Adaptation Initiative)
  • 59 countries → gain_score (0-100), vulnerability (0-1), readiness (0-1).
  • Derived from the CarbonLens dataset + curated ND-GAIN 2022 values.

Usage
─────
    from cri.connectors.hf_carbontlens import HFEnricher

    # By ticker (S&P 500 companies)
    r = HFEnricher.by_ticker("AAPL")
    # → {'credit_score': 0.408, 'risk_tier': 'Medium', 'gain_score': 66.8,
    #    'vulnerability': 0.312, 'readiness': 0.647, 'esg_level': 'Low', 'high_carbon': 0}

    # By country name
    r = HFEnricher.by_country("Germany")
    # → {'gain_score': 70.21, 'vulnerability': 0.260, 'readiness': 0.714,
    #    'climate_readiness_label': 'High', 'ndgain_tier': 'Resilient'}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

# ── Data paths ───────────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"
_TICKERS_PATH  = _DATA_DIR / "carbontlens_tickers.json"
_NDGAIN_PATH   = _DATA_DIR / "ndgain_extended.json"

# ── Lazy singletons ──────────────────────────────────────────────────────────
_tickers: Optional[Dict[str, Any]] = None
_ndgain:  Optional[Dict[str, Any]] = None


def _load_tickers() -> Dict[str, Any]:
    global _tickers
    if _tickers is None:
        _tickers = json.loads(_TICKERS_PATH.read_text())
    return _tickers


def _load_ndgain() -> Dict[str, Any]:
    global _ndgain
    if _ndgain is None:
        raw = json.loads(_NDGAIN_PATH.read_text())
        # Normalise format: stored as either dict or list [gain, vuln, readiness]
        out: Dict[str, Any] = {}
        for country, val in raw.items():
            if isinstance(val, list):
                out[country] = {
                    "gain_score": val[0],
                    "vulnerability_score": val[1],
                    "readiness_score": val[2],
                }
            else:
                out[country] = val
        _ndgain = out
    return _ndgain


def _ndgain_tier(gain: float) -> str:
    """Classify ND-GAIN score into resilience tier."""
    if gain >= 70:  return "Resilient"
    if gain >= 55:  return "Adapting"
    if gain >= 45:  return "Vulnerable"
    return "Highly Vulnerable"


def _readiness_label(r: float) -> str:
    if r >= 0.65: return "High"
    if r >= 0.45: return "Medium"
    return "Low"


class HFEnricher:
    """Static helper — no instantiation needed."""

    # ── Ticker lookup ─────────────────────────────────────────────────────────

    @staticmethod
    def by_ticker(symbol: str) -> Optional[Dict[str, Any]]:
        """Return CarbonLens record for a ticker, or None if not found.

        Returned dict keys:
            credit_score     float  0-1  XGBoost climate credit risk score
            risk_tier        str        Low / Medium / High / Critical
            gain_score       float  0-100 ND-GAIN composite
            vulnerability    float  0-1  country climate vulnerability
            readiness        float  0-1  country climate readiness
            esg_level        str        Low / Medium / High / Severe / n/a
            high_carbon      int        1 if high-carbon sector, else 0
            ndgain_tier      str        Resilient / Adapting / Vulnerable / Highly Vulnerable
        """
        tickers = _load_tickers()
        rec = tickers.get(symbol.upper())
        if rec is None:
            return None
        gain = rec.get("gain_score") or 0.0
        return {
            "credit_score":  rec.get("credit_score"),
            "risk_tier":     rec.get("risk_tier", "Unknown"),
            "gain_score":    gain,
            "vulnerability": rec.get("vulnerability"),
            "readiness":     rec.get("readiness"),
            "esg_level":     rec.get("esg_level", "n/a"),
            "high_carbon":   rec.get("high_carbon", 0),
            "ndgain_tier":   _ndgain_tier(gain),
        }

    # ── Country lookup ────────────────────────────────────────────────────────

    @staticmethod
    def by_country(country: str) -> Optional[Dict[str, Any]]:
        """Return ND-GAIN scores for a country name, or None if unknown.

        Returned dict keys:
            gain_score              float  0-100
            vulnerability_score     float  0-1
            readiness_score         float  0-1
            climate_readiness_label str    High / Medium / Low
            ndgain_tier             str    Resilient / Adapting / Vulnerable / Highly Vulnerable
        """
        ndgain = _load_ndgain()
        # Try exact match first, then case-insensitive
        rec = ndgain.get(country) or next(
            (v for k, v in ndgain.items() if k.lower() == country.lower()), None
        )
        if rec is None:
            return None
        gain  = rec.get("gain_score", 0.0)
        vuln  = rec.get("vulnerability_score", 0.0)
        ready = rec.get("readiness_score", 0.0)
        return {
            "gain_score":              gain,
            "vulnerability_score":     vuln,
            "readiness_score":         ready,
            "climate_readiness_label": _readiness_label(ready),
            "ndgain_tier":             _ndgain_tier(gain),
        }

    # ── Jurisdiction code → country name map (ISO-2 subset) ─────────────────

    _JURISDICTION_MAP: Dict[str, str] = {
        "US": "United States",  "GB": "United Kingdom",  "DE": "Germany",
        "FR": "France",         "JP": "Japan",            "CA": "Canada",
        "AU": "Australia",      "CH": "Switzerland",      "NL": "Netherlands",
        "SE": "Sweden",         "NO": "Norway",           "DK": "Denmark",
        "FI": "Finland",        "IE": "Ireland",          "SG": "Singapore",
        "KR": "South Korea",    "CN": "China",            "IN": "India",
        "BR": "Brazil",         "RU": "Russia",           "MX": "Mexico",
        "ID": "Indonesia",      "TR": "Turkey",           "SA": "Saudi Arabia",
        "AE": "UAE",            "ZA": "South Africa",     "NG": "Nigeria",
        "AR": "Argentina",      "CO": "Colombia",         "CL": "Chile",
        "PL": "Poland",         "CZ": "Czech Republic",   "HU": "Hungary",
        "GR": "Greece",         "PT": "Portugal",         "ES": "Spain",
        "IT": "Italy",          "BE": "Belgium",          "AT": "Austria",
        "NZ": "New Zealand",    "TH": "Thailand",         "VN": "Vietnam",
        "PH": "Philippines",    "MY": "Malaysia",         "PK": "Pakistan",
        "BD": "Bangladesh",     "EG": "Egypt",            "KE": "Kenya",
        "GH": "Ghana",          "ET": "Ethiopia",         "TZ": "Tanzania",
        "MA": "Morocco",        "IL": "Israel",           "UA": "Ukraine",
        "RO": "Romania",        "TW": "Taiwan",           "HK": "Hong Kong",
        "LU": "Luxembourg",     "IS": "Iceland",
    }

    @classmethod
    def by_jurisdiction(cls, iso2: str) -> Optional[Dict[str, Any]]:
        """Convenience: look up by ISO-2 country code (e.g. 'DE', 'IN')."""
        country = cls._JURISDICTION_MAP.get(iso2.upper())
        if not country:
            return None
        return cls.by_country(country)

    # ── Composite enrichment (best available data) ────────────────────────────

    @classmethod
    def enrich(
        cls,
        ticker: Optional[str] = None,
        country: Optional[str] = None,
        jurisdiction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the richest available enrichment, merging ticker + country data.

        Priority: ticker record (has both company and country data)
                  → country name lookup
                  → jurisdiction code lookup
                  → empty dict (caller handles gracefully)
        """
        result: Dict[str, Any] = {}

        # Try ticker first — it already carries country ND-GAIN fields
        if ticker:
            rec = cls.by_ticker(ticker)
            if rec:
                result.update(rec)
                return result

        # Fall back to country / jurisdiction
        country_rec: Optional[Dict[str, Any]] = None
        if country:
            country_rec = cls.by_country(country)
        if country_rec is None and jurisdiction:
            country_rec = cls.by_jurisdiction(jurisdiction)
        if country_rec:
            result.update(country_rec)

        return result
