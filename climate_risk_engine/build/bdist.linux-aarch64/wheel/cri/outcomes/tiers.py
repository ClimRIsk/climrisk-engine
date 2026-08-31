"""
CRI Freemium Tier System.

Defines what each tier can access and provides a gate function that strips
paid-only fields from RunResults before returning them to free-tier users.

Tiers
-----
FREE
    - Climate risk rating (A–E letter grade) + pillar labels
    - 1 scenario (Current Policies only)
    - 5-year horizon (2026–2030)
    - No asset-level breakdown
    - No financial numbers (EV, equity, FCF)
    - No disclosure reports

ANALYST    ($500–$1,500 / month)
    - All 3 scenarios
    - Full 25-year horizon
    - Asset-level physical risk breakdown
    - Full financial trajectory (revenue, EBITDA, FCF, EV, equity)
    - Pillar sub-scores with driver narratives
    - CSV/JSON export

PROFESSIONAL    ($2,500–$5,000 / month)
    - Everything in Analyst
    - TCFD-aligned disclosure report (PDF)
    - ISSB S2 metrics table
    - EU CSRD E1 data points
    - Sensitivity / scenario sensitivity tables
    - Share-price impact analysis
    - Priority email support

ENTERPRISE    ($50,000–$150,000 / year)
    - Everything in Professional
    - Full API access (white-label ready)
    - Custom scenario builder
    - Portfolio-level aggregation across multiple companies
    - Dedicated analyst support + consultancy retainer
    - SLA, audit trail, SSO
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class Tier(str, Enum):
    FREE         = "free"
    ANALYST      = "analyst"
    PROFESSIONAL = "professional"
    ENTERPRISE   = "enterprise"

    @classmethod
    def from_str(cls, s: str) -> "Tier":
        try:
            return cls(s.lower())
        except ValueError:
            return cls.FREE


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierFeatures:
    scenarios_allowed: int          # max scenario runs visible
    horizon_years: int              # max horizon shown (from start year)
    asset_breakdown: bool           # per-asset physical/financial breakdown
    full_financials: bool           # EV, equity, FCF, share price
    pillar_scores: bool             # numeric sub-scores (not just labels)
    disclosure_tcfd: bool           # TCFD report generation
    disclosure_issb: bool           # ISSB S2 metrics
    disclosure_csrd: bool           # EU CSRD E1 data points
    sensitivity: bool               # sensitivity tables
    api_access: bool                # REST API access
    export: bool                    # CSV/JSON/PDF export
    custom_scenarios: bool          # custom scenario builder
    portfolio: bool                 # multi-company aggregation


TIER_FEATURES: dict[Tier, TierFeatures] = {
    Tier.FREE: TierFeatures(
        scenarios_allowed=1,
        horizon_years=5,
        asset_breakdown=False,
        full_financials=False,
        pillar_scores=False,
        disclosure_tcfd=False,
        disclosure_issb=False,
        disclosure_csrd=False,
        sensitivity=False,
        api_access=False,
        export=False,
        custom_scenarios=False,
        portfolio=False,
    ),
    Tier.ANALYST: TierFeatures(
        scenarios_allowed=3,
        horizon_years=25,
        asset_breakdown=True,
        full_financials=True,
        pillar_scores=True,
        disclosure_tcfd=False,
        disclosure_issb=False,
        disclosure_csrd=False,
        sensitivity=False,
        api_access=True,
        export=True,
        custom_scenarios=False,
        portfolio=False,
    ),
    Tier.PROFESSIONAL: TierFeatures(
        scenarios_allowed=3,
        horizon_years=25,
        asset_breakdown=True,
        full_financials=True,
        pillar_scores=True,
        disclosure_tcfd=True,
        disclosure_issb=True,
        disclosure_csrd=True,
        sensitivity=True,
        api_access=True,
        export=True,
        custom_scenarios=False,
        portfolio=False,
    ),
    Tier.ENTERPRISE: TierFeatures(
        scenarios_allowed=99,
        horizon_years=30,
        asset_breakdown=True,
        full_financials=True,
        pillar_scores=True,
        disclosure_tcfd=True,
        disclosure_issb=True,
        disclosure_csrd=True,
        sensitivity=True,
        api_access=True,
        export=True,
        custom_scenarios=True,
        portfolio=True,
    ),
}


TIER_PRICING: dict[Tier, dict] = {
    Tier.FREE: {
        "label": "Free",
        "price": "Free",
        "cta": "Get your climate rating",
        "description": "Instant A–E climate risk rating for your company. No card required.",
    },
    Tier.ANALYST: {
        "label": "Analyst",
        "price": "$500–$1,500 / month",
        "cta": "Start free trial",
        "description": (
            "Full multi-scenario DCF, 25-year trajectories, asset-level breakdown, "
            "and data export. Ideal for in-house analyst teams."
        ),
    },
    Tier.PROFESSIONAL: {
        "label": "Professional",
        "price": "$2,500–$5,000 / month",
        "cta": "Book a demo",
        "description": (
            "Everything in Analyst plus TCFD, ISSB S2, and EU CSRD disclosure reports, "
            "sensitivity analysis, and priority support."
        ),
    },
    Tier.ENTERPRISE: {
        "label": "Enterprise",
        "price": "$50,000–$150,000 / year",
        "cta": "Contact us",
        "description": (
            "Full API, white-label, custom scenarios, portfolio aggregation, "
            "and a dedicated analyst consultancy retainer."
        ),
    },
}


# ---------------------------------------------------------------------------
# Gate class
# ---------------------------------------------------------------------------


class TierGate:
    """
    Apply tier-based access control to result dictionaries.

    Usage:
        gate = TierGate(Tier.FREE)
        safe_results = gate.apply(full_results_dict)
    """

    _LOCKED_PLACEHOLDER = "🔒 Upgrade to unlock"
    _LOCKED_NUMERIC = None

    def __init__(self, tier: Tier = Tier.FREE):
        self.tier = tier
        self.features = TIER_FEATURES[tier]

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Strip or mask fields that are not allowed at the current tier."""
        out = dict(data)

        if not self.features.full_financials:
            for field in [
                "enterprise_value", "equity_value", "implied_share_price",
                "npv_fcf", "terminal_value", "wacc_used", "npv_impact_pct",
                "ebitda_compression_2030_pct", "ebitda_compression_2040_pct",
                "baseline_npv",
            ]:
                if field in out:
                    out[field] = self._LOCKED_NUMERIC

        if not self.features.pillar_scores:
            for field in ["exposure_score", "transition_score", "financial_score", "adaptive_score"]:
                if field in out:
                    out[field] = self._LOCKED_NUMERIC

        if not self.features.asset_breakdown and "asset_breakdown" in out:
            out["asset_breakdown"] = self._LOCKED_PLACEHOLDER

        # Truncate trajectory
        if "years" in out and isinstance(out["years"], list):
            max_yr = self.features.horizon_years
            start = out.get("start_year", 2026)
            cutoff = start + max_yr
            out["years"] = [y for y in out["years"] if y.get("year", 0) < cutoff]
            if not self.features.full_financials:
                out["years"] = [self._strip_financials(y) for y in out["years"]]

        out["_tier"] = self.tier.value
        out["_locked_features"] = self._locked_list()
        return out

    def _strip_financials(self, year_dict: dict) -> dict:
        y = dict(year_dict)
        for field in ["fcf", "nopat", "ebit", "da", "transition_capex",
                      "adaptation_capex", "maintenance_capex", "working_capital_change"]:
            if field in y:
                y[field] = self._LOCKED_NUMERIC
        return y

    def _locked_list(self) -> list[str]:
        locked = []
        if not self.features.full_financials:
            locked.append("Full financial trajectory (EV, equity, FCF)")
        if not self.features.asset_breakdown:
            locked.append("Asset-level physical & financial breakdown")
        if not self.features.pillar_scores:
            locked.append("Pillar risk sub-scores")
        if not self.features.disclosure_tcfd:
            locked.append("TCFD disclosure report")
        if not self.features.disclosure_issb:
            locked.append("ISSB S2 metrics")
        if not self.features.disclosure_csrd:
            locked.append("EU CSRD E1 data points")
        if not self.features.sensitivity:
            locked.append("Sensitivity & scenario analysis tables")
        if not self.features.export:
            locked.append("CSV / JSON / PDF export")
        if not self.features.custom_scenarios:
            locked.append("Custom scenario builder")
        if not self.features.portfolio:
            locked.append("Portfolio-level aggregation")
        return locked

    def can_access(self, feature: str) -> bool:
        """Check if a named feature is available at this tier."""
        return bool(getattr(self.features, feature, False))

    @staticmethod
    def upgrade_prompt(feature: str) -> dict:
        """Return a structured upgrade prompt for a locked feature."""
        feature_tier_map = {
            "full_financials":  Tier.ANALYST,
            "asset_breakdown":  Tier.ANALYST,
            "pillar_scores":    Tier.ANALYST,
            "disclosure_tcfd":  Tier.PROFESSIONAL,
            "disclosure_issb":  Tier.PROFESSIONAL,
            "disclosure_csrd":  Tier.PROFESSIONAL,
            "sensitivity":      Tier.PROFESSIONAL,
            "custom_scenarios": Tier.ENTERPRISE,
            "portfolio":        Tier.ENTERPRISE,
        }
        min_tier = feature_tier_map.get(feature, Tier.ANALYST)
        pricing = TIER_PRICING[min_tier]
        return {
            "locked": True,
            "required_tier": min_tier.value,
            "tier_label": pricing["label"],
            "price": pricing["price"],
            "cta": pricing["cta"],
            "message": f"Available from the {pricing['label']} plan ({pricing['price']})",
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def gate_results(
    results: dict[str, Any],
    tier: str | Tier = Tier.FREE,
) -> dict[str, Any]:
    """Apply tier gating to a results dict. Accepts string or Tier enum."""
    if isinstance(tier, str):
        tier = Tier.from_str(tier)
    return TierGate(tier).apply(results)
