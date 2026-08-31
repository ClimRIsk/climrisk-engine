"""
EventAlert — the structured output object produced by PredictiveEventEngine.

One EventAlert represents a single predicted climate event at one asset,
with full financial quantification and provenance metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .events import EventDetection, EventType, Severity
from .impact_matrix import ImpactRecord


@dataclass
class EventAlert:
    """
    A ranked, financially-quantified climate event alert for one asset.

    Financial figures are in USD unless noted.  All $ figures are probability-
    weighted (i.e., multiplied by detection confidence).
    """
    # ── Identity ──────────────────────────────────────────────────────────
    company_id:    str
    company_name:  str
    asset_id:      str
    asset_name:    str
    asset_lat:     float
    asset_lon:     float
    asset_sector:  str

    # ── Event ─────────────────────────────────────────────────────────────
    event_type:         EventType
    severity:           Severity
    time_horizon_days:  float       # days until onset
    duration_days:      float       # expected impact length
    confidence:         float       # 0–1
    peak_date:          Optional[str]
    event_metadata:     dict = field(default_factory=dict)

    # ── Impact (from ImpactRecord × daily revenue) ─────────────────────────
    daily_revenue_usd:              float = 0.0
    revenue_at_risk_per_day_usd:    float = 0.0   # at peak severity
    total_revenue_at_risk_usd:      float = 0.0   # expected_value = p×impact
    capex_at_risk_usd:              float = 0.0   # prob-weighted unplanned capex
    ebitda_margin_hit_pp:           float = 0.0

    # ── Ranking ────────────────────────────────────────────────────────────
    urgency_score: float = 0.0   # higher = act sooner

    # ── Provenance ─────────────────────────────────────────────────────────
    impact_notes: str = ""

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def horizon_label(self) -> str:
        d = self.time_horizon_days
        if d <= 0:
            return "Active now"
        if d <= 1:
            return f"Within 24 hours"
        if d <= 7:
            return f"~{int(d)} days"
        if d <= 30:
            return f"~{int(d // 7)} week{'s' if d >= 14 else ''}"
        return f"~{int(d // 30)} month{'s' if d >= 60 else ''}"

    @property
    def severity_label(self) -> str:
        return {
            Severity.NEGLIGIBLE:  "Negligible",
            Severity.MINOR:       "Minor",
            Severity.MODERATE:    "Moderate",
            Severity.SIGNIFICANT: "Significant",
            Severity.SEVERE:      "Severe",
            Severity.EXTREME:     "Extreme",
        }.get(self.severity, "Unknown")

    @property
    def event_label(self) -> str:
        return {
            EventType.TROPICAL_CYCLONE:    "Tropical Cyclone",
            EventType.EXTRATROPICAL_STORM: "Windstorm",
            EventType.HEAT_WAVE:           "Heat Wave",
            EventType.COLD_SNAP:           "Cold Snap",
            EventType.HEAVY_RAIN_FLOOD:    "Heavy Rain / Flood",
            EventType.DROUGHT:             "Drought",
            EventType.WILDFIRE_RISK:       "Wildfire Conditions",
            EventType.ACTIVE_DISASTER:     "Active Disaster (GDACS)",
        }.get(self.event_type, str(self.event_type))

    @property
    def summary_line(self) -> str:
        rev = self.total_revenue_at_risk_usd
        rev_str = f"${rev/1e6:.1f}M" if rev >= 1e6 else f"${rev/1e3:.0f}K"
        return (
            f"[{self.severity_label.upper()} | {self.horizon_label}] "
            f"{self.event_label} @ {self.asset_name} ({self.company_name}) — "
            f"Revenue at risk: {rev_str} | "
            f"Disruption: ~{self.duration_days:.0f} days | "
            f"Confidence: {self.confidence*100:.0f}%"
        )

    def to_dict(self) -> dict:
        return {
            "company_id":               self.company_id,
            "company_name":             self.company_name,
            "asset_id":                 self.asset_id,
            "asset_name":               self.asset_name,
            "asset_lat":                self.asset_lat,
            "asset_lon":                self.asset_lon,
            "asset_sector":             self.asset_sector,
            "event_type":               self.event_type.value,
            "event_label":              self.event_label,
            "severity":                 self.severity.value,
            "severity_label":           self.severity_label,
            "time_horizon_days":        round(self.time_horizon_days, 1),
            "horizon_label":            self.horizon_label,
            "duration_days":            round(self.duration_days, 1),
            "peak_date":                self.peak_date,
            "confidence":               round(self.confidence, 2),
            "daily_revenue_usd":        round(self.daily_revenue_usd),
            "revenue_at_risk_per_day":  round(self.revenue_at_risk_per_day_usd),
            "total_revenue_at_risk":    round(self.total_revenue_at_risk_usd),
            "capex_at_risk_usd":        round(self.capex_at_risk_usd),
            "ebitda_margin_hit_pp":     round(self.ebitda_margin_hit_pp, 2),
            "urgency_score":            round(self.urgency_score, 3),
            "impact_notes":             self.impact_notes,
            "event_metadata":           self.event_metadata,
            "summary_line":             self.summary_line,
        }


def build_alert(
    detection: EventDetection,
    impact: ImpactRecord,
    company_id: str,
    company_name: str,
    asset_name: str,
    asset_sector: str,
    annual_revenue_usd: float,
    annual_revenue_usd_company: float,
) -> EventAlert:
    """
    Combine an EventDetection + ImpactRecord + revenue context into an EventAlert.

    annual_revenue_usd          : revenue attributable to this specific asset
    annual_revenue_usd_company  : total company revenue (for capex calc)
    """
    daily_rev = annual_revenue_usd / 365.0

    # Revenue at risk per day (peak, not probability-weighted)
    rev_per_day = daily_rev * impact.revenue_at_risk_fraction

    # Expected total revenue at risk = daily × disruption_days × confidence
    total_rev_risk = rev_per_day * impact.disruption_days * detection.confidence

    # Capex at risk (company-level, probability-weighted)
    capex_risk = (
        annual_revenue_usd_company
        * impact.capex_fraction_of_revenue
        * impact.capex_trigger_prob
        * detection.confidence
    )

    urgency = detection.urgency_score

    return EventAlert(
        company_id   = company_id,
        company_name = company_name,
        asset_id     = detection.asset_id,
        asset_name   = asset_name,
        asset_lat    = detection.asset_lat,
        asset_lon    = detection.asset_lon,
        asset_sector = asset_sector,
        event_type           = detection.event_type,
        severity             = detection.severity,
        time_horizon_days    = detection.time_horizon_days,
        duration_days        = detection.duration_days,
        confidence           = detection.confidence,
        peak_date            = detection.peak_date,
        event_metadata       = detection.metadata,
        daily_revenue_usd             = daily_rev,
        revenue_at_risk_per_day_usd   = rev_per_day,
        total_revenue_at_risk_usd     = total_rev_risk,
        capex_at_risk_usd             = capex_risk,
        ebitda_margin_hit_pp          = impact.ebitda_margin_hit_pp,
        urgency_score        = urgency,
        impact_notes         = impact.notes,
    )
