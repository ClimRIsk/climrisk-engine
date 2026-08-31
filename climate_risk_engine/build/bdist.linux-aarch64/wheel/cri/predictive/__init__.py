"""
CRI Predictive Event Engine
===========================

Short-to-medium horizon climate event prediction and financial impact
translation for company assets.

Horizons
--------
  0–16 days   : Open-Meteo NWP deterministic (storm, heat wave, heavy rain, cold snap)
  17–90 days  : Open-Meteo SEAS5 seasonal ensemble (drought, temperature extremes)
  0–5 days    : NOAA NHC tropical cyclone advisories (Atlantic + E Pacific)
  0–7 days    : GDACS active disaster alerts (flood, cyclone, tsunami)

Entry point
-----------
    from cri.predictive import PredictiveEventEngine

    engine = PredictiveEventEngine()
    alerts = engine.assess_company(
        company_id="carnival_corp",
        assets=[
            {"id": "miami_port",    "lat": 25.7741, "lon": -80.1842, "sector": "cruise_terminal"},
            {"id": "southampton",   "lat": 50.9097, "lon": -1.4044,  "sector": "cruise_terminal"},
        ],
        annual_revenue_usd=21_600_000_000,
    )
    for alert in alerts:
        print(alert.summary_line)
"""

from .engine import PredictiveEventEngine
from .events import EventDetection, EventType, Severity
from .alerts import EventAlert

__all__ = [
    "PredictiveEventEngine",
    "EventDetection",
    "EventType",
    "Severity",
    "EventAlert",
]
