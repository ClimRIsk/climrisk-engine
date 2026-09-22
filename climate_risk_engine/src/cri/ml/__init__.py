"""
CRI Machine Learning Layer
──────────────────────────────────────────────────────────────────────────────
Three modules that make the engine predictive and self-learning:

1. disclosure_scanner  — ClimateBERT NLP: reads ESG reports/filings,
                         extracts risk signals, adjusts ITR + commitment score
2. risk_predictor      — GradientBoosting: predicts CRI score trajectory
                         2025–2050 with confidence bands per NGFS scenario
3. emissions_estimator — XGBoost: imputes Scope 1/2/3 when CDP/EUTL data
                         is absent, far better than EEIO sector averages

All three degrade gracefully if optional ML packages (transformers, torch,
sklearn, xgboost) are not installed — the engine still runs, but AI-enhanced
fields will be absent from the output.

Install for full capability:
    pip install transformers torch scikit-learn xgboost
"""

from .disclosure_scanner import DisclosureScanner, DisclosureResult
from .risk_predictor import RiskPredictor, TrajectoryResult
from .emissions_estimator import EmissionsEstimator, EmissionsEstimate

__all__ = [
    "DisclosureScanner",
    "DisclosureResult",
    "RiskPredictor",
    "TrajectoryResult",
    "EmissionsEstimator",
    "EmissionsEstimate",
]
