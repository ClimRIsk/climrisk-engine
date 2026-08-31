"""CRI Outcomes layer — ratings, tiers, and financial disclosures."""

from .ratings import ClimateRating, RatingEngine, RatingResult
from .tiers import Tier, TierGate, gate_results
from .disclosure import (
    DisclosureReport,
    TCFDReport,
    ISSBReport,
    CSRDReport,
    generate_tcfd,
    generate_issb,
    generate_csrd,
)

__all__ = [
    "ClimateRating",
    "RatingEngine",
    "RatingResult",
    "Tier",
    "TierGate",
    "gate_results",
    "DisclosureReport",
    "TCFDReport",
    "ISSBReport",
    "CSRDReport",
    "generate_tcfd",
    "generate_issb",
    "generate_csrd",
]
