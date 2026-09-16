"""
provenance.py — Source tagging for every data field acquired by the engine.

Every piece of data the engine uses MUST be wrapped in a ProvenanceField so
practitioners can audit where a number came from, how fresh it is, and how
confident the system is.

No silent fallbacks.  If data cannot be obtained, the field is None and the
reason is recorded in data_gaps on the CompanyProfile.

ConfidenceTier
--------------
VERIFIED   — Primary regulatory source (ETS, EDGAR, Companies House)
REPORTED   — Company self-reported but publicly filed (CDP, sustainability report)
ESTIMATED  — Sector benchmark / proxy model (explicitly flagged low-confidence)
MISSING    — Field could not be obtained; gap recorded
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Optional


class ConfidenceTier(str, Enum):
    VERIFIED  = "verified"    # Regulatory registry (ETS, EDGAR, Companies House)
    REPORTED  = "reported"    # Company self-reported, publicly filed
    ESTIMATED = "estimated"   # Sector/country benchmark proxy
    MISSING   = "missing"     # Could not be obtained


@dataclass
class ProvenanceField:
    """
    A single data field with full source attribution.

    Attributes
    ----------
    value           : The actual data value (None → field is missing)
    source          : Short human-readable source label, e.g. "SEC EDGAR 10-K"
    url             : Direct URL to the source document or API response
    retrieval_date  : ISO date when the value was fetched
    confidence_tier : ConfidenceTier enum member
    notes           : Optional explanation (e.g. "interpolated from 2022 filing")
    """
    value:           Any
    source:          str
    url:             Optional[str]  = None
    retrieval_date:  Optional[str]  = None   # ISO-8601 date string
    confidence_tier: ConfidenceTier = ConfidenceTier.REPORTED
    notes:           Optional[str]  = None

    def is_missing(self) -> bool:
        return self.value is None or self.confidence_tier == ConfidenceTier.MISSING

    def to_dict(self) -> dict:
        return {
            "value":           self.value,
            "source":          self.source,
            "url":             self.url,
            "retrieval_date":  self.retrieval_date,
            "confidence_tier": self.confidence_tier.value,
            "notes":           self.notes,
        }

    @classmethod
    def missing(cls, field_name: str, reason: str) -> "ProvenanceField":
        """Factory for an explicit gap record."""
        return cls(
            value=None,
            source=f"NOT FOUND — {reason}",
            url=None,
            retrieval_date=date.today().isoformat(),
            confidence_tier=ConfidenceTier.MISSING,
            notes=field_name,
        )

    @classmethod
    def estimated(
        cls,
        value: Any,
        basis: str,
        notes: str = "",
    ) -> "ProvenanceField":
        """Factory for a benchmark/proxy estimate — always ConfidenceTier.ESTIMATED."""
        return cls(
            value=value,
            source=f"Sector/country benchmark: {basis}",
            url=None,
            retrieval_date=date.today().isoformat(),
            confidence_tier=ConfidenceTier.ESTIMATED,
            notes=notes or "Estimated — explicit primary data preferred",
        )


@dataclass
class DataGap:
    """Records a field that could not be sourced."""
    field_name:  str
    reason:      str
    impact:      str    # e.g. "physical hazard uses lat/lon fallback"
    suggested_action: str  # e.g. "Submit annual Scope 1 GHG inventory"


def today_iso() -> str:
    return date.today().isoformat()
