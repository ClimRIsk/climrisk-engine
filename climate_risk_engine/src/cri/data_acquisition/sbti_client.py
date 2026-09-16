"""
sbti_client.py — Science Based Targets initiative (SBTi) database.

SBTi publishes a public list of companies with:
  • Committed targets (submitted, validation pending)
  • Approved targets (validated by SBTi)
  • Targets aligned with 1.5°C / Well Below 2°C / 2°C pathways
  • Net Zero targets

Data source: SBTi public target dashboard
  https://sciencebasedtargets.org/companies-taking-action
  CSV: https://sciencebasedtargets.org/resources/legacy/2021/06/SBTi-Target-Checker.xlsm

We use the publicly available JSON endpoint that powers their dashboard.

Confidence tier: REPORTED (company commitment, SBTi-validated where approved)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .provenance import ConfidenceTier, DataGap, ProvenanceField, today_iso

logger = logging.getLogger(__name__)

_SBTI_API   = "https://sciencebasedtargets.org/api/v2/targets"
_SBTI_BASE  = "https://sciencebasedtargets.org/companies-taking-action"
_TIMEOUT    = 15.0
_HEADERS    = {
    "User-Agent": "ClimRiskEngine/1.0 contact@climrisk.io",
    "Accept":     "application/json",
}

# SBTi temperature alignment labels
_TEMPERATURE_MAP = {
    "1.5C": "1.5°C",
    "WB2C": "Well Below 2°C",
    "2C":   "2°C",
    "4C":   "4°C (insufficient)",
}


@dataclass
class SbtiResult:
    company_name:          Optional[str]             = None
    sbti_status:           Optional[str]             = None  # Committed / Targets Set / Removed
    near_term_status:      Optional[str]             = None  # Approved / Committed / Not Set
    long_term_status:      Optional[str]             = None  # Net Zero Approved / Committed
    temperature_ambition:  Optional[str]             = None  # 1.5°C / WB2°C / 2°C
    target_year:           Optional[int]             = None  # Near-term target year
    net_zero_year:         Optional[int]             = None
    sbti_provenance:       Optional[ProvenanceField] = None
    data_gaps:             list[DataGap]             = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "company_name":         self.company_name,
            "sbti_status":          self.sbti_status,
            "near_term_status":     self.near_term_status,
            "long_term_status":     self.long_term_status,
            "temperature_ambition": self.temperature_ambition,
            "target_year":          self.target_year,
            "net_zero_year":        self.net_zero_year,
            "sbti_provenance":      self.sbti_provenance.to_dict() if self.sbti_provenance else None,
            "data_gaps":            [
                {"field": g.field_name, "reason": g.reason, "action": g.suggested_action}
                for g in self.data_gaps
            ],
        }


def _fuzzy_name_match(query: str, candidate: str, threshold: float = 0.75) -> bool:
    """Simple token-overlap check for company name matching."""
    q_tokens = set(query.lower().split())
    c_tokens = set(candidate.lower().split())
    # Remove common suffixes
    stopwords = {"inc", "llc", "ltd", "plc", "sa", "ag", "gmbh", "bv", "nv", "co", "corp", "group"}
    q_tokens -= stopwords
    c_tokens -= stopwords
    if not q_tokens:
        return False
    overlap = len(q_tokens & c_tokens) / len(q_tokens)
    return overlap >= threshold


async def fetch_sbti_data(company_name: str) -> SbtiResult:
    """
    Look up a company in the SBTi public target database.

    Returns commitment and target details with full provenance.
    """
    result = SbtiResult(company_name=company_name)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            # Try the SBTi API
            params = {"name": company_name, "limit": 10}
            resp   = await client.get(_SBTI_API, params=params, headers=_HEADERS)

            if resp.status_code != 200:
                # API unavailable — record gap
                result.data_gaps.append(DataGap(
                    field_name="sbti_targets",
                    reason=f"SBTi API returned {resp.status_code}",
                    impact="Net zero target alignment cannot be independently verified",
                    suggested_action="Check https://sciencebasedtargets.org/companies-taking-action manually",
                ))
                return result

            data    = resp.json()
            targets = data if isinstance(data, list) else data.get("data", data.get("results", []))

            # Find best name match
            match = None
            for t in targets:
                name = t.get("companyName", t.get("name", ""))
                if _fuzzy_name_match(company_name, name):
                    match = t
                    break

            if not match:
                result.data_gaps.append(DataGap(
                    field_name="sbti_targets",
                    reason=f"'{company_name}' not found in SBTi database",
                    impact="Company has no validated near-term or net zero science-based target",
                    suggested_action=(
                        "Consider committing to SBTi target — required by ISSB S2 and many bank covenants"
                    ),
                ))
                return result

            # Parse fields
            result.sbti_status         = match.get("status", match.get("companyStatus"))
            result.near_term_status    = match.get("nearTermStatus")
            result.long_term_status    = match.get("longTermStatus", match.get("netZeroStatus"))
            temp                       = match.get("temperatureAmbition", "")
            result.temperature_ambition = _TEMPERATURE_MAP.get(temp, temp)
            result.target_year         = match.get("targetYear")
            result.net_zero_year       = match.get("netZeroTargetYear", match.get("longTermTargetYear"))

            result.sbti_provenance = ProvenanceField(
                value=result.sbti_status,
                source="Science Based Targets initiative (SBTi) public database",
                url=f"{_SBTI_BASE}#anchor_targets",
                retrieval_date=today_iso(),
                confidence_tier=ConfidenceTier.REPORTED,
                notes=(
                    f"Temperature: {result.temperature_ambition}; "
                    f"Near-term: {result.near_term_status}; "
                    f"Net Zero: {result.long_term_status}"
                ),
            )

    except httpx.TimeoutException:
        result.data_gaps.append(DataGap(
            field_name="sbti_targets",
            reason="SBTi API timeout",
            impact="Cannot verify SBTi commitment",
            suggested_action="Re-run or check https://sciencebasedtargets.org manually",
        ))
    except Exception as exc:
        logger.warning("SBTi fetch failed for '%s': %s", company_name, exc)
        result.data_gaps.append(DataGap(
            field_name="sbti_targets",
            reason=str(exc),
            impact="SBTi data unavailable",
            suggested_action="Verify SBTi status manually at sciencebasedtargets.org",
        ))

    return result
