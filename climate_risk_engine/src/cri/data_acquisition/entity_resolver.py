"""
entity_resolver.py — Company name → canonical identity via GLEIF API.

The Global LEI Foundation (GLEIF) publishes the Legal Entity Identifier (LEI)
registry which covers ~2.5M legal entities worldwide.  For every counterparty
the engine assesses, we resolve to a canonical LEI record which gives us:

  • Official registered legal name
  • Jurisdiction (country + subdivision)
  • Registered address (used for geocoding)
  • Entity status (ACTIVE / INACTIVE / MERGED)

API: https://api.gleif.org/api/v1/  (public, no auth required)
Docs: https://www.gleif.org/en/lei-data/gleif-lei-search-api

Fallback chain:
  1. GLEIF exact name match
  2. GLEIF fuzzy search (top candidate by relevance)
  3. Return None → caller records a DataGap
"""
from __future__ import annotations

import logging
import urllib.parse
from typing import Optional

import httpx

from .provenance import ConfidenceTier, DataGap, ProvenanceField, today_iso

logger = logging.getLogger(__name__)

_GLEIF_BASE = "https://api.gleif.org/api/v1"
_TIMEOUT    = 15.0   # seconds


class EntityResolverResult:
    """
    Result of a GLEIF entity resolution attempt.

    Attributes
    ----------
    resolved        : True if a matching LEI record was found
    lei             : Legal Entity Identifier (20-char alphanumeric)
    legal_name      : Official registered legal name
    jurisdiction    : ISO 3166-1 alpha-2 country code (+ optional subdivision)
    registered_address : Street address from LEI record
    entity_status   : "ACTIVE", "INACTIVE", "ANNULLED", etc.
    provenance      : ProvenanceField wrapping the LEI value
    data_gap        : DataGap if resolution failed
    """

    def __init__(
        self,
        resolved: bool,
        lei: Optional[str]   = None,
        legal_name: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        registered_address: Optional[str] = None,
        entity_status: Optional[str] = None,
        source_url: Optional[str] = None,
        data_gap: Optional[DataGap] = None,
        confidence: ConfidenceTier = ConfidenceTier.VERIFIED,
    ) -> None:
        self.resolved           = resolved
        self.lei                = lei
        self.legal_name         = legal_name
        self.jurisdiction       = jurisdiction
        self.registered_address = registered_address
        self.entity_status      = entity_status
        self.data_gap           = data_gap

        self.provenance = ProvenanceField(
            value=lei,
            source="GLEIF LEI Registry",
            url=source_url or (
                f"https://search.gleif.org/#/record/{lei}" if lei else None
            ),
            retrieval_date=today_iso(),
            confidence_tier=confidence,
            notes=f"legal_name={legal_name}; jurisdiction={jurisdiction}",
        )

    def to_dict(self) -> dict:
        return {
            "resolved":           self.resolved,
            "lei":                self.lei,
            "legal_name":         self.legal_name,
            "jurisdiction":       self.jurisdiction,
            "registered_address": self.registered_address,
            "entity_status":      self.entity_status,
            "provenance":         self.provenance.to_dict(),
            "data_gap":           (
                {"field": self.data_gap.field_name,
                 "reason": self.data_gap.reason,
                 "action": self.data_gap.suggested_action}
                if self.data_gap else None
            ),
        }


def _parse_gleif_record(record: dict) -> dict:
    """Extract useful fields from a GLEIF v1 LEI record."""
    attrs   = record.get("attributes", {})
    entity  = attrs.get("entity", {})
    reg_adr = entity.get("legalAddress", {})

    lines   = reg_adr.get("addressLines", [])
    city    = reg_adr.get("city", "")
    country = reg_adr.get("country", "")
    post    = reg_adr.get("postalCode", "")
    address = ", ".join(filter(None, lines + [city, post, country]))

    jurisdiction = entity.get("jurisdiction", country)

    return {
        "lei":          attrs.get("lei"),
        "legal_name":   entity.get("legalName", {}).get("name", ""),
        "jurisdiction": jurisdiction,
        "address":      address,
        "status":       entity.get("status", "ACTIVE"),
    }


async def resolve_entity(
    company_name: str,
    *,
    jurisdiction_hint: Optional[str] = None,
    timeout: float = _TIMEOUT,
) -> EntityResolverResult:
    """
    Resolve a company name to a GLEIF LEI record.

    Parameters
    ----------
    company_name       : Company name as known to the practitioner
    jurisdiction_hint  : ISO-2 country code to narrow search (e.g. "GB", "US")
    timeout            : HTTP timeout in seconds

    Returns
    -------
    EntityResolverResult
    """
    query = urllib.parse.quote(company_name)
    url   = f"{_GLEIF_BASE}/lei-records?filter[entity.names]={query}&page[size]=5"
    if jurisdiction_hint:
        url += f"&filter[entity.jurisdiction]={jurisdiction_hint}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                url,
                headers={"Accept": "application/vnd.api+json"},
            )
            resp.raise_for_status()
            data = resp.json()

        records = data.get("data", [])
        if not records:
            # Try broader fuzzy search
            url2 = f"{_GLEIF_BASE}/fuzzycompletions?field=fulltext&q={query}&page[size]=3"
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp2 = await client.get(url2, headers={"Accept": "application/json"})
                resp2.raise_for_status()
                suggestions = resp2.json()

            if not suggestions:
                return EntityResolverResult(
                    resolved=False,
                    data_gap=DataGap(
                        field_name="lei",
                        reason=f"GLEIF returned 0 records for '{company_name}'",
                        impact="Entity resolution failed — address-based fallback only",
                        suggested_action="Provide LEI code or registered company number",
                    ),
                )

            # Pull the top fuzzy suggestion
            top     = suggestions[0]
            lei_val = top.get("lei")
            if not lei_val:
                return EntityResolverResult(
                    resolved=False,
                    data_gap=DataGap(
                        field_name="lei",
                        reason="Fuzzy match returned no LEI",
                        impact="Entity resolution failed",
                        suggested_action="Provide LEI code or registered company number",
                    ),
                )

            # Fetch the full record by LEI
            lei_url = f"{_GLEIF_BASE}/lei-records/{lei_val}"
            async with httpx.AsyncClient(timeout=timeout) as client:
                r3 = await client.get(lei_url, headers={"Accept": "application/vnd.api+json"})
                r3.raise_for_status()
                full = r3.json().get("data", {})

            parsed = _parse_gleif_record(full)
            return EntityResolverResult(
                resolved=True,
                lei=parsed["lei"],
                legal_name=parsed["legal_name"],
                jurisdiction=parsed["jurisdiction"],
                registered_address=parsed["address"],
                entity_status=parsed["status"],
                source_url=f"https://search.gleif.org/#/record/{parsed['lei']}",
                confidence=ConfidenceTier.VERIFIED,
            )

        # Primary exact match
        parsed = _parse_gleif_record(records[0])
        return EntityResolverResult(
            resolved=True,
            lei=parsed["lei"],
            legal_name=parsed["legal_name"],
            jurisdiction=parsed["jurisdiction"],
            registered_address=parsed["address"],
            entity_status=parsed["status"],
            source_url=f"https://search.gleif.org/#/record/{parsed['lei']}",
            confidence=ConfidenceTier.VERIFIED,
        )

    except httpx.TimeoutException:
        logger.warning("GLEIF API timed out for '%s'", company_name)
        return EntityResolverResult(
            resolved=False,
            data_gap=DataGap(
                field_name="lei",
                reason="GLEIF API timeout",
                impact="Entity resolution failed — using name as-is",
                suggested_action="Re-run or provide LEI code directly",
            ),
        )
    except Exception as exc:
        logger.warning("GLEIF resolution failed for '%s': %s", company_name, exc)
        return EntityResolverResult(
            resolved=False,
            data_gap=DataGap(
                field_name="lei",
                reason=str(exc),
                impact="Entity resolution failed",
                suggested_action="Provide LEI code or verify company name spelling",
            ),
        )


def resolve_entity_sync(
    company_name: str,
    *,
    jurisdiction_hint: Optional[str] = None,
) -> EntityResolverResult:
    """Synchronous wrapper for use outside an async context."""
    import asyncio
    return asyncio.run(
        resolve_entity(company_name, jurisdiction_hint=jurisdiction_hint)
    )
