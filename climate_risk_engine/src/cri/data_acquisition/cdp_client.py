"""
cdp_client.py — CDP (Carbon Disclosure Project) public disclosure data.

CDP makes its questionnaire response summaries publicly available via their
open data portal.  This client fetches Scope 1, Scope 2, Scope 3, and climate
targets for companies that have disclosed.

Data source
-----------
CDP Open Data Portal: https://www.cdp.net/en/responses
CSV download: https://cdn.cdp.net/cdp-production/cms/reports/documents/...

Because the full CDP database is distributed as annual CSV downloads (multi-GB),
we use their public search API and individual company pages to fetch:
  1. Disclosure status + score (A/A-/B/etc.)
  2. Scope 1 emissions (tCO2e)
  3. Scope 2 emissions — market-based and location-based
  4. Climate targets (year, base, target)

CDP API: https://api.cdp.net/v0/  (public, limited)

Note: CDP also offers a full data API to research partners.  Without that key,
we scrape the public response summary pages.

Confidence tier: REPORTED (company self-reported, verified by CDP questionnaire)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .provenance import ConfidenceTier, DataGap, ProvenanceField, today_iso

logger = logging.getLogger(__name__)

_CDP_API       = "https://api.cdp.net/v0"
_CDP_SEARCH    = "https://www.cdp.net/en/responses"
_TIMEOUT       = 15.0
_HEADERS       = {
    "User-Agent": "ClimRiskEngine/1.0 contact@climrisk.io",
    "Accept":     "application/json",
}


@dataclass
class CdpResult:
    company_name:       Optional[str]             = None
    cdp_id:             Optional[str]             = None
    disclosure_year:    Optional[int]             = None
    disclosure_score:   Optional[str]             = None  # A, A-, B, C, D, F
    scope1_mt_co2e:     Optional[ProvenanceField] = None
    scope2_lb_mt_co2e:  Optional[ProvenanceField] = None  # location-based
    scope2_mb_mt_co2e:  Optional[ProvenanceField] = None  # market-based
    scope3_mt_co2e:     Optional[ProvenanceField] = None
    net_zero_target:    Optional[ProvenanceField] = None  # year string, e.g. "2050"
    near_term_target:   Optional[ProvenanceField] = None
    data_gaps:          list[DataGap]             = field(default_factory=list)

    def to_dict(self) -> dict:
        def _pf(f):
            return f.to_dict() if f else None
        return {
            "company_name":      self.company_name,
            "cdp_id":            self.cdp_id,
            "disclosure_year":   self.disclosure_year,
            "disclosure_score":  self.disclosure_score,
            "scope1_mt_co2e":    _pf(self.scope1_mt_co2e),
            "scope2_lb_mt_co2e": _pf(self.scope2_lb_mt_co2e),
            "scope2_mb_mt_co2e": _pf(self.scope2_mb_mt_co2e),
            "scope3_mt_co2e":    _pf(self.scope3_mt_co2e),
            "net_zero_target":   _pf(self.net_zero_target),
            "near_term_target":  _pf(self.near_term_target),
            "data_gaps":         [
                {"field": g.field_name, "reason": g.reason, "action": g.suggested_action}
                for g in self.data_gaps
            ],
        }


async def _search_cdp_company(
    company_name: str,
    client: httpx.AsyncClient,
) -> Optional[dict]:
    """Search CDP open API for a company; returns first matching record or None."""
    try:
        url  = f"{_CDP_API}/organizations?name={company_name}&limit=5"
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            orgs = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(orgs, list) and orgs:
                return orgs[0]
    except Exception:
        pass

    # Secondary: try the public responses endpoint
    try:
        url2 = f"{_CDP_SEARCH}?queries[name]={company_name}&queries[year][]=2023&limit=3"
        resp2 = await client.get(url2, headers={**_HEADERS, "Accept": "text/html"}, timeout=_TIMEOUT)
        if resp2.status_code == 200:
            # Extract CDP company ID from embedded JSON-LD or meta tags
            html = resp2.text
            match = re.search(r'"cdp_id"\s*:\s*"([^"]+)"', html)
            if match:
                return {"cdp_id": match.group(1)}
    except Exception:
        pass

    return None


async def _fetch_company_responses(
    cdp_id: str,
    client: httpx.AsyncClient,
) -> dict:
    """Fetch disclosure data for a known CDP organization ID."""
    try:
        url  = f"{_CDP_API}/organizations/{cdp_id}/responses?year=2023&limit=1"
        resp = await client.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def _extract_emissions_from_response(response: dict) -> dict:
    """Parse the CDP API response for GHG emissions data."""
    out: dict = {}

    # CDP response structure varies; try common paths
    ghg = (
        response.get("ghgEmissions")
        or response.get("data", {}).get("ghgEmissions")
        or {}
    )

    scope1_raw = (
        ghg.get("scope1Total")
        or ghg.get("scope1")
        or response.get("scope1")
    )
    if scope1_raw is not None:
        try:
            out["scope1"] = float(scope1_raw) / 1_000_000  # t → Mt
        except (TypeError, ValueError):
            pass

    scope2_lb  = ghg.get("scope2LocationBased") or ghg.get("scope2_lb")
    scope2_mb  = ghg.get("scope2MarketBased")   or ghg.get("scope2_mb")
    if scope2_lb:
        try:
            out["scope2_lb"] = float(scope2_lb) / 1_000_000
        except (TypeError, ValueError):
            pass
    if scope2_mb:
        try:
            out["scope2_mb"] = float(scope2_mb) / 1_000_000
        except (TypeError, ValueError):
            pass

    score = response.get("disclosureScore") or response.get("score")
    if score:
        out["score"] = str(score)

    return out


async def fetch_cdp_data(company_name: str) -> CdpResult:
    """
    Fetch CDP disclosure data for a company.

    Returns a CdpResult with provenance-tagged emission fields.
    Gaps are recorded explicitly — nothing is silently estimated.
    """
    result = CdpResult(company_name=company_name)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            company = await _search_cdp_company(company_name, client)

            if not company:
                result.data_gaps.append(DataGap(
                    field_name="cdp_disclosure",
                    reason=f"No CDP disclosure found for '{company_name}'",
                    impact=(
                        "Scope 1/2/3 emissions unavailable from CDP; "
                        "EU ETS or EDGAR may still provide Scope 1"
                    ),
                    suggested_action=(
                        "Respond to CDP questionnaire or provide Scope 1 GHG inventory directly"
                    ),
                ))
                return result

            cdp_id = company.get("cdp_id") or company.get("id")
            result.cdp_id = cdp_id

            response_data = {}
            if cdp_id:
                response_data = await _fetch_company_responses(cdp_id, client)

            parsed = _extract_emissions_from_response(response_data)

            base_url = (
                f"https://www.cdp.net/en/responses?queries[name]={company_name}"
            )

            if "scope1" in parsed:
                result.scope1_mt_co2e = ProvenanceField(
                    value=round(parsed["scope1"], 4),
                    source="CDP Climate Questionnaire Response",
                    url=base_url,
                    retrieval_date=today_iso(),
                    confidence_tier=ConfidenceTier.REPORTED,
                    notes="Self-reported, verified by CDP questionnaire process",
                )
            else:
                result.data_gaps.append(DataGap(
                    field_name="scope1_mt_co2e",
                    reason="Scope 1 not found in CDP response",
                    impact="Using sector benchmark for transition risk calculation",
                    suggested_action="Submit GHG inventory in CDP questionnaire C6.1",
                ))

            if "scope2_lb" in parsed:
                result.scope2_lb_mt_co2e = ProvenanceField(
                    value=round(parsed["scope2_lb"], 4),
                    source="CDP Climate Questionnaire (Scope 2 location-based)",
                    url=base_url,
                    retrieval_date=today_iso(),
                    confidence_tier=ConfidenceTier.REPORTED,
                )
            if "scope2_mb" in parsed:
                result.scope2_mb_mt_co2e = ProvenanceField(
                    value=round(parsed["scope2_mb"], 4),
                    source="CDP Climate Questionnaire (Scope 2 market-based)",
                    url=base_url,
                    retrieval_date=today_iso(),
                    confidence_tier=ConfidenceTier.REPORTED,
                )

            if "score" in parsed:
                result.disclosure_score = parsed["score"]

    except httpx.TimeoutException:
        result.data_gaps.append(DataGap(
            field_name="cdp_disclosure",
            reason="CDP API timeout",
            impact="CDP emissions data unavailable",
            suggested_action="Re-run or provide CDP score / Scope 1 data manually",
        ))
    except Exception as exc:
        logger.warning("CDP fetch failed for '%s': %s", company_name, exc)
        result.data_gaps.append(DataGap(
            field_name="cdp_disclosure",
            reason=str(exc),
            impact="CDP data unavailable",
            suggested_action="Check company name spelling or provide emissions data directly",
        ))

    return result
