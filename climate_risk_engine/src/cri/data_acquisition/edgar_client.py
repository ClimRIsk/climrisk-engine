"""
edgar_client.py — SEC EDGAR data acquisition for US-listed companies.

Fetches annual 10-K filings to extract:
  • Revenue (most recent fiscal year)
  • Total assets / enterprise value proxy
  • GHG emissions disclosures (Scope 1/2 where reported)
  • Climate risk disclosures (post-SEC climate rule 2024)

APIs used
---------
1. EDGAR Company Search:  https://efts.sec.gov/LATEST/search-index?q=...
2. EDGAR Submissions API: https://data.sec.gov/submissions/CIK{cik}.json
3. EDGAR XBRL API:        https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json

All APIs are public, no API key required.
Rate limit: 10 requests/second (we stay well below with async).

Confidence tiers
----------------
Revenue / assets from XBRL structured data → VERIFIED
GHG from XBRL (few companies tag it)       → VERIFIED
GHG extracted from text disclosure          → REPORTED
No filing found                             → MISSING (DataGap recorded)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .provenance import ConfidenceTier, DataGap, ProvenanceField, today_iso

logger = logging.getLogger(__name__)

_EDGAR_BASE     = "https://data.sec.gov"
_EDGAR_SEARCH   = "https://efts.sec.gov/LATEST/search-index"
_USER_AGENT     = "ClimRiskEngine/1.0 contact@climrisk.io"
_TIMEOUT        = 20.0


@dataclass
class EdgarResult:
    """Structured output from an EDGAR 10-K pull."""
    cik:                   Optional[str]             = None
    company_name:          Optional[str]             = None
    ticker:                Optional[str]             = None
    fiscal_year:           Optional[int]             = None
    revenue_usd_m:         Optional[ProvenanceField] = None
    total_assets_usd_m:    Optional[ProvenanceField] = None
    scope1_mt_co2e:        Optional[ProvenanceField] = None
    scope2_mt_co2e:        Optional[ProvenanceField] = None
    filing_url:            Optional[str]             = None
    data_gaps:             list[DataGap]             = field(default_factory=list)

    def to_dict(self) -> dict:
        def _pf(f):
            return f.to_dict() if f else None
        return {
            "cik":                self.cik,
            "company_name":       self.company_name,
            "ticker":             self.ticker,
            "fiscal_year":        self.fiscal_year,
            "revenue_usd_m":      _pf(self.revenue_usd_m),
            "total_assets_usd_m": _pf(self.total_assets_usd_m),
            "scope1_mt_co2e":     _pf(self.scope1_mt_co2e),
            "scope2_mt_co2e":     _pf(self.scope2_mt_co2e),
            "filing_url":         self.filing_url,
            "data_gaps":          [
                {"field": g.field_name, "reason": g.reason, "action": g.suggested_action}
                for g in self.data_gaps
            ],
        }


def _headers() -> dict:
    return {
        "User-Agent":  _USER_AGENT,
        "Accept":      "application/json",
        "Accept-Encoding": "gzip",
    }


def _usd_to_millions(value: float, unit: str) -> float:
    """Convert XBRL unit to USD millions."""
    unit = (unit or "").upper()
    if unit in ("USD", "US-DOLLAR"):
        return value / 1_000_000
    if unit in ("THOUSANDS", "USD-THOUSANDS"):
        return value / 1_000
    if unit in ("MILLIONS", "USD-MILLIONS"):
        return value
    if unit in ("BILLIONS", "USD-BILLIONS"):
        return value * 1_000
    # Default: assume raw USD
    return value / 1_000_000


async def _search_cik(company_name: str, client: httpx.AsyncClient) -> Optional[str]:
    """Search EDGAR for a company by name; return CIK string (zero-padded to 10 digits)."""
    url  = f"https://efts.sec.gov/LATEST/search-index?q=%22{company_name}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
    resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    if hits:
        cik = hits[0].get("_source", {}).get("entity_id", "")
        return cik.zfill(10) if cik else None

    # Fallback: company search endpoint
    url2  = f"https://efts.sec.gov/LATEST/search-index?q=%22{company_name}%22&forms=10-K"
    resp2 = await client.get(url2, headers=_headers())
    resp2.raise_for_status()
    hits2 = resp2.json().get("hits", {}).get("hits", [])
    if hits2:
        cik = hits2[0].get("_source", {}).get("entity_id", "")
        return cik.zfill(10) if cik else None

    return None


async def _get_xbrl_facts(cik: str, client: httpx.AsyncClient) -> dict:
    """Fetch the full XBRL company facts blob."""
    url  = f"{_EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    resp = await client.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


def _extract_revenue(facts: dict, cik: str, company_name: str) -> Optional[ProvenanceField]:
    """Pull the most recent annual revenue from XBRL us-gaap facts."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    # Candidates in order of preference
    candidates = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ]
    for tag in candidates:
        if tag not in us_gaap:
            continue
        units_data = us_gaap[tag].get("units", {})
        usd_entries = units_data.get("USD", [])
        # Filter for annual (10-K) filings
        annual = [e for e in usd_entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
        if not annual:
            continue
        # Most recent by filed date
        latest = max(annual, key=lambda e: e.get("filed", ""))
        value_m = _usd_to_millions(latest["val"], "USD")
        fy_year = int(latest.get("end", "2024")[:4])
        filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=1"
        return ProvenanceField(
            value=round(value_m, 2),
            source=f"SEC EDGAR 10-K (XBRL tag: {tag})",
            url=filing_url,
            retrieval_date=today_iso(),
            confidence_tier=ConfidenceTier.VERIFIED,
            notes=f"Fiscal year {fy_year}; raw USD value: {latest['val']:,.0f}",
        )
    return None


def _extract_total_assets(facts: dict, cik: str) -> Optional[ProvenanceField]:
    """Pull total assets from XBRL."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    tag      = "Assets"
    if tag not in us_gaap:
        return None
    entries = us_gaap[tag].get("units", {}).get("USD", [])
    annual  = [e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
    if not annual:
        return None
    latest  = max(annual, key=lambda e: e.get("filed", ""))
    value_m = _usd_to_millions(latest["val"], "USD")
    fy_year = int(latest.get("end", "2024")[:4])
    return ProvenanceField(
        value=round(value_m, 2),
        source="SEC EDGAR 10-K (XBRL tag: Assets)",
        url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K",
        retrieval_date=today_iso(),
        confidence_tier=ConfidenceTier.VERIFIED,
        notes=f"Fiscal year {fy_year}",
    )


def _extract_ghg(facts: dict, cik: str) -> tuple[Optional[ProvenanceField], Optional[ProvenanceField]]:
    """
    Attempt to pull Scope 1 / Scope 2 GHG from XBRL.

    Most companies do NOT tag GHG in XBRL yet.  Returns (None, None) if not found;
    caller records a DataGap.  The SEC 2024 climate disclosure rule will improve coverage.
    """
    # Try ecd (emerging SEC climate tags) and custom extensions
    for namespace in ["ecd", "us-gaap"]:
        ns_facts = facts.get("facts", {}).get(namespace, {})
        for tag, scope in [
            ("GHGEmissionsScope1", "scope1"),
            ("GreenhouseGasEmissionsScope1", "scope1"),
            ("GHGEmissionsScope2", "scope2"),
            ("GreenhouseGasEmissionsScope2", "scope2"),
        ]:
            if tag in ns_facts:
                # Find metric-ton unit entries
                for unit_key, entries in ns_facts[tag].get("units", {}).items():
                    if "ton" in unit_key.lower() or "T" in unit_key:
                        annual = [e for e in entries if e.get("form") in ("10-K", "20-F")]
                        if annual:
                            latest = max(annual, key=lambda e: e.get("filed", ""))
                            pf = ProvenanceField(
                                value=round(latest["val"] / 1_000_000, 4),  # → Mt CO2e
                                source=f"SEC EDGAR XBRL ({namespace}:{tag})",
                                url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                                retrieval_date=today_iso(),
                                confidence_tier=ConfidenceTier.VERIFIED,
                                notes=f"Unit: {unit_key}; converted to Mt CO2e",
                            )
                            if scope == "scope1":
                                return pf, None
                            else:
                                return None, pf
    return None, None


async def fetch_edgar_data(
    company_name: str,
    *,
    cik: Optional[str] = None,
    timeout: float = _TIMEOUT,
) -> EdgarResult:
    """
    Main entry point: fetch 10-K data for a US company.

    Parameters
    ----------
    company_name : Name to search in EDGAR
    cik          : If known, skip the name search and use directly
    """
    result = EdgarResult(company_name=company_name)

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            # Step 1: resolve CIK
            if not cik:
                cik = await _search_cik(company_name, client)

            if not cik:
                result.data_gaps.append(DataGap(
                    field_name="cik",
                    reason=f"No EDGAR company found for '{company_name}'",
                    impact="No SEC filing data available; UK Companies House or manual data required",
                    suggested_action="Confirm company is SEC-registered (US-listed), or provide CIK directly",
                ))
                return result

            result.cik = cik

            # Step 2: fetch XBRL facts
            try:
                facts = await _get_xbrl_facts(cik, client)
            except Exception as e:
                result.data_gaps.append(DataGap(
                    field_name="xbrl_facts",
                    reason=f"XBRL facts fetch failed: {e}",
                    impact="Financial data unavailable from EDGAR",
                    suggested_action="Check CIK is correct or company files structured data",
                ))
                return result

            # Extract company metadata
            entity = facts.get("entityName", company_name)
            result.company_name = entity

            # Step 3: Revenue
            rev = _extract_revenue(facts, cik, company_name)
            if rev:
                result.revenue_usd_m = rev
            else:
                result.data_gaps.append(DataGap(
                    field_name="revenue_usd_m",
                    reason="Revenue tag not found in XBRL us-gaap facts",
                    impact="Transition VaR % of revenue will use sector benchmark",
                    suggested_action="Provide most recent annual revenue (USD millions)",
                ))

            # Step 4: Total assets
            assets = _extract_total_assets(facts, cik)
            if assets:
                result.total_assets_usd_m = assets
            else:
                result.data_gaps.append(DataGap(
                    field_name="total_assets_usd_m",
                    reason="Assets tag not found in XBRL",
                    impact="Will use revenue × sector multiplier as proxy EV",
                    suggested_action="Provide total assets (USD millions)",
                ))

            # Step 5: GHG emissions
            s1, s2 = _extract_ghg(facts, cik)
            result.scope1_mt_co2e = s1
            result.scope2_mt_co2e = s2

            if not s1:
                result.data_gaps.append(DataGap(
                    field_name="scope1_mt_co2e",
                    reason="Scope 1 GHG not found in XBRL (most companies don't tag yet)",
                    impact="Transition risk uses CDP or sector emissions benchmark",
                    suggested_action=(
                        "Submit Scope 1 GHG inventory in tonnes CO2e; "
                        "or ensure CDP disclosure is filed"
                    ),
                ))

            result.filing_url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=5"
            )

    except httpx.TimeoutException:
        result.data_gaps.append(DataGap(
            field_name="edgar",
            reason="EDGAR API timeout",
            impact="All financial data unavailable from SEC",
            suggested_action="Re-run assessment; EDGAR may be temporarily slow",
        ))
    except Exception as exc:
        logger.exception("EDGAR fetch failed for '%s': %s", company_name, exc)
        result.data_gaps.append(DataGap(
            field_name="edgar",
            reason=str(exc),
            impact="SEC EDGAR data unavailable",
            suggested_action="Verify company name and SEC registration status",
        ))

    return result
