"""
eutl_client.py — EU ETS (European Union Transaction Log) verified emissions client.

The EUTL is the official EU ETS registry operated by the European Commission.
It contains verified Scope 1 emissions (in tCO2e) for every ETS installation
in Europe, covering ~10,000 installations across energy-intensive sectors.

This data is:
  - Annually verified by accredited verifiers (highest confidence — VERIFIED tier)
  - Publicly accessible via the EU EUTL open data portal
  - More granular and reliable than CDP self-reported or EDGAR XBRL estimates
    for EU-regulated emitters

Coverage
--------
ETS-covered sectors (EU ETS Phase 4, 2021-2030):
  - Electricity and heat generation (all combustion ≥ 20 MW thermal)
  - Energy-intensive industries: cement, glass, ceramics, lime, paper,
    aluminium, petrochemicals, steel (BF-BOF), fertilisers, aviation (intra-EEA)

Non-covered by ETS:
  - Road transport (until EU ETS2 from 2027)
  - Buildings (until EU ETS2 from 2027)
  - Agriculture
  - Non-EEA aviation

API
---
EU EUTL Open Data API:
  https://eutl.eea.europa.eu/api/v1/

Company search endpoint:
  GET https://eutl.eea.europa.eu/api/v1/accounts?companyName={name}&pageSize=20

Installation emissions endpoint:
  GET https://eutl.eea.europa.eu/api/v1/emissions/{accountID}

Note: The EEA EUTL API is public and does not require authentication.
Rate limiting: be respectful; max 5 req/s.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

from .provenance import ConfidenceTier, DataGap, ProvenanceField, today_iso

logger = logging.getLogger(__name__)

_EUTL_BASE        = "https://eutl.eea.europa.eu/api/v1"
_SEARCH_ENDPOINT  = f"{_EUTL_BASE}/accounts"
_EMIT_ENDPOINT    = f"{_EUTL_BASE}/emissions"
_TIMEOUT_S        = 12.0
_USER_AGENT       = "ClimRisk-Engine/1.0 (climate-risk-research; contact=climrisk@example.com)"

# Most recent complete reporting year (ETS submissions lag by ~14 months)
_LATEST_REPORTING_YEAR = 2024


@dataclass
class EutlResult:
    """Result from EU ETS EUTL lookup for a company."""
    company_name:        str
    eutl_account_ids:    list[str]              = field(default_factory=list)
    installation_count:  int                    = 0
    scope1_mt_co2e:      Optional[ProvenanceField] = None   # Total verified across installations
    scope1_by_year:      dict[int, float]       = field(default_factory=dict)  # year → Mt CO2e
    latest_year:         Optional[int]          = None
    ets_covered:         bool                   = False
    data_gaps:           list[DataGap]          = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "company_name":       self.company_name,
            "eutl_account_ids":   self.eutl_account_ids,
            "installation_count": self.installation_count,
            "scope1_mt_co2e":     self.scope1_mt_co2e.to_dict() if self.scope1_mt_co2e else None,
            "scope1_by_year":     {str(y): round(v, 4) for y, v in self.scope1_by_year.items()},
            "latest_year":        self.latest_year,
            "ets_covered":        self.ets_covered,
        }


async def fetch_eutl_emissions(company_name: str) -> EutlResult:
    """
    Search the EU EUTL for a company and retrieve verified annual Scope 1 emissions.

    Returns
    -------
    EutlResult
        scope1_mt_co2e carries VERIFIED confidence tier if data found.
        EutlResult.scope1_by_year contains the full time-series.
    """
    result = EutlResult(company_name=company_name)

    if not _HTTPX_AVAILABLE:
        result.data_gaps.append(DataGap(
            field_name="eutl_emissions",
            reason="httpx not installed — install with: pip install httpx",
            impact="EU ETS verified emissions unavailable",
            suggested_action="pip install 'cri[api]'",
        ))
        return result

    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S, headers=headers) as client:
            # Step 1: Search for accounts matching company name
            search_params = {
                "companyName": company_name,
                "pageSize": 20,
                "registryCode": "EU",   # restrict to EU registry
            }
            resp = await client.get(_SEARCH_ENDPOINT, params=search_params)

            if resp.status_code != 200:
                logger.warning("EUTL search returned %d for '%s'", resp.status_code, company_name)
                result.data_gaps.append(DataGap(
                    field_name="eutl_emissions",
                    reason=f"EUTL API returned HTTP {resp.status_code}",
                    impact="Verified EU ETS emissions not available",
                    suggested_action="Retry or check EUTL at https://eutl.eea.europa.eu",
                ))
                return result

            data = resp.json()
            accounts = data if isinstance(data, list) else data.get("accounts", data.get("content", []))

            if not accounts:
                logger.info("EUTL: no accounts found for '%s'", company_name)
                result.data_gaps.append(DataGap(
                    field_name="eutl_emissions",
                    reason=f"No EU ETS accounts found for '{company_name}'",
                    impact="Company may not be an EU ETS-regulated operator, "
                           "or name mismatch in registry",
                    suggested_action=(
                        "Search manually at https://eutl.eea.europa.eu/eutl/ "
                        "and provide the EUTL account ID"
                    ),
                ))
                return result

            # Step 2: Collect account IDs and fetch emission series
            account_ids = []
            for acc in accounts[:10]:   # limit to top 10 matches
                aid = acc.get("accountIdentifier") or acc.get("id") or acc.get("accountID")
                if aid:
                    account_ids.append(str(aid))

            if not account_ids:
                result.data_gaps.append(DataGap(
                    field_name="eutl_emissions",
                    reason="EUTL accounts found but no account identifiers parseable",
                    impact="Cannot retrieve emission series",
                    suggested_action="Check EUTL API response format",
                ))
                return result

            result.eutl_account_ids = account_ids
            result.installation_count = len(account_ids)
            result.ets_covered = True

            # Step 3: Fetch emission series for each installation, aggregate
            year_totals: dict[int, float] = {}

            async def _fetch_one(aid: str) -> None:
                try:
                    r = await client.get(f"{_EMIT_ENDPOINT}/{aid}")
                    if r.status_code != 200:
                        return
                    emissions_data = r.json()
                    if isinstance(emissions_data, list):
                        rows = emissions_data
                    else:
                        rows = emissions_data.get("emissions", [])

                    for row in rows:
                        year_raw = row.get("year") or row.get("reportingYear")
                        co2_raw  = row.get("verifiedEmissions") or row.get("totalVerifiedEmissions") or row.get("co2Emissions")
                        if year_raw is None or co2_raw is None:
                            continue
                        try:
                            year_int = int(year_raw)
                            co2_t    = float(co2_raw)    # in tCO2e
                            year_totals[year_int] = year_totals.get(year_int, 0.0) + co2_t
                        except (ValueError, TypeError):
                            continue
                except Exception as e:
                    logger.debug("EUTL emission fetch failed for account %s: %s", aid, e)

            # Fetch all installations in parallel (with concurrency limit)
            sem = asyncio.Semaphore(3)
            async def _guarded(aid):
                async with sem:
                    await _fetch_one(aid)

            await asyncio.gather(*[_guarded(aid) for aid in account_ids])

            if not year_totals:
                result.data_gaps.append(DataGap(
                    field_name="eutl_scope1",
                    reason="EUTL installation accounts found but no emission records retrieved",
                    impact="Verified Scope 1 not available",
                    suggested_action="Check EUTL API or provide from EU ETS registry directly",
                ))
                return result

            # Convert tCO2e → Mt CO2e; store by year
            result.scope1_by_year = {y: v / 1_000_000 for y, v in year_totals.items()}
            latest_year = max(result.scope1_by_year.keys())
            result.latest_year = latest_year
            latest_mt = result.scope1_by_year[latest_year]

            result.scope1_mt_co2e = ProvenanceField(
                value=round(latest_mt, 4),
                source="EU ETS EUTL verified emissions registry",
                url=f"https://eutl.eea.europa.eu/api/v1/emissions/{account_ids[0]}",
                retrieval_date=today_iso(),
                confidence_tier=ConfidenceTier.VERIFIED,
                notes=(
                    f"Verified Scope 1 emissions from EU ETS registry "
                    f"({result.installation_count} installations, reporting year {latest_year}). "
                    "Annually audited by accredited verifiers per EU MRV Regulation 2018/2066."
                ),
            )

            logger.info(
                "EUTL: found %d installations for '%s', total Scope 1 (latest=%d): %.4f Mt CO2e",
                result.installation_count, company_name, latest_year, latest_mt,
            )

    except asyncio.TimeoutError:
        result.data_gaps.append(DataGap(
            field_name="eutl_emissions",
            reason="EUTL API timeout after 12s",
            impact="Verified EU ETS emissions unavailable",
            suggested_action="Retry; check EUTL availability at https://eutl.eea.europa.eu",
        ))
    except Exception as exc:
        logger.warning("EUTL fetch failed for '%s': %s", company_name, exc)
        result.data_gaps.append(DataGap(
            field_name="eutl_emissions",
            reason=str(exc),
            impact="Verified EU ETS emissions unavailable",
            suggested_action="Check logs and retry",
        ))

    return result
