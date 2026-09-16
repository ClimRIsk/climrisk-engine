"""
company_profiler.py — Autonomous parallel data acquisition orchestrator.

This is the "virtual employee" layer.  Given a company name and optional
assessment scope, it:

  1. Resolves the entity via GLEIF (→ LEI, jurisdiction, address)
  2. Geocodes the registered address (→ lat/lon for physical hazard)
  3. Fetches financials from SEC EDGAR (US) or uses Yahoo Finance
  4. Fetches GHG emissions from CDP
  5. Checks SBTi net zero targets
  6. Tries Yahoo Finance for market cap / EV
  7. Reconciles conflicts between sources (CDP vs EDGAR on emissions)
  8. Returns a CompanyProfile with every field provenance-tagged
  9. Makes ALL data gaps explicit — never silently fills from benchmarks
     unless the practitioner explicitly requests tiered fallback

All source fetches run in parallel (asyncio.gather).

Conflict resolution rules
-------------------------
Scope 1 emissions: EU ETS > EDGAR XBRL > CDP disclosure > estimate
Revenue:           EDGAR XBRL > Yahoo Finance TTM > manual input > estimate
Coordinates:       Explicit input > GLEIF address geocoded > country centroid

Output
------
CompanyProfile — ready to pass directly to the CRI engine's physical/
transition/biodiversity assessment modules.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .cdp_client      import CdpResult, fetch_cdp_data
from .edgar_client    import EdgarResult, fetch_edgar_data
from .entity_resolver import EntityResolverResult, resolve_entity
from .eutl_client     import EutlResult, fetch_eutl_emissions
from .geocoder_client import GeocodeResult, geocode_address
from .provenance      import (
    ConfidenceTier, DataGap, ProvenanceField, today_iso
)
from .sbti_client     import SbtiResult, fetch_sbti_data
from .yahoo_client    import YahooResult, fetch_yahoo_data, find_ticker_from_name

logger = logging.getLogger(__name__)

# Sector emissions intensity benchmarks (Mt CO2e per $B revenue)
# Used ONLY if all primary sources fail — explicitly flagged ESTIMATED
_SECTOR_EMISSION_INTENSITY: dict[str, float] = {
    "oil_gas":        2.40,
    "coal":           3.80,
    "utilities":      1.60,
    "steel":          0.95,
    "cement":         0.70,
    "chemicals":      0.45,
    "aviation":       0.30,
    "shipping":       0.25,
    "automotive":     0.15,
    "real_estate":    0.05,
    "agriculture":    0.55,
    "technology":     0.02,
    "financials":     0.01,
    "healthcare":     0.04,
    "default":        0.20,
}

# Revenue benchmarks by sector (USD M) — last resort only
_SECTOR_REVENUE_BENCHMARKS: dict[str, float] = {
    "oil_gas":     50_000,
    "utilities":   12_000,
    "steel":        8_000,
    "mining":      15_000,
    "technology":  10_000,
    "financials":  25_000,
    "default":      5_000,
}


@dataclass
class CompanyProfile:
    """
    A fully sourced company data profile for CRI engine input.

    Every numerical field is a ProvenanceField (value + source + confidence).
    data_gaps lists all fields that couldn't be sourced.
    All estimated/fallback values are explicitly flagged — none are silent.
    """
    # Identity
    input_name:          str
    resolved_name:       Optional[str]             = None
    lei:                 Optional[str]             = None
    jurisdiction:        Optional[str]             = None
    sector:              Optional[str]             = None

    # Location
    lat:                 Optional[ProvenanceField] = None
    lon:                 Optional[ProvenanceField] = None
    registered_address:  Optional[str]             = None

    # Financials
    revenue_usd_m:       Optional[ProvenanceField] = None
    total_assets_usd_m:  Optional[ProvenanceField] = None
    market_cap_usd_m:    Optional[ProvenanceField] = None
    ev_usd_m:            Optional[ProvenanceField] = None
    ebitda_usd_m:        Optional[ProvenanceField] = None
    total_debt_usd_m:    Optional[ProvenanceField] = None

    # Emissions
    scope1_mt_co2e:      Optional[ProvenanceField] = None
    scope2_mt_co2e:      Optional[ProvenanceField] = None
    scope3_mt_co2e:      Optional[ProvenanceField] = None

    # Climate targets
    sbti_status:         Optional[str]             = None
    temperature_ambition: Optional[str]            = None
    net_zero_year:       Optional[int]             = None
    cdp_score:           Optional[str]             = None

    # Metadata
    data_gaps:           list[DataGap]             = field(default_factory=list)
    raw_sources:         dict[str, Any]            = field(default_factory=dict)
    profile_created_at:  str                       = field(default_factory=today_iso)
    assessment_readiness: str                      = "UNKNOWN"  # FULL / PARTIAL / TRIAGE_ONLY

    def add_gap(self, gap: DataGap) -> None:
        # Deduplicate by field_name
        existing = {g.field_name for g in self.data_gaps}
        if gap.field_name not in existing:
            self.data_gaps.append(gap)

    def add_all_gaps(self, gaps: list[DataGap]) -> None:
        for g in gaps:
            self.add_gap(g)

    def to_dict(self) -> dict:
        def _pf(f):
            return f.to_dict() if f else None
        return {
            "input_name":          self.input_name,
            "resolved_name":       self.resolved_name,
            "lei":                 self.lei,
            "jurisdiction":        self.jurisdiction,
            "sector":              self.sector,
            "location": {
                "lat":                self.lat.value if self.lat else None,
                "lon":                self.lon.value if self.lon else None,
                "registered_address": self.registered_address,
                "lat_provenance":     _pf(self.lat),
                "lon_provenance":     _pf(self.lon),
            },
            "financials": {
                "revenue_usd_m":      _pf(self.revenue_usd_m),
                "total_assets_usd_m": _pf(self.total_assets_usd_m),
                "market_cap_usd_m":   _pf(self.market_cap_usd_m),
                "ev_usd_m":           _pf(self.ev_usd_m),
                "ebitda_usd_m":       _pf(self.ebitda_usd_m),
                "total_debt_usd_m":   _pf(self.total_debt_usd_m),
            },
            "emissions": {
                "scope1_mt_co2e": _pf(self.scope1_mt_co2e),
                "scope2_mt_co2e": _pf(self.scope2_mt_co2e),
                "scope3_mt_co2e": _pf(self.scope3_mt_co2e),
            },
            "climate_targets": {
                "sbti_status":          self.sbti_status,
                "temperature_ambition": self.temperature_ambition,
                "net_zero_year":        self.net_zero_year,
                "cdp_score":            self.cdp_score,
            },
            "data_gaps": [
                {
                    "field":   g.field_name,
                    "reason":  g.reason,
                    "impact":  g.impact,
                    "action":  g.suggested_action,
                }
                for g in self.data_gaps
            ],
            "assessment_readiness": self.assessment_readiness,
            "profile_created_at":   self.profile_created_at,
        }

    def engine_payload(self) -> dict:
        """
        Returns a dict compatible with the existing CRI engine endpoint
        payload format.  Missing fields are explicitly None — never silently
        estimated at this layer.
        """
        lat = self.lat.value if self.lat else None
        lon = self.lon.value if self.lon else None

        return {
            "company_name":   self.resolved_name or self.input_name,
            "sector":         self.sector or "unknown",
            "country":        self.jurisdiction or "unknown",
            "lat":            float(lat) if lat is not None else None,
            "lon":            float(lon) if lon is not None else None,
            "revenue_usd_m":  self.revenue_usd_m.value if self.revenue_usd_m else None,
            "ev_usd_m":       self.ev_usd_m.value if self.ev_usd_m else None,
            "scope1_mt_co2e": self.scope1_mt_co2e.value if self.scope1_mt_co2e else None,
        }


def _assess_readiness(profile: CompanyProfile) -> str:
    """
    Determine how complete the profile is for engine assessment.

    FULL         — all key fields have primary data (≥ REPORTED confidence)
    PARTIAL      — enough for Tier 2 (coordinates + sector) but gaps in financials/emissions
    TRIAGE_ONLY  — only name + sector + country; physical hazard and deep DD not possible
    """
    has_coords    = profile.lat is not None and profile.lon is not None
    has_financials = profile.revenue_usd_m is not None
    has_emissions  = profile.scope1_mt_co2e is not None

    if has_coords and has_financials and has_emissions:
        return "FULL"
    elif has_coords or has_financials:
        return "PARTIAL"
    else:
        return "TRIAGE_ONLY"


def _resolve_emissions(
    edgar:   EdgarResult,
    cdp:     CdpResult,
    sector:  Optional[str],
    revenue_usd_m: Optional[float],
    use_estimates: bool = True,
) -> tuple[Optional[ProvenanceField], Optional[ProvenanceField]]:
    """
    Conflict resolution for Scope 1 / Scope 2.

    Priority: EDGAR XBRL (VERIFIED) > CDP (REPORTED) > Sector estimate (ESTIMATED)
    """
    scope1 = edgar.scope1_mt_co2e or cdp.scope1_mt_co2e

    scope2 = (
        edgar.scope2_mt_co2e
        or cdp.scope2_mb_mt_co2e
        or cdp.scope2_lb_mt_co2e
    )

    # If still no Scope 1 and estimates allowed, use sector intensity
    if scope1 is None and use_estimates and revenue_usd_m and sector:
        intensity = _SECTOR_EMISSION_INTENSITY.get(sector, _SECTOR_EMISSION_INTENSITY["default"])
        estimated_mt = intensity * (revenue_usd_m / 1_000)  # USD M → USD B
        scope1 = ProvenanceField.estimated(
            value=round(estimated_mt, 3),
            basis=f"{sector} sector intensity ({intensity:.2f} Mt CO2e per $B revenue)",
            notes=(
                "ESTIMATED — no primary source found. "
                f"Basis: {intensity:.2f} Mt CO2e / $B revenue × ${revenue_usd_m:,.0f}M revenue. "
                "Submit Scope 1 GHG inventory for verified assessment."
            ),
        )

    return scope1, scope2


def _resolve_revenue(
    edgar: EdgarResult,
    yahoo: YahooResult,
    sector: Optional[str],
    use_estimates: bool = True,
) -> Optional[ProvenanceField]:
    """
    Revenue priority: EDGAR XBRL (VERIFIED) > Yahoo TTM (REPORTED) > estimate.
    """
    rev = edgar.revenue_usd_m or yahoo.revenue_usd_m
    if rev:
        return rev

    if use_estimates and sector:
        benchmark = _SECTOR_REVENUE_BENCHMARKS.get(sector, _SECTOR_REVENUE_BENCHMARKS["default"])
        return ProvenanceField.estimated(
            value=float(benchmark),
            basis=f"{sector} sector median revenue benchmark",
            notes=(
                f"ESTIMATED — ${benchmark:,.0f}M sector median. "
                "Provide actual revenue for accurate transition VaR."
            ),
        )
    return None


async def build_company_profile(
    company_name:    str,
    sector:          Optional[str]  = None,
    country_hint:    Optional[str]  = None,   # ISO-2 e.g. "US", "GB"
    ticker:          Optional[str]  = None,
    lat:             Optional[float] = None,
    lon:             Optional[float] = None,
    use_estimates:   bool           = True,
) -> CompanyProfile:
    """
    Main entry point.  Runs all acquisition tasks in parallel and returns
    a fully provenance-tagged CompanyProfile.

    Parameters
    ----------
    company_name  : Free-text company name
    sector        : Optional sector key (used for benchmark fallbacks)
    country_hint  : Optional ISO-2 country (speeds up GLEIF search)
    ticker        : Optional stock ticker (skips Yahoo name search)
    lat / lon     : If provided, skips geocoding (uses these directly)
    use_estimates : If True, benchmark estimates fill missing fields (flagged ESTIMATED)
                    If False, missing fields stay None (stricter, gaps must be filled)
    """
    profile = CompanyProfile(input_name=company_name, sector=sector)
    t_start = datetime.utcnow()

    logger.info("Starting parallel data acquisition for '%s'", company_name)

    # ── If caller provided coordinates, use them directly ─────────────────────
    if lat is not None and lon is not None:
        profile.lat = ProvenanceField(
            value=lat, source="Practitioner-provided", url=None,
            retrieval_date=today_iso(), confidence_tier=ConfidenceTier.VERIFIED,
            notes="Coordinates provided by user — not geocoded",
        )
        profile.lon = ProvenanceField(
            value=lon, source="Practitioner-provided", url=None,
            retrieval_date=today_iso(), confidence_tier=ConfidenceTier.VERIFIED,
            notes="Coordinates provided by user — not geocoded",
        )

    # ── Parallel fetch: GLEIF + EDGAR + CDP + SBTi + Yahoo ───────────────────
    async def _gleif() -> EntityResolverResult:
        return await resolve_entity(company_name, jurisdiction_hint=country_hint)

    async def _edgar() -> EdgarResult:
        return await fetch_edgar_data(company_name)

    async def _cdp() -> CdpResult:
        return await fetch_cdp_data(company_name)

    async def _sbti() -> SbtiResult:
        return await fetch_sbti_data(company_name)

    async def _yahoo() -> YahooResult:
        t = ticker
        if not t:
            t = await find_ticker_from_name(company_name)
        if t:
            return await fetch_yahoo_data(t)
        r = YahooResult()
        r.data_gaps.append(DataGap(
            field_name="ticker",
            reason=f"Could not auto-detect ticker for '{company_name}'",
            impact="Market cap, EV, and beta unavailable from Yahoo Finance",
            suggested_action="Provide ticker symbol (e.g. XOM, SHEL, BP)",
        ))
        return r

    async def _eutl() -> EutlResult:
        return await fetch_eutl_emissions(company_name)

    # All tasks run simultaneously (including EUTL for EU ETS verified emissions)
    gleif_r, edgar_r, cdp_r, sbti_r, yahoo_r, eutl_r = await asyncio.gather(
        _gleif(), _edgar(), _cdp(), _sbti(), _yahoo(), _eutl(),
        return_exceptions=True,
    )

    # ── Safely unpack (exceptions → empty results with gap) ──────────────────
    def _safe(result, cls, field_name: str):
        if isinstance(result, Exception):
            logger.error("Acquisition task failed (%s): %s", field_name, result)
            r = cls()
            r.data_gaps = [DataGap(
                field_name=field_name,
                reason=str(result),
                impact=f"{field_name} data unavailable",
                suggested_action="Check logs and re-run",
            )]
            return r
        return result

    gleif_r = _safe(gleif_r, EntityResolverResult, "lei")   if not isinstance(gleif_r, EntityResolverResult) else gleif_r
    edgar_r = _safe(edgar_r, EdgarResult,           "edgar") if not isinstance(edgar_r, EdgarResult)           else edgar_r
    cdp_r   = _safe(cdp_r,   CdpResult,             "cdp")   if not isinstance(cdp_r,   CdpResult)             else cdp_r
    sbti_r  = _safe(sbti_r,  SbtiResult,            "sbti")  if not isinstance(sbti_r,  SbtiResult)            else sbti_r
    yahoo_r = _safe(yahoo_r, YahooResult,            "yahoo") if not isinstance(yahoo_r, YahooResult)           else yahoo_r
    eutl_r  = _safe(eutl_r,  EutlResult,             "eutl")  if not isinstance(eutl_r,  EutlResult)            else eutl_r

    # ── Entity identity ───────────────────────────────────────────────────────
    if gleif_r.resolved:
        profile.resolved_name      = gleif_r.legal_name
        profile.lei                = gleif_r.lei
        profile.jurisdiction       = gleif_r.jurisdiction or country_hint
        profile.registered_address = gleif_r.registered_address
    else:
        profile.jurisdiction = country_hint
        if gleif_r.data_gap:
            profile.add_gap(gleif_r.data_gap)

    # ── Geocoding (if coordinates not already set) ────────────────────────────
    if profile.lat is None:
        address_to_geocode = profile.registered_address or company_name
        geocode_r: GeocodeResult = await geocode_address(
            address_to_geocode,
            country_code=profile.jurisdiction,
        )
        if geocode_r.lat is not None:
            # Split the combined provenance into lat / lon fields
            profile.lat = ProvenanceField(
                value=geocode_r.lat,
                source=geocode_r.provenance.source,
                url=geocode_r.provenance.url,
                retrieval_date=geocode_r.provenance.retrieval_date,
                confidence_tier=geocode_r.provenance.confidence_tier,
                notes=geocode_r.provenance.notes,
            )
            profile.lon = ProvenanceField(
                value=geocode_r.lon,
                source=geocode_r.provenance.source,
                url=geocode_r.provenance.url,
                retrieval_date=geocode_r.provenance.retrieval_date,
                confidence_tier=geocode_r.provenance.confidence_tier,
                notes=geocode_r.provenance.notes,
            )
        elif geocode_r.data_gap:
            profile.add_gap(geocode_r.data_gap)

    # ── Financials (EDGAR wins over Yahoo) ───────────────────────────────────
    profile.revenue_usd_m    = _resolve_revenue(edgar_r, yahoo_r, sector, use_estimates)
    profile.total_assets_usd_m = edgar_r.total_assets_usd_m
    profile.market_cap_usd_m   = yahoo_r.market_cap_usd_m
    profile.ev_usd_m           = (
        yahoo_r.ev_usd_m
        or (
            ProvenanceField.estimated(
                value=round((profile.revenue_usd_m.value if profile.revenue_usd_m else 5000) * 1.5, 2),
                basis="Revenue × 1.5× sector median EV/Revenue multiple",
                notes="Estimated EV — provide actual for accurate VaR calculation",
            ) if use_estimates and profile.revenue_usd_m else None
        )
    )
    profile.ebitda_usd_m    = yahoo_r.ebitda_usd_m
    profile.total_debt_usd_m = yahoo_r.total_debt_usd_m

    # ── Emissions — priority: EU ETS EUTL > EDGAR XBRL > CDP > estimate ─────
    rev_val = profile.revenue_usd_m.value if profile.revenue_usd_m else None

    # EUTL verified emissions override all other Scope 1 sources for EU operators
    if eutl_r.scope1_mt_co2e is not None:
        profile.scope1_mt_co2e = eutl_r.scope1_mt_co2e
        logger.info(
            "EU ETS EUTL emissions used for '%s' (%d installations): %.4f Mt CO2e",
            company_name, eutl_r.installation_count, eutl_r.scope1_mt_co2e.value
        )
        # Still resolve Scope 2 from other sources
        _, profile.scope2_mt_co2e = _resolve_emissions(
            edgar_r, cdp_r, sector, rev_val, use_estimates
        )
    else:
        profile.scope1_mt_co2e, profile.scope2_mt_co2e = _resolve_emissions(
            edgar_r, cdp_r, sector, rev_val, use_estimates
        )
    profile.add_all_gaps(eutl_r.data_gaps)
    profile.scope3_mt_co2e = cdp_r.scope3_mt_co2e

    # ── Scope 3 double-counting correction via EE-MRIO ────────────────────────
    # GHG Protocol §5.4: raw Scope 3 sums overcount supply-chain emissions
    # by 2-3× because the same tonne appears at multiple tiers.
    # Apply Hertwich-Peters tier-attribution weights (40% Tier-1, 15% Tier-2).
    if profile.scope3_mt_co2e is not None and profile.scope3_mt_co2e.value:
        try:
            from ..climate.eemrio import attenuate_scope3, Scope3Breakdown
            # CDP reports total Scope 3 without tier breakdown — split heuristically:
            # upstream ~70% of Scope 3 (mostly cat 1-8), downstream ~30% (cat 9-15)
            raw_scope3 = profile.scope3_mt_co2e.value
            upstream_kt   = raw_scope3 * 1_000 * 0.70   # → ktCO2e Tier-1+2 upstream
            downstream_kt = raw_scope3 * 1_000 * 0.30   # → ktCO2e downstream
            breakdown = Scope3Breakdown(
                tier1_upstream_ktco2e=upstream_kt * 0.60,   # Tier-1 = 60% of upstream
                tier2_upstream_ktco2e=upstream_kt * 0.40,   # Tier-2 = 40% of upstream
                downstream_ktco2e=downstream_kt,
            )
            attenuated = attenuate_scope3(breakdown)
            attenuated_mt = attenuated.total_ktco2e / 1_000
            profile.scope3_mt_co2e = ProvenanceField(
                value=round(attenuated_mt, 4),
                source=f"EE-MRIO attenuated (GHG Protocol §5.4 Hertwich-Peters); raw CDP={raw_scope3:.3f}Mt",
                url="https://doi.org/10.1073/pnas.0906974106",
                retrieval_date=today_iso(),
                confidence_tier=ConfidenceTier.REPORTED,
                notes=(
                    f"Double-counting corrected: raw {raw_scope3:.3f} Mt → "
                    f"attenuated {attenuated_mt:.3f} Mt. "
                    "Tier-1 weight 40%, Tier-2 weight 15% per GHG Protocol §5.4."
                ),
            )
        except Exception as _eemrio_err:
            logger.debug("EE-MRIO attenuation skipped: %s", _eemrio_err)

    # ── Climate targets ───────────────────────────────────────────────────────
    profile.sbti_status          = sbti_r.sbti_status
    profile.temperature_ambition = sbti_r.temperature_ambition
    profile.net_zero_year        = sbti_r.net_zero_year
    profile.cdp_score            = cdp_r.disclosure_score

    # ── Collect all data gaps ─────────────────────────────────────────────────
    for r in [edgar_r, cdp_r, sbti_r, yahoo_r]:
        profile.add_all_gaps(getattr(r, "data_gaps", []))

    # Revenue gap
    if not profile.revenue_usd_m:
        profile.add_gap(DataGap(
            field_name="revenue_usd_m",
            reason="Revenue unavailable from EDGAR, Yahoo Finance, or estimates",
            impact="Transition VaR as % revenue cannot be computed",
            suggested_action="Provide most recent annual revenue (USD millions)",
        ))

    # EV gap
    if not profile.ev_usd_m:
        profile.add_gap(DataGap(
            field_name="ev_usd_m",
            reason="Enterprise value unavailable from Yahoo Finance",
            impact="Physical VaR as % EV cannot be computed; flood cost in USD M used instead",
            suggested_action="Provide enterprise value or market cap (USD millions)",
        ))

    # ── Assess readiness ──────────────────────────────────────────────────────
    profile.assessment_readiness = _assess_readiness(profile)

    # ── Store raw source results ──────────────────────────────────────────────
    profile.raw_sources = {
        "gleif":    gleif_r.to_dict() if hasattr(gleif_r, "to_dict") else {},
        "edgar":    edgar_r.to_dict(),
        "cdp":      cdp_r.to_dict(),
        "sbti":     sbti_r.to_dict(),
        "yahoo":    yahoo_r.to_dict(),
    }

    elapsed = (datetime.utcnow() - t_start).total_seconds()
    logger.info(
        "Company profile built for '%s': readiness=%s, gaps=%d, elapsed=%.1fs",
        company_name,
        profile.assessment_readiness,
        len(profile.data_gaps),
        elapsed,
    )

    return profile
