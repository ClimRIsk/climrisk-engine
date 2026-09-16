"""
data_acquisition — autonomous data fetching layer for the ClimRisk engine.

Every field produced here carries a ProvenanceField: {value, source, url,
retrieval_date, confidence_tier}.  No silent fallbacks; every gap is made
explicit in CompanyProfile.data_gaps.

Modules
-------
provenance       — ProvenanceField + ConfidenceTier definitions
entity_resolver  — GLEIF API: company name → LEI + legal entity metadata
edgar_client     — SEC EDGAR: 10-K / 10-Q financials + GHG disclosures
cdp_client       — CDP public disclosure scraper: Scope 1/2/3 + targets
yahoo_client     — Yahoo Finance: revenue, EV, market cap (listed companies)
sbti_client      — SBTi database: net zero targets
geocoder_client  — Nominatim / OSM: address → lat/lon
company_profiler — Orchestrates all acquisition tools in parallel
"""
