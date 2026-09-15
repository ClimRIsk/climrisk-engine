"""
Company Lookup / Payload Auto-Fill.

Given a company name or ticker, returns a pre-populated ClimRisk
API payload so the user does not have to hunt for lat/lon, revenue
benchmarks, or sector codes before running the engine.

Data hierarchy:
    1. COMPANY_REGISTRY — hand-curated major names (instant, offline)
    2. Sector/country benchmark fallback — estimates from sector centroid
       coordinates and GICS revenue benchmarks

All coordinates are public-domain or derived from country centroids —
no licensed data required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math


# ── Country centroids (latitude, longitude) ──────────────────────────────────
_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "United States": (37.09, -95.71),
    "United Kingdom": (55.38, -3.44),
    "Germany": (51.17, 10.45),
    "France": (46.23, 2.21),
    "Japan": (36.20, 138.25),
    "China": (35.86, 104.19),
    "India": (20.59, 78.96),
    "Australia": (-25.27, 133.77),
    "Brazil": (-14.24, -51.93),
    "Canada": (56.13, -106.35),
    "Netherlands": (52.13, 5.29),
    "Switzerland": (46.82, 8.23),
    "Sweden": (60.13, 18.64),
    "South Korea": (35.91, 127.77),
    "Italy": (41.87, 12.57),
    "Spain": (40.46, -3.75),
    "Mexico": (23.63, -102.55),
    "Indonesia": (-0.79, 113.92),
    "Saudi Arabia": (23.89, 45.08),
    "Norway": (60.47, 8.47),
    "Denmark": (56.26, 9.50),
    "Belgium": (50.50, 4.47),
    "Singapore": (1.35, 103.82),
    "South Africa": (-30.56, 22.94),
    "Nigeria": (9.08, 8.68),
    "UAE": (23.42, 53.85),
}

# ── Sector revenue benchmarks (USD M) — median listed company ─────────────────
_SECTOR_REVENUE_BENCHMARKS: dict[str, dict] = {
    "oil_gas":        {"revenue_usd_m": 8000, "ev_usd_m": 12000, "scope1_mt_co2e": 2500},
    "coal":           {"revenue_usd_m": 2000, "ev_usd_m": 1500,  "scope1_mt_co2e": 5000},
    "utilities":      {"revenue_usd_m": 4000, "ev_usd_m": 6000,  "scope1_mt_co2e": 8000},
    "chemicals":      {"revenue_usd_m": 3000, "ev_usd_m": 4500,  "scope1_mt_co2e": 800},
    "metals_mining":  {"revenue_usd_m": 5000, "ev_usd_m": 7000,  "scope1_mt_co2e": 1200},
    "mining":         {"revenue_usd_m": 3500, "ev_usd_m": 5000,  "scope1_mt_co2e": 900},
    "cement":         {"revenue_usd_m": 2500, "ev_usd_m": 3500,  "scope1_mt_co2e": 3000},
    "agriculture":    {"revenue_usd_m": 1500, "ev_usd_m": 2000,  "scope1_mt_co2e": 400},
    "automotive":     {"revenue_usd_m": 15000, "ev_usd_m": 20000, "scope1_mt_co2e": 300},
    "aviation":       {"revenue_usd_m": 6000, "ev_usd_m": 5000,  "scope1_mt_co2e": 1800},
    "shipping":       {"revenue_usd_m": 4000, "ev_usd_m": 3500,  "scope1_mt_co2e": 600},
    "industrials":    {"revenue_usd_m": 3000, "ev_usd_m": 4500,  "scope1_mt_co2e": 200},
    "real_estate":    {"revenue_usd_m": 500,  "ev_usd_m": 2000,  "scope1_mt_co2e": 50},
    "construction":   {"revenue_usd_m": 2000, "ev_usd_m": 2500,  "scope1_mt_co2e": 150},
    "food_beverage":  {"revenue_usd_m": 4000, "ev_usd_m": 5500,  "scope1_mt_co2e": 180},
    "consumer":       {"revenue_usd_m": 5000, "ev_usd_m": 7000,  "scope1_mt_co2e": 80},
    "healthcare":     {"revenue_usd_m": 3000, "ev_usd_m": 5000,  "scope1_mt_co2e": 60},
    "technology":     {"revenue_usd_m": 5000, "ev_usd_m": 12000, "scope1_mt_co2e": 40},
    "financials":     {"revenue_usd_m": 8000, "ev_usd_m": 20000, "scope1_mt_co2e": 30},
    "transport":      {"revenue_usd_m": 2000, "ev_usd_m": 2500,  "scope1_mt_co2e": 300},
    "default":        {"revenue_usd_m": 2000, "ev_usd_m": 3000,  "scope1_mt_co2e": 200},
}

# ── Curated company registry ──────────────────────────────────────────────────
# Each entry: name_variants, sector_key, country, hq_lat, hq_lon,
#             revenue_usd_m, ev_usd_m, scope1_mt_co2e, ticker
COMPANY_REGISTRY: list[dict] = [
    # ─ Energy / Oil & Gas ─
    {
        "names": ["exxonmobil", "exxon", "xom"],
        "company_name": "ExxonMobil Corporation",
        "ticker": "XOM",
        "sector_key": "oil_gas",
        "country": "United States",
        "hq_lat": 32.91, "hq_lon": -97.05,  # Spring, TX
        "revenue_usd_m": 398321, "ev_usd_m": 520000, "scope1_mt_co2e": 112000,
    },
    {
        "names": ["shell", "royal dutch shell", "rdsa", "rdsb", "shel"],
        "company_name": "Shell plc",
        "ticker": "SHEL",
        "sector_key": "oil_gas",
        "country": "United Kingdom",
        "hq_lat": 51.51, "hq_lon": -0.09,
        "revenue_usd_m": 302000, "ev_usd_m": 220000, "scope1_mt_co2e": 67000,
    },
    {
        "names": ["bp", "british petroleum", "bpa"],
        "company_name": "BP plc",
        "ticker": "BP",
        "sector_key": "oil_gas",
        "country": "United Kingdom",
        "hq_lat": 51.52, "hq_lon": -0.15,
        "revenue_usd_m": 241000, "ev_usd_m": 92000, "scope1_mt_co2e": 49000,
    },
    {
        "names": ["totalenergies", "total", "tte"],
        "company_name": "TotalEnergies SE",
        "ticker": "TTE",
        "sector_key": "oil_gas",
        "country": "France",
        "hq_lat": 48.87, "hq_lon": 2.33,
        "revenue_usd_m": 213000, "ev_usd_m": 140000, "scope1_mt_co2e": 45000,
    },
    {
        "names": ["chevron", "cvx"],
        "company_name": "Chevron Corporation",
        "ticker": "CVX",
        "sector_key": "oil_gas",
        "country": "United States",
        "hq_lat": 37.93, "hq_lon": -122.12,
        "revenue_usd_m": 200000, "ev_usd_m": 300000, "scope1_mt_co2e": 60000,
    },
    {
        "names": ["equinor", "statoil", "eqnr"],
        "company_name": "Equinor ASA",
        "ticker": "EQNR",
        "sector_key": "oil_gas",
        "country": "Norway",
        "hq_lat": 58.90, "hq_lon": 5.72,
        "revenue_usd_m": 116000, "ev_usd_m": 85000, "scope1_mt_co2e": 12000,
    },
    # ─ Utilities ─
    {
        "names": ["enel", "enl"],
        "company_name": "Enel SpA",
        "ticker": "ENEL.MI",
        "sector_key": "utilities",
        "country": "Italy",
        "hq_lat": 41.90, "hq_lon": 12.49,
        "revenue_usd_m": 88000, "ev_usd_m": 70000, "scope1_mt_co2e": 38000,
    },
    {
        "names": ["rwe", "rwe ag"],
        "company_name": "RWE AG",
        "ticker": "RWE.DE",
        "sector_key": "utilities",
        "country": "Germany",
        "hq_lat": 51.45, "hq_lon": 7.01,
        "revenue_usd_m": 36000, "ev_usd_m": 28000, "scope1_mt_co2e": 45000,
    },
    {
        "names": ["nextera energy", "nextera", "nee"],
        "company_name": "NextEra Energy Inc",
        "ticker": "NEE",
        "sector_key": "utilities",
        "country": "United States",
        "hq_lat": 25.77, "hq_lon": -80.19,
        "revenue_usd_m": 22000, "ev_usd_m": 120000, "scope1_mt_co2e": 5000,
    },
    # ─ Metals & Mining ─
    {
        "names": ["rio tinto", "rio", "rto"],
        "company_name": "Rio Tinto Group",
        "ticker": "RIO",
        "sector_key": "metals_mining",
        "country": "Australia",
        "hq_lat": -31.95, "hq_lon": 115.86,
        "revenue_usd_m": 54000, "ev_usd_m": 80000, "scope1_mt_co2e": 27000,
    },
    {
        "names": ["bhp", "bhp group", "bhp billiton"],
        "company_name": "BHP Group Limited",
        "ticker": "BHP",
        "sector_key": "metals_mining",
        "country": "Australia",
        "hq_lat": -37.81, "hq_lon": 144.96,
        "revenue_usd_m": 53000, "ev_usd_m": 130000, "scope1_mt_co2e": 15000,
    },
    {
        "names": ["vale", "vale sa"],
        "company_name": "Vale S.A.",
        "ticker": "VALE",
        "sector_key": "metals_mining",
        "country": "Brazil",
        "hq_lat": -22.90, "hq_lon": -43.17,
        "revenue_usd_m": 42000, "ev_usd_m": 55000, "scope1_mt_co2e": 14000,
    },
    {
        "names": ["glencore", "glen"],
        "company_name": "Glencore plc",
        "ticker": "GLEN",
        "sector_key": "metals_mining",
        "country": "Switzerland",
        "hq_lat": 47.10, "hq_lon": 8.60,
        "revenue_usd_m": 256000, "ev_usd_m": 58000, "scope1_mt_co2e": 20000,
    },
    # ─ Chemicals ─
    {
        "names": ["basf", "basf se"],
        "company_name": "BASF SE",
        "ticker": "BAS.DE",
        "sector_key": "chemicals",
        "country": "Germany",
        "hq_lat": 49.48, "hq_lon": 8.47,
        "revenue_usd_m": 68000, "ev_usd_m": 42000, "scope1_mt_co2e": 16000,
    },
    {
        "names": ["dow", "dow inc"],
        "company_name": "Dow Inc",
        "ticker": "DOW",
        "sector_key": "chemicals",
        "country": "United States",
        "hq_lat": 43.62, "hq_lon": -83.89,
        "revenue_usd_m": 45000, "ev_usd_m": 35000, "scope1_mt_co2e": 8500,
    },
    # ─ Automotive ─
    {
        "names": ["volkswagen", "vw", "vow", "vow3"],
        "company_name": "Volkswagen AG",
        "ticker": "VOW3.DE",
        "sector_key": "automotive",
        "country": "Germany",
        "hq_lat": 52.43, "hq_lon": 10.79,
        "revenue_usd_m": 293000, "ev_usd_m": 80000, "scope1_mt_co2e": 8000,
    },
    {
        "names": ["toyota", "toyota motor", "tm"],
        "company_name": "Toyota Motor Corporation",
        "ticker": "TM",
        "sector_key": "automotive",
        "country": "Japan",
        "hq_lat": 35.08, "hq_lon": 137.15,
        "revenue_usd_m": 274000, "ev_usd_m": 220000, "scope1_mt_co2e": 9000,
    },
    {
        "names": ["tesla", "tsla"],
        "company_name": "Tesla Inc",
        "ticker": "TSLA",
        "sector_key": "automotive",
        "country": "United States",
        "hq_lat": 30.23, "hq_lon": -97.62,
        "revenue_usd_m": 97700, "ev_usd_m": 580000, "scope1_mt_co2e": 900,
    },
    # ─ Agriculture / Food ─
    {
        "names": ["nestle", "nesn"],
        "company_name": "Nestlé S.A.",
        "ticker": "NESN.SW",
        "sector_key": "food_beverage",
        "country": "Switzerland",
        "hq_lat": 46.81, "hq_lon": 6.64,
        "revenue_usd_m": 94000, "ev_usd_m": 260000, "scope1_mt_co2e": 4000,
    },
    {
        "names": ["unilever", "ul", "ulvr"],
        "company_name": "Unilever PLC",
        "ticker": "ULVR",
        "sector_key": "food_beverage",
        "country": "United Kingdom",
        "hq_lat": 51.50, "hq_lon": -0.12,
        "revenue_usd_m": 60000, "ev_usd_m": 110000, "scope1_mt_co2e": 2000,
    },
    # ─ Technology ─
    {
        "names": ["apple", "aapl", "apple inc"],
        "company_name": "Apple Inc",
        "ticker": "AAPL",
        "sector_key": "technology",
        "country": "United States",
        "hq_lat": 37.33, "hq_lon": -122.01,
        "revenue_usd_m": 383000, "ev_usd_m": 3100000, "scope1_mt_co2e": 50,
    },
    {
        "names": ["microsoft", "msft"],
        "company_name": "Microsoft Corporation",
        "ticker": "MSFT",
        "sector_key": "technology",
        "country": "United States",
        "hq_lat": 47.64, "hq_lon": -122.13,
        "revenue_usd_m": 212000, "ev_usd_m": 3000000, "scope1_mt_co2e": 140,
    },
    {
        "names": ["alphabet", "google", "googl", "goog"],
        "company_name": "Alphabet Inc",
        "ticker": "GOOGL",
        "sector_key": "technology",
        "country": "United States",
        "hq_lat": 37.42, "hq_lon": -122.08,
        "revenue_usd_m": 307000, "ev_usd_m": 2000000, "scope1_mt_co2e": 300,
    },
    # ─ Financials ─
    {
        "names": ["jpmorgan", "jp morgan", "jpm", "jpmorgan chase"],
        "company_name": "JPMorgan Chase & Co",
        "ticker": "JPM",
        "sector_key": "financials",
        "country": "United States",
        "hq_lat": 40.75, "hq_lon": -73.98,
        "revenue_usd_m": 158000, "ev_usd_m": 480000, "scope1_mt_co2e": 120,
    },
    {
        "names": ["hsbc", "hsba", "hsbc holdings"],
        "company_name": "HSBC Holdings plc",
        "ticker": "HSBA",
        "sector_key": "financials",
        "country": "United Kingdom",
        "hq_lat": 51.51, "hq_lon": -0.08,
        "revenue_usd_m": 67000, "ev_usd_m": 160000, "scope1_mt_co2e": 80,
    },
    {
        "names": ["blackrock", "blk"],
        "company_name": "BlackRock Inc",
        "ticker": "BLK",
        "sector_key": "financials",
        "country": "United States",
        "hq_lat": 40.71, "hq_lon": -74.01,
        "revenue_usd_m": 17900, "ev_usd_m": 110000, "scope1_mt_co2e": 25,
    },
    # ─ Real Estate ─
    {
        "names": ["vonovia", "vna"],
        "company_name": "Vonovia SE",
        "ticker": "VNA.DE",
        "sector_key": "real_estate",
        "country": "Germany",
        "hq_lat": 51.49, "hq_lon": 7.22,
        "revenue_usd_m": 2500, "ev_usd_m": 12000, "scope1_mt_co2e": 300,
    },
    # ─ Healthcare ─
    {
        "names": ["johnson & johnson", "jnj", "j&j"],
        "company_name": "Johnson & Johnson",
        "ticker": "JNJ",
        "sector_key": "healthcare",
        "country": "United States",
        "hq_lat": 40.57, "hq_lon": -74.46,
        "revenue_usd_m": 85200, "ev_usd_m": 390000, "scope1_mt_co2e": 900,
    },
    {
        "names": ["novartis", "nvs", "novn"],
        "company_name": "Novartis AG",
        "ticker": "NVS",
        "sector_key": "healthcare",
        "country": "Switzerland",
        "hq_lat": 47.56, "hq_lon": 7.58,
        "revenue_usd_m": 45400, "ev_usd_m": 200000, "scope1_mt_co2e": 380,
    },
    # ─ Aviation / Shipping ─
    {
        "names": ["lufthansa", "lha", "deutsche lufthansa"],
        "company_name": "Deutsche Lufthansa AG",
        "ticker": "LHA.DE",
        "sector_key": "aviation",
        "country": "Germany",
        "hq_lat": 50.87, "hq_lon": 6.96,
        "revenue_usd_m": 36000, "ev_usd_m": 10000, "scope1_mt_co2e": 22000,
    },
    {
        "names": ["maersk", "a.p. moller maersk", "maerskb"],
        "company_name": "A.P. Møller - Mærsk A/S",
        "ticker": "MAERSKB.CO",
        "sector_key": "shipping",
        "country": "Denmark",
        "hq_lat": 55.68, "hq_lon": 12.57,
        "revenue_usd_m": 81500, "ev_usd_m": 42000, "scope1_mt_co2e": 32000,
    },
    # ─ Cement / Construction ─
    {
        "names": ["holcim", "holcim ltd", "holn"],
        "company_name": "Holcim Ltd",
        "ticker": "HOLN.SW",
        "sector_key": "cement",
        "country": "Switzerland",
        "hq_lat": 47.09, "hq_lon": 8.56,
        "revenue_usd_m": 28000, "ev_usd_m": 35000, "scope1_mt_co2e": 120000,
    },
]


@dataclass
class CompanyResolveResult:
    found:               bool
    confidence:          str          # "registry" | "sector_estimate" | "country_default"
    company_name:        str
    ticker:              Optional[str]
    sector_key:          str
    country:             str
    hq_lat:              float
    hq_lon:              float
    revenue_usd_m:       float
    ev_usd_m:            float
    scope1_mt_co2e:      float
    distance_to_coast_km: float       # rough estimate
    elevation_m:         float        # rough estimate (country/sector heuristic)
    pre_filled_payload:  dict         # ready-to-POST ClimRisk payload
    notes:               list[str]


def _levenshtein(a: str, b: str) -> int:
    """Simple Levenshtein distance for fuzzy name matching."""
    if len(b) < len(a):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (ca != cb)))
        prev = curr
    return prev[len(b)]


def _normalize(s: str) -> str:
    return s.lower().strip().replace(",", "").replace(".", "").replace("-", " ")


def _coast_distance_heuristic(lat: float, lon: float, country: str) -> float:
    """Very rough coastal distance — landlocked countries get 500 km."""
    _landlocked = {
        "Switzerland", "Austria", "Czech Republic", "Hungary", "Bolivia",
        "Ethiopia", "Niger", "Chad", "Central African Republic",
    }
    if country in _landlocked:
        return 500.0
    # Simple approximation based on lat/lon extremity
    # Real implementation would use a proximity raster; this is a placeholder
    return 25.0  # coastal default for registered HQ cities


def _elevation_heuristic(lat: float, lon: float, country: str) -> float:
    """Very rough elevation estimate — river deltas low, alpine countries high."""
    _high_elev = {"Switzerland": 800, "Austria": 600, "Nepal": 2000,
                  "Bolivia": 3500, "Colombia": 600, "Peru": 1200}
    _low_elev  = {"Netherlands": 2, "Bangladesh": 5, "Vietnam": 10,
                  "Maldives": 1, "Philippines": 20}
    base = _high_elev.get(country, _low_elev.get(country, 50))
    return float(base)


def resolve_company(
    query: str,
    hint_sector: Optional[str] = None,
    hint_country: Optional[str] = None,
) -> CompanyResolveResult:
    """
    Look up a company by name or ticker and return a pre-filled payload.

    Parameters
    ----------
    query : str          Company name, ticker, or fragment
    hint_sector : str    Optional sector override if name is ambiguous
    hint_country : str   Optional country override
    """
    q = _normalize(query)

    # 1) Exact / substring match in registry
    best_entry: Optional[dict] = None
    best_score = 9999

    for entry in COMPANY_REGISTRY:
        for alias in entry["names"]:
            alias_n = _normalize(alias)
            if q == alias_n or q in alias_n or alias_n in q:
                best_entry = entry
                best_score = 0
                break
            dist = _levenshtein(q, alias_n)
            # Allow fuzzy match within ~30% of string length
            if dist < best_score and dist <= max(3, int(len(alias_n) * 0.30)):
                best_entry = entry
                best_score = dist

    notes: list[str] = []

    if best_entry:
        entry = best_entry
        company_name = entry["company_name"]
        ticker       = entry.get("ticker")
        sector_key   = hint_sector or entry["sector_key"]
        country      = hint_country or entry["country"]
        hq_lat       = entry["hq_lat"]
        hq_lon       = entry["hq_lon"]
        revenue      = entry["revenue_usd_m"]
        ev           = entry["ev_usd_m"]
        scope1       = entry["scope1_mt_co2e"]
        confidence   = "registry"
        if best_score > 0:
            notes.append(f"Fuzzy match (edit distance {best_score}) — verify company identity")
    else:
        # 2) Sector/country estimate fallback
        company_name = query.strip().title()
        ticker       = None
        sector_key   = hint_sector or "default"
        country      = hint_country or "default"
        benchmarks   = _SECTOR_REVENUE_BENCHMARKS.get(
            sector_key, _SECTOR_REVENUE_BENCHMARKS["default"]
        )
        revenue  = benchmarks["revenue_usd_m"]
        ev       = benchmarks["ev_usd_m"]
        scope1   = benchmarks["scope1_mt_co2e"]
        centroid = _COUNTRY_CENTROIDS.get(country, (51.5, 0.0))
        hq_lat, hq_lon = centroid
        confidence = "sector_estimate"
        notes.append(
            f"Company not found in registry — using {sector_key} sector benchmarks "
            f"and {country} country centroid. Provide lat/lon/revenue overrides for accuracy."
        )

    coast_km   = _coast_distance_heuristic(hq_lat, hq_lon, country)
    elevation  = _elevation_heuristic(hq_lat, hq_lon, country)

    pre_filled = {
        "company_id":           query.lower().replace(" ", "_"),
        "company_name":         company_name,
        "sector_key":           sector_key,
        "country":              country,
        "asset_lat":            hq_lat,
        "asset_lon":            hq_lon,
        "asset_elevation_m":    elevation,
        "distance_to_coast_km": coast_km,
        "revenue_usd_m":        revenue,
        "ev_usd_m":             ev,
        "scope1_emissions_mt_co2e": scope1,
        "_confidence":          confidence,
        "_notes":               notes,
    }

    return CompanyResolveResult(
        found=bool(best_entry),
        confidence=confidence,
        company_name=company_name,
        ticker=ticker,
        sector_key=sector_key,
        country=country,
        hq_lat=hq_lat,
        hq_lon=hq_lon,
        revenue_usd_m=revenue,
        ev_usd_m=ev,
        scope1_mt_co2e=scope1,
        distance_to_coast_km=coast_km,
        elevation_m=elevation,
        pre_filled_payload=pre_filled,
        notes=notes,
    )
