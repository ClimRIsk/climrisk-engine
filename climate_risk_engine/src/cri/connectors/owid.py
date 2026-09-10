"""Our World in Data connector for energy and commodity data.

Fossil share:  live from OWID energy CSV on GitHub
  https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv
  Column: fossil_share_energy (% of primary energy from fossil fuels)
  Downloaded once, cached on disk; fallback to regional hardcoded table.

Commodity demand indices: IEA WEO 2023-aligned static curves (no public API).
"""

from pathlib import Path
from typing import Optional, Dict
import csv
import io
import json
import urllib.request

from .base import BaseConnector

# ---------------------------------------------------------------------------
# OWID energy CSV — fossil share lookup
# ---------------------------------------------------------------------------
_OWID_ENERGY_URL = (
    "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
)

# ISO-3166-1 alpha-3 and common name aliases → OWID country name
_ISO_TO_OWID: dict[str, str] = {
    "AUS": "Australia",  "AUT": "Austria",   "BEL": "Belgium",
    "BRA": "Brazil",     "CAN": "Canada",    "CHE": "Switzerland",
    "CHN": "China",      "DEU": "Germany",   "DNK": "Denmark",
    "ESP": "Spain",      "FIN": "Finland",   "FRA": "France",
    "GBR": "United Kingdom", "IDN": "Indonesia", "IND": "India",
    "ITA": "Italy",      "JPN": "Japan",     "KOR": "South Korea",
    "MEX": "Mexico",     "NLD": "Netherlands", "NOR": "Norway",
    "POL": "Poland",     "RUS": "Russia",    "SAU": "Saudi Arabia",
    "SWE": "Sweden",     "TUR": "Turkey",    "UKR": "Ukraine",
    "USA": "United States", "ZAF": "South Africa",
    # Two-letter aliases
    "AU": "Australia", "BR": "Brazil",  "CA": "Canada",  "CN": "China",
    "DE": "Germany",   "FR": "France",  "GB": "United Kingdom",
    "ID": "Indonesia", "IN": "India",   "IT": "Italy",   "JP": "Japan",
    "KR": "South Korea","MX": "Mexico", "NO": "Norway",  "PL": "Poland",
    "RU": "Russia",    "SA": "Saudi Arabia","SE": "Sweden","TR": "Turkey",
    "UA": "Ukraine",   "US": "United States","ZA": "South Africa",
}

# Fallback regional averages (fraction, 0-1) for countries not in OWID data
_REGIONAL_FALLBACK: dict[str, float] = {
    "China": 0.83, "India": 0.74, "Australia": 0.78, "United States": 0.81,
    "Germany": 0.65, "France": 0.13, "Brazil": 0.44, "Canada": 0.64,
    "United Kingdom": 0.72, "Russia": 0.88, "Japan": 0.83,
    "South Korea": 0.83, "Saudi Arabia": 0.99, "Indonesia": 0.72,
    "South Africa": 0.87, "Norway": 0.29, "Sweden": 0.23,
}


# IEA WEO 2023-aligned demand indices by commodity and scenario
# Base year 2025 = 100
# Scenarios: nze_2050 (Net Zero by 2050), delayed_transition, current_policies
DEMAND_INDICES = {
    "crude_oil": {
        "nze_2050": {2025: 100, 2030: 75, 2035: 55, 2040: 38, 2045: 25, 2050: 15},
        "delayed_transition": {2025: 100, 2030: 88, 2035: 72, 2040: 58, 2045: 42, 2050: 30},
        "current_policies": {2025: 100, 2030: 98, 2035: 97, 2040: 96, 2045: 95, 2050: 93},
        "below_2c_orderly": {2025: 100, 2030: 68, 2035: 45, 2040: 28, 2045: 18, 2050: 10},
    },
    "natural_gas": {
        "nze_2050": {2025: 100, 2030: 95, 2035: 75, 2040: 50, 2045: 30, 2050: 15},
        "delayed_transition": {2025: 100, 2030: 105, 2035: 85, 2040: 65, 2045: 45, 2050: 28},
        "current_policies": {2025: 100, 2030: 110, 2035: 115, 2040: 118, 2045: 120, 2050: 122},
        "below_2c_orderly": {2025: 100, 2030: 85, 2035: 60, 2040: 38, 2045: 22, 2050: 12},
    },
    "coal_thermal": {
        "nze_2050": {2025: 100, 2030: 65, 2035: 30, 2040: 12, 2045: 4, 2050: 1},
        "delayed_transition": {2025: 100, 2030: 85, 2035: 55, 2040: 32, 2045: 15, 2050: 7},
        "current_policies": {2025: 100, 2030: 98, 2035: 95, 2040: 92, 2045: 88, 2050: 82},
        "below_2c_orderly": {2025: 100, 2030: 55, 2035: 20, 2040: 8, 2045: 2, 2050: 0},
    },
    "coal_metallurgical": {
        "nze_2050": {2025: 100, 2030: 90, 2035: 75, 2040: 55, 2045: 35, 2050: 20},
        "delayed_transition": {2025: 100, 2030: 98, 2035: 90, 2040: 78, 2045: 62, 2050: 48},
        "current_policies": {2025: 100, 2030: 105, 2035: 112, 2040: 120, 2045: 125, 2050: 128},
        "below_2c_orderly": {2025: 100, 2030: 85, 2035: 68, 2040: 48, 2045: 28, 2050: 15},
    },
    "iron_ore": {
        "nze_2050": {2025: 100, 2030: 95, 2035: 85, 2040: 75, 2045: 68, 2050: 62},
        "delayed_transition": {2025: 100, 2030: 105, 2035: 98, 2040: 88, 2045: 78, 2050: 68},
        "current_policies": {2025: 100, 2030: 108, 2035: 120, 2040: 132, 2045: 140, 2050: 145},
        "below_2c_orderly": {2025: 100, 2030: 92, 2035: 82, 2040: 70, 2045: 62, 2050: 55},
    },
    "copper": {
        "nze_2050": {2025: 100, 2030: 145, 2035: 190, 2040: 240, 2045: 280, 2050: 310},
        "delayed_transition": {2025: 100, 2030: 120, 2035: 160, 2040: 200, 2045: 235, 2050: 260},
        "current_policies": {2025: 100, 2030: 110, 2035: 125, 2040: 140, 2045: 150, 2050: 155},
        "below_2c_orderly": {2025: 100, 2030: 155, 2035: 210, 2040: 270, 2045: 310, 2050: 345},
    },
    "aluminium": {
        "nze_2050": {2025: 100, 2030: 115, 2035: 145, 2040: 175, 2045: 200, 2050: 220},
        "delayed_transition": {2025: 100, 2030: 108, 2035: 130, 2040: 155, 2045: 175, 2050: 190},
        "current_policies": {2025: 100, 2030: 110, 2035: 128, 2040: 145, 2045: 160, 2050: 170},
        "below_2c_orderly": {2025: 100, 2030: 120, 2035: 155, 2040: 190, 2045: 220, 2050: 245},
    },
    "lithium": {
        "nze_2050": {2025: 100, 2030: 280, 2035: 450, 2040: 620, 2045: 750, 2050: 850},
        "delayed_transition": {2025: 100, 2030: 210, 2035: 320, 2040: 420, 2045: 500, 2050: 560},
        "current_policies": {2025: 100, 2030: 140, 2035: 180, 2040: 210, 2045: 235, 2050: 250},
        "below_2c_orderly": {2025: 100, 2030: 310, 2035: 500, 2040: 680, 2045: 820, 2050: 920},
    },
    "nickel": {
        "nze_2050": {2025: 100, 2030: 170, 2035: 240, 2040: 310, 2045: 370, 2050: 420},
        "delayed_transition": {2025: 100, 2030: 130, 2035: 180, 2040: 230, 2045: 270, 2050: 300},
        "current_policies": {2025: 100, 2030: 115, 2035: 135, 2040: 155, 2045: 170, 2050: 180},
        "below_2c_orderly": {2025: 100, 2030: 185, 2035: 270, 2040: 360, 2045: 430, 2050: 490},
    },
}


class OWIDConnector(BaseConnector):
    """Connector for Our World in Data energy and commodity data.

    Provides historical energy data and forward commodity demand indices
    aligned with IEA World Energy Outlook scenarios.
    """

    ENERGY_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

    def fetch(self, **kwargs) -> dict:
        """Not used directly; use get_fossil_share or get_commodity_demand_index."""
        raise NotImplementedError(
            "Use get_fossil_share() or get_commodity_demand_index()"
        )

    # ── OWID energy CSV cache (class-level, shared across instances) ──────────
    _owid_index: dict[tuple[str, int], float] | None = None

    def _load_owid_index(self) -> dict[tuple[str, int], float]:
        """Download + parse OWID energy CSV; cache in class variable.

        Returns dict keyed by (country_name, year) → fossil_share_energy (0-100 → 0-1).
        Falls back to empty dict on network failure.
        """
        if OWIDConnector._owid_index is not None:
            return OWIDConnector._owid_index

        # Try disk cache first (CSV is ~5 MB; refresh daily)
        disk_key = "owid_energy_csv_index"
        cached = self._load_cache(disk_key)
        if cached:
            OWIDConnector._owid_index = {
                (k.split("|")[0], int(k.split("|")[1])): v
                for k, v in cached.items()
            }
            return OWIDConnector._owid_index

        try:
            with urllib.request.urlopen(_OWID_ENERGY_URL, timeout=20) as resp:
                text = resp.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(text))
            index: dict[tuple[str, int], float] = {}
            for row in reader:
                raw = row.get("fossil_share_energy", "").strip()
                if not raw:
                    continue
                try:
                    yr = int(row["year"])
                    share = float(raw) / 100.0   # OWID stores as percentage
                    index[(row["country"], yr)] = share
                except (ValueError, KeyError):
                    continue
            OWIDConnector._owid_index = index
            # Persist to disk (keyed as "country|year")
            self._save_cache(disk_key, {f"{c}|{y}": v for (c, y), v in index.items()})
            return index
        except Exception:
            OWIDConnector._owid_index = {}
            return {}

    def get_fossil_share(self, country: str, year: int) -> float:
        """Get fossil fuel share of primary energy for a country/year.

        Primary source: OWID energy CSV (live download, disk-cached daily).
        Fallback: regional hardcoded table.

        Args:
            country: Country name, ISO-3166-1 alpha-2 or alpha-3 code
            year:    Year (historical: available; future: last known + linear decline)

        Returns:
            Fossil fuel share as fraction 0-1.
        """
        cache_key = f"fossil_share_{country}_{year}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached.get("fossil_share", 0.0)

        # Normalise country name
        owid_name = _ISO_TO_OWID.get(country.upper(), country)

        index = self._load_owid_index()

        # Exact year lookup
        if (owid_name, year) in index:
            share = index[(owid_name, year)]
        else:
            # Find nearest available historical year for this country
            country_years = sorted(
                {y for (c, y) in index if c == owid_name}, reverse=True
            )
            if country_years:
                latest_yr = country_years[0]
                base_share = index[(owid_name, latest_yr)]
                if year > latest_yr:
                    # Forward-project: linear decline assumption (transition)
                    elapsed = year - latest_yr
                    share = max(0.0, base_share - elapsed * 0.006)   # ~0.6%/yr decline
                else:
                    share = base_share
            else:
                # Country not in OWID data — use regional fallback
                share = _REGIONAL_FALLBACK.get(owid_name, 0.72)
                if year > 2022:
                    share = max(0.0, share - (year - 2022) * 0.006)

        result = {"fossil_share": round(share, 4), "country": country, "year": year,
                  "source": "owid_live" if index else "regional_fallback"}
        self._save_cache(cache_key, result)
        return result["fossil_share"]

    def get_commodity_demand_index(
        self, commodity: str, scenario: str, year: int
    ) -> float:
        """Get commodity demand index for a given scenario and year.

        Args:
            commodity: Commodity name (e.g. 'copper', 'crude_oil')
            scenario: Scenario name (e.g. 'nze_2050', 'current_policies')
            year: Target year

        Returns:
            Demand index with 2025 = 100
        """
        if commodity not in DEMAND_INDICES:
            raise ValueError(
                f"Unknown commodity {commodity!r}. "
                f"Available: {list(DEMAND_INDICES.keys())}"
            )

        scenario_data = DEMAND_INDICES[commodity]
        if scenario not in scenario_data:
            raise ValueError(
                f"Unknown scenario {scenario!r} for {commodity}. "
                f"Available: {list(scenario_data.keys())}"
            )

        prices = scenario_data[scenario]

        # Check if year is directly available
        if year in prices:
            return float(prices[year])

        # Find surrounding years for interpolation
        sorted_years = sorted(prices.keys())

        # Before first year: use first value
        if year < sorted_years[0]:
            return float(prices[sorted_years[0]])

        # After last year: use last value
        if year > sorted_years[-1]:
            return float(prices[sorted_years[-1]])

        # Linear interpolation between two points
        for i in range(len(sorted_years) - 1):
            y1, y2 = sorted_years[i], sorted_years[i + 1]
            if y1 <= year <= y2:
                p1, p2 = prices[y1], prices[y2]
                # Linear interpolation
                weight = (year - y1) / (y2 - y1)
                return float(p1 + weight * (p2 - p1))

        # Fallback (should not reach here)
        return float(prices[sorted_years[-1]])
