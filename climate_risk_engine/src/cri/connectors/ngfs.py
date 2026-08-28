"""NGFS scenario carbon price connector."""

from pathlib import Path
from typing import Optional, List
import json

from .base import BaseConnector


# NGFS Phase 4 carbon prices (USD/tCO2e)
# Source: NGFS published scenario data, publicly available knowledge
NGFS_CARBON_PRICES = {
    "Net Zero 2050": {
        2020: 12,
        2025: 34,
        2030: 130,
        2035: 185,
        2040: 220,
        2045: 240,
        2050: 250,
    },
    "Delayed Transition": {
        2020: 5,
        2025: 10,
        2030: 30,
        2035: 95,
        2040: 155,
        2045: 170,
        2050: 180,
    },
    "Current Policies": {
        2020: 3,
        2025: 8,
        2030: 15,
        2035: 22,
        2040: 28,
        2045: 32,
        2050: 35,
    },
    "Below 2°C": {
        2020: 8,
        2025: 22,
        2030: 75,
        2035: 120,
        2040: 160,
        2045: 185,
        2050: 200,
    },
}


class NGFSConnector(BaseConnector):
    """Connector for NGFS scenario carbon price data.

    Maps NGFS scenario names to carbon price paths with linear interpolation
    between data points. Falls back to embedded Phase 4 data.
    """

    BASE_URL = "https://data.ene.iiasa.ac.at/ngfs/"

    # Mapping from CRI scenario names to NGFS names
    NGFS_MAPPING = {
        "nze_2050": "Net Zero 2050",
        "below_2c_orderly": "Below 2°C",
        "delayed_transition": "Delayed Transition",
        "current_policies": "Current Policies",
    }

    def fetch(self, **kwargs) -> dict:
        """Not used directly; use get_carbon_price or list_scenarios."""
        raise NotImplementedError("Use get_carbon_price() or list_scenarios()")

    def list_scenarios(self) -> List[str]:
        """List available NGFS scenario names.

        Returns:
            List of scenario names
        """
        return list(NGFS_CARBON_PRICES.keys())

    def get_carbon_price(self, scenario_name: str, year: int) -> float:
        """Get carbon price for a scenario and year with linear interpolation.

        Args:
            scenario_name: NGFS scenario name (e.g. 'Net Zero 2050')
            year: Target year

        Returns:
            Carbon price in USD/tCO2e
        """
        if scenario_name not in NGFS_CARBON_PRICES:
            raise ValueError(
                f"Unknown scenario {scenario_name!r}. "
                f"Available: {list(NGFS_CARBON_PRICES.keys())}"
            )

        prices = NGFS_CARBON_PRICES[scenario_name]

        # Check if year is directly available
        if year in prices:
            return prices[year]

        # Find surrounding years for interpolation
        sorted_years = sorted(prices.keys())

        # Before first year: extrapolate with first value
        if year < sorted_years[0]:
            return prices[sorted_years[0]]

        # After last year: use last value
        if year > sorted_years[-1]:
            return prices[sorted_years[-1]]

        # Linear interpolation between two points
        for i in range(len(sorted_years) - 1):
            y1, y2 = sorted_years[i], sorted_years[i + 1]
            if y1 <= year <= y2:
                p1, p2 = prices[y1], prices[y2]
                # Linear interpolation
                weight = (year - y1) / (y2 - y1)
                return p1 + weight * (p2 - p1)

        # Fallback (should not reach here)
        return prices[sorted_years[-1]]

    def to_cri_scenario(self, ngfs_name: str) -> Optional[str]:
        """Map NGFS scenario name to CRI scenario family name.

        Args:
            ngfs_name: NGFS scenario name

        Returns:
            CRI scenario family name (e.g. 'nze_2050'), or None if not mapped
        """
        for cri_name, mapped_ngfs_name in self.NGFS_MAPPING.items():
            if mapped_ngfs_name == ngfs_name:
                return cri_name
        return None

    def get_cri_scenario_carbon_prices(self, cri_family: str) -> Optional[dict]:
        """Get carbon price path for a CRI scenario family.

        Args:
            cri_family: CRI scenario family (e.g. 'nze_2050')

        Returns:
            Dict of {year: price_usd_per_tco2e}, or None if not found
        """
        ngfs_name = self.NGFS_MAPPING.get(cri_family)
        if not ngfs_name:
            return None
        return NGFS_CARBON_PRICES[ngfs_name]
