"""Transition-risk driver derivation.

For a given (scenario, year), return the carbon price and the per-commodity
demand / price adjustment an asset or segment should experience.

This module is deliberately thin — it reads the Scenario, interpolates, and
returns structured drivers. No company logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..data.schemas import Commodity, Scenario


@dataclass
class TransitionDrivers:
    """Transition drivers for one (scenario, year)."""

    year: int
    carbon_price: float                                    # USD / tCO2e
    commodity_demand_index: dict[Commodity, float]         # 2026 = 100
    commodity_price: dict[Commodity, float]                # USD / unit
    commodity_elasticity: dict[Commodity, float]


def resolve(scenario: Scenario, year: int, region: str = "global") -> TransitionDrivers:
    """Resolve transition drivers for a specific year under a scenario."""
    carbon = scenario.carbon_price(year, region=region)

    demand: dict[Commodity, float] = {}
    price: dict[Commodity, float] = {}
    elast: dict[Commodity, float] = {}

    for curve in scenario.commodity_curves:
        demand[curve.commodity] = curve.demand_index.get(year, 100.0)
        price[curve.commodity] = curve.price_path.get(year, 0.0)
        elast[curve.commodity] = curve.price_elasticity

    return TransitionDrivers(
        year=year,
        carbon_price=carbon,
        commodity_demand_index=demand,
        commodity_price=price,
        commodity_elasticity=elast,
    )


def demand_shift(drivers: TransitionDrivers, commodity: Commodity) -> float:
    """Fractional change in demand vs. base year (index=100).

    e.g., if commodity demand index is 135 at year Y, this returns 0.35.
    """
    idx = drivers.commodity_demand_index.get(commodity, 100.0)
    return (idx - 100.0) / 100.0


def price_at(drivers: TransitionDrivers, commodity: Commodity) -> Optional[float]:
    return drivers.commodity_price.get(commodity)
