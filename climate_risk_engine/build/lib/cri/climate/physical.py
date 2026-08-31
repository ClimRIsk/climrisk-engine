"""Physical-risk resolution: hazard → expected production loss, per asset, per year.

This module is a thin orchestrator over a `PhysicalRiskProvider`
(see `providers.py`). That abstraction is what lets us swap between
internal scenario data, Continuuiti's REST API, Climate X, WRI
Aqueduct, or any future provider — without the rest of the engine
caring.

Aggregation rule (used by the internal provider): treat each hazard as
an independent annual occurrence with probability p_i; joint
no-disruption probability is Π(1 - p_i), so expected loss = 1 - Π(1 - p_i).
This caps at 100% and handles overlapping hazards sensibly.
"""

from __future__ import annotations

from typing import Optional

from ..data.schemas import Asset, Scenario
from .providers import HazardQuery, PhysicalRiskProvider, ScenarioHazardProvider


def expected_loss_fraction(
    scenario: Scenario,
    asset: Asset,
    year: int,
    provider: Optional[PhysicalRiskProvider] = None,
) -> float:
    """Expected fractional production loss for `asset` in `year`.

    Defaults to the internal `ScenarioHazardProvider`, which reads the
    hazards embedded in the given Scenario. Pass an alternative provider
    to plug in Continuuiti / Climate X / etc.
    """
    prov = provider or ScenarioHazardProvider(scenario)
    result = prov.resolve(
        HazardQuery(asset=asset, year=year, scenario_family=scenario.family)
    )
    return result.expected_loss_fraction


def adaptation_capex(
    scenario: Scenario,
    asset: Asset,
    year: int,
    per_loss_spend: float = 0.5,
) -> float:
    """Adaptation capex: fraction of the asset's hazard-weighted production
    exposure scaled by unit cost. Models the capital required to harden an
    asset against its expected physical loss (flood barriers, cooling systems,
    water recycling, fire breaks). Per-loss spend defaults to 50% of exposed
    value — calibrated to industry adaptation cost benchmarks.
    """
    loss = expected_loss_fraction(scenario, asset, year)
    exposed_value = asset.baseline_production * asset.baseline_unit_cost * loss
    return per_loss_spend * exposed_value
