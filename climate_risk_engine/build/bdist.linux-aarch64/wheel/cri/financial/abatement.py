"""Transition capex — a simple, transparent abatement model.

We model each scenario as demanding a company achieve a cumulative
abatement fraction vs. baseline emissions by given milestone years.
Transition capex is a convex function of abatement level (it gets harder
and more expensive to abate the 'last tonne' than the first).

This is deliberately simple for v0.1. In Phase 2 we'll replace it with
sector-specific marginal abatement cost curves (MACC) sourced from IEA /
McKinsey style datasets.
"""

from __future__ import annotations

from ..data.schemas import ScenarioFamily


# Milestone year -> required cumulative abatement vs. baseline emissions
_SCENARIO_TARGETS: dict[ScenarioFamily, dict[int, float]] = {
    ScenarioFamily.NZE_2050: {2030: 0.40, 2040: 0.75, 2050: 0.95},
    ScenarioFamily.BELOW_2C_ORDERLY: {2030: 0.25, 2040: 0.55, 2050: 0.80},
    ScenarioFamily.DELAYED_TRANSITION: {2030: 0.05, 2040: 0.50, 2050: 0.85},
    ScenarioFamily.CURRENT_POLICIES: {2030: 0.05, 2040: 0.15, 2050: 0.25},
    ScenarioFamily.CUSTOM: {2030: 0.20, 2040: 0.50, 2050: 0.75},  # default fallback only
}


def required_abatement_fraction(
    family: ScenarioFamily,
    year: int,
    custom_targets: dict[int, float] | None = None,
) -> float:
    """Linear interpolation between scenario milestones.

    If `custom_targets` is provided AND family is CUSTOM, uses the supplied
    targets dict instead of the hard-coded _SCENARIO_TARGETS lookup.
    Format: {milestone_year: cumulative_abatement_fraction}
    e.g. {2030: 0.30, 2040: 0.65, 2050: 0.90}
    """
    if custom_targets and family == ScenarioFamily.CUSTOM:
        targets = custom_targets
    else:
        targets = _SCENARIO_TARGETS.get(family, _SCENARIO_TARGETS[ScenarioFamily.CUSTOM])
    years = sorted(targets)
    if year <= years[0]:
        # Pre-milestone ramp from 0 at 2025 to target at first milestone
        anchor = years[0]
        return max(0.0, targets[anchor] * (year - 2025) / (anchor - 2025))
    if year >= years[-1]:
        return targets[years[-1]]
    for i in range(len(years) - 1):
        y0, y1 = years[i], years[i + 1]
        if y0 <= year <= y1:
            w = (year - y0) / (y1 - y0)
            return targets[y0] + w * (targets[y1] - targets[y0])
    return targets[years[-1]]


def transition_capex(
    scenario_family: ScenarioFamily,
    year: int,
    baseline_emissions_tCO2: float,
    carbon_price: float,
    convexity: float = 1.8,
    capex_per_tonne_factor: float = 180.0,
    custom_targets: dict[int, float] | None = None,
) -> float:
    """USD of transition capex this year.

    Design:
      - `abatement_frac` grows along the scenario trajectory.
      - Unit capex per tonne abated scales with `carbon_price * convexity`
        factor — marginal abatement gets harder as the fraction rises.
      - Total capex = unit_cost × tonnes_abated_this_step.

    The `capex_per_tonne_factor` (USD/tCO2 at low abatement) is roughly
    grounded in IEA NZE estimates; will be sector-calibrated in Phase 2.

    Pass `custom_targets` when running a CUSTOM scenario with user-specified
    abatement milestones. Ignored for all named NGFS scenarios.
    """
    frac_now = required_abatement_fraction(scenario_family, year, custom_targets)
    frac_prev = required_abatement_fraction(scenario_family, year - 1, custom_targets)
    delta_frac = max(0.0, frac_now - frac_prev)

    tonnes_abated_this_year = baseline_emissions_tCO2 * delta_frac

    # Convex cost ramp: cost per tonne grows with current abatement level.
    # Normalized so that at frac=0 -> capex_per_tonne_factor, at frac=1
    # -> capex_per_tonne_factor * (1 + convexity).
    avg_frac = 0.5 * (frac_now + frac_prev)
    unit_cost = capex_per_tonne_factor * (1 + convexity * avg_frac)

    # Carbon-price-indexed uplift (companies in high-price worlds face
    # tighter supply chains for abatement tech)
    price_uplift = 1.0 + max(0.0, carbon_price - 50.0) / 500.0

    return tonnes_abated_this_year * unit_cost * price_uplift


def coverage_after_abatement(
    scenario_family: ScenarioFamily,
    year: int,
    custom_targets: dict[int, float] | None = None,
) -> float:
    """Fraction of baseline emissions STILL emitted after abatement.

    Pass `custom_targets` when running a CUSTOM scenario with user-specified
    abatement milestones. Ignored for all named NGFS scenarios.
    """
    return max(0.0, 1.0 - required_abatement_fraction(scenario_family, year, custom_targets))
