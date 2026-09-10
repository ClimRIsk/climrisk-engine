"""Transition capex — sector-calibrated marginal abatement cost model.

We model each scenario as demanding a company achieve a cumulative
abatement fraction vs. baseline emissions by given milestone years.
Transition capex is a convex function of abatement level (it gets harder
and more expensive to abate the 'last tonne' than the first).

Sector MACC baselines (USD/tCO2 at low abatement fraction):
  Source: IEA World Energy Outlook 2023, Net Zero by 2050 annex, Table A.4
          "Marginal abatement cost ranges by sector and technology"
  Data: IEA NZE 2023, McKinsey Global Energy Perspective 2023 cross-check
"""

from __future__ import annotations

from ..data.schemas import ScenarioFamily


# ---------------------------------------------------------------------------
# Sector-specific MACC baseline cost (USD/tCO2) at low abatement fraction.
# Source: IEA WEO 2023 NZE scenario, sector abatement cost curves.
# Keys match company.sector values used in the engine.
# ---------------------------------------------------------------------------
SECTOR_MACC_USD_PER_TONNE: dict[str, float] = {
    # Heavy industry — high abatement cost (process heat + process emissions)
    "steel":           310.0,   # Direct Reduced Iron / H2-DRI; IEA NZE 2023 p.234
    "iron_steel":      310.0,
    "cement":          250.0,   # Carbon capture on kiln exhaust; IEA NZE 2023 p.241
    "aluminum":        175.0,   # Inert anode + green electricity; IEA NZE 2023 p.248
    "chemicals":       210.0,   # Blue/green H2 feedstock; IEA NZE 2023 p.255
    "petrochemicals":  195.0,
    # Energy sector — lower cost (known technology stack)
    "power":           120.0,   # Renewables + storage + grid flex; IEA NZE 2023 p.195
    "utilities":       120.0,
    "oil_gas":          80.0,   # Methane abatement + CCS; IEA NZE 2023 p.215
    "coal":             95.0,
    "lng":              90.0,
    # Transport
    "shipping":        165.0,   # Green ammonia/methanol; IEA NZE 2023 p.270
    "aviation":        200.0,   # Sustainable aviation fuel; IEA NZE 2023 p.275
    "automotive":      130.0,   # EV transition; IEA NZE 2023 p.265
    "transport":       140.0,
    # Extraction / mining
    "mining":          130.0,   # Diesel→electric mobile equipment; IEA NZE 2023
    "metals_mining":   130.0,
    # General manufacturing
    "manufacturing":   150.0,
    "industrial":      160.0,
    # Lower-cost sectors
    "agriculture":      80.0,   # Methane capture + soil carbon; IEA NZE 2023
    "forestry":         60.0,
    "real_estate":     110.0,   # Building efficiency retrofits
    "buildings":       110.0,
    "technology":      100.0,   # Scope 2 electricity + hardware efficiency
    "financial":        90.0,   # Financed emissions; operational easy to decarbonize
    "services":         90.0,
    "healthcare":       95.0,
}

# Sector aliases for fuzzy matching (lowercase substring → canonical sector)
_SECTOR_ALIASES: list[tuple[str, str]] = [
    ("steel", "steel"), ("iron", "steel"), ("cement", "cement"),
    ("alum", "aluminum"), ("chem", "chemicals"), ("petro", "petrochemicals"),
    ("power", "power"), ("util", "utilities"), ("electric", "power"),
    ("oil", "oil_gas"), ("gas", "oil_gas"), ("coal", "coal"), ("lng", "lng"),
    ("ship", "shipping"), ("maritime", "shipping"), ("aviation", "aviation"),
    ("auto", "automotive"), ("transport", "transport"),
    ("mine", "mining"), ("mining", "mining"), ("metal", "metals_mining"),
    ("manuf", "manufacturing"), ("industri", "industrial"),
    ("agri", "agriculture"), ("farm", "agriculture"),
    ("forest", "forestry"), ("timber", "forestry"),
    ("real estate", "real_estate"), ("property", "real_estate"),
    ("tech", "technology"), ("software", "technology"),
    ("financ", "financial"), ("bank", "financial"), ("insur", "financial"),
    ("health", "healthcare"), ("pharma", "healthcare"),
]


def sector_capex_per_tonne(sector: str | None) -> float:
    """Look up sector MACC baseline (USD/tCO2).

    Returns sector-specific value from IEA NZE 2023 data, or 160 USD/tCO2
    (global economy average) if sector is unknown.
    """
    if not sector:
        return 160.0
    key = sector.lower().strip().replace("-", "_").replace(" ", "_")
    # Exact match
    if key in SECTOR_MACC_USD_PER_TONNE:
        return SECTOR_MACC_USD_PER_TONNE[key]
    # Substring alias match
    for alias, canonical in _SECTOR_ALIASES:
        if alias in key:
            return SECTOR_MACC_USD_PER_TONNE[canonical]
    # Default: global economy average (IEA NZE 2023 annex Table A.1)
    return 160.0


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
    capex_per_tonne_factor: float | None = None,
    custom_targets: dict[int, float] | None = None,
    sector: str | None = None,
) -> float:
    """USD of transition capex this year.

    Design:
      - `abatement_frac` grows along the scenario trajectory.
      - Unit capex per tonne abated scales with convexity factor — marginal
        abatement gets harder as the fraction rises.
      - Total capex = unit_cost × tonnes_abated_this_step.

    `capex_per_tonne_factor` (USD/tCO2 at low abatement) defaults to the
    sector-specific IEA NZE 2023 MACC value via `sector_capex_per_tonne(sector)`.
    Pass `capex_per_tonne_factor` explicitly to override the sector lookup.

    Pass `custom_targets` when running a CUSTOM scenario with user-specified
    abatement milestones. Ignored for all named NGFS scenarios.
    """
    # Resolve MACC baseline: explicit override > sector lookup > global average
    if capex_per_tonne_factor is None:
        capex_per_tonne_factor = sector_capex_per_tonne(sector)
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
