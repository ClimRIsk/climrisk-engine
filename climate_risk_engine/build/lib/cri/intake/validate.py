"""Validation logic for Company data quality."""

from cri.data.schemas import Company


def validate_company(c: Company) -> list[str]:
    """
    Return a list of warning strings for any suspicious values in a Company.

    Checks for:
    - Energy cost share exceeding 0.8
    - Emissions intensities being negative or unusually high
    - Production <= 0
    - Carrying value <= 0
    - Financial metrics being negative or unreasonable
    """
    warnings = []

    # Validate financials
    if c.financials.revenue < 0:
        warnings.append(f"{c.id}: revenue is negative (${c.financials.revenue:.0f}M)")

    if c.financials.ebitda < 0:
        warnings.append(f"{c.id}: EBITDA is negative (${c.financials.ebitda:.0f}M)")

    if c.financials.shares_outstanding <= 0:
        warnings.append(f"{c.id}: shares_outstanding must be > 0")

    if c.financials.market_cap is not None and c.financials.market_cap < 0:
        warnings.append(f"{c.id}: market_cap is negative")

    # Validate assets
    for asset in c.assets:
        if asset.baseline_production <= 0:
            warnings.append(f"{c.id}/{asset.id}: baseline_production must be > 0")

        if asset.carrying_value < 0:
            warnings.append(
                f"{c.id}/{asset.id}: carrying_value is negative (${asset.carrying_value:.0f}M)"
            )

        if asset.energy_cost_share < 0 or asset.energy_cost_share > 1:
            warnings.append(
                f"{c.id}/{asset.id}: energy_cost_share must be in [0, 1], got {asset.energy_cost_share}"
            )

        if asset.energy_cost_share > 0.8:
            warnings.append(
                f"{c.id}/{asset.id}: energy_cost_share is very high ({asset.energy_cost_share:.1%})"
            )

        # Check emissions
        if asset.emissions.scope1_intensity < 0:
            warnings.append(
                f"{c.id}/{asset.id}: scope1_intensity is negative ({asset.emissions.scope1_intensity})"
            )

        if asset.emissions.scope2_intensity < 0:
            warnings.append(
                f"{c.id}/{asset.id}: scope2_intensity is negative ({asset.emissions.scope2_intensity})"
            )

        if asset.emissions.scope3_intensity < 0:
            warnings.append(
                f"{c.id}/{asset.id}: scope3_intensity is negative ({asset.emissions.scope3_intensity})"
            )

        # High emission intensities might be errors
        if asset.emissions.scope1_intensity > 100:
            warnings.append(
                f"{c.id}/{asset.id}: scope1_intensity seems very high ({asset.emissions.scope1_intensity} tCO2e/unit)"
            )

        if asset.emissions.scope2_intensity > 100:
            warnings.append(
                f"{c.id}/{asset.id}: scope2_intensity seems very high ({asset.emissions.scope2_intensity} tCO2e/unit)"
            )

        if asset.emissions.scope3_intensity > 100:
            warnings.append(
                f"{c.id}/{asset.id}: scope3_intensity seems very high ({asset.emissions.scope3_intensity} tCO2e/unit)"
            )

        # Check coverage fractions
        if asset.emissions.carbon_price_coverage < 0 or asset.emissions.carbon_price_coverage > 1:
            warnings.append(
                f"{c.id}/{asset.id}: carbon_price_coverage must be in [0, 1], got {asset.emissions.carbon_price_coverage}"
            )

        if asset.emissions.free_allocation < 0 or asset.emissions.free_allocation > 1:
            warnings.append(
                f"{c.id}/{asset.id}: free_allocation must be in [0, 1], got {asset.emissions.free_allocation}"
            )

        if asset.emissions.free_allocation > asset.emissions.carbon_price_coverage:
            warnings.append(
                f"{c.id}/{asset.id}: free_allocation ({asset.emissions.free_allocation}) "
                f"should not exceed carbon_price_coverage ({asset.emissions.carbon_price_coverage})"
            )

        # Validate remaining life
        if asset.remaining_life_years is not None and asset.remaining_life_years < 0:
            warnings.append(
                f"{c.id}/{asset.id}: remaining_life_years is negative ({asset.remaining_life_years})"
            )

        if asset.baseline_unit_cost < 0:
            warnings.append(
                f"{c.id}/{asset.id}: baseline_unit_cost is negative (${asset.baseline_unit_cost})"
            )

    return warnings
