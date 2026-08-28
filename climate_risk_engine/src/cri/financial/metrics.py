"""Financial-layer computations: EBITDA, FCF, climate-adjusted WACC."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.schemas import Company, Scenario, YearResult
from ..operations.company import OperationalYear


@dataclass
class ClimateWACC:
    base: float
    scenario_premium: float
    exposure_premium: float

    @property
    def total(self) -> float:
        return self.base + self.scenario_premium + self.exposure_premium


def climate_adjusted_wacc(company: Company, scenario: Scenario) -> ClimateWACC:
    """Base WACC + scenario risk premium + company-specific exposure premium.

    The exposure premium is a simple, transparent function of the company's
    sector weights. In Phase 2 we'll calibrate empirically to CDS / bond
    spreads where available.
    """
    scenario_premium = scenario.risk_premium_bps / 10_000.0  # bps -> decimal
    exposure_premium = 0.002 * company.exposure_weight + 0.003 * company.transition_weight
    return ClimateWACC(
        base=company.financials.wacc_base,
        scenario_premium=scenario_premium,
        exposure_premium=exposure_premium,
    )


def compute_year(
    company: Company,
    op: OperationalYear,
    prev_revenue: float | None = None,
) -> YearResult:
    """Turn an OperationalYear into a fully-populated YearResult.

    Key choices (and departures from V2):
      - EBITDA is *derived* (revenue - opex - carbon - physical), not margin×revenue.
      - Capex is split into maintenance (a function of baseline capex) +
        adaptation (from physical module) + transition (MACC-based, live in v0.3).
      - Tax is applied to EBIT, not to (EBITDA - carbon).
      - Working-capital change: 10% of ΔRevenue (requires prev_revenue from caller).
        Pass prev_revenue=None for the first year (wc_change=0 assumed).
    """
    fin = company.financials

    ebitda = op.revenue - op.opex - op.carbon_cost - op.physical_loss_cost

    # Depreciation: baseline plus straight-line on new adaptation capex (15y life)
    baseline_da = 0.6 * fin.capex * fin.maintenance_capex_share
    adapt_da = op.adaptation_capex / 15.0
    da = baseline_da + adapt_da

    ebit = ebitda - da
    nopat = ebit * (1.0 - fin.tax_rate)

    # Capex split
    maintenance_capex = fin.capex * fin.maintenance_capex_share
    adaptation_capex = op.adaptation_capex
    transition_capex = op.transition_capex

    # Working-capital change: 10% of revenue growth/decline.
    # Positive ΔRev → WC investment (cash outflow); negative → WC release (cash inflow).
    if prev_revenue is not None:
        wc_change = 0.10 * (op.revenue - prev_revenue)
    else:
        wc_change = 0.0

    fcf = (
        nopat
        + da
        - maintenance_capex
        - adaptation_capex
        - transition_capex
        - wc_change
    )

    return YearResult(
        year=op.year,
        revenue=op.revenue,
        opex=op.opex,
        carbon_cost=op.carbon_cost,
        physical_loss_cost=op.physical_loss_cost,
        ebitda=ebitda,
        da=da,
        ebit=ebit,
        nopat=nopat,
        transition_capex=transition_capex,
        adaptation_capex=adaptation_capex,
        maintenance_capex=maintenance_capex,
        working_capital_change=wc_change,
        fcf=fcf,
        revenue_by_commodity=op.revenue_by_commodity,
        emissions_by_scope={
            "scope_1": op.emissions_scope1,
            "scope_2": op.emissions_scope2,
            "scope_3": op.emissions_scope3,
        },
        physical_loss_by_hazard=op.physical_loss_by_hazard,
        stranded_writedown=op.stranded_writedown,
        stranded_assets=op.stranded_assets,
    )
