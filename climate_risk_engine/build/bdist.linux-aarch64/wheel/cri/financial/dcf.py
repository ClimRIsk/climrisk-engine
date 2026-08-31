"""Climate-adjusted DCF valuation.

Explicit horizon + Gordon-growth terminal value. WACC is climate-adjusted
per-scenario. No hidden constants — every lever is a named parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.schemas import Company, YearResult
from .metrics import ClimateWACC


@dataclass
class DCFOutput:
    npv_fcf: float
    terminal_value: float
    enterprise_value: float
    equity_value: float
    implied_share_price: float
    wacc_used: float


def value(
    company: Company,
    years: list[YearResult],
    wacc: ClimateWACC,
    terminal_growth: float = 0.015,
) -> DCFOutput:
    if not years:
        raise ValueError("No year results provided for DCF.")

    wacc_total = wacc.total
    if wacc_total <= terminal_growth:
        raise ValueError(
            f"WACC ({wacc_total:.3%}) must exceed terminal growth "
            f"({terminal_growth:.3%}) for a Gordon terminal."
        )

    # Discount each year's FCF to t=0 (start of horizon - 1 = 2025)
    start_year = years[0].year
    npv = 0.0
    for y in years:
        t = y.year - start_year + 1
        npv += y.fcf / (1.0 + wacc_total) ** t

    # Terminal value on the final year's FCF
    final_fcf = years[-1].fcf
    tv = final_fcf * (1.0 + terminal_growth) / (wacc_total - terminal_growth)
    n = years[-1].year - start_year + 1
    tv_discounted = tv / (1.0 + wacc_total) ** n

    enterprise = npv + tv_discounted
    equity = enterprise - company.financials.net_debt
    shares = max(company.financials.shares_outstanding, 1e-9)
    implied = equity / shares

    return DCFOutput(
        npv_fcf=npv,
        terminal_value=tv_discounted,
        enterprise_value=enterprise,
        equity_value=equity,
        implied_share_price=implied,
        wacc_used=wacc_total,
    )
