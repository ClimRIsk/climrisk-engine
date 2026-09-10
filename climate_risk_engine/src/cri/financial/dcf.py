"""Climate-adjusted DCF valuation.

Explicit horizon + regime-based terminal value. WACC is climate-adjusted
per-scenario. No hidden constants — every lever is a named parameter.

Terminal value regime dispatch (replaces plain Gordon growth perpetuity):
  1. final_fcf ≤ 0           → 0   (no going-concern value; limited-liability floor)
  2. FCF declining > 10%/yr  → finite declining annuity to 0 over `phase_out_years`
  3. otherwise               → Gordon growth, capped at 25× FCF

Regime 1 fixes the Gordon-growth failure mode where a negative or near-zero
FCF_final in a stressed carbon-price scenario produced a negative or near-zero
perpetuity, understating residual going-concern value (or misleadingly negative EV).
Regime 2 handles orderly wind-down of fossil-fuel assets under transition scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    tv_regime: str = field(default="gordon")   # "gordon" | "phase_out" | "zero"


def _terminal_value_undiscounted(
    fcf_series: list[float],
    wacc: float,
    terminal_growth: float = 0.015,
    phase_out_years: int = 10,
    gordon_cap_multiple: float = 25.0,
) -> tuple[float, str]:
    """Regime-based terminal value (undiscounted sum at the horizon year).

    Returns (tv_undiscounted, regime_label).
    """
    if not fcf_series:
        return 0.0, "zero"

    final_fcf = fcf_series[-1]

    # ── Regime 1: negative or zero terminal FCF ──────────────────────────────
    if final_fcf <= 0.0:
        return 0.0, "zero"

    # ── Detect FCF trend (2-year CAGR using the last 3 values) ───────────────
    cagr = 0.0
    if len(fcf_series) >= 3:
        prev = fcf_series[-3]
        if prev > 1e-9:
            cagr = (final_fcf / prev) ** 0.5 - 1.0  # annualised over 2 years

    # ── Regime 2: steeply declining FCF — finite phase-out annuity ───────────
    if cagr < -0.10:
        # FCF assumed to decline linearly from final_fcf → 0 over phase_out_years.
        # Discounted at WACC (no growth — asset is in run-off).
        tv = 0.0
        for k in range(1, phase_out_years + 1):
            f_k = final_fcf * (1.0 - k / phase_out_years)
            if f_k > 0.0:
                tv += f_k / (1.0 + wacc) ** k
        return tv, "phase_out"

    # ── Regime 3: Gordon growth, capped ──────────────────────────────────────
    if wacc <= terminal_growth:
        # WACC constraint violated; fall back to flat finite annuity (no growth)
        tv = sum(final_fcf / (1.0 + wacc) ** k for k in range(1, phase_out_years + 1))
        return tv, "phase_out"

    gordon_tv = final_fcf * (1.0 + terminal_growth) / (wacc - terminal_growth)
    tv = min(gordon_tv, final_fcf * gordon_cap_multiple)
    return tv, "gordon"


def value(
    company: Company,
    years: list[YearResult],
    wacc: ClimateWACC,
    terminal_growth: float = 0.015,
    phase_out_years: int = 10,
) -> DCFOutput:
    if not years:
        raise ValueError("No year results provided for DCF.")

    wacc_total = wacc.total

    # Discount each year's FCF to t=0 (start of horizon - 1 = 2025)
    start_year = years[0].year
    npv = 0.0
    for y in years:
        t = y.year - start_year + 1
        npv += y.fcf / (1.0 + wacc_total) ** t

    # Terminal value — regime-based
    fcf_series = [y.fcf for y in years]
    tv_raw, tv_regime = _terminal_value_undiscounted(
        fcf_series,
        wacc=wacc_total,
        terminal_growth=terminal_growth,
        phase_out_years=phase_out_years,
    )
    # Discount TV back to t=0 at the final projection year
    n = years[-1].year - start_year + 1
    tv_discounted = tv_raw / (1.0 + wacc_total) ** n

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
        tv_regime=tv_regime,
    )
