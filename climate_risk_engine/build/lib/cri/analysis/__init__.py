"""CRI analysis layer — post-run analytics on top of engine outputs.

Two modules:

sensitivity
    Carbon price stress test. Sweeps carbon price across a user-defined
    range (e.g. $10–$300/tCO2e) and shows how EV haircut, WACC uplift, and
    annual carbon liability respond. Single-variable sensitivity designed for
    analyst use and regulatory stress-test disclosure.

portfolio
    Portfolio-level climate risk aggregation. Takes multiple company
    FullRunResults with portfolio weights and produces weighted average CRI,
    pairwise loss correlation matrix, per-scenario VaR, and concentration
    metrics (HHI, diversification ratio).

Usage
-----
from cri.analysis.sensitivity import run_carbon_stress_test, plot_sensitivity
from cri.analysis.portfolio import PortfolioHolding, PortfolioAnalyser, plot_portfolio
"""

from .sensitivity import (
    CarbonSensitivityResult,
    PricePoint,
    run_carbon_stress_test,
    plot_sensitivity,
    DEFAULT_PRICES,
)

from .portfolio import (
    PortfolioHolding,
    PortfolioResult,
    ScenarioPortfolioMetrics,
    PortfolioAnalyser,
    plot_portfolio,
)

__all__ = [
    # sensitivity
    "CarbonSensitivityResult",
    "PricePoint",
    "run_carbon_stress_test",
    "plot_sensitivity",
    "DEFAULT_PRICES",
    # portfolio
    "PortfolioHolding",
    "PortfolioResult",
    "ScenarioPortfolioMetrics",
    "PortfolioAnalyser",
    "plot_portfolio",
]
