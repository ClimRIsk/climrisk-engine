"""Carbon price stress test — single-variable sensitivity analysis.

Sweeps carbon price (USD/tCO2e) across a user-defined range while holding
all other scenario parameters constant at the CURRENT_POLICIES base. Every
point in the sweep reveals exactly how the company's carbon liability,
climate-adjusted WACC, and enterprise value respond to that single variable.

This is the direct answer to a credit committee's question:
  "If the EU ETS hits €150/t next year, what happens to this borrower's EV?"

Methodology
-----------
- Base scenario: CURRENT_POLICIES (commodity curves, hazard paths, SSP3-7.0).
- Variable: global carbon price (flat rate, time-invariant across the horizon).
  A flat path isolates the carbon price level effect from the escalation path
  effect — the latter is captured by the full NGFS scenario runs.
- For each price point we create a custom Scenario that is identical to CP
  except for the carbon_prices list, then call engine.run() on the company.
- EV haircut = baseline_ev − ev_at_price_point, where baseline_ev is the EV
  under CURRENT_POLICIES at the stated market_carbon_price.
- WACC uplift = wacc_used − company.financials.wacc_base, expressed in bps.

Key output: CarbonSensitivityResult with a list of PricePoint dataclasses
plus helpers summary() and a plot helper (plot_sensitivity).

Usage
-----
from cri.analysis.sensitivity import run_carbon_stress_test, plot_sensitivity

result = run_carbon_stress_test(company)
print(result.summary())

import matplotlib
matplotlib.use('Agg')
fig = plot_sensitivity(result)
fig.savefig('carbon_stress.png', dpi=180, bbox_inches='tight')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..data.schemas import (
    CarbonPricePath,
    Company,
    Scenario,
    ScenarioFamily,
)
from ..engine.orchestrator import run
from .. import scenarios as _base


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

# 14 price points: granular at low end (policy range), coarse at extreme stress
DEFAULT_PRICES: list[float] = [
    10, 20, 30, 40, 50, 60, 75, 90, 110, 130, 160, 200, 250, 300
]

# Market carbon price used as the baseline (CP scenario reference point).
# Corresponds to approximate current EU ETS / global carbon price circa 2026.
DEFAULT_MARKET_PRICE: float = 30.0


# ---------------------------------------------------------------------------
# Output data types
# ---------------------------------------------------------------------------

@dataclass
class PricePoint:
    """Results for one carbon price in the sweep."""

    carbon_price: float          # USD/tCO2e — flat rate applied to all years

    # Carbon liability
    total_carbon_cost_npv: float   # NPV of all annual carbon costs over horizon (USD)
    carbon_cost_yr1: float         # annual carbon cost in first year (USD)
    carbon_cost_2030: float        # annual carbon cost in 2030 (USD)
    carbon_cost_2040: float        # annual carbon cost in 2040 (USD)
    carbon_cost_2050: float        # annual carbon cost in 2050 (USD)

    # Valuation impact
    enterprise_value: float        # DCF EV under this carbon price (USD)
    ev_haircut: float              # EV loss vs CP baseline at market_carbon_price (USD)
    ev_haircut_pct: float          # ev_haircut as fraction of baseline EV (e.g. 0.12)

    # Cost of capital
    wacc_used: float               # climate-adjusted WACC (decimal, e.g. 0.102)
    wacc_uplift_bps: float         # uplift vs company.financials.wacc_base (bps)

    # EBITDA impact
    ebitda_compression_2030_pct: Optional[float] = None  # vs baseline, may be None


@dataclass
class CarbonSensitivityResult:
    """Full carbon price stress test output for one company.

    Attributes
    ----------
    company_id          : Company.id
    company_name        : Company.name
    sector              : Company.sector
    base_wacc           : Company.financials.wacc_base (decimal)
    baseline_ev         : EV under CURRENT_POLICIES at market_carbon_price (USD)
    market_carbon_price : Carbon price used as baseline (default $30/t)
    points              : Ordered list of PricePoint, one per sweep step
    """

    company_id: str
    company_name: str
    sector: str
    base_wacc: float
    baseline_ev: float
    market_carbon_price: float
    points: list[PricePoint] = field(default_factory=list)

    # ── Convenience accessors ─────────────────────────────────────────────

    @property
    def prices(self) -> list[float]:
        return [p.carbon_price for p in self.points]

    @property
    def ev_haircuts_usd(self) -> list[float]:
        """EV haircut in USD for each price point."""
        return [p.ev_haircut for p in self.points]

    @property
    def ev_haircuts_pct(self) -> list[float]:
        """EV haircut as percentage (0–100) for each price point."""
        return [p.ev_haircut_pct * 100 for p in self.points]

    @property
    def wacc_bps(self) -> list[float]:
        """WACC uplift in basis points for each price point."""
        return [p.wacc_uplift_bps for p in self.points]

    @property
    def carbon_costs_2030_m(self) -> list[float]:
        """Annual carbon cost in 2030 in $M for each price point."""
        return [p.carbon_cost_2030 for p in self.points]  # engine $M

    @property
    def carbon_costs_2050_m(self) -> list[float]:
        """Annual carbon cost in 2050 in $M for each price point."""
        return [p.carbon_cost_2050 for p in self.points]  # engine $M

    def at_price(self, price: float) -> Optional[PricePoint]:
        """Return the PricePoint closest to the requested carbon price."""
        if not self.points:
            return None
        return min(self.points, key=lambda p: abs(p.carbon_price - price))

    def summary(self) -> str:
        """Print-friendly sensitivity table."""
        lines = [
            f"Carbon Price Sensitivity  —  {self.company_name}  [{self.sector}]",
            f"Baseline EV (CP @ ${self.market_carbon_price:.0f}/t): "
            f"${self.baseline_ev / 1e3:.2f}B   "
            f"Base WACC: {self.base_wacc * 100:.1f}%",
            "",
            f"{'Price ($/t)':>12}  {'Carbon Cost 2030':>18}  "
            f"{'WACC (bps)':>12}  {'EV Haircut ($B)':>16}  {'Haircut (%)':>12}",
            "─" * 76,
        ]
        for p in self.points:
            lines.append(
                f"{p.carbon_price:>12.0f}  "
                f"${p.carbon_cost_2030:>15.0f}M  "
                f"{p.wacc_uplift_bps:>12.1f}  "
                f"{p.ev_haircut / 1e3:>16.2f}  "
                f"{p.ev_haircut_pct * 100:>12.1f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------

def _make_flat_price_scenario(base: Scenario, flat_price: float, idx: int) -> Scenario:
    """Clone base scenario with a single flat carbon price for all years."""
    years = list(range(base.horizon[0], base.horizon[1] + 1))
    flat_path = {y: flat_price for y in years}

    return Scenario(
        id=f"carbon_stress_{idx}",
        name=f"Carbon Stress ${flat_price:.0f}/t",
        family=ScenarioFamily.CUSTOM,
        horizon=base.horizon,
        description=(
            f"Single-variable stress test: flat carbon price ${flat_price:.0f}/tCO2e, "
            f"all other parameters from CURRENT_POLICIES base."
        ),
        version=base.version,
        carbon_prices=[CarbonPricePath(region="global", path=flat_path)],
        commodity_curves=list(base.commodity_curves),
        hazards=list(base.hazards),
        risk_premium_bps=base.risk_premium_bps,
    )


# ---------------------------------------------------------------------------
# Year-result helpers
# ---------------------------------------------------------------------------

def _year_carbon_cost(years, target_year: int) -> float:
    """Return carbon_cost for a specific year, or 0 if not present."""
    for yr in years:
        if yr.year == target_year:
            return yr.carbon_cost
    return 0.0


def _npv_carbon_cost(years, wacc: float, start_year: int) -> float:
    """NPV of the entire carbon cost trajectory at the given WACC."""
    total = 0.0
    for yr in years:
        t = yr.year - start_year + 1
        total += yr.carbon_cost / (1.0 + wacc) ** t
    return total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_carbon_stress_test(
    company: Company,
    prices: list[float] | None = None,
    market_carbon_price: float = DEFAULT_MARKET_PRICE,
    model_version: str = "0.4.0",
) -> CarbonSensitivityResult:
    """Run a carbon price sweep and return the full sensitivity result.

    Parameters
    ----------
    company              : Fully-populated Company object.
    prices               : Carbon prices to sweep (USD/tCO2e). Defaults to
                           DEFAULT_PRICES = [10, 20, 30, …, 300].
    market_carbon_price  : Reference price for the baseline EV calculation.
                           Default 30 USD/t (approx. CP 2026 anchor).
    model_version        : Engine version string for provenance.

    Returns
    -------
    CarbonSensitivityResult with one PricePoint per price in the sweep,
    plus baseline_ev set to the EV at market_carbon_price.
    """
    if prices is None:
        prices = DEFAULT_PRICES

    base_scenario = _base.CURRENT_POLICIES
    start_year = base_scenario.horizon[0]

    # Step 1: compute baseline EV at market carbon price
    baseline_scenario = _make_flat_price_scenario(base_scenario, market_carbon_price, 0)
    baseline_run = run(company, baseline_scenario, model_version=model_version)
    baseline_ev = baseline_run.enterprise_value

    result = CarbonSensitivityResult(
        company_id=company.id,
        company_name=company.name,
        sector=company.sector,
        base_wacc=company.financials.wacc_base,
        baseline_ev=baseline_ev,
        market_carbon_price=market_carbon_price,
    )

    # Step 2: sweep
    for idx, price in enumerate(sorted(prices), start=1):
        scenario = _make_flat_price_scenario(base_scenario, price, idx)
        r = run(company, scenario, model_version=model_version)

        wacc_uplift_bps = (r.wacc_used - company.financials.wacc_base) * 10_000
        ev_haircut = baseline_ev - r.enterprise_value
        ev_haircut_pct = ev_haircut / baseline_ev if baseline_ev != 0 else 0.0

        years = r.years
        carbon_npv = _npv_carbon_cost(years, r.wacc_used, start_year)
        cc_yr1 = years[0].carbon_cost if years else 0.0

        point = PricePoint(
            carbon_price=price,
            total_carbon_cost_npv=carbon_npv,
            carbon_cost_yr1=cc_yr1,
            carbon_cost_2030=_year_carbon_cost(years, 2030),
            carbon_cost_2040=_year_carbon_cost(years, 2040),
            carbon_cost_2050=_year_carbon_cost(years, 2050),
            enterprise_value=r.enterprise_value,
            ev_haircut=ev_haircut,
            ev_haircut_pct=ev_haircut_pct,
            wacc_used=r.wacc_used,
            wacc_uplift_bps=wacc_uplift_bps,
            ebitda_compression_2030_pct=r.ebitda_compression_2030_pct,
        )
        result.points.append(point)

    return result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_sensitivity(
    result: CarbonSensitivityResult,
    output_path: Optional[str] = None,
    dpi: int = 180,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Dark-theme 4-panel sensitivity dashboard.

    Panels
    ------
    1 (top-left)   : Annual carbon cost ($M) in 2030 and 2050 vs carbon price.
                     Reveals the operating leverage to carbon price.
    2 (top-right)  : WACC uplift (bps) vs carbon price.
                     Key input for cost-of-equity and credit spread models.
    3 (bottom-left): EV haircut ($B) vs carbon price — absolute impact.
    4 (bottom-right): EV haircut (%) at key regulatory anchor prices
                      ($30, $75, $130, $200, $250) as a horizontal bar chart.

    Parameters
    ----------
    result      : CarbonSensitivityResult from run_carbon_stress_test().
    output_path : If provided, saves the figure to this path.
    dpi         : Resolution for file output.

    Returns
    -------
    matplotlib.figure.Figure  (caller may savefig or display inline)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # ── Palette (matches ClimRisk brand) ────────────────────────────────────
    BG    = "#060f1e"
    CARD  = "#0b1f38"
    BORD  = "#1e3a5f"
    GREEN = "#22c55e"
    BLUE  = "#38bdf8"
    AMBER = "#f59e0b"
    RED   = "#ef4444"
    WHITE = "#f8fafc"
    SLATE = "#94a3b8"
    DIM   = "#475569"
    ORANGE= "#f97316"
    PURPLE= "#a855f7"

    prices = result.prices
    costs_2030 = result.carbon_costs_2030_m
    costs_2050 = result.carbon_costs_2050_m
    wacc_bps   = result.wacc_bps
    ev_haircut = [h / 1e3 for h in result.ev_haircuts_usd]
    ev_pct     = result.ev_haircuts_pct

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(CARD)
        ax.tick_params(colors=SLATE, labelsize=8)
        ax.xaxis.label.set_color(SLATE)
        ax.yaxis.label.set_color(SLATE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORD)

    # ── Header ───────────────────────────────────────────────────────────────
    fig.text(
        0.012, 0.97,
        f"ClimRisk  |  Carbon Price Stress Test  —  {result.company_name}  [{result.sector}]",
        fontsize=11, color=BLUE, fontweight="bold",
        fontfamily="monospace", va="top",
    )
    fig.text(
        0.012, 0.935,
        (f"Baseline EV: USD {result.baseline_ev/1e3:.1f}B  "
         f"Base WACC: {result.base_wacc*100:.1f}%  "
         f"Reference: USD {result.market_carbon_price:.0f}/tCO2e  "
         f"NGFS Phase 4 CURRENT POLICIES base"),
        fontsize=8, color=SLATE, fontfamily="monospace", va="top",
    )

    # ── Panel 1: Annual carbon cost vs price ─────────────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(prices, costs_2030, color=AMBER, linewidth=2.0, marker="o", markersize=4,
             label="2030")
    ax1.plot(prices, costs_2050, color=RED,   linewidth=2.0, marker="s", markersize=4,
             label="2050")
    ax1.fill_between(prices, costs_2030, costs_2050, color=RED, alpha=0.07)
    ax1.set_title("Annual Carbon Cost (USD M)", color=WHITE, fontsize=9, pad=6)
    ax1.set_xlabel("Carbon Price (USD/tCO2e)", fontsize=8)
    ax1.set_ylabel("USD M / year", fontsize=8)
    ax1.legend(fontsize=8, facecolor=CARD, edgecolor=BORD, labelcolor=SLATE)
    ax1.grid(True, color=BORD, linewidth=0.5, alpha=0.6)
    # Vertical reference lines at regulatory anchors
    for anchor, lbl, col in [(50, "USD50", GREEN), (130, "USD130", AMBER), (200, "USD200", RED)]:
        if min(prices) <= anchor <= max(prices):
            ax1.axvline(anchor, color=col, linewidth=0.8, linestyle="--", alpha=0.6)
            ax1.text(anchor + 2, ax1.get_ylim()[1] * 0.9, lbl,
                     color=col, fontsize=7, fontfamily="monospace")

    # ── Panel 2: WACC uplift vs price ────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.plot(prices, wacc_bps, color=PURPLE, linewidth=2.2, marker="o", markersize=4)
    ax2.fill_between(prices, 0, wacc_bps, color=PURPLE, alpha=0.12)
    ax2.set_title("WACC Uplift (basis points)", color=WHITE, fontsize=9, pad=6)
    ax2.set_xlabel("Carbon Price (USD/tCO2e)", fontsize=8)
    ax2.set_ylabel("bps above base WACC", fontsize=8)
    ax2.grid(True, color=BORD, linewidth=0.5, alpha=0.6)
    # Annotate current scenario anchors
    for anchor, lbl, col in [
        (50, "NZE 2030", GREEN), (130, "NZE 2050", BLUE), (200, "Stress", RED)
    ]:
        pt = result.at_price(anchor)
        if pt:
            ax2.annotate(
                f"{pt.wacc_uplift_bps:.0f} bps",
                xy=(pt.carbon_price, pt.wacc_uplift_bps),
                xytext=(pt.carbon_price + 12, pt.wacc_uplift_bps + 1),
                color=col, fontsize=7, fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color=col, lw=0.8),
            )

    # ── Panel 3: EV haircut (USD B) vs price ─────────────────────────────────
    ax3 = axes[1, 0]
    ax3.plot(prices, ev_haircut, color=RED, linewidth=2.2, marker="o", markersize=4)
    ax3.fill_between(prices, 0, ev_haircut, color=RED, alpha=0.12)
    ax3.axhline(0, color=SLATE, linewidth=0.6, alpha=0.5)
    ax3.set_title("Enterprise Value Haircut (USD B)", color=WHITE, fontsize=9, pad=6)
    ax3.set_xlabel("Carbon Price (USD/tCO2e)", fontsize=8)
    ax3.set_ylabel("EV loss vs baseline (USD B)", fontsize=8)
    ax3.grid(True, color=BORD, linewidth=0.5, alpha=0.6)
    # Baseline marker
    ax3.axvline(result.market_carbon_price, color=GREEN, linewidth=1.0,
                linestyle=":", alpha=0.8)
    ax3.text(
        result.market_carbon_price + 2,
        max(ev_haircut) * 0.05,
        f"Baseline\nUSD{result.market_carbon_price:.0f}/t",
        color=GREEN, fontsize=7, fontfamily="monospace",
    )

    # ── Panel 4: EV haircut (%) at regulatory anchors — horizontal bars ──────
    ax4 = axes[1, 1]
    anchors = [30, 50, 75, 130, 160, 200, 250]
    anchor_pts = [result.at_price(a) for a in anchors if result.at_price(a)]
    bar_labels = [f"USD{int(p.carbon_price)}/t" for p in anchor_pts]
    bar_vals   = [p.ev_haircut_pct * 100 for p in anchor_pts]
    bar_colors = []
    for v in bar_vals:
        if v < 5:
            bar_colors.append(GREEN)
        elif v < 15:
            bar_colors.append(AMBER)
        elif v < 25:
            bar_colors.append(ORANGE)
        else:
            bar_colors.append(RED)

    y_pos = range(len(anchor_pts))
    bars = ax4.barh(list(y_pos), bar_vals, color=bar_colors, height=0.55, alpha=0.85)
    ax4.set_yticks(list(y_pos))
    ax4.set_yticklabels(bar_labels, fontsize=8)
    ax4.set_title("EV Haircut (%) at Key Carbon Prices", color=WHITE, fontsize=9, pad=6)
    ax4.set_xlabel("EV haircut vs baseline (%)", fontsize=8)
    ax4.grid(True, axis="x", color=BORD, linewidth=0.5, alpha=0.6)
    ax4.axvline(0, color=SLATE, linewidth=0.6)
    for bar, val in zip(bars, bar_vals):
        ax4.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", color=WHITE, fontsize=7.5, fontfamily="monospace",
        )

    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=2.0)
    fig.subplots_adjust(top=0.90, wspace=0.32, hspace=0.38)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG)

    return fig
