"""Portfolio-level climate risk aggregation and correlation analysis.

Takes the per-company FullRunResult outputs and aggregates them into a
portfolio view that institutional investors, banks, and asset managers need
for capital allocation, climate stress testing, and regulatory disclosure
(ECB climate stress test, PRA SS5/25, NGFS portfolio exercise).

Core computations
-----------------
1. Weighted portfolio EV impact per NGFS scenario
   - Weighted average EV haircut across holdings under NZE and Delayed
   - Absolute portfolio EV at risk in $ under each scenario

2. Pairwise loss correlation matrix
   - Per company, extract the annual FCF deviation series relative to CP
     baseline across NZE + Delayed scenarios (50 data points per company)
   - Pearson correlation between all company pairs
   - Reveals diversification quality: low cross-sector correlation = genuine
     risk diversification; high correlation = concentrated climate exposure

3. Portfolio VaR (parametric)
   - Construct the portfolio variance from the correlation matrix + individual
     EV haircuts + portfolio weights
   - Report VaR_95 and VaR_99 in USD at the portfolio level
   - Scenario-conditioned: separate VaR for NZE, Delayed, and combined

4. Concentration metrics
   - Herfindahl-Hirschman Index (HHI) on portfolio weights
   - Climate-HHI: HHI weighted by EV haircut intensity (concentration in
     high-climate-risk names is worse than weight concentration alone)
   - Diversification ratio: sqrt(sum(w_i² × σ_i²)) / σ_portfolio

5. Sector exposure summary
   - Aggregate weight per sector
   - Sector average CRI score
   - Stranded asset exposure by sector ($B)

Usage
-----
from cri.engine.orchestrator import run_full
from cri.analysis.portfolio import PortfolioHolding, PortfolioAnalyser, plot_portfolio

holdings = [
    PortfolioHolding("RIO.L",  "Rio Tinto",    "Mining",   0.25, run_full(rio)),
    PortfolioHolding("TSL.NS", "Tata Steel",   "Steel",    0.20, run_full(tata)),
    PortfolioHolding("HNK.AE", "Heineken",     "Beverages",0.20, run_full(heineken)),
    PortfolioHolding("ING.AS", "ING Group",    "Finance",  0.20, run_full(ing)),
    PortfolioHolding("RNG.L",  "Ryanair",      "Aviation", 0.15, run_full(ryanair)),
]
result = PortfolioAnalyser(holdings).analyse()
print(result.summary())
fig = plot_portfolio(result)
fig.savefig("portfolio_risk.png", dpi=180, bbox_inches="tight")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..engine.orchestrator import FullRunResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PortfolioHolding:
    """One position in the portfolio.

    Attributes
    ----------
    company_id   : Ticker or slug — must match Company.id used in the run.
    company_name : Display name.
    sector       : Free-text sector label.
    weight       : Portfolio weight, 0–1. Weights across all holdings should
                   sum to 1.0 (normalised internally if they do not).
    full_result  : FullRunResult from run_full(company). Provides NZE /
                   Delayed / CP scenario results and composite CRI rating.
    """
    company_id: str
    company_name: str
    sector: str
    weight: float
    full_result: FullRunResult


@dataclass
class ScenarioPortfolioMetrics:
    """Aggregated portfolio metrics under one NGFS scenario.

    Attributes
    ----------
    scenario_label          : "nze" | "delayed" | "cp"
    weighted_ev_impact_pct  : Portfolio-weighted average EV haircut (% of CP baseline)
    portfolio_ev_at_risk    : Absolute EV at risk across all holdings (USD)
    portfolio_carbon_cost_yr1 : Portfolio-weighted aggregate annual carbon cost, year 1 (USD)
    portfolio_physical_loss : Portfolio-weighted physical loss cost, year 1 (USD)
    var_95                  : Portfolio VaR at 95% confidence (USD, positive = loss)
    var_99                  : Portfolio VaR at 99% confidence (USD, positive = loss)
    """
    scenario_label: str
    weighted_ev_impact_pct: float
    portfolio_ev_at_risk: float
    portfolio_carbon_cost_yr1: float
    portfolio_physical_loss_yr1: float
    var_95: float
    var_99: float


@dataclass
class PortfolioResult:
    """Full portfolio climate risk output.

    Attributes
    ----------
    holdings                : Original holdings list (after weight normalisation).
    weights_normalised      : Normalised weights (sum = 1.0).
    weighted_cri_score      : Holdings-weighted composite CRI score (0–100).
    weighted_cri_rating     : Letter rating implied by weighted_cri_score.
    scenario_metrics        : Dict keyed by scenario label → ScenarioPortfolioMetrics.
    correlation_matrix      : Dict company_id → Dict company_id → float (−1 to 1).
    hhi                     : Herfindahl-Hirschman Index on portfolio weights (0–1).
    climate_hhi             : HHI weighted by individual EV haircut intensity.
    diversification_ratio   : Undiversified σ / portfolio σ (Delayed scenario).
    sector_exposures        : Dict sector → dict with weight, avg_cri, ev_at_risk_b.
    """
    holdings: list[PortfolioHolding]
    weights_normalised: list[float]
    weighted_cri_score: float
    weighted_cri_rating: str
    scenario_metrics: dict[str, ScenarioPortfolioMetrics] = field(default_factory=dict)
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    hhi: float = 0.0
    climate_hhi: float = 0.0
    diversification_ratio: float = 1.0
    sector_exposures: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Portfolio Climate Risk Summary  —  ClimRisk CRI Engine v0.4",
            "═" * 65,
            f"Holdings: {len(self.holdings)}   "
            f"Weighted CRI: {self.weighted_cri_score:.1f}/100  "
            f"Rating: {self.weighted_cri_rating}   "
            f"HHI: {self.hhi:.3f}",
            "",
            f"{'Scenario':<14} {'EV Impact (%)':>14} {'EV at Risk ($B)':>16} "
            f"{'VaR 95% ($B)':>14} {'VaR 99% ($B)':>14}",
            "─" * 74,
        ]
        for label in ("nze", "delayed", "cp"):
            m = self.scenario_metrics.get(label)
            if m:
                lines.append(
                    f"{label.upper():<14} "
                    f"{m.weighted_ev_impact_pct*100:>13.1f}%  "
                    f"{m.portfolio_ev_at_risk/1e3:>15.2f}  "
                    f"{m.var_95/1e3:>13.2f}  "
                    f"{m.var_99/1e3:>13.2f}"
                )
        lines += ["", "Holdings:", "─" * 74]
        for h, w in zip(self.holdings, self.weights_normalised):
            rating = str(h.full_result.rating.rating)
            score  = h.full_result.rating.composite_score
            lines.append(
                f"  {h.company_name:<22}  {h.sector:<18}  "
                f"wt={w*100:.1f}%  CRI={score:.0f}  [{rating}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _composite_score(full: FullRunResult) -> float:
    """Extract composite CRI score from FullRunResult.rating."""
    try:
        return float(full.rating.composite_score)
    except Exception:
        return 50.0  # fallback: Moderate


def _ev_impact_pct(full: FullRunResult, scenario: str) -> float:
    """EV impact % vs CP baseline. Positive = EV lower than baseline (haircut)."""
    cp_ev  = full.cp.enterprise_value
    sc_ev  = getattr(full, scenario).enterprise_value
    if cp_ev == 0:
        return 0.0
    return (cp_ev - sc_ev) / abs(cp_ev)


def _annual_fcf_deviation(full: FullRunResult, scenario: str) -> list[float]:
    """Annual FCF deviation from CP baseline, normalised by company revenue."""
    revenue = full.cp.years[0].revenue if full.cp.years else 1.0
    cp_fcf  = {yr.year: yr.fcf for yr in full.cp.years}
    sc_fcf  = {yr.year: yr.fcf for yr in getattr(full, scenario).years}
    common_years = sorted(set(cp_fcf) & set(sc_fcf))
    if not common_years or revenue == 0:
        return []
    return [(sc_fcf[y] - cp_fcf[y]) / revenue for y in common_years]


def _cri_to_rating(score: float) -> str:
    """Convert composite score (0–100) to letter rating."""
    if score <= 20:
        return "A"
    if score <= 40:
        return "B"
    if score <= 60:
        return "C"
    if score <= 80:
        return "D"
    return "E"


def _normalise_weights(holdings: list[PortfolioHolding]) -> list[float]:
    """Return weights normalised to sum = 1.0."""
    total = sum(h.weight for h in holdings)
    if total <= 0:
        n = len(holdings)
        return [1.0 / n] * n
    return [h.weight / total for h in holdings]


# ---------------------------------------------------------------------------
# VaR helpers
# ---------------------------------------------------------------------------

_Z_95 = 1.6449    # one-tailed 95th percentile
_Z_99 = 2.3263    # one-tailed 99th percentile


def _portfolio_var(
    ev_haircuts: list[float],
    weights: list[float],
    corr_matrix: np.ndarray,
    z: float,
) -> float:
    """Parametric portfolio VaR at z-score confidence level.

    Uses variance-covariance method:
      σ_portfolio² = Σ_i Σ_j w_i w_j σ_i σ_j ρ_ij
    where σ_i = ev_haircut_i (treated as the standard deviation of position i)

    VaR = z × σ_portfolio

    Note: this is a structural / scenario-conditioned VaR — it does not
    assume a probability distribution over outcomes but uses the scenario
    EV haircuts as the shock magnitudes. Appropriate for TCFD scenario
    analysis; not a historical or Monte Carlo VaR.
    """
    n = len(weights)
    w = np.array(weights)
    sigma = np.array([abs(h) for h in ev_haircuts])

    # Covariance matrix: Σ_ij = σ_i × σ_j × ρ_ij
    cov = np.outer(sigma, sigma) * corr_matrix

    portfolio_variance = float(w @ cov @ w)
    portfolio_std = math.sqrt(max(portfolio_variance, 0.0))
    return z * portfolio_std


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PortfolioAnalyser:
    """Aggregate company-level CRI outputs into a full portfolio view.

    Parameters
    ----------
    holdings : List of PortfolioHolding objects, each with a FullRunResult.

    Example
    -------
    result = PortfolioAnalyser(holdings).analyse()
    """

    def __init__(self, holdings: list[PortfolioHolding]) -> None:
        if not holdings:
            raise ValueError("At least one holding required.")
        self.holdings = holdings
        self.weights = _normalise_weights(holdings)

    def analyse(self) -> PortfolioResult:
        """Run all portfolio computations and return PortfolioResult."""

        n = len(self.holdings)

        # ── 1. Weighted CRI score ────────────────────────────────────────────
        cri_scores = [_composite_score(h.full_result) for h in self.holdings]
        weighted_cri = sum(w * s for w, s in zip(self.weights, cri_scores))
        weighted_rating = _cri_to_rating(weighted_cri)

        # ── 2. Per-scenario aggregation ──────────────────────────────────────
        scenario_metrics: dict[str, ScenarioPortfolioMetrics] = {}

        for label in ("nze", "delayed", "cp"):
            ev_impacts_pct  = []
            ev_at_risk_abs  = []
            cc_yr1_list     = []
            phys_yr1_list   = []

            for h in self.holdings:
                fr = h.full_result
                cp_ev = fr.cp.enterprise_value

                if label == "cp":
                    impact_pct = 0.0
                    ev_at_risk = 0.0
                else:
                    impact_pct = _ev_impact_pct(fr, label)
                    ev_at_risk = impact_pct * abs(cp_ev)

                ev_impacts_pct.append(impact_pct)
                ev_at_risk_abs.append(ev_at_risk)

                sc_res = getattr(fr, label)
                cc_yr1_list.append(sc_res.years[0].carbon_cost if sc_res.years else 0.0)
                phys_yr1_list.append(
                    sc_res.years[0].physical_loss_cost if sc_res.years else 0.0
                )

            weighted_impact_pct = sum(w * p for w, p in zip(self.weights, ev_impacts_pct))
            portfolio_ev_at_risk = sum(w * a for w, a in zip(self.weights, ev_at_risk_abs))
            portfolio_cc_yr1     = sum(w * c for w, c in zip(self.weights, cc_yr1_list))
            portfolio_phys_yr1   = sum(w * p for w, p in zip(self.weights, phys_yr1_list))

            # VaR — uses correlation matrix below; placeholder for now, filled after
            scenario_metrics[label] = ScenarioPortfolioMetrics(
                scenario_label=label,
                weighted_ev_impact_pct=weighted_impact_pct,
                portfolio_ev_at_risk=portfolio_ev_at_risk,
                portfolio_carbon_cost_yr1=portfolio_cc_yr1,
                portfolio_physical_loss_yr1=portfolio_phys_yr1,
                var_95=0.0,  # filled after correlation matrix
                var_99=0.0,
            )

        # ── 3. Correlation matrix ────────────────────────────────────────────
        # Stack NZE + Delayed FCF deviations to get a combined (2×horizon)-length
        # vector per company. Pearson correlation between these vectors.
        series_per_company: dict[str, list[float]] = {}
        for h in self.holdings:
            nze_dev = _annual_fcf_deviation(h.full_result, "nze")
            dly_dev = _annual_fcf_deviation(h.full_result, "delayed")
            combined = nze_dev + dly_dev
            series_per_company[h.company_id] = combined

        # Align lengths (trim to shortest)
        min_len = min((len(v) for v in series_per_company.values()), default=0)
        if min_len > 1:
            aligned = {k: v[:min_len] for k, v in series_per_company.items()}
        else:
            # Fallback: no time-series data available; use identity correlation
            aligned = {k: [0.0, 1.0] for k in series_per_company}

        ids = [h.company_id for h in self.holdings]
        mat = np.zeros((n, n))
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if i == j:
                    mat[i, j] = 1.0
                elif i < j:
                    vi = np.array(aligned[id_i])
                    vj = np.array(aligned[id_j])
                    # Handle zero-variance series
                    std_i, std_j = vi.std(), vj.std()
                    if std_i < 1e-12 or std_j < 1e-12:
                        rho = 0.0
                    else:
                        rho = float(np.corrcoef(vi, vj)[0, 1])
                        rho = max(-1.0, min(1.0, rho))  # clamp numerical noise
                    mat[i, j] = rho
                    mat[j, i] = rho

        corr_dict: dict[str, dict[str, float]] = {}
        for i, id_i in enumerate(ids):
            corr_dict[id_i] = {id_j: float(mat[i, j]) for j, id_j in enumerate(ids)}

        # ── 4. Back-fill VaR into scenario metrics ───────────────────────────
        for label in ("nze", "delayed"):
            ev_haircuts_usd = []
            for h in self.holdings:
                pct = _ev_impact_pct(h.full_result, label)
                cp_ev = abs(h.full_result.cp.enterprise_value)
                ev_haircuts_usd.append(pct * cp_ev)

            var95 = _portfolio_var(ev_haircuts_usd, self.weights, mat, _Z_95)
            var99 = _portfolio_var(ev_haircuts_usd, self.weights, mat, _Z_99)
            scenario_metrics[label].var_95 = var95
            scenario_metrics[label].var_99 = var99

        # ── 5. Concentration metrics ─────────────────────────────────────────
        hhi = sum(w ** 2 for w in self.weights)

        # Climate-HHI: weight each position by its EV haircut intensity
        nze_haircuts = [
            abs(_ev_impact_pct(h.full_result, "nze")) for h in self.holdings
        ]
        max_haircut = max(nze_haircuts) if max(nze_haircuts) > 0 else 1.0
        climate_weights = [
            w * (h / max_haircut) for w, h in zip(self.weights, nze_haircuts)
        ]
        climate_w_total = sum(climate_weights) or 1.0
        climate_weights = [cw / climate_w_total for cw in climate_weights]
        climate_hhi = sum(cw ** 2 for cw in climate_weights)

        # Diversification ratio (Delayed scenario)
        dly_haircuts_usd = [
            abs(_ev_impact_pct(h.full_result, "delayed"))
            * abs(h.full_result.cp.enterprise_value)
            for h in self.holdings
        ]
        undiv_variance = sum(
            (w * abs(hc)) ** 2 for w, hc in zip(self.weights, dly_haircuts_usd)
        )
        undiv_std = math.sqrt(undiv_variance) if undiv_variance > 0 else 1.0
        port_var_dly = _portfolio_var(dly_haircuts_usd, self.weights, mat, 1.0)
        div_ratio = undiv_std / max(port_var_dly, 1e-12) if port_var_dly > 0 else 1.0
        div_ratio = max(1.0, div_ratio)  # diversification ratio ≥ 1 by construction

        # ── 6. Sector exposure summary ───────────────────────────────────────
        sector_data: dict[str, dict] = {}
        for h, w in zip(self.holdings, self.weights):
            sec = h.sector
            if sec not in sector_data:
                sector_data[sec] = {"weight": 0.0, "cri_scores": [], "ev_at_risk": 0.0}
            sector_data[sec]["weight"] += w
            sector_data[sec]["cri_scores"].append(_composite_score(h.full_result))
            nze_impact = _ev_impact_pct(h.full_result, "nze")
            sector_data[sec]["ev_at_risk"] += (
                nze_impact * abs(h.full_result.cp.enterprise_value) * w
            )

        sector_exposures = {
            sec: {
                "weight": data["weight"],
                "avg_cri": (
                    sum(data["cri_scores"]) / len(data["cri_scores"])
                    if data["cri_scores"]
                    else 0.0
                ),
                "ev_at_risk_b": data["ev_at_risk"] / 1e3,
            }
            for sec, data in sector_data.items()
        }

        return PortfolioResult(
            holdings=self.holdings,
            weights_normalised=self.weights,
            weighted_cri_score=weighted_cri,
            weighted_cri_rating=weighted_rating,
            scenario_metrics=scenario_metrics,
            correlation_matrix=corr_dict,
            hhi=hhi,
            climate_hhi=climate_hhi,
            diversification_ratio=div_ratio,
            sector_exposures=sector_exposures,
        )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_portfolio(
    result: PortfolioResult,
    output_path: Optional[str] = None,
    dpi: int = 180,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Dark-theme portfolio risk dashboard.

    Panels
    ------
    1 (top-left)    : Weighted CRI score per holding — horizontal bar chart,
                      coloured by rating band. Shows which positions drive
                      portfolio climate risk.
    2 (top-right)   : EV at risk ($B) under NZE and Delayed Transition, per
                      holding. Grouped bars — side-by-side scenario comparison.
    3 (middle-left) : Pairwise loss correlation heatmap (company × company).
                      Reds = high correlation (sector concentration risk);
                      blues = low/negative correlation (genuine diversification).
    4 (middle-right): Sector exposure wheel — pie chart of portfolio weight by
                      sector, annotated with sector average CRI.
    5 (bottom)      : Portfolio summary stats strip: weighted CRI, HHI,
                      climate-HHI, diversification ratio, VaR_95 (Delayed).

    Parameters
    ----------
    result      : PortfolioResult from PortfolioAnalyser.analyse().
    output_path : If provided, saves the figure to this path.
    dpi         : Resolution for file output.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import numpy as np

    # ── Palette ──────────────────────────────────────────────────────────────
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

    def _rating_color(rating: str) -> str:
        return {
            "A": GREEN, "B": BLUE, "C": AMBER, "D": ORANGE, "E": RED
        }.get(rating, SLATE)

    def _score_color(score: float) -> str:
        if score <= 20:
            return GREEN
        if score <= 40:
            return BLUE
        if score <= 60:
            return AMBER
        if score <= 80:
            return ORANGE
        return RED

    n = len(result.holdings)
    names  = [h.company_name for h in result.holdings]
    ids    = [h.company_id for h in result.holdings]
    scores = [_composite_score(h.full_result) for h in result.holdings]
    weights_pct = [w * 100 for w in result.weights_normalised]

    fig = plt.figure(figsize=(14, 16))
    fig.patch.set_facecolor(BG)

    # Layout: 3 rows — top row 2 panels, middle row 2 panels, bottom strip
    gs = fig.add_gridspec(
        3, 2, height_ratios=[3, 3, 0.9],
        hspace=0.42, wspace=0.35,
        left=0.07, right=0.97, top=0.92, bottom=0.04,
    )

    def _style_ax(ax):
        ax.set_facecolor(CARD)
        ax.tick_params(colors=SLATE, labelsize=8)
        ax.xaxis.label.set_color(SLATE)
        ax.yaxis.label.set_color(SLATE)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORD)

    # ── Header ───────────────────────────────────────────────────────────────
    fig.text(
        0.012, 0.965,
        "ClimRisk  |  Portfolio Climate Risk Dashboard  —  CRI Engine v0.4",
        fontsize=11, color=BLUE, fontweight="bold",
        fontfamily="monospace", va="top",
    )
    fig.text(
        0.012, 0.944,
        f"Holdings: {n}   "
        f"Weighted CRI: {result.weighted_cri_score:.1f}/100  [{result.weighted_cri_rating}]   "
        f"HHI: {result.hhi:.3f}   "
        f"Diversification ratio: {result.diversification_ratio:.2f}   "
        f"NGFS Phase 4",
        fontsize=8, color=SLATE, fontfamily="monospace", va="top",
    )

    # ── Panel 1: Per-holding CRI scores ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1)

    y_pos = range(n)
    bar_colors = [_score_color(s) for s in scores]
    bars1 = ax1.barh(list(y_pos), scores, color=bar_colors, height=0.55, alpha=0.85)
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(names, fontsize=7.5)
    ax1.set_xlabel("CRI Score (0–100, higher = more risk)", fontsize=7.5)
    ax1.set_title("CRI Score by Holding", color=WHITE, fontsize=9, pad=6)
    ax1.set_xlim(0, 105)
    ax1.grid(True, axis="x", color=BORD, linewidth=0.5, alpha=0.6)
    ax1.axvline(result.weighted_cri_score, color=PURPLE, linewidth=1.2,
                linestyle="--", alpha=0.8)
    ax1.text(
        result.weighted_cri_score + 1, n - 0.5,
        f"Wtd avg\n{result.weighted_cri_score:.0f}",
        color=PURPLE, fontsize=7, fontfamily="monospace",
    )
    for bar, score, w in zip(bars1, scores, weights_pct):
        ax1.text(
            score + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{score:.0f}  ({w:.0f}%)",
            va="center", color=WHITE, fontsize=7, fontfamily="monospace",
        )

    # ── Panel 2: EV at risk per holding — NZE vs Delayed ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style_ax(ax2)

    nze_ev   = [
        _ev_impact_pct(h.full_result, "nze")
        * abs(h.full_result.cp.enterprise_value) / 1e3
        for h in result.holdings
    ]
    dly_ev   = [
        _ev_impact_pct(h.full_result, "delayed")
        * abs(h.full_result.cp.enterprise_value) / 1e3
        for h in result.holdings
    ]

    x_idx = np.arange(n)
    bar_w = 0.35
    ax2.barh(x_idx - bar_w / 2, nze_ev, bar_w, color=BLUE,  alpha=0.85, label="NZE 2050")
    ax2.barh(x_idx + bar_w / 2, dly_ev, bar_w, color=AMBER, alpha=0.85, label="Delayed")
    ax2.set_yticks(x_idx)
    ax2.set_yticklabels(names, fontsize=7.5)
    ax2.set_xlabel("EV at Risk (USD B, weighted by portfolio share)", fontsize=7.5)
    ax2.set_title("EV at Risk by Holding (NGFS)", color=WHITE, fontsize=9, pad=6)
    ax2.legend(fontsize=8, facecolor=CARD, edgecolor=BORD, labelcolor=SLATE, loc="lower right")
    ax2.grid(True, axis="x", color=BORD, linewidth=0.5, alpha=0.6)
    ax2.axvline(0, color=SLATE, linewidth=0.6)

    # ── Panel 3: Correlation heatmap ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3)

    corr_mat = np.array([[result.correlation_matrix[id_i][id_j]
                          for id_j in ids] for id_i in ids])

    # Diverging colormap: blue (low corr) → white → red (high corr)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "climrisk_div",
        ["#1e40af", "#0b1f38", "#f8fafc", "#7f1d1d", "#ef4444"],
        N=256,
    )
    im = ax3.imshow(corr_mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    short_names = [name.split()[0] for name in names]
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7.5, color=SLATE)
    ax3.set_yticklabels(short_names, fontsize=7.5, color=SLATE)
    ax3.set_title("Loss Correlation Matrix (NZE + Delayed FCF deviations)",
                  color=WHITE, fontsize=8.5, pad=6)

    # Cell annotations
    for i in range(n):
        for j in range(n):
            val = corr_mat[i, j]
            txt_col = WHITE if abs(val) > 0.5 else DIM
            ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=6.5, color=txt_col, fontfamily="monospace")

    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04,
                 label="Correlation").ax.yaxis.label.set_color(SLATE)

    # ── Panel 4: Sector exposure pie ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(CARD)
    ax4.set_title("Portfolio Weight by Sector", color=WHITE, fontsize=9, pad=6)

    sector_names = list(result.sector_exposures.keys())
    sector_wts   = [result.sector_exposures[s]["weight"] * 100 for s in sector_names]
    sector_cris  = [result.sector_exposures[s]["avg_cri"] for s in sector_names]

    pie_colors = [
        GREEN, BLUE, AMBER, ORANGE, RED, PURPLE,
        "#06b6d4", "#84cc16", "#f43f5e", "#8b5cf6",
    ][:len(sector_names)]

    wedges, texts, autotexts = ax4.pie(
        sector_wts, labels=None,
        colors=pie_colors, autopct="%1.0f%%",
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(linewidth=0.8, edgecolor=CARD),
    )
    for autotext in autotexts:
        autotext.set_fontsize(7)
        autotext.set_color(BG)
        autotext.set_fontweight("bold")

    legend_labels = [
        f"{s}  (CRI {result.sector_exposures[s]['avg_cri']:.0f})"
        for s in sector_names
    ]
    ax4.legend(
        wedges, legend_labels, loc="lower left", fontsize=7,
        facecolor=CARD, edgecolor=BORD, labelcolor=SLATE,
    )

    # ── Panel 5: Summary stats strip ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor(BORD)
    ax5.axis("off")
    for spine in ax5.spines.values():
        spine.set_edgecolor(BLUE)

    dly_m = result.scenario_metrics.get("delayed")
    nze_m = result.scenario_metrics.get("nze")

    stats = [
        ("PORTFOLIO CRI", f"{result.weighted_cri_score:.1f}  [{result.weighted_cri_rating}]",
         _rating_color(result.weighted_cri_rating)),
        ("HHI (WEIGHT)", f"{result.hhi:.3f}", SLATE),
        ("CLIMATE HHI", f"{result.climate_hhi:.3f}", AMBER),
        ("DIVERSIF. RATIO", f"{result.diversification_ratio:.2f}×", GREEN),
        ("NZE EV IMPACT",
         f"{nze_m.weighted_ev_impact_pct * 100:.1f}%" if nze_m else "—", BLUE),
        ("DELAYED EV IMPACT",
         f"{dly_m.weighted_ev_impact_pct * 100:.1f}%" if dly_m else "—", AMBER),
        ("VAR 95% (DELAYED)",
         f"USD {dly_m.var_95 / 1e3:.2f}B" if dly_m else "—", RED),
        ("VAR 99% (DELAYED)",
         f"USD {dly_m.var_99 / 1e3:.2f}B" if dly_m else "—", RED),
    ]

    x_step = 1.0 / len(stats)
    for i, (label, val, col) in enumerate(stats):
        x = (i + 0.5) * x_step
        ax5.text(x, 0.68, label, ha="center", va="top",
                 fontsize=6.5, color=SLATE, fontfamily="monospace",
                 transform=ax5.transAxes)
        ax5.text(x, 0.28, val, ha="center", va="top",
                 fontsize=9.5, color=col, fontweight="bold",
                 fontfamily="monospace", transform=ax5.transAxes)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=BG)

    return fig
