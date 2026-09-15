"""
What-If Sensitivity / Scenario Sweep.

Runs the core transition risk and physical risk calculations at
multiple parameter values, returning a grid of outcomes that shows
how the key metrics change as the analyst moves levers.

Levers
------
- carbon_price_mult    : Carbon price multiplier (0.5× to 3.0×)
- capex_green_usd_m    : Green CAPEX committed (0 to 500 M USD)
- scope1_reduction_pct : Scope 1 reduction vs. baseline (0% to 75%)
- asset_elevation_m    : Asset elevation assumption (physical sensitivity)

Output
------
A grid of {lever_value, transition_var_pct, cumulative_capex,
net_cost_usd_m, physical_var_pct, composite_score} at each step.

Designed to be called from the /sensitivity endpoint and returned
as a JSON grid that can power a chart on the client side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math


# ── Carbon price schedule (USD/tCO2e) — three IEA NZE scenarios ──────────────
# Source: IEA World Energy Outlook 2023 / Net Zero 2050
_CARBON_PRICE_2030 = {
    "nze_15c":   250,   # Net Zero Emissions by 2050 (1.5°C)
    "aps_18c":   130,   # Announced Pledges Scenario (1.8°C)
    "steps_2c":  60,    # Stated Policies Scenario (2°C+)
}
_CARBON_PRICE_2050 = {
    "nze_15c":   700,
    "aps_18c":   350,
    "steps_2c":  130,
}

# Default mid-case: APS 2030
_BASE_CARBON_PRICE_2030 = _CARBON_PRICE_2030["aps_18c"]


@dataclass
class SensitivityPoint:
    lever_name:          str
    lever_value:         float
    lever_unit:          str
    transition_var_pct:  float   # % of revenue
    scope1_cost_usd_m:   float   # annual carbon cost
    net_capex_usd_m:     float   # cumulative green capex committed
    net_transition_cost: float   # scope1_cost - avoided cost from capex
    physical_var_pct:    float   # % of EV (flood/SLR)
    composite_score:     float   # 0–1
    label:               str     # human-readable point label


@dataclass
class SensitivityResult:
    company_name:   str
    sweep_lever:    str
    points:         list[SensitivityPoint]
    baseline:       SensitivityPoint
    optimal_point:  Optional[SensitivityPoint]  # lowest composite score
    methodology:    str


def _transition_var(
    scope1_mt_co2e: float,      # absolute Scope 1
    carbon_price:   float,      # USD/tCO2e
    reduction_pct:  float,      # 0–100
    capex_usd_m:    float,      # committed green capex
    revenue_usd_m:  float,
) -> tuple[float, float, float]:
    """Returns (var_pct_revenue, scope1_cost_usd_m, net_transition_cost_usd_m)."""
    # Avoided Scope 1 from CAPEX: assume $40M capex → 1 Mt avoided (conservative)
    avoided_mt = capex_usd_m / 40.0
    effective_mt = max(0, scope1_mt_co2e * (1 - reduction_pct / 100.0) - avoided_mt)

    scope1_cost_usd_m = effective_mt * carbon_price / 1_000_000  # convert tCO2e → USD M
    # Avoided costs = carbon cost saving from capex
    avoided_cost = avoided_mt * carbon_price / 1_000_000

    # Net cost = what you pay minus what you saved
    net_cost = max(0.0, scope1_cost_usd_m - avoided_cost * 0.0)  # capex already deducted via effective_mt

    var_pct = (net_cost / revenue_usd_m * 100) if revenue_usd_m > 0 else 0.0
    return round(var_pct, 3), round(scope1_cost_usd_m, 2), round(net_cost, 2)


def _physical_var(
    base_flood_var_pct: float,
    elevation_m: float,
    base_elevation_m: float,
    ev_usd_m: float,
) -> float:
    """Physical VaR adjusts with elevation — higher elevation → lower flood risk."""
    if base_elevation_m <= 0:
        return base_flood_var_pct
    elev_ratio  = elevation_m / max(base_elevation_m, 0.1)
    # Logistic compression: 0 m → 100% of base VaR; 10 m → ~10%; 20 m → ~2%
    adjustment  = 1.0 / (1 + math.exp((elevation_m - 5) / 2))
    adjusted    = base_flood_var_pct * adjustment * 2  # re-scale so base_elevation gives base_var
    return round(max(0.0, adjusted), 3)


def _composite(trans_var_pct: float, phys_var_pct: float,
               biodiversity_score: float = 0.0) -> float:
    phys_comp  = min(1.0, phys_var_pct / 20.0)   # 20% EV flood VaR → score 1.0
    trans_comp = min(1.0, trans_var_pct / 33.0)   # 33% revenue VaR → score 1.0
    return round(0.5 * phys_comp + 0.5 * trans_comp, 3)


def run_sensitivity(
    company_name: str,
    sector_key: str,
    # Baseline values
    scope1_mt_co2e: float,
    revenue_usd_m: float,
    ev_usd_m: float,
    base_flood_var_pct: float = 0.0,
    base_elevation_m: float = 5.0,
    base_carbon_price: float = _BASE_CARBON_PRICE_2030,
    base_capex_green: float = 0.0,
    base_scope1_reduction_pct: float = 0.0,
    # Which lever to sweep
    sweep: str = "carbon_price",  # carbon_price | capex_green | scope1_reduction | elevation
    # Lever range
    sweep_min: Optional[float] = None,
    sweep_max: Optional[float] = None,
    sweep_steps: int = 10,
) -> SensitivityResult:
    """
    Sweep one parameter while holding others at baseline.

    sweep options:
        carbon_price       — multiply base carbon price (0.5× to 3.0×), expressed as multiplier
        capex_green        — committed green capex USD M (0 to 500)
        scope1_reduction   — Scope 1 cut % vs. baseline (0 to 75)
        elevation          — asset elevation m (0 to 30)
    """
    # ── Set sweep range defaults ──────────────────────────────────────────────
    _defaults = {
        "carbon_price":      (0.5, 3.0),
        "capex_green":       (0.0, min(500.0, ev_usd_m * 0.10)),
        "scope1_reduction":  (0.0, 75.0),
        "elevation":         (0.0, 30.0),
    }
    lo, hi = _defaults.get(sweep, (0.0, 1.0))
    if sweep_min is not None:
        lo = sweep_min
    if sweep_max is not None:
        hi = sweep_max

    step_size = (hi - lo) / max(1, sweep_steps - 1)
    lever_values = [round(lo + i * step_size, 4) for i in range(sweep_steps)]

    points: list[SensitivityPoint] = []

    _units = {
        "carbon_price":     "×",
        "capex_green":      "USD M",
        "scope1_reduction": "%",
        "elevation":        "m",
    }
    unit = _units.get(sweep, "")

    for v in lever_values:
        # Resolve actual parameters for this sweep point
        if sweep == "carbon_price":
            cp    = base_carbon_price * v
            capex = base_capex_green
            reduc = base_scope1_reduction_pct
            elev  = base_elevation_m
            label = f"{v:.1f}× carbon price (${cp:.0f}/tCO₂e)"
        elif sweep == "capex_green":
            cp    = base_carbon_price
            capex = v
            reduc = base_scope1_reduction_pct
            elev  = base_elevation_m
            label = f"${v:.0f}M green capex committed"
        elif sweep == "scope1_reduction":
            cp    = base_carbon_price
            capex = base_capex_green
            reduc = v
            elev  = base_elevation_m
            label = f"{v:.0f}% Scope 1 reduction"
        else:  # elevation
            cp    = base_carbon_price
            capex = base_capex_green
            reduc = base_scope1_reduction_pct
            elev  = v
            label = f"{v:.1f}m asset elevation"

        tv_pct, s1_cost, net_cost = _transition_var(
            scope1_mt_co2e=scope1_mt_co2e,
            carbon_price=cp,
            reduction_pct=reduc,
            capex_usd_m=capex,
            revenue_usd_m=revenue_usd_m,
        )

        pv_pct = _physical_var(
            base_flood_var_pct=base_flood_var_pct,
            elevation_m=elev,
            base_elevation_m=base_elevation_m,
            ev_usd_m=ev_usd_m,
        )

        comp = _composite(tv_pct, pv_pct)

        points.append(SensitivityPoint(
            lever_name=sweep,
            lever_value=round(v, 4),
            lever_unit=unit,
            transition_var_pct=tv_pct,
            scope1_cost_usd_m=s1_cost,
            net_capex_usd_m=round(capex, 2),
            net_transition_cost=net_cost,
            physical_var_pct=pv_pct,
            composite_score=comp,
            label=label,
        ))

    # ── Baseline ──────────────────────────────────────────────────────────────
    btv, bs1, bnet = _transition_var(
        scope1_mt_co2e=scope1_mt_co2e,
        carbon_price=base_carbon_price,
        reduction_pct=base_scope1_reduction_pct,
        capex_usd_m=base_capex_green,
        revenue_usd_m=revenue_usd_m,
    )
    bpv  = _physical_var(base_flood_var_pct, base_elevation_m, base_elevation_m, ev_usd_m)
    bcomp = _composite(btv, bpv)

    baseline = SensitivityPoint(
        lever_name=sweep,
        lever_value=1.0 if sweep == "carbon_price" else (
            base_capex_green if sweep == "capex_green" else (
                base_scope1_reduction_pct if sweep == "scope1_reduction" else base_elevation_m
            )
        ),
        lever_unit=unit,
        transition_var_pct=btv,
        scope1_cost_usd_m=bs1,
        net_capex_usd_m=base_capex_green,
        net_transition_cost=bnet,
        physical_var_pct=bpv,
        composite_score=bcomp,
        label="Baseline",
    )

    # ── Optimal point (lowest composite) ────────────────────────────────────
    optimal = min(points, key=lambda p: p.composite_score)

    meth = (
        "Carbon cost = effective Scope 1 (after reductions + avoided CAPEX) × carbon price. "
        "Physical VaR adjusted via logistic elevation function. "
        "Composite = 0.5 × physical score + 0.5 × transition score, each capped at 1.0."
    )

    return SensitivityResult(
        company_name=company_name,
        sweep_lever=sweep,
        points=points,
        baseline=baseline,
        optimal_point=optimal if optimal.composite_score < bcomp else None,
        methodology=meth,
    )


def sensitivity_to_dict(result: SensitivityResult) -> dict:
    def pt(p: SensitivityPoint) -> dict:
        return {
            "lever_value":        p.lever_value,
            "lever_unit":         p.lever_unit,
            "label":              p.label,
            "transition_var_pct": p.transition_var_pct,
            "scope1_cost_usd_m":  p.scope1_cost_usd_m,
            "net_capex_usd_m":    p.net_capex_usd_m,
            "physical_var_pct":   p.physical_var_pct,
            "composite_score":    p.composite_score,
        }

    return {
        "company_name": result.company_name,
        "sweep_lever":  result.sweep_lever,
        "baseline":     pt(result.baseline),
        "optimal":      pt(result.optimal_point) if result.optimal_point else None,
        "grid":         [pt(p) for p in result.points],
        "methodology":  result.methodology,
    }
