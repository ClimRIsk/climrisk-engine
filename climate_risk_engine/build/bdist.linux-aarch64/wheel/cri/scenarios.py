"""Built-in canonical NGFS-aligned scenarios — CRI Engine v0.2.

Three scenarios covering the TCFD-recommended range of climate outcomes:

  NZE 2050          — Front-loaded transition compatible with Paris 1.5°C.
                      Carbon price path anchored to NGFS Phase 4 NZE.
  Delayed Transition — Policy inaction to ~2030, then emergency repricing.
                      Represents disorderly transition risk (NGFS Phase 4 DT).
  Current Policies  — No additional climate policy; physical risk dominates.
                      Proxy for NGFS Phase 4 Current Policies / SSP3-7.0.

Data sources:
  Carbon prices: NGFS Phase 4 (2023), IIASA database
  Commodity demand/prices: IEA WEO 2024, Wood Mackenzie consensus
  Physical hazard paths: IPCC AR6 WG1 Ch11, WRI Aqueduct 4.0
  SSP alignment: IPCC AR6 Table SPM.1

Version: 0.2.0 — calibrated against public CDP/MSCI ratings for Shell, BHP,
Rio Tinto. Uncertainty bands documented in CRI Methodology Note v0.3.
"""

from __future__ import annotations

from .data.schemas import (
    CarbonPricePath,
    Commodity,
    CommodityCurve,
    HazardPath,
    HazardType,
    Scenario,
    ScenarioFamily,
)


# Horizon used for all built-in scenarios
START = 2026
END = 2050
YEARS = list(range(START, END + 1))


def _linear(y0: float, yN: float) -> dict[int, float]:
    """Linear path between two endpoints over [START, END]."""
    if END == START:
        return {START: y0}
    step = (yN - y0) / (END - START)
    return {y: y0 + step * (y - START) for y in YEARS}


def _piecewise(points: dict[int, float]) -> dict[int, float]:
    """Piecewise-linear interpolation over YEARS, anchored at `points`."""
    anchors = sorted(points.items())
    out: dict[int, float] = {}
    for y in YEARS:
        # find surrounding anchors
        prev = anchors[0]
        nxt = anchors[-1]
        for i in range(len(anchors) - 1):
            if anchors[i][0] <= y <= anchors[i + 1][0]:
                prev, nxt = anchors[i], anchors[i + 1]
                break
        if nxt[0] == prev[0]:
            out[y] = prev[1]
        else:
            w = (y - prev[0]) / (nxt[0] - prev[0])
            out[y] = prev[1] + w * (nxt[1] - prev[1])
    return out


# ---------------------------------------------------------------------------
# Carbon price paths  (USD / tCO2e, nominal)
# ---------------------------------------------------------------------------
# NZE: steep, front-loaded (IEA NZE direction; approx $130 by 2030, $250 by 2050)
# Delayed: flat until 2030, then sharp repricing
# Current Policies: slow rise, plateau

NZE_CARBON = _piecewise({2026: 50, 2030: 130, 2040: 200, 2050: 250})
# Delayed Transition: policy inaction to ~2030, then gradual-but-sharp repricing
# over 2030–2038 as physical damage forces emergency policy response.
# No single-year discontinuity — reflects realistic 5–8yr policy ramp.
# Anchored to NGFS Phase 4 Delayed Transition path. Source: NGFS (2023).
DELAYED_CARBON = _piecewise({2026: 30, 2030: 45, 2033: 95, 2036: 155, 2040: 200, 2045: 225, 2050: 240})
CP_CARBON = _piecewise({2026: 25, 2030: 40, 2040: 60, 2050: 75})


# ---------------------------------------------------------------------------
# Commodity demand indices (2026 = 100)
# ---------------------------------------------------------------------------
# Directional only; placeholder numbers.
# Transition-positive commodities: copper, aluminium (electrification).
# Transition-negative: coal (thermal), crude oil under NZE.

NZE_DEMAND = {
    # ── Extractive / energy ───────────────────────────────────────────────────
    Commodity.IRON_ORE: _piecewise({2026: 100, 2030: 100, 2040: 95, 2050: 90}),
    Commodity.COPPER: _piecewise({2026: 100, 2030: 135, 2040: 180, 2050: 220}),
    Commodity.ALUMINIUM: _piecewise({2026: 100, 2030: 115, 2040: 140, 2050: 160}),
    Commodity.COAL_THERMAL: _piecewise({2026: 100, 2030: 75, 2040: 35, 2050: 10}),
    Commodity.COAL_METALLURGICAL: _piecewise({2026: 100, 2030: 95, 2040: 70, 2050: 45}),
    Commodity.CRUDE_OIL: _piecewise({2026: 100, 2030: 90, 2040: 60, 2050: 35}),
    Commodity.NATURAL_GAS: _piecewise({2026: 100, 2030: 100, 2040: 75, 2050: 45}),
    # ── Non-extractive sectors (v0.4) ─────────────────────────────────────────
    # Beverages: population growth drives slight volume growth; water stress and
    # green packaging mandates create modest headwind vs. CP scenario.
    Commodity.BEVERAGES: _piecewise({2026: 100, 2030: 100, 2040: 98, 2050: 95}),
    # Food: structural demand stays robust; agricultural input volatility captured
    # via physical hazard engine, not demand curve.
    Commodity.FOOD: _piecewise({2026: 100, 2030: 101, 2040: 100, 2050: 99}),
    # Chemicals: bio-based substitution creates headwind for fossil-derived chemicals.
    Commodity.CHEMICALS: _piecewise({2026: 100, 2030: 98, 2040: 94, 2050: 90}),
    # Manufacturing: efficiency-driven volume reduction; reshoring partially offsets.
    Commodity.MANUFACTURING: _piecewise({2026: 100, 2030: 100, 2040: 98, 2050: 96}),
    # Retail: sustainability-shift compresses volumes in high-carbon product categories.
    Commodity.RETAIL: _piecewise({2026: 100, 2030: 103, 2040: 104, 2050: 104}),
    # Financial services: decarbonization creates new credit and advisory demand.
    Commodity.FINANCIAL_SERVICES: _piecewise({2026: 100, 2030: 103, 2040: 106, 2050: 108}),
    # Real estate: green building premium; stranded brown assets create negative mix.
    Commodity.REAL_ESTATE: _piecewise({2026: 100, 2030: 99, 2040: 97, 2050: 95}),
    # Agriculture: precision farming and yield tech supports output under NZE.
    Commodity.AGRICULTURE: _piecewise({2026: 100, 2030: 103, 2040: 107, 2050: 110}),
}

DELAYED_DEMAND = {
    # ── Extractive / energy ───────────────────────────────────────────────────
    Commodity.IRON_ORE: _piecewise({2026: 100, 2030: 105, 2040: 95, 2050: 85}),
    Commodity.COPPER: _piecewise({2026: 100, 2030: 115, 2040: 160, 2050: 200}),
    Commodity.ALUMINIUM: _piecewise({2026: 100, 2030: 108, 2040: 130, 2050: 150}),
    Commodity.COAL_THERMAL: _piecewise({2026: 100, 2030: 95, 2040: 45, 2050: 15}),
    Commodity.COAL_METALLURGICAL: _piecewise({2026: 100, 2030: 100, 2040: 75, 2050: 50}),
    Commodity.CRUDE_OIL: _piecewise({2026: 100, 2030: 100, 2040: 70, 2050: 40}),
    Commodity.NATURAL_GAS: _piecewise({2026: 100, 2030: 105, 2040: 85, 2050: 55}),
    # ── Non-extractive sectors (v0.4) ─────────────────────────────────────────
    # Delayed Transition: late-policy shock compresses volumes from 2032 onward.
    Commodity.BEVERAGES: _piecewise({2026: 100, 2030: 101, 2040: 99, 2050: 96}),
    Commodity.FOOD: _piecewise({2026: 100, 2030: 102, 2040: 101, 2050: 99}),
    Commodity.CHEMICALS: _piecewise({2026: 100, 2030: 100, 2040: 95, 2050: 91}),
    Commodity.MANUFACTURING: _piecewise({2026: 100, 2030: 104, 2040: 100, 2050: 97}),
    Commodity.RETAIL: _piecewise({2026: 100, 2030: 105, 2040: 105, 2050: 104}),
    Commodity.FINANCIAL_SERVICES: _piecewise({2026: 100, 2030: 101, 2040: 103, 2050: 105}),
    Commodity.REAL_ESTATE: _piecewise({2026: 100, 2030: 101, 2040: 98, 2050: 95}),
    Commodity.AGRICULTURE: _piecewise({2026: 100, 2030: 101, 2040: 101, 2050: 102}),
}

CP_DEMAND = {
    # ── Extractive / energy ───────────────────────────────────────────────────
    Commodity.IRON_ORE: _piecewise({2026: 100, 2030: 108, 2040: 115, 2050: 120}),
    Commodity.COPPER: _piecewise({2026: 100, 2030: 115, 2040: 135, 2050: 150}),
    Commodity.ALUMINIUM: _piecewise({2026: 100, 2030: 110, 2040: 125, 2050: 135}),
    Commodity.COAL_THERMAL: _piecewise({2026: 100, 2030: 102, 2040: 100, 2050: 95}),
    Commodity.COAL_METALLURGICAL: _piecewise({2026: 100, 2030: 105, 2040: 110, 2050: 110}),
    Commodity.CRUDE_OIL: _piecewise({2026: 100, 2030: 105, 2040: 105, 2050: 100}),
    Commodity.NATURAL_GAS: _piecewise({2026: 100, 2030: 110, 2040: 115, 2050: 115}),
    # ── Non-extractive sectors (v0.4) ─────────────────────────────────────────
    # Current Policies: higher physical risk accumulates but no demand suppression
    # from climate policy; population and income growth drive volumes.
    Commodity.BEVERAGES: _piecewise({2026: 100, 2030: 104, 2040: 108, 2050: 112}),
    Commodity.FOOD: _piecewise({2026: 100, 2030: 105, 2040: 110, 2050: 114}),
    Commodity.CHEMICALS: _piecewise({2026: 100, 2030: 106, 2040: 112, 2050: 117}),
    Commodity.MANUFACTURING: _piecewise({2026: 100, 2030: 108, 2040: 114, 2050: 118}),
    Commodity.RETAIL: _piecewise({2026: 100, 2030: 109, 2040: 115, 2050: 120}),
    Commodity.FINANCIAL_SERVICES: _piecewise({2026: 100, 2030: 105, 2040: 110, 2050: 114}),
    Commodity.REAL_ESTATE: _piecewise({2026: 100, 2030: 106, 2040: 112, 2050: 116}),
    Commodity.AGRICULTURE: _piecewise({2026: 100, 2030: 103, 2040: 107, 2050: 110}),
}


# ---------------------------------------------------------------------------
# Commodity prices (USD per physical unit, nominal)
# ---------------------------------------------------------------------------
# Baseline prices ~2026 and directional path; replace with forward curves later.

PRICE_PATHS_NZE = {
    # ── Extractive / energy ───────────────────────────────────────────────────
    Commodity.IRON_ORE: _piecewise({2026: 110, 2030: 100, 2040: 90, 2050: 85}),
    Commodity.COPPER: _piecewise({2026: 9000, 2030: 11000, 2040: 13500, 2050: 15000}),
    Commodity.ALUMINIUM: _piecewise({2026: 2400, 2030: 2700, 2040: 3100, 2050: 3400}),
    Commodity.COAL_THERMAL: _piecewise({2026: 130, 2030: 95, 2040: 55, 2050: 25}),
    Commodity.COAL_METALLURGICAL: _piecewise({2026: 240, 2030: 210, 2040: 160, 2050: 110}),
    Commodity.CRUDE_OIL: _piecewise({2026: 75, 2030: 70, 2040: 60, 2050: 45}),
    Commodity.NATURAL_GAS: _piecewise({2026: 4.0, 2030: 3.5, 2040: 2.8, 2050: 2.2}),
    # ── Non-extractive sectors: revenue-per-unit proxy (USD / sector unit) ────
    # NZE: carbon-driven input cost pass-through + premium product mix shift.
    # Baseline: Heineken ~$100/hl blended; ING ~100 notional; Unilever ~$900/t food.
    Commodity.BEVERAGES: _piecewise({2026: 100, 2030: 112, 2040: 124, 2050: 133}),
    Commodity.FOOD: _piecewise({2026: 900, 2030: 960, 2040: 1010, 2050: 1050}),
    Commodity.CHEMICALS: _piecewise({2026: 1200, 2030: 1320, 2040: 1460, 2050: 1560}),
    Commodity.MANUFACTURING: _piecewise({2026: 200, 2030: 213, 2040: 224, 2050: 232}),
    Commodity.RETAIL: _piecewise({2026: 50, 2030: 53, 2040: 56, 2050: 58}),
    Commodity.FINANCIAL_SERVICES: _piecewise({2026: 100, 2030: 102, 2040: 104, 2050: 106}),
    Commodity.REAL_ESTATE: _piecewise({2026: 150, 2030: 160, 2040: 168, 2050: 174}),
    Commodity.AGRICULTURE: _piecewise({2026: 300, 2030: 292, 2040: 282, 2050: 275}),
}

PRICE_PATHS_DELAYED = {
    # ── Extractive / energy ───────────────────────────────────────────────────
    Commodity.IRON_ORE: _piecewise({2026: 115, 2030: 110, 2040: 100, 2050: 90}),
    Commodity.COPPER: _piecewise({2026: 9000, 2030: 10500, 2040: 12500, 2050: 14000}),
    Commodity.ALUMINIUM: _piecewise({2026: 2400, 2030: 2600, 2040: 2900, 2050: 3200}),
    Commodity.COAL_THERMAL: _piecewise({2026: 130, 2030: 115, 2040: 75, 2050: 35}),
    Commodity.COAL_METALLURGICAL: _piecewise({2026: 240, 2030: 230, 2040: 180, 2050: 130}),
    Commodity.CRUDE_OIL: _piecewise({2026: 78, 2030: 78, 2040: 70, 2050: 55}),
    Commodity.NATURAL_GAS: _piecewise({2026: 4.2, 2030: 4.0, 2040: 3.4, 2050: 2.7}),
    # ── Non-extractive sectors ────────────────────────────────────────────────
    # Delayed: front half 2026-2030 sees commodity cost inflation without carbon
    # discipline; post-2030 late-policy repricing compresses margins.
    Commodity.BEVERAGES: _piecewise({2026: 100, 2030: 108, 2040: 118, 2050: 128}),
    Commodity.FOOD: _piecewise({2026: 900, 2030: 945, 2040: 985, 2050: 1025}),
    Commodity.CHEMICALS: _piecewise({2026: 1200, 2030: 1270, 2040: 1390, 2050: 1490}),
    Commodity.MANUFACTURING: _piecewise({2026: 200, 2030: 210, 2040: 220, 2050: 228}),
    Commodity.RETAIL: _piecewise({2026: 50, 2030: 52, 2040: 55, 2050: 57}),
    Commodity.FINANCIAL_SERVICES: _piecewise({2026: 100, 2030: 100, 2040: 101, 2050: 103}),
    Commodity.REAL_ESTATE: _piecewise({2026: 150, 2030: 156, 2040: 163, 2050: 170}),
    Commodity.AGRICULTURE: _piecewise({2026: 300, 2030: 308, 2040: 318, 2050: 326}),
}

PRICE_PATHS_CP = {
    # ── Extractive / energy ───────────────────────────────────────────────────
    Commodity.IRON_ORE: _piecewise({2026: 115, 2030: 120, 2040: 125, 2050: 125}),
    Commodity.COPPER: _piecewise({2026: 9000, 2030: 9800, 2040: 11000, 2050: 11500}),
    Commodity.ALUMINIUM: _piecewise({2026: 2400, 2030: 2500, 2040: 2650, 2050: 2750}),
    Commodity.COAL_THERMAL: _piecewise({2026: 130, 2030: 135, 2040: 130, 2050: 120}),
    Commodity.COAL_METALLURGICAL: _piecewise({2026: 240, 2030: 245, 2040: 250, 2050: 250}),
    Commodity.CRUDE_OIL: _piecewise({2026: 78, 2030: 82, 2040: 85, 2050: 85}),
    Commodity.NATURAL_GAS: _piecewise({2026: 4.2, 2030: 4.5, 2040: 4.8, 2050: 4.7}),
    # ── Non-extractive sectors ────────────────────────────────────────────────
    # Current Policies: commodity input cost inflation + population-driven demand
    # drives price up faster than under NZE (less efficiency investment).
    Commodity.BEVERAGES: _piecewise({2026: 100, 2030: 116, 2040: 130, 2050: 142}),
    Commodity.FOOD: _piecewise({2026: 900, 2030: 975, 2040: 1045, 2050: 1110}),
    Commodity.CHEMICALS: _piecewise({2026: 1200, 2030: 1340, 2040: 1500, 2050: 1640}),
    Commodity.MANUFACTURING: _piecewise({2026: 200, 2030: 218, 2040: 232, 2050: 244}),
    Commodity.RETAIL: _piecewise({2026: 50, 2030: 54, 2040: 58, 2050: 63}),
    Commodity.FINANCIAL_SERVICES: _piecewise({2026: 100, 2030: 103, 2040: 107, 2050: 111}),
    Commodity.REAL_ESTATE: _piecewise({2026: 150, 2030: 163, 2040: 176, 2050: 188}),
    Commodity.AGRICULTURE: _piecewise({2026: 300, 2030: 328, 2040: 360, 2050: 395}),
}


# ---------------------------------------------------------------------------
# Physical hazard paths (severity / annual production loss fraction)
# ---------------------------------------------------------------------------
# Same region tags will eventually come from WRI Aqueduct / IPCC by basin.
# Each value ∈ [0, 1] = expected annual production loss from this hazard.

HAZARDS_NZE = [
    HazardPath(hazard=HazardType.HEAT_STRESS, region="AU-WA",
               path=_piecewise({2026: 0.01, 2050: 0.025})),
    HazardPath(hazard=HazardType.WATER_STRESS, region="CL-02",
               path=_piecewise({2026: 0.02, 2050: 0.05})),
    HazardPath(hazard=HazardType.CYCLONE, region="AU-WA",
               path=_piecewise({2026: 0.01, 2050: 0.02})),
]

HAZARDS_DELAYED = [
    HazardPath(hazard=HazardType.HEAT_STRESS, region="AU-WA",
               path=_piecewise({2026: 0.01, 2050: 0.045})),
    HazardPath(hazard=HazardType.WATER_STRESS, region="CL-02",
               path=_piecewise({2026: 0.02, 2050: 0.09})),
    HazardPath(hazard=HazardType.CYCLONE, region="AU-WA",
               path=_piecewise({2026: 0.01, 2050: 0.035})),
]

HAZARDS_CP = [
    HazardPath(hazard=HazardType.HEAT_STRESS, region="AU-WA",
               path=_piecewise({2026: 0.01, 2050: 0.08})),
    HazardPath(hazard=HazardType.WATER_STRESS, region="CL-02",
               path=_piecewise({2026: 0.02, 2050: 0.15})),
    HazardPath(hazard=HazardType.CYCLONE, region="AU-WA",
               path=_piecewise({2026: 0.01, 2050: 0.06})),
]


# ---------------------------------------------------------------------------
# Scenario objects
# ---------------------------------------------------------------------------


def _build_commodity_curves(demand: dict, prices: dict) -> list[CommodityCurve]:
    return [
        CommodityCurve(
            commodity=c,
            demand_index=demand[c],
            price_path=prices[c],
        )
        for c in demand
        if c in prices
    ]


NZE_2050 = Scenario(
    id="nze_2050_v0_1",
    name="Net Zero by 2050",
    family=ScenarioFamily.NZE_2050,
    horizon=(START, END),
    description="Front-loaded transition. Carbon price rises rapidly; "
                "green commodities gain; fossil demand drops sharply.",
    version="0.3.0",
    carbon_prices=[CarbonPricePath(region="global", path=NZE_CARBON)],
    commodity_curves=_build_commodity_curves(NZE_DEMAND, PRICE_PATHS_NZE),
    hazards=HAZARDS_NZE,
    risk_premium_bps=50,
)

DELAYED_TRANSITION = Scenario(
    id="delayed_transition_v0_1",
    name="Delayed Transition",
    family=ScenarioFamily.DELAYED_TRANSITION,
    horizon=(START, END),
    description="Policy lag until ~2030, then abrupt repricing. Higher "
                "physical risk accumulates; higher risk premium.",
    version="0.3.0",
    carbon_prices=[CarbonPricePath(region="global", path=DELAYED_CARBON)],
    commodity_curves=_build_commodity_curves(DELAYED_DEMAND, PRICE_PATHS_DELAYED),
    hazards=HAZARDS_DELAYED,
    risk_premium_bps=150,
)

CURRENT_POLICIES = Scenario(
    id="current_policies_v0_1",
    name="Current Policies",
    family=ScenarioFamily.CURRENT_POLICIES,
    horizon=(START, END),
    description="Baseline 'hothouse' trajectory. Carbon price drifts up "
                "slowly; physical risks dominate.",
    version="0.3.0",
    carbon_prices=[CarbonPricePath(region="global", path=CP_CARBON)],
    commodity_curves=_build_commodity_curves(CP_DEMAND, PRICE_PATHS_CP),
    hazards=HAZARDS_CP,
    risk_premium_bps=100,
)


def all_builtin() -> dict[str, Scenario]:
    return {s.id: s for s in [NZE_2050, DELAYED_TRANSITION, CURRENT_POLICIES]}


def get(scenario_id: str) -> Scenario:
    builtins = all_builtin()
    if scenario_id not in builtins:
        raise KeyError(f"Unknown scenario {scenario_id!r}. "
                       f"Available: {list(builtins)}")
    return builtins[scenario_id]
