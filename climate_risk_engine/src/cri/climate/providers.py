"""Physical-risk data provider abstraction.

CRI is designed to be **data-provider-agnostic** for physical risk. A
Provider implementation converts (asset, scenario_family, year) into an
expected fractional production loss, using whatever underlying data it
has access to.

Built-in providers:
    - ScenarioHazardProvider: reads the hazards baked into a Scenario
      object (the default for tests and demos).
    - ContinuuitiProvider: wraps the Continuuiti REST API (stub — wire
      up in Phase 2 once we have an API key).

Additional providers (Climate X, Mitiga, Munich Re, in-house Aqueduct
pulls) slot in behind the same interface without any engine changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..data.schemas import Asset, Scenario, ScenarioFamily


@dataclass(frozen=True)
class HazardQuery:
    asset: Asset
    year: int
    scenario_family: ScenarioFamily


@dataclass(frozen=True)
class HazardResult:
    """Response from a physical-risk provider for one (asset, year, scenario)."""

    expected_loss_fraction: float          # ∈ [0, 1] — share of production lost
    hazards: dict[str, float]              # hazard_name → severity 0..1
    provider: str                          # e.g., "internal", "continuuiti"
    raw: Optional[dict] = None             # vendor payload for lineage


class PhysicalRiskProvider(ABC):
    """Abstract interface for a physical-risk data source."""

    name: str = "abstract"

    @abstractmethod
    def resolve(self, query: HazardQuery) -> HazardResult:
        ...


# ---------------------------------------------------------------------------
# Default: read from the Scenario object (good for tests and demos)
# ---------------------------------------------------------------------------


class ScenarioHazardProvider(PhysicalRiskProvider):
    """Uses the HazardPath data embedded in a Scenario object.

    This is the provider the engine defaults to. It produces identical
    results to `cri.climate.physical.expected_loss_fraction`, wrapped in
    the provider contract so other modules can consume it uniformly.
    """

    name = "scenario_internal"

    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario

    def resolve(self, query: HazardQuery) -> HazardResult:
        survival = 1.0
        hazards: dict[str, float] = {}
        for hp in self._scenario.hazards:
            if hp.region != query.asset.region:
                continue
            sev = hp.path.get(query.year, 0.0)
            hazards[hp.hazard.value] = sev
            survival *= (1.0 - sev)
        return HazardResult(
            expected_loss_fraction=1.0 - survival,
            hazards=hazards,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# Continuuiti wrapper  (Phase 2 — stubbed for now)
# ---------------------------------------------------------------------------


class ContinuuitiProvider(PhysicalRiskProvider):
    """Wraps the Continuuiti REST API.

    Continuuiti returns 12-hazard scores (0–10) and a composite score
    for each coordinate, under SSP2-4.5 / SSP5-8.5, projected to
    baseline / 2030 / 2040 / 2050 / 2060.

    Mapping notes:
      - NZE 2050           ≈ SSP1-1.9 (not available) -> fall back to SSP2-4.5.
      - Below 2C Orderly   -> SSP2-4.5.
      - Delayed Transition -> interpolate between SSP2-4.5 and SSP5-8.5.
      - Current Policies   -> SSP5-8.5.

    A hazard score 0–10 is converted to a fractional production loss via
    a calibrated mapping (stub here; to be tuned against historical
    disruption data in Phase 2).
    """

    name = "continuuiti"

    _FAMILY_TO_SSP: dict[ScenarioFamily, str] = {
        ScenarioFamily.NZE_2050: "ssp245",
        ScenarioFamily.BELOW_2C_ORDERLY: "ssp245",
        ScenarioFamily.DELAYED_TRANSITION: "ssp245",   # interpolate once wired
        ScenarioFamily.CURRENT_POLICIES: "ssp585",
        ScenarioFamily.CUSTOM: "ssp245",
    }

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://continuuiti.com/api/v1") -> None:
        self._api_key = api_key
        self._base_url = base_url

    def resolve(self, query: HazardQuery) -> HazardResult:
        if not self._api_key:
            # In Phase 1 we don't raise — we just fall back to a zero-loss
            # result with a clear provenance flag, so pipelines don't break.
            return HazardResult(
                expected_loss_fraction=0.0,
                hazards={},
                provider=f"{self.name}:stub",
                raw={"reason": "no_api_key_configured"},
            )

        # Phase 2: actually call the Continuuiti REST API here.
        raise NotImplementedError(
            "ContinuuitiProvider.resolve is wired in Phase 2. Configure "
            "CRI_PHYSICAL_PROVIDER=scenario_internal for now."
        )

    @staticmethod
    def score_to_loss_fraction(score_0_10: float) -> float:
        """Convex map from a 0–10 hazard score to an annual-loss fraction.

        Calibration target (to be tuned in Phase 2):
            score 2  -> 0.5% loss
            score 5  -> 3.0% loss
            score 8  -> 9.0% loss
            score 10 -> 15.0% loss
        A simple power curve that hits these anchors well:
        """
        s = max(0.0, min(10.0, score_0_10))
        return 0.0015 * (s ** 2)


# ---------------------------------------------------------------------------
# Enriched provider: wraps scenario logic with real open-source data
# ---------------------------------------------------------------------------


class EnrichedHazardProvider(PhysicalRiskProvider):
    """
    Uses real open-source data to enrich scenario hazard paths.

    Augments the base ScenarioHazardProvider with:
      - WRI Aqueduct water/flood/drought risk scores
      - NASA NEX-GDDP temperature deltas (via IPCC AR6 proxy)
      - OWID aridity/drought trends (Phase 2)

    This provider is transparent about data lineage via the HazardResult.raw field.
    """

    name = "enriched_scenario"

    def __init__(self, scenario: Scenario, use_live_data: bool = False) -> None:
        """
        Initialize EnrichedHazardProvider.

        Args:
            scenario: Base Scenario object containing embedded hazard paths.
            use_live_data: If True, attempt live WRI API calls; else use lookup table.
        """
        self.base = ScenarioHazardProvider(scenario)
        self.use_live = use_live_data
        self._scenario = scenario

        # Import here to avoid circular dependency
        from . import nasa_heat  # noqa: F401
        from ..connectors.wri_aqueduct import WRIAqueductConnector  # noqa: F401

    def resolve(self, query: HazardQuery) -> HazardResult:
        """
        Resolve hazard with enrichment from real data sources.

        Args:
            query: HazardQuery specifying asset, year, scenario family.

        Returns:
            HazardResult with enriched expected_loss_fraction and detailed hazards.
        """
        # Start with scenario baseline
        base_result = self.base.resolve(query)

        # Augment with real data
        enriched_loss = self._enrich_loss(query, base_result)

        return HazardResult(
            expected_loss_fraction=enriched_loss,
            hazards=base_result.hazards,
            provider=self.name,
            raw={
                "base_scenario_loss": base_result.expected_loss_fraction,
                "enriched_loss": enriched_loss,
                "asset_region": query.asset.region,
                "year": query.year,
            },
        )

    def _enrich_loss(self, query: HazardQuery, base_result: HazardResult) -> float:
        """Blend scenario loss with real-data enrichments."""
        from . import nasa_heat
        from ..connectors.wri_aqueduct import WRIAqueductConnector

        # Get heat stress adjustment from NASA GDDP
        connector = WRIAqueductConnector()
        water_risks = connector.get_region_risk(query.asset.region, year=query.year)

        # Temperature delta for this region/year/scenario
        delta_c = nasa_heat.warming_delta(
            query.asset.region, query.year, query.scenario_family.value
        )

        # Heat stress probability
        baseline_heat = 0.015 * (water_risks.get("water_stress", 2.5) / 2.5)
        heat_prob = nasa_heat.heat_stress_probability(delta_c, baseline_heat)

        # Aggregate: treat scenario loss and heat stress as independent hazards
        # Joint survival = (1 - scenario_loss) * (1 - heat_stress)
        scenario_survival = 1.0 - base_result.expected_loss_fraction
        joint_survival = scenario_survival * (1.0 - heat_prob)
        enriched_loss = 1.0 - joint_survival

        return min(1.0, enriched_loss)
