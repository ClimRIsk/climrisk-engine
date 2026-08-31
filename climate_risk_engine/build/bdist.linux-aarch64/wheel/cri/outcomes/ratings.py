"""
CRI Climate Risk Rating Engine.

Produces a composite A–E climate risk rating from multi-scenario RunResults.
This is the FREE-TIER output: the letter grade and sub-scores are visible to
all users; the underlying numbers and full trajectory are behind the paywall.

Methodology
-----------
Three pillar scores (0–100, higher = worse risk) feed into the composite:

  Physical Risk Score (P)
    - Average annual physical loss cost as % of baseline revenue, across
      all three scenarios, weighted by scenario likelihood.
    - Normalised against sector benchmarks.

  Transition Risk Score (T)
    - EBITDA compression by 2035 under NZE_2050 (most demanding scenario).
    - Carbon cost as % of EBITDA by 2030 under NZE_2050.
    - Carbon intensity trajectory vs. net-zero pathway.

  Financial Impact Score (F)
    - NPV impact % from Current Policies → NZE_2050 (downside delta).
    - WACC uplift from scenario risk premium.
    - Equity value at risk under tail scenario.

Composite = 0.35 × P + 0.35 × T + 0.30 × F  (weights reflect TCFD emphasis)

Rating scale:
  A  → composite ≤ 20   (Low risk, well-aligned or minimal exposure)
  B  → composite ≤ 40   (Moderate risk, manageable transition pathway)
  C  → composite ≤ 60   (Elevated risk, material action required)
  D  → composite ≤ 80   (High risk, significant financial exposure)
  E  → composite  > 80   (Critical risk, near-term financial impact likely)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..data.schemas import RunResults


# ---------------------------------------------------------------------------
# Firm-configurable weight profiles
# ---------------------------------------------------------------------------

class WeightProfile(str, Enum):
    """
    Named weight presets for the composite CRI score.

    No regulator prescribes specific weights. The right distribution depends
    on the firm's use case. Four profiles are provided; firms may also supply
    fully custom weights via the API.

    Profile          Physical  Transition  Financial  Primary use case
    ─────────────────────────────────────────────────────────────────────
    EQUAL            33 %      33 %        33 %       Default — no view
    PHYSICAL_FOCUS   50 %      25 %        25 %       Insurers, real-estate lenders
    TRANSITION_FOCUS 25 %      50 %        25 %       Equity analysts, ESG mandates
    FINANCIAL_FOCUS  25 %      25 %        50 %       Credit committees, M&A
    """
    EQUAL            = "equal"
    PHYSICAL_FOCUS   = "physical_focus"
    TRANSITION_FOCUS = "transition_focus"
    FINANCIAL_FOCUS  = "financial_focus"
    CUSTOM           = "custom"


# (physical_weight, transition_weight, financial_weight) — must sum to 1.0
_PROFILE_WEIGHTS: dict[WeightProfile, tuple[float, float, float]] = {
    WeightProfile.EQUAL:            (1/3,  1/3,  1/3),
    WeightProfile.PHYSICAL_FOCUS:   (0.50, 0.25, 0.25),
    WeightProfile.TRANSITION_FOCUS: (0.25, 0.50, 0.25),
    WeightProfile.FINANCIAL_FOCUS:  (0.25, 0.25, 0.50),
    WeightProfile.CUSTOM:           (1/3,  1/3,  1/3),   # overridden by caller
}


# ---------------------------------------------------------------------------
# Rating output types
# ---------------------------------------------------------------------------


class ClimateRating(str):
    """A letter rating A–E. Subclasses str so it serialises cleanly."""
    VALID = ("A", "B", "C", "D", "E")

    def __new__(cls, value: str) -> "ClimateRating":
        v = str(value).upper()
        if v not in cls.VALID:
            raise ValueError(f"ClimateRating must be one of {cls.VALID}, got {v!r}")
        return super().__new__(cls, v)


@dataclass
class PillarScore:
    name: str
    score: float          # 0–100, higher = more risk
    label: str            # "Low" / "Moderate" / "Elevated" / "High" / "Critical"
    drivers: list[str] = field(default_factory=list)   # top 3 driver strings


@dataclass
class RatingResult:
    """Full rating output — free tier shows rating + pillar labels only."""

    # Composite
    composite_score: float          # 0–100
    rating: ClimateRating
    rating_label: str               # e.g. "Moderate Climate Risk"
    confidence: str                 # "High" / "Medium" / "Low" (driven by data quality)

    # Pillars (sub-scores — shown in paid tier, pillar *labels* shown in free)
    physical: PillarScore
    transition: PillarScore
    financial: PillarScore

    # Scenario used for worst-case calculation
    stress_scenario: str

    # Peer context
    sector_avg_composite: Optional[float] = None
    sector_rank: Optional[str] = None      # e.g. "Top 30%"

    # Summary narrative (1–2 sentences, free tier)
    summary: str = ""

    def as_free_tier(self) -> dict:
        """Return only what a free-tier user should see."""
        return {
            "rating": str(self.rating),
            "rating_label": self.rating_label,
            "confidence": self.confidence,
            "summary": self.summary,
            "pillars": {
                "physical_risk": self.physical.label,
                "transition_risk": self.transition.label,
                "financial_impact": self.financial.label,
            },
            "sector_rank": self.sector_rank,
            "upgrade_prompt": (
                "Unlock full scores, asset-level breakdown, scenario trajectories, "
                "and TCFD / ISSB S2 disclosure reports with CRI Professional."
            ),
        }

    def as_paid_tier(self) -> dict:
        """Return the full rating detail for paid subscribers."""
        return {
            "rating": str(self.rating),
            "rating_label": self.rating_label,
            "composite_score": round(self.composite_score, 1),
            "confidence": self.confidence,
            "summary": self.summary,
            "stress_scenario": self.stress_scenario,
            "sector_avg_composite": self.sector_avg_composite,
            "sector_rank": self.sector_rank,
            "pillars": {
                "physical_risk": {
                    "score": round(self.physical.score, 1),
                    "label": self.physical.label,
                    "key_drivers": self.physical.drivers,
                },
                "transition_risk": {
                    "score": round(self.transition.score, 1),
                    "label": self.transition.label,
                    "key_drivers": self.transition.drivers,
                },
                "financial_impact": {
                    "score": round(self.financial.score, 1),
                    "label": self.financial.label,
                    "key_drivers": self.financial.drivers,
                },
            },
        }


# ---------------------------------------------------------------------------
# Sector benchmarks (simplified — replace with live peer data in production)
# ---------------------------------------------------------------------------

_SECTOR_BENCHMARKS: dict[str, dict] = {
    # ── Extractive / energy ───────────────────────────────────────────────────
    "oil_gas":            {"avg_composite": 68, "stdev": 12},
    "mining":             {"avg_composite": 52, "stdev": 15},
    "utilities":          {"avg_composite": 58, "stdev": 14},
    "steel":              {"avg_composite": 55, "stdev": 13},
    "cement":             {"avg_composite": 60, "stdev": 11},
    "diversified":        {"avg_composite": 50, "stdev": 13},
    # ── Non-extractive sectors (v0.4) ─────────────────────────────────────────
    # Benchmarks derived from MSCI ESG peer data and CDP sector distributions.
    # Physical risk dominated by water/heat; transition risk mainly input costs.
    "beverages":          {"avg_composite": 42, "stdev": 11},
    "food":               {"avg_composite": 38, "stdev": 11},
    "chemicals":          {"avg_composite": 52, "stdev": 13},
    "manufacturing":      {"avg_composite": 45, "stdev": 14},
    "retail":             {"avg_composite": 32, "stdev": 10},
    "financial_services": {"avg_composite": 38, "stdev": 12},
    "real_estate":        {"avg_composite": 48, "stdev": 14},
    "agriculture":        {"avg_composite": 50, "stdev": 16},
    # ── Fallback ──────────────────────────────────────────────────────────────
    "default":            {"avg_composite": 50, "stdev": 13},
}


def _sector_key(sector: str) -> str:
    s = sector.lower()
    # Extractive / energy
    if any(k in s for k in ["oil", "gas", "energy", "petro"]):
        return "oil_gas"
    if any(k in s for k in ["mine", "mining", "iron", "copper", "nickel"]):
        return "mining"
    if "util" in s or "power" in s or "electric" in s:
        return "utilities"
    if "steel" in s or "metal" in s:
        return "steel"
    if "cement" in s or "concrete" in s:
        return "cement"
    # Non-extractive sectors
    if any(k in s for k in ["bever", "brew", "beer", "spirit", "drink", "distil"]):
        return "beverages"
    if any(k in s for k in ["food", "nutri", "packag"]):
        return "food"
    if "chem" in s or "pharma" in s:
        return "chemicals"
    if any(k in s for k in ["manufactur", "industri", "assembl"]):
        return "manufacturing"
    if any(k in s for k in ["retail", "consumer", "fmcg", "wholesale"]):
        return "retail"
    if any(k in s for k in ["bank", "financ", "insur", "invest", "asset manag"]):
        return "financial_services"
    if any(k in s for k in ["real estate", "property", "reit", "constructi"]):
        return "real_estate"
    if any(k in s for k in ["agricult", "farm", "crop", "livestock", "agro"]):
        return "agriculture"
    return "default"


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------


def _score_label(score: float) -> str:
    if score <= 20:
        return "Low"
    if score <= 40:
        return "Moderate"
    if score <= 60:
        return "Elevated"
    if score <= 80:
        return "High"
    return "Critical"


def _rating_label(rating: ClimateRating) -> str:
    return {
        "A": "Low Climate Risk",
        "B": "Moderate Climate Risk",
        "C": "Elevated Climate Risk",
        "D": "High Climate Risk",
        "E": "Critical Climate Risk",
    }[str(rating)]


def _letter(composite: float) -> ClimateRating:
    if composite <= 20:
        return ClimateRating("A")
    if composite <= 40:
        return ClimateRating("B")
    if composite <= 60:
        return ClimateRating("C")
    if composite <= 80:
        return ClimateRating("D")
    return ClimateRating("E")


# ---------------------------------------------------------------------------
# Physical risk score
# ---------------------------------------------------------------------------


def _physical_score(
    nze: RunResults,
    dt: RunResults,
    cp: RunResults,
) -> PillarScore:
    """
    Physical risk score based on the expected annual physical loss cost
    as a share of baseline revenue, stress-tested across scenarios.
    """
    def avg_physical_pct(r: RunResults) -> float:
        """Average physical loss cost / revenue across 2026–2050."""
        if not r.years:
            return 0.0
        ratios = [
            y.physical_loss_cost / max(y.revenue, 1e-9)
            for y in r.years
            if y.revenue > 0
        ]
        return sum(ratios) / len(ratios) if ratios else 0.0

    # Weighted average: NZE (0.30), DT (0.40), CP (0.30)
    # NZE tends to have lower physical risk (faster transition slows warming)
    # CP tends to have highest physical risk by 2050
    p_nze = avg_physical_pct(nze)
    p_dt  = avg_physical_pct(dt)
    p_cp  = avg_physical_pct(cp)
    weighted_avg = 0.30 * p_nze + 0.40 * p_dt + 0.30 * p_cp

    # Normalise: 0% loss → score 5; ≥1.5% loss → score 100
    # Calibrated to recalibrated hazard model (v0.3) where realistic annual
    # physical losses range 0–1.5% of revenue (was 0–8% in v0.1, hence old cap).
    raw_score = min(100.0, max(0.0, (weighted_avg / 0.015) * 100.0))

    # Identify top drivers — thresholds scaled to recalibrated loss range
    drivers: list[str] = []
    if p_cp > 0.006:
        drivers.append(f"Physical loss under Current Policies: {p_cp*100:.2f}% of revenue p.a.")
    if p_dt > 0.005:
        drivers.append(f"Physical loss under Delayed Transition: {p_dt*100:.2f}% of revenue p.a.")
    # Check heat + water stress peaks
    for r, sc in [(nze, "NZE"), (dt, "DT"), (cp, "CP")]:
        peak = max((y.physical_loss_cost for y in r.years), default=0.0)
        peak_rev = max((y.revenue for y in r.years), default=1.0)
        if peak / max(peak_rev, 1e-9) > 0.008:
            drivers.append(f"Peak physical loss {sc} 2050: {peak/peak_rev*100:.2f}% of revenue")
            break
    if not drivers:
        drivers.append("Physical hazard exposure within manageable range")

    return PillarScore(
        name="Physical Risk",
        score=raw_score,
        label=_score_label(raw_score),
        drivers=drivers[:3],
    )


# ---------------------------------------------------------------------------
# Transition risk score
# ---------------------------------------------------------------------------


def _transition_score(nze: RunResults, dt: RunResults) -> PillarScore:
    """
    Transition risk score — three components, all normalised to revenue not EBITDA.

    Using revenue as the denominator (not EBITDA) is critical: EBITDA can be
    zero or negative under NZE compression, which would silence every component
    and produce a spurious Transition score of 0 for companies with real carbon
    exposure. Revenue is always a positive, stable denominator.

    Components
    ----------
    carbon_burden   Carbon cost as % of revenue at 2030 under NZE (0–50 pts).
                    Captures absolute carbon cost load regardless of EBITDA level.

    ebitda_compress EBITDA compression 2026→2035 under NZE as % of baseline
                    revenue — avoids division by zero when EBITDA ≤ 0 (0–35 pts).

    rev_decline     Revenue decline 2026→2040 under NZE, capturing demand-shift
                    and stranded-asset effects on the top line (0–15 pts).
    """

    def yr(r: RunResults, year: int):
        return next((y for y in r.years if y.year == year), None)

    base   = yr(nze, 2026)
    y2030  = yr(nze, 2030)
    y2035  = yr(nze, 2035)
    y2040  = yr(nze, 2040)

    base_rev = base.revenue if base and base.revenue > 0 else 1.0

    # ── Component 1: carbon cost burden (revenue-denominated) ─────────────────
    carbon_burden = 0.0
    if y2030 and y2030.carbon_cost > 0:
        carbon_burden = y2030.carbon_cost / max(y2030.revenue, base_rev)
    # Score: 5% carbon/revenue → ~10 pts; 30%+ → 50 pts
    carbon_pts = min(50.0, carbon_burden * 167.0)

    # ── Component 2: EBITDA compression vs baseline revenue ───────────────────
    # Using base_rev in denominator (not base EBITDA) keeps this non-zero even
    # when a company starts with low margins or when EBITDA turns negative.
    compress_pts = 0.0
    if base and y2035 is not None:
        ebitda_drop = (base.ebitda - y2035.ebitda) if base.ebitda > y2035.ebitda else 0.0
        compress_frac = ebitda_drop / base_rev   # absolute EBITDA drop / revenue
        compress_pts = min(35.0, compress_frac * 175.0)

    # ── Component 3: revenue decline (demand shift / stranded assets) ─────────
    rev_decline_pts = 0.0
    if y2040 and base_rev > 0:
        rev_drop = max(0.0, base_rev - y2040.revenue)
        rev_decline_pts = min(15.0, (rev_drop / base_rev) * 30.0)

    score = min(100.0, carbon_pts + compress_pts + rev_decline_pts)

    # ── Drivers ───────────────────────────────────────────────────────────────
    drivers: list[str] = []
    if carbon_burden > 0.05:
        drivers.append(
            f"Carbon cost {carbon_burden*100:.1f}% of revenue by 2030 (NZE)"
        )
    if compress_pts > 5.0 and base and y2035 is not None:
        ebitda_drop_pct = ((base.ebitda - y2035.ebitda) / base_rev) * 100
        drivers.append(
            f"EBITDA down {ebitda_drop_pct:.0f}% of revenue by 2035 under NZE"
        )
    if rev_decline_pts > 3.0 and y2040:
        rev_drop_pct = ((base_rev - y2040.revenue) / base_rev) * 100
        drivers.append(
            f"Revenue decline {rev_drop_pct:.0f}% by 2040 under NZE (demand shift)"
        )
    if not drivers:
        drivers.append("Transition exposure contained; low carbon cost burden relative to revenue")
    # Tag whether transition risk is from own emissions or commodity demand
    if carbon_burden < 0.01 and compress_pts > 3:
        drivers.append("EBITDA impact driven by commodity demand shift, not own emissions")
    elif carbon_burden >= 0.01:
        drivers.append(f"Own Scope 1+2 carbon cost contributes {min(100, int(carbon_burden*100))}% of revenue under NZE")

    return PillarScore(
        name="Transition Risk",
        score=score,
        label=_score_label(score),
        drivers=drivers[:3],
    )


# ---------------------------------------------------------------------------
# Financial impact score
# ---------------------------------------------------------------------------


def _financial_score(nze: RunResults, cp: RunResults) -> PillarScore:
    """
    Financial impact score from the NPV delta between NZE and Current Policies.
    Current Policies = baseline (no transition action); NZE = stress case.
    """
    # NPV impact (positive = NZE is *worse* than CP, i.e. stranded assets)
    if cp.enterprise_value > 0:
        ev_delta_pct = (cp.enterprise_value - nze.enterprise_value) / cp.enterprise_value
    else:
        ev_delta_pct = 0.0
    ev_delta_pct = max(0.0, ev_delta_pct)   # clamp: we only count downside here

    # WACC uplift
    # WACC delta: CP has higher scenario premium than NZE.
    # Correct sign: measure how much more capital cost the CP scenario carries.
    wacc_delta = max(0.0, cp.wacc_used - nze.wacc_used)

    # Equity value at risk %
    if cp.equity_value > 0:
        equity_var = max(0.0, (cp.equity_value - nze.equity_value) / cp.equity_value)
    else:
        equity_var = 0.0

    # Scoring
    # ev_delta_pct ≥ 60% → ~50 pts; wacc_delta ≥ 3% → ~30 pts; equity_var → 20 pts
    score = (
        min(50.0, ev_delta_pct * 83.3) +
        min(30.0, wacc_delta * 1000.0) +
        min(20.0, equity_var * 33.3)
    )
    score = min(100.0, score)

    drivers: list[str] = []
    if ev_delta_pct > 0.10:
        drivers.append(f"Enterprise value at risk: {ev_delta_pct*100:.0f}% (NZE vs Current Policies)")
    if wacc_delta > 0.005:
        drivers.append(f"WACC uplift {wacc_delta*100:.1f}pp (Current Policies vs NZE) — scenario risk premium")
    if equity_var > 0.15:
        drivers.append(f"Equity value at risk: {equity_var*100:.0f}% under NZE stress")
    if not drivers:
        drivers.append("Financial exposure to transition is moderate and priced in")

    return PillarScore(
        name="Financial Impact",
        score=score,
        label=_score_label(score),
        drivers=drivers[:3],
    )


# ---------------------------------------------------------------------------
# Summary narrative
# ---------------------------------------------------------------------------


def _summary(
    company_name: str,
    rating: ClimateRating,
    physical: PillarScore,
    transition: PillarScore,
    financial: PillarScore,
) -> str:
    r = str(rating)
    p, t, f = physical.label, transition.label, financial.label
    sector_phrase = {
        "A": "well positioned for the net-zero transition",
        "B": "broadly manageable climate exposure with targeted action required",
        "C": "material climate-related financial risks requiring active management",
        "D": "significant near-term financial exposure to both physical and transition risks",
        "E": "critical climate-related risks with potential for near-term asset impairment",
    }[r]
    return (
        f"{company_name} receives a CRI Rating of {r} ({_rating_label(ClimateRating(r))}), "
        f"reflecting {sector_phrase}. "
        f"Physical hazard exposure is {p.lower()}, transition risk is {t.lower()}, "
        f"and financial impact is {f.lower()} across modelled scenarios."
    )


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class RatingEngine:
    """
    Compute a CRI Climate Risk Rating from multi-scenario RunResults.

    Weights are firm-configurable. No regulator prescribes specific weights;
    the right distribution depends on the firm's mandate.

    Usage — named profile:
        engine = RatingEngine()
        result = engine.rate(
            company_name="BHP Group",
            sector="Mining",
            nze_results=nze,
            dt_results=dt,
            cp_results=cp,
            weight_profile=WeightProfile.PHYSICAL_FOCUS,  # insurer use case
        )

    Usage — fully custom weights (must sum to 1.0):
        result = engine.rate(
            ...,
            weight_profile=WeightProfile.CUSTOM,
            custom_weights=(0.40, 0.40, 0.20),
        )
    """

    # Default profile — equal weight; no arbitrary view imposed
    DEFAULT_PROFILE = WeightProfile.EQUAL

    def rate(
        self,
        company_name: str,
        sector: str,
        nze_results:    RunResults,
        dt_results:     RunResults,
        cp_results:     RunResults,
        data_quality:   str = "medium",
        weight_profile: WeightProfile = WeightProfile.EQUAL,
        custom_weights: tuple[float, float, float] | None = None,
    ) -> RatingResult:
        """
        Compute a full RatingResult.

        Args:
            company_name:    Display name for narratives.
            sector:          Sector string for peer benchmarking.
            nze_results:     RunResults under NZE 2050.
            dt_results:      RunResults under Delayed Transition.
            cp_results:      RunResults under Current Policies.
            data_quality:    "low" | "medium" | "high" — controls confidence label.
            weight_profile:  Named weight preset (see WeightProfile enum).
                             Defaults to EQUAL (33/33/33).
            custom_weights:  (physical, transition, financial) tuple summing to 1.0.
                             Only used when weight_profile=CUSTOM.
        """
        # Resolve weights
        if weight_profile == WeightProfile.CUSTOM:
            if custom_weights is None:
                raise ValueError(
                    "custom_weights must be provided when weight_profile=CUSTOM"
                )
            w_p, w_t, w_f = custom_weights
            if abs(w_p + w_t + w_f - 1.0) > 1e-6:
                raise ValueError(
                    f"custom_weights must sum to 1.0, got {w_p + w_t + w_f:.4f}"
                )
        else:
            w_p, w_t, w_f = _PROFILE_WEIGHTS[weight_profile]

        physical   = _physical_score(nze_results, dt_results, cp_results)
        transition = _transition_score(nze_results, dt_results)
        financial  = _financial_score(nze_results, cp_results)

        composite = (
            w_p * physical.score +
            w_t * transition.score +
            w_f * financial.score
        )

        rating = _letter(composite)

        # Confidence from data quality
        confidence = {"low": "Low", "medium": "Medium", "high": "High"}.get(
            data_quality.lower(), "Medium"
        )

        # Sector context
        sk = _sector_key(sector)
        bench = _SECTOR_BENCHMARKS.get(sk, _SECTOR_BENCHMARKS["default"])
        sector_avg = bench["avg_composite"]
        if composite < sector_avg - bench["stdev"]:
            sector_rank = "Top 15% of sector peers"
        elif composite < sector_avg:
            sector_rank = "Above-average vs. sector peers"
        elif composite < sector_avg + bench["stdev"]:
            sector_rank = "Average vs. sector peers"
        else:
            sector_rank = "Below-average vs. sector peers"

        summary = _summary(company_name, rating, physical, transition, financial)

        return RatingResult(
            composite_score=composite,
            rating=rating,
            rating_label=_rating_label(rating),
            confidence=confidence,
            physical=physical,
            transition=transition,
            financial=financial,
            stress_scenario="NZE 2050",
            sector_avg_composite=float(sector_avg),
            sector_rank=sector_rank,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Module-level convenience wrapper (used by orchestrator and backward compat)
# ---------------------------------------------------------------------------

def rate(
    nze_results: RunResults,
    dt_results:  RunResults,
    cp_results:  RunResults,
    company_name: str = "",
    sector: str = "default",
    data_quality: str = "medium",
    weight_profile: WeightProfile = WeightProfile.EQUAL,
    custom_weights: tuple[float, float, float] | None = None,
) -> RatingResult:
    """Module-level convenience wrapper around RatingEngine.rate().

    Signature matches the positional call used in orchestrator.py:
        rate(nze_r2, dly_r2, cp_r)
    """
    engine = RatingEngine()
    return engine.rate(
        company_name=company_name,
        sector=sector,
        nze_results=nze_results,
        dt_results=dt_results,
        cp_results=cp_results,
        data_quality=data_quality,
        weight_profile=weight_profile,
        custom_weights=custom_weights,
    )
