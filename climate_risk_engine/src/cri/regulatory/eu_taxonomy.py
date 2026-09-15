"""
EU Taxonomy Alignment Scoring.

Computes turnover / CAPEX / OPEX alignment with the EU Taxonomy for
Sustainable Activities (Regulation (EU) 2020/852) across all six
environmental objectives, using sector-level technical screening criteria
mapped from NACE Rev. 2 codes.

Environmental objectives
------------------------
1. Climate change mitigation (CCM)
2. Climate change adaptation (CCA)
3. Sustainable use / protection of water and marine resources (WTR)
4. Transition to a circular economy (CE)
5. Pollution prevention and control (PPC)
6. Protection and restoration of biodiversity and ecosystems (BIO)

Alignment requires: (a) Substantial Contribution to ≥1 objective,
(b) Do No Significant Harm (DNSH) to all others,
(c) Minimum Social Safeguards (UNGC / OECD).

References
----------
EU Taxonomy Climate Delegated Act (EU) 2021/2139 (CCM + CCA)
EU Taxonomy Environmental Delegated Act (EU) 2023/2485 (WTR, CE, PPC, BIO)
ESMA / Platform on Sustainable Finance guidance notes 2023
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── NACE code → sector key mapping ───────────────────────────────────────────

_SECTOR_TO_NACE: dict[str, str] = {
    "agriculture":     "A01",
    "mining":          "B",
    "oil_gas":         "B06",
    "metals_mining":   "B07",
    "chemicals":       "C20",
    "automotive":      "C29",
    "food_beverage":   "C10",
    "cement":          "C23",
    "energy":          "D35",
    "utilities":       "E36",
    "construction":    "F",
    "transport":       "H",
    "shipping":        "H50",
    "aviation":        "H51",
    "real_estate":     "L68",
    "technology":      "J",
    "financials":      "K64",
    "healthcare":      "Q",
    "consumer":        "G47",
    "industrials":     "C",
    "telecom":         "J61",
    "default":         "C",
}


@dataclass
class ObjectiveAlignment:
    """Alignment result for one environmental objective."""
    objective_id: int
    objective_name: str
    eligible: bool                          # Activity in taxonomy scope
    aligned_turnover_pct: float             # % of turnover substantially contributing
    aligned_capex_pct: float
    aligned_opex_pct: float
    dnsh_compliant: bool                    # Do No Significant Harm
    screening_criteria: str                 # Human-readable basis
    data_quality: int                       # 1-5


@dataclass
class EUTaxonomyReport:
    """Full EU Taxonomy alignment report for one company."""
    company_id: str
    company_name: str
    sector: str
    nace_code: str
    reference_period: str
    objectives: list[ObjectiveAlignment] = field(default_factory=list)
    minimum_social_safeguards: bool = True  # Assumed unless UNGC violation flagged
    methodology_note: str = (
        "EU Taxonomy alignment estimated from sector technical screening criteria "
        "(Climate DA 2021/2139; Environmental DA 2023/2485). "
        "Alignment % is a sector-level proxy — company-level revenue segment "
        "data required for verified disclosure."
    )

    @property
    def total_eligible_turnover_pct(self) -> float:
        return max((o.aligned_turnover_pct for o in self.objectives), default=0.0)

    @property
    def total_aligned_turnover_pct(self) -> float:
        eligible = [o for o in self.objectives if o.eligible and o.dnsh_compliant]
        return max((o.aligned_turnover_pct for o in eligible), default=0.0)

    @property
    def total_aligned_capex_pct(self) -> float:
        eligible = [o for o in self.objectives if o.eligible and o.dnsh_compliant]
        return max((o.aligned_capex_pct for o in eligible), default=0.0)


# ── Sector-level taxonomy lookup table ───────────────────────────────────────
# Structure: sector_key → {objective_id: (eligible, turnover%, capex%, opex%, dnsh, note)}
# Values are sector-level proxies based on Delegated Act technical criteria.

_TAXONOMY_LOOKUP: dict[str, dict[int, tuple[bool, float, float, float, bool, str]]] = {
    # (eligible, turnover_pct, capex_pct, opex_pct, dnsh, note)

    "energy": {
        1: (True,  65.0, 75.0, 60.0, True,
            "Low-carbon electricity generation (solar/wind/hydro) — CCM 4.1/4.2/4.3. "
            "Assumes 65% renewable generation mix; adjust per actual asset portfolio."),
        2: (True,  40.0, 50.0, 35.0, True,
            "Climate-resilient generation infrastructure — CCA 4.1"),
        3: (False,  0.0,  0.0,  0.0, True,  "Not in scope for WTR"),
        4: (False,  0.0,  0.0,  0.0, True,  "Not in scope for CE"),
        5: (True,  20.0, 25.0, 20.0, True,  "Pollution controls at generation sites"),
        6: (False,  0.0,  0.0,  0.0, True,  "Not primary scope for BIO"),
    },
    "utilities": {
        1: (True,  50.0, 60.0, 45.0, True,
            "District heating/cooling with renewables — CCM 4.15; water supply — CCM 5.1"),
        2: (True,  30.0, 40.0, 25.0, True,  "Climate-resilient infrastructure — CCA"),
        3: (True,  35.0, 45.0, 30.0, True,  "Water treatment / supply — WTR 5.1/5.2"),
        4: (True,  20.0, 30.0, 15.0, True,  "Circular waste management — CE 5.5"),
        5: (True,  15.0, 20.0, 10.0, True,  "Pollution prevention at water/waste sites"),
        6: (False,  0.0,  0.0,  0.0, True,  "Not primary scope"),
    },
    "transport": {
        1: (True,  35.0, 55.0, 30.0, True,
            "Zero-emission rail/bus/urban mobility — CCM 6.1-6.5. "
            "55% CAPEX aligned assumes electrification investment programme."),
        2: (True,  20.0, 35.0, 15.0, True,  "Climate-resilient transport infrastructure"),
        3: (False,  0.0,  0.0,  0.0, True,  "Not in scope"),
        4: (True,  10.0, 15.0,  8.0, True,  "Circular fleet management"),
        5: (False,  0.0,  0.0,  0.0, True,  "Not primary scope"),
        6: (False,  0.0,  0.0,  0.0, True,  "Not primary scope"),
    },
    "construction": {
        1: (True,  40.0, 50.0, 35.0, True,
            "Nearly zero-energy buildings — CCM 7.1/7.2. "
            "Eligible where construction meets EPBD nZEB or 10% better than NZEB benchmark."),
        2: (True,  25.0, 35.0, 20.0, True,  "Climate-adaptive building design — CCA 7.1"),
        3: (True,  15.0, 20.0, 10.0, True,  "Water-efficient buildings — WTR 7.1"),
        4: (True,  20.0, 30.0, 15.0, True,  "Construction and demolition waste — CE 3.1"),
        5: (True,  10.0, 15.0,  8.0, True,  "Pollution prevention in construction"),
        6: (True,  10.0, 15.0,  8.0, True,  "Biodiversity-positive site design"),
    },
    "real_estate": {
        1: (True,  55.0, 65.0, 50.0, True,
            "Energy-efficient buildings — CCM 7.7 (renovation) + 7.1 (acquisition). "
            "Aligned where EPC ≥ B or top 15% of national building stock."),
        2: (True,  30.0, 40.0, 25.0, True,  "Climate-resilient real estate — CCA 7.1"),
        3: (True,  15.0, 20.0, 12.0, True,  "Water-efficient buildings — WTR 7.1"),
        4: (True,  15.0, 20.0, 10.0, True,  "Construction waste minimisation — CE 3.1"),
        5: (True,   8.0, 12.0,  6.0, True,  "Pollution prevention in buildings"),
        6: (True,  10.0, 15.0,  8.0, True,  "Biodiversity in building/site design"),
    },
    "agriculture": {
        1: (True,  20.0, 30.0, 15.0, True,
            "Low-carbon farming — CCM 1.1/1.2 (soil carbon sequestration, agroforestry). "
            "Eligible where regenerative practices documented."),
        2: (True,  25.0, 35.0, 20.0, True,  "Climate-resilient agriculture — CCA 1.1"),
        3: (True,  30.0, 40.0, 25.0, True,  "Sustainable irrigation / water use — WTR 1.1"),
        4: (True,  15.0, 20.0, 12.0, True,  "Circular nutrient management — CE 1.1"),
        5: (True,  20.0, 25.0, 15.0, True,  "Reduction of pesticide / fertiliser pollution"),
        6: (True,  25.0, 35.0, 20.0, True,  "Biodiversity-supportive farming — BIO 1.1"),
    },
    "technology": {
        1: (True,  30.0, 45.0, 25.0, True,
            "Data centres powered by renewables — CCM 8.1. "
            "Aligned where PUE < 1.5 and ≥90% renewable energy."),
        2: (False,  0.0,  0.0,  0.0, True,  "Not in scope unless climate solutions provider"),
        3: (False,  0.0,  0.0,  0.0, True,  "Not in scope"),
        4: (True,  15.0, 20.0, 12.0, True,  "Circular electronics / e-waste — CE 8.1"),
        5: (True,   8.0, 10.0,  6.0, True,  "Reduction of electronic waste pollution"),
        6: (False,  0.0,  0.0,  0.0, True,  "Not primary scope"),
    },
    "financials": {
        1: (True,  45.0, 50.0, 40.0, True,
            "Green lending / investment aligned with taxonomy — CCM across sectors. "
            "Aligned assets = green loan book % × counterparty taxonomy alignment."),
        2: (True,  20.0, 25.0, 15.0, True,  "Climate-resilient financial products"),
        3: (True,  10.0, 12.0,  8.0, True,  "Water-related finance"),
        4: (True,  10.0, 12.0,  8.0, True,  "Circular economy finance"),
        5: (True,   8.0, 10.0,  6.0, True,  "Pollution prevention finance"),
        6: (True,   8.0, 10.0,  6.0, True,  "Biodiversity finance / nature finance"),
    },
    "chemicals": {
        1: (True,  15.0, 25.0, 10.0, False,
            "Low-carbon chemical processes — CCM 3.1. "
            "DNSH concern: chemical pollution risk under PPC objective."),
        2: (True,  10.0, 15.0,  8.0, False, "Climate adaptation — DNSH concern (pollution)"),
        3: (False,  0.0,  0.0,  0.0, False, "Water pollution risk — DNSH likely fails"),
        4: (True,  20.0, 30.0, 15.0, True,  "Circular chemistry — CE 3.1"),
        5: (False,  0.0,  0.0,  0.0, False, "Primary pollution sector — fails DNSH"),
        6: (False,  0.0,  0.0,  0.0, False, "Biodiversity risk — DNSH likely fails"),
    },
    "oil_gas": {
        1: (False,  0.0,  0.0,  0.0, False,
            "Fossil fuel extraction / refining not taxonomy-eligible for CCM. "
            "Exception: CH4 emissions reduction investments may be eligible under CCM 4.29."),
        2: (True,   5.0, 10.0,  3.0, False, "Minimal CCA eligibility; DNSH concerns"),
        3: (False,  0.0,  0.0,  0.0, False, "Water pollution risk"),
        4: (False,  0.0,  0.0,  0.0, False, "Not in scope"),
        5: (False,  0.0,  0.0,  0.0, False, "Pollution sector — fails DNSH"),
        6: (False,  0.0,  0.0,  0.0, False, "Biodiversity harm — fails DNSH"),
    },
    "industrials": {
        1: (True,  25.0, 40.0, 20.0, True,
            "Industrial decarbonisation — CCM 3.1-3.18. "
            "Eligible where processes below EU ETS benchmark or on transition pathway."),
        2: (True,  15.0, 25.0, 10.0, True,  "Climate-resilient manufacturing"),
        3: (True,   8.0, 12.0,  6.0, True,  "Water efficiency in manufacturing — WTR"),
        4: (True,  20.0, 30.0, 15.0, True,  "Circular manufacturing — CE 3.1-3.6"),
        5: (True,  10.0, 15.0,  8.0, True,  "Pollution prevention in manufacturing"),
        6: (False,  0.0,  0.0,  0.0, True,  "Not primary scope"),
    },
    "healthcare": {
        1: (True,  20.0, 30.0, 15.0, True,  "Low-carbon healthcare infrastructure — CCM 7.7"),
        2: (True,  15.0, 20.0, 10.0, True,  "Climate-resilient hospitals — CCA 7.1"),
        3: (True,  10.0, 15.0,  8.0, True,  "Water-efficient medical facilities"),
        4: (True,   8.0, 12.0,  6.0, True,  "Medical waste circular management"),
        5: (True,  10.0, 15.0,  8.0, True,  "Pharmaceutical pollution prevention"),
        6: (False,  0.0,  0.0,  0.0, True,  "Not primary scope"),
    },
}

# Default for unmapped sectors
_DEFAULT_LOOKUP: dict[int, tuple[bool, float, float, float, bool, str]] = {
    1: (True,  15.0, 20.0, 12.0, True,  "Generic sector — limited CCM eligibility"),
    2: (True,  10.0, 15.0,  8.0, True,  "Climate adaptation eligibility"),
    3: (False,  0.0,  0.0,  0.0, True,  "Not in scope"),
    4: (False,  0.0,  0.0,  0.0, True,  "Not in scope"),
    5: (False,  0.0,  0.0,  0.0, True,  "Not in scope"),
    6: (False,  0.0,  0.0,  0.0, True,  "Not in scope"),
}

_OBJECTIVE_NAMES = {
    1: "Climate change mitigation (CCM)",
    2: "Climate change adaptation (CCA)",
    3: "Water and marine resources (WTR)",
    4: "Circular economy (CE)",
    5: "Pollution prevention (PPC)",
    6: "Biodiversity and ecosystems (BIO)",
}


def compute_eu_taxonomy(
    company_id: str,
    company_name: str,
    sector_key: str,
    revenue_eur_m: float,
    capex_eur_m: float,
    opex_eur_m: float,
    ungc_compliant: bool = True,
    reference_period: str = "2024",
    override_alignment: Optional[dict[int, float]] = None,
) -> EUTaxonomyReport:
    """
    Estimate EU Taxonomy alignment for a company.

    Parameters
    ----------
    company_id, company_name : str
    sector_key : str
        CRI internal sector key.
    revenue_eur_m, capex_eur_m, opex_eur_m : float
        Financial base values in EUR millions.
    ungc_compliant : bool
        Whether minimum social safeguards are met.
    reference_period : str
        Reporting year.
    override_alignment : dict[int, float], optional
        Per-objective turnover alignment override (company-reported). If provided,
        replaces sector proxy for that objective.

    Returns
    -------
    EUTaxonomyReport
    """
    lookup = _TAXONOMY_LOOKUP.get(sector_key, _DEFAULT_LOOKUP)
    data_quality = 3 if sector_key in _TAXONOMY_LOOKUP else 1

    objectives: list[ObjectiveAlignment] = []
    for obj_id in range(1, 7):
        eligible, t_pct, c_pct, o_pct, dnsh, note = lookup.get(
            obj_id, _DEFAULT_LOOKUP[obj_id]
        )

        # Apply company override if provided
        if override_alignment and obj_id in override_alignment:
            t_pct = override_alignment[obj_id]
            data_quality = 4  # company-reported

        # Minimum social safeguards gate
        if not ungc_compliant:
            dnsh = False
            note += " [MSS FAIL: UNGC non-compliance]"

        objectives.append(ObjectiveAlignment(
            objective_id=obj_id,
            objective_name=_OBJECTIVE_NAMES[obj_id],
            eligible=eligible,
            aligned_turnover_pct=t_pct if (eligible and dnsh) else 0.0,
            aligned_capex_pct=c_pct if (eligible and dnsh) else 0.0,
            aligned_opex_pct=o_pct if (eligible and dnsh) else 0.0,
            dnsh_compliant=dnsh,
            screening_criteria=note,
            data_quality=data_quality,
        ))

    return EUTaxonomyReport(
        company_id=company_id,
        company_name=company_name,
        sector=sector_key,
        nace_code=_SECTOR_TO_NACE.get(sector_key, "C"),
        reference_period=reference_period,
        objectives=objectives,
        minimum_social_safeguards=ungc_compliant,
    )


def taxonomy_to_dict(report: EUTaxonomyReport) -> dict:
    """Serialise EUTaxonomyReport to a JSON-ready dict."""
    return {
        "company_id":               report.company_id,
        "company_name":             report.company_name,
        "sector":                   report.sector,
        "nace_code":                report.nace_code,
        "reference_period":         report.reference_period,
        "minimum_social_safeguards": report.minimum_social_safeguards,
        "summary": {
            "eligible_turnover_pct":  round(report.total_eligible_turnover_pct, 1),
            "aligned_turnover_pct":   round(report.total_aligned_turnover_pct, 1),
            "aligned_capex_pct":      round(report.total_aligned_capex_pct, 1),
        },
        "objectives": [
            {
                "id":                    o.objective_id,
                "name":                  o.objective_name,
                "eligible":              o.eligible,
                "dnsh_compliant":        o.dnsh_compliant,
                "aligned_turnover_pct":  round(o.aligned_turnover_pct, 1),
                "aligned_capex_pct":     round(o.aligned_capex_pct, 1),
                "aligned_opex_pct":      round(o.aligned_opex_pct, 1),
                "screening_criteria":    o.screening_criteria,
                "data_quality":          o.data_quality,
            }
            for o in report.objectives
        ],
        "methodology_note": report.methodology_note,
    }
