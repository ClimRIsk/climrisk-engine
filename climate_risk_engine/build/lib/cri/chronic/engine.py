"""
Chronic Physical Risk Engine — ESRS E1 / TCFD aligned.

Quantifies how progressive climate shifts erode asset viability over
short (≤2030), medium (≤2040), and long (≤2050) ESRS time horizons.

Physical hazards modelled
--------------------------
1. Temperature rise       — OPEX (cooling/heating energy), labour productivity
2. Sea-level rise         — asset impairment, flood frequency amplification
3. Water stress           — input cost increase, operational constraint
4. Drought frequency      — supply disruption multiplier
5. Heat stress days       — labour hour loss (WBGT >33°C threshold)

Financial impact linkages
--------------------------
OPEX impact    = revenue_m × energy_intensity × temp_rise × energy_cost_factor
SLR impairment = book_value × coastal_exposure × slr_damage_fraction(elevation, slr_mm)
Water OPEX     = revenue_m × water_intensity × (stress_mult - 1) × cost_factor
Drought loss   = revenue_m × supply_exposure × (drought_mult - 1) × disruption_factor
Labour loss    = revenue_m × heat_days_fraction × labour_intensity × productivity_loss

All coefficients are CSRD/ESRS-grade estimates drawn from:
- McKinsey Global Institute: Climate Risk and Response (2020)
- NGFS Scenario Analysis for Financial Institutions (2022)
- IPCC AR6 WGII (Chapter 16 — key economic sectors and services)
- World Bank: The Changing Wealth of Nations 2021

Usage
-----
    from cri.chronic import ChronicRiskEngine

    engine = ChronicRiskEngine()
    result = engine.assess(
        lat=17.6868, lon=83.2185,
        asset_name='Vizag Steel Plant',
        asset_rev_m=4200,
        book_value_m=1800,
        elevation_m=51,
        sector='heavy_industry',
        scenario='SSP3-7.0',
    )
    print(result.summary)
    d = result.to_dict()   # ESRS E1-9 ready JSON
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .climate_zones import (
    get_region, _t, _slr, WATER_STRESS_BASELINE,
    WATER_STRESS_2050_MULT, DROUGHT_FREQ_MULT
)


# ---------------------------------------------------------------------------
# Sector coefficients
# ---------------------------------------------------------------------------

# Format: sector → (energy_intensity, water_intensity, labour_intensity,
#                   coastal_exposure_adj, supply_chain_exposure)
# energy_intensity  : fraction of revenue exposed to energy cost changes per °C
# water_intensity   : fraction of revenue exposed to water cost changes
# labour_intensity  : fraction of revenue from outdoor/heat-exposed labour
# coastal_adj       : multiplier on SLR impairment (1.0 = standard, 2.0 = very exposed)
# supply_exposure   : fraction of revenue from supply chains vulnerable to drought

SECTOR_COEFFICIENTS: dict[str, dict[str, float]] = {
    "heavy_industry": {
        "energy_intensity":  0.035,   # 3.5% rev per °C (steel/cement are very energy intensive)
        "water_intensity":   0.020,   # 2% rev exposed to water cost
        "labour_intensity":  0.040,   # 4% outdoor/hot work exposure
        "coastal_adj":       1.2,
        "supply_exposure":   0.15,
    },
    "chemicals": {
        "energy_intensity":  0.028,
        "water_intensity":   0.030,   # high water use in chemical processes
        "labour_intensity":  0.025,
        "coastal_adj":       1.3,
        "supply_exposure":   0.20,
    },
    "aviation": {
        "energy_intensity":  0.010,   # jet fuel mostly, less grid exposure
        "water_intensity":   0.005,
        "labour_intensity":  0.015,
        "coastal_adj":       1.5,     # airports often coastal/low-lying
        "supply_exposure":   0.12,
    },
    "shipping_port": {
        "energy_intensity":  0.015,
        "water_intensity":   0.005,
        "labour_intensity":  0.020,
        "coastal_adj":       2.5,     # directly exposed to SLR
        "supply_exposure":   0.10,
    },
    "cement": {
        "energy_intensity":  0.040,   # cement is extremely energy intensive
        "water_intensity":   0.015,
        "labour_intensity":  0.035,
        "coastal_adj":       1.1,
        "supply_exposure":   0.18,
    },
    "generic": {
        "energy_intensity":  0.020,
        "water_intensity":   0.010,
        "labour_intensity":  0.015,
        "coastal_adj":       1.0,
        "supply_exposure":   0.10,
    },
}


# ---------------------------------------------------------------------------
# SLR damage function
# ---------------------------------------------------------------------------

def _slr_damage_fraction(elevation_m: float, slr_mm: float,
                          coastal_adj: float) -> float:
    """
    Estimate fraction of asset value impaired by SLR at a given horizon.

    Logic
    -----
    - If asset elevation > SLR + 3m safety buffer: no impairment
    - As SLR approaches asset elevation, impairment rises steeply
    - Above-water assets are impaired through increased flood frequency
      (storm surge + SLR) even before permanent inundation

    Equation (logistic):
        impairment = coastal_adj × sigmoid((SLR_m - effective_head) / 0.5)

    where effective_head = max(0, elevation_m - 2.0)  [2m baseline storm buffer]
    """
    slr_m = slr_mm / 1000.0
    effective_head = max(0.0, elevation_m - 2.0)    # 2m baseline storm surge assumed
    # If elevation is very high, no SLR exposure
    if elevation_m > 20 and slr_m < 0.3:
        return 0.0
    # Logistic impairment: full damage when SLR = effective_head
    x = (slr_m - effective_head) / 0.5
    sigmoid = 1.0 / (1.0 + math.exp(-x))
    return min(0.80, coastal_adj * sigmoid * 0.25)   # cap at 80% impairment


# ---------------------------------------------------------------------------
# Heat stress days estimate
# ---------------------------------------------------------------------------

def _heat_stress_days(lat: float, temp_rise: float, baseline_hot_days: int) -> int:
    """
    Estimate additional WBGT-threshold (>33°C) days per year at this latitude
    given a temperature rise above baseline.

    A rough 2× per °C warming for hot days (Fischer & Knutti 2015 scaling).
    """
    return int(baseline_hot_days * (2.0 ** (temp_rise / 1.5)))


def _baseline_hot_days(lat: float) -> int:
    """
    Approximate baseline days per year exceeding WBGT 33°C by latitude band.
    """
    alat = abs(lat)
    if alat < 15:  return 80    # tropical: very frequent high WBGT
    if alat < 25:  return 45    # sub-tropical
    if alat < 35:  return 20    # warm temperate
    if alat < 45:  return 5     # temperate
    return 1                    # boreal / polar


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ChronicRiskResult:
    """
    Chronic physical risk assessment for a single asset, ESRS E1-9 aligned.

    Monetary values are in USD millions.
    """
    asset_lat:   float
    asset_lon:   float
    asset_name:  str
    sector:      str
    scenario:    str
    region:      str

    # ── Climate projections ────────────────────────────────────────────────
    temp_rise_2030: float   # °C
    temp_rise_2040: float
    temp_rise_2050: float
    slr_mm_2030:    float   # mm above 1995-2014
    slr_mm_2040:    float
    slr_mm_2050:    float
    water_stress_baseline: float   # 0–5
    water_stress_2050:     float
    drought_freq_mult_2050: float  # multiplier on 1-in-10yr drought

    # ── Annual OPEX uplift (USD m/yr at each horizon) ─────────────────────
    opex_temp_2030:   float   # additional cooling/energy OPEX from temp rise
    opex_temp_2040:   float
    opex_temp_2050:   float
    opex_water_2030:  float   # additional water procurement OPEX
    opex_water_2040:  float
    opex_water_2050:  float

    # ── Revenue at risk ────────────────────────────────────────────────────
    rev_heat_loss_2030: float    # labour-hour loss → revenue impact
    rev_heat_loss_2040: float
    rev_heat_loss_2050: float
    rev_drought_loss_2050: float # supply disruption revenue loss

    # ── Asset impairment (book value fraction and USD m) ──────────────────
    slr_impairment_frac_2050: float
    slr_impairment_m_2050:    float
    is_coastal:               bool   # elevation < 10m

    # ── Stranded asset flags ───────────────────────────────────────────────
    stranded_risk_2040:  str   # LOW / MEDIUM / HIGH / CRITICAL
    stranded_risk_2050:  str
    uninsurable_flag:    bool  # SLR impairment > 40% of book value

    # ── Total chronic CAPEX/OPEX exposure ─────────────────────────────────
    total_annual_opex_2050_m:  float    # total additional annual OPEX by 2050
    total_revenue_risk_2050_m: float    # total revenue at risk by 2050
    gross_var_2050_m:          float    # Gross VaR (no insurance offset)

    # ── Risk ratings ──────────────────────────────────────────────────────
    temperature_risk:   str
    slr_risk:           str
    water_stress_risk:  str
    drought_risk:       str
    overall_chronic:    str

    # ── Narrative ─────────────────────────────────────────────────────────
    key_findings:    list[str] = field(default_factory=list)
    esrs_disclosures: list[str] = field(default_factory=list)
    adaptation_actions: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        lines = [
            f"Chronic Risk — {self.asset_name}  [{self.scenario}]",
            f"  Region: {self.region}   Sector: {self.sector}",
            f"  ─── Climate projections ───────────────────────────",
            f"  Temp rise:      +{self.temp_rise_2030:.1f}°C (2030) / +{self.temp_rise_2040:.1f}°C (2040) / +{self.temp_rise_2050:.1f}°C (2050)",
            f"  SLR:            {self.slr_mm_2030:.0f}mm (2030) / {self.slr_mm_2040:.0f}mm (2040) / {self.slr_mm_2050:.0f}mm (2050)",
            f"  Water stress:   {self.water_stress_baseline:.1f}→{self.water_stress_2050:.1f} (2050)  drought×{self.drought_freq_mult_2050:.1f}",
            f"  ─── Financial exposure ────────────────────────────",
            f"  OPEX uplift/yr: ${self.total_annual_opex_2050_m:.1f}M by 2050",
            f"  Revenue risk:   ${self.total_revenue_risk_2050_m:.1f}M by 2050",
            f"  SLR impairment: ${self.slr_impairment_m_2050:.1f}M  ({self.slr_impairment_frac_2050*100:.0f}% of book)",
            f"  Gross VaR 2050: ${self.gross_var_2050_m:.1f}M",
            f"  ─── Risk ratings ──────────────────────────────────",
            f"  Temperature: {self.temperature_risk:8s}  SLR: {self.slr_risk:8s}",
            f"  Water stress:{self.water_stress_risk:8s}  Drought: {self.drought_risk}",
            f"  OVERALL CHRONIC: {self.overall_chronic}",
            f"  Stranded 2040: {self.stranded_risk_2040}   Stranded 2050: {self.stranded_risk_2050}",
        ]
        if self.uninsurable_flag:
            lines.append("  ⚠  UNINSURABLE FLAG: SLR impairment exceeds 40% of book value")
        for f_ in self.key_findings:
            lines.append(f"  • {f_}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "asset_lat":             self.asset_lat,
            "asset_lon":             self.asset_lon,
            "asset_name":            self.asset_name,
            "sector":                self.sector,
            "scenario":              self.scenario,
            "region":                self.region,
            "projections": {
                "temp_rise_c":    {"2030": self.temp_rise_2030, "2040": self.temp_rise_2040, "2050": self.temp_rise_2050},
                "slr_mm":         {"2030": self.slr_mm_2030,    "2040": self.slr_mm_2040,    "2050": self.slr_mm_2050},
                "water_stress":   {"baseline": self.water_stress_baseline, "2050": self.water_stress_2050},
                "drought_freq_mult_2050": self.drought_freq_mult_2050,
            },
            "financial": {
                "opex_temp_m":        {"2030": self.opex_temp_2030, "2040": self.opex_temp_2040, "2050": self.opex_temp_2050},
                "opex_water_m":       {"2030": self.opex_water_2030,"2040": self.opex_water_2040,"2050": self.opex_water_2050},
                "rev_heat_loss_m":    {"2030": self.rev_heat_loss_2030, "2040": self.rev_heat_loss_2040, "2050": self.rev_heat_loss_2050},
                "rev_drought_loss_m": self.rev_drought_loss_2050,
                "slr_impairment_frac":self.slr_impairment_frac_2050,
                "slr_impairment_m":   self.slr_impairment_m_2050,
                "total_annual_opex_m":self.total_annual_opex_2050_m,
                "total_rev_risk_m":   self.total_revenue_risk_2050_m,
                "gross_var_2050_m":   self.gross_var_2050_m,
            },
            "ratings": {
                "temperature":    self.temperature_risk,
                "slr":            self.slr_risk,
                "water_stress":   self.water_stress_risk,
                "drought":        self.drought_risk,
                "overall_chronic":self.overall_chronic,
            },
            "stranded": {
                "risk_2040":       self.stranded_risk_2040,
                "risk_2050":       self.stranded_risk_2050,
                "uninsurable":     self.uninsurable_flag,
                "is_coastal":      self.is_coastal,
            },
            "key_findings":       self.key_findings,
            "esrs_disclosures":   self.esrs_disclosures,
            "adaptation_actions": self.adaptation_actions,
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ChronicRiskEngine:
    """
    Chronic physical risk engine — ESRS E1 / TCFD aligned.

    Parameters
    ----------
    scenario    : IPCC SSP scenario for projections (default SSP3-7.0 per ESRS materiality)
    """

    SCENARIOS = ("SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5")

    def __init__(self, scenario: str = "SSP3-7.0") -> None:
        if scenario not in self.SCENARIOS:
            raise ValueError(f"scenario must be one of {self.SCENARIOS}")
        self.scenario = scenario

    def assess(
        self,
        lat:           float,
        lon:           float,
        asset_name:    str   = "Asset",
        asset_rev_m:   float = 1000.0,   # annual revenue USD millions
        book_value_m:  float = 500.0,    # net book value USD millions
        elevation_m:   float = 50.0,     # metres above sea level
        sector:        str   = "generic",
        scenario:      Optional[str] = None,
    ) -> ChronicRiskResult:
        """
        Assess chronic physical climate risk for an asset.

        Parameters
        ----------
        lat, lon        : asset coordinates
        asset_rev_m     : annual revenue (USD millions)
        book_value_m    : net book value (USD millions)
        elevation_m     : elevation above MSL (metres) — key for SLR
        sector          : one of heavy_industry / chemicals / aviation /
                          shipping_port / cement / generic
        scenario        : override instance scenario for this call
        """
        sc   = scenario or self.scenario
        coef = SECTOR_COEFFICIENTS.get(sector, SECTOR_COEFFICIENTS["generic"])
        region = get_region(lat, lon)

        # ── 1. Climate projections ──────────────────────────────────────────
        t30, t40, t50 = _t(region, sc)
        s30, s40, s50 = _slr(region, sc)
        ws_base = WATER_STRESS_BASELINE.get(region, 2.0)
        ws_mult = WATER_STRESS_2050_MULT.get(region, 1.2)
        ws_2050 = min(5.0, ws_base * ws_mult)
        dr_mult = DROUGHT_FREQ_MULT.get(region, {}).get(sc, 1.5)

        # ── 2. Temperature → OPEX ──────────────────────────────────────────
        # Energy cost uplift: each °C raises industrial energy OPEX by ~energy_intensity×revenue
        # Source: McKinsey Global Institute (2020) sector calibration
        ei = coef["energy_intensity"]
        opex_t30 = round(asset_rev_m * ei * t30, 2)
        opex_t40 = round(asset_rev_m * ei * t40, 2)
        opex_t50 = round(asset_rev_m * ei * t50, 2)

        # ── 3. Water stress → OPEX ─────────────────────────────────────────
        # Water stress increase from baseline amplified by stress multiplier
        wi = coef["water_intensity"]
        ws_delta_frac = (ws_2050 / max(ws_base, 0.1) - 1.0)     # fractional increase
        opex_w30 = round(asset_rev_m * wi * ws_delta_frac * 0.3, 2)   # partial at 2030
        opex_w40 = round(asset_rev_m * wi * ws_delta_frac * 0.6, 2)
        opex_w50 = round(asset_rev_m * wi * ws_delta_frac, 2)

        # ── 4. Heat stress → labour productivity → revenue loss ─────────────
        # WBGT >33°C → ~20% productivity loss per affected hour (IPCC AR6 WGII 16.4)
        li = coef["labour_intensity"]
        bd = _baseline_hot_days(lat)
        hd30 = _heat_stress_days(lat, t30, bd)
        hd40 = _heat_stress_days(lat, t40, bd)
        hd50 = _heat_stress_days(lat, t50, bd)
        heat_factor = 0.20   # 20% productivity loss per heat-stress day
        rev_h30 = round(asset_rev_m * li * (hd30 - bd) / 365.0 * heat_factor, 2)
        rev_h40 = round(asset_rev_m * li * (hd40 - bd) / 365.0 * heat_factor, 2)
        rev_h50 = round(asset_rev_m * li * (hd50 - bd) / 365.0 * heat_factor, 2)

        # ── 5. Drought → supply disruption → revenue loss ───────────────────
        se = coef["supply_exposure"]
        # Each doubling of drought frequency → 15% revenue disruption per event
        disruption_per_mult = 0.10
        rev_dr50 = round(asset_rev_m * se * (dr_mult - 1.0) * disruption_per_mult, 2)

        # ── 6. SLR → asset impairment ──────────────────────────────────────
        ca = coef["coastal_adj"]
        slr_frac_2050 = _slr_damage_fraction(elevation_m, s50, ca)
        slr_m_2050    = round(book_value_m * slr_frac_2050, 2)
        is_coastal    = elevation_m < 10

        # ── 7. Gross VaR (ESRS E1-9 — no insurance offset) ─────────────────
        total_opex_2050  = round(opex_t50 + opex_w50, 2)
        total_rev_2050   = round(rev_h50 + rev_dr50, 2)
        # Gross VaR = 10-yr cumulative OPEX uplift + revenue loss + asset impairment
        gross_var_2050   = round(10 * (total_opex_2050 + total_rev_2050) + slr_m_2050, 2)

        # ── 8. Risk ratings ─────────────────────────────────────────────────
        temp_risk  = self._rate_temp(t50)
        slr_risk   = self._rate_slr(elevation_m, s50, slr_frac_2050)
        water_risk = self._rate_water(ws_2050)
        dr_risk    = self._rate_drought(dr_mult)
        overall    = self._rate_overall(temp_risk, slr_risk, water_risk, dr_risk)

        # ── 9. Stranded asset flags ─────────────────────────────────────────
        strand40 = self._stranded_risk(
            elevation_m, _slr(region, sc)[1], slr_frac_2050 * 0.6,
            book_value_m, ws_2050, overall
        )
        strand50 = self._stranded_risk(
            elevation_m, s50, slr_frac_2050,
            book_value_m, ws_2050, overall
        )
        uninsurable = slr_frac_2050 > 0.40

        # ── 10. Narratives ──────────────────────────────────────────────────
        findings   = self._build_findings(
            asset_name, region, sc, t50, s50, slr_frac_2050, slr_m_2050,
            ws_base, ws_2050, dr_mult, total_opex_2050, total_rev_2050,
            gross_var_2050, strand50, uninsurable, elevation_m
        )
        disclosures = self._build_esrs(
            asset_name, sc, t50, s50, slr_frac_2050, slr_m_2050,
            gross_var_2050, uninsurable, book_value_m, asset_rev_m, strand50
        )
        actions = self._build_actions(temp_risk, slr_risk, water_risk, dr_risk,
                                       elevation_m, ws_2050)

        return ChronicRiskResult(
            asset_lat=lat, asset_lon=lon, asset_name=asset_name,
            sector=sector, scenario=sc, region=region,
            temp_rise_2030=t30, temp_rise_2040=t40, temp_rise_2050=t50,
            slr_mm_2030=s30, slr_mm_2040=s40, slr_mm_2050=s50,
            water_stress_baseline=ws_base, water_stress_2050=round(ws_2050, 1),
            drought_freq_mult_2050=round(dr_mult, 1),
            opex_temp_2030=opex_t30, opex_temp_2040=opex_t40, opex_temp_2050=opex_t50,
            opex_water_2030=opex_w30, opex_water_2040=opex_w40, opex_water_2050=opex_w50,
            rev_heat_loss_2030=rev_h30, rev_heat_loss_2040=rev_h40, rev_heat_loss_2050=rev_h50,
            rev_drought_loss_2050=rev_dr50,
            slr_impairment_frac_2050=round(slr_frac_2050, 3),
            slr_impairment_m_2050=slr_m_2050,
            is_coastal=is_coastal,
            stranded_risk_2040=strand40, stranded_risk_2050=strand50,
            uninsurable_flag=uninsurable,
            total_annual_opex_2050_m=total_opex_2050,
            total_revenue_risk_2050_m=total_rev_2050,
            gross_var_2050_m=gross_var_2050,
            temperature_risk=temp_risk, slr_risk=slr_risk,
            water_stress_risk=water_risk, drought_risk=dr_risk,
            overall_chronic=overall,
            key_findings=findings,
            esrs_disclosures=disclosures,
            adaptation_actions=actions,
        )

    # ── Risk rating functions ─────────────────────────────────────────────────

    @staticmethod
    def _rate_temp(t50: float) -> str:
        if t50 >= 2.5: return "HIGH"
        if t50 >= 1.5: return "MEDIUM"
        if t50 >= 0.8: return "LOW"
        return "NEGLIGIBLE"

    @staticmethod
    def _rate_slr(elev: float, slr_mm: float, frac: float) -> str:
        if frac > 0.25 or (elev < 3 and slr_mm > 150): return "HIGH"
        if frac > 0.05 or elev < 10:                    return "MEDIUM"
        if elev < 20:                                    return "LOW"
        return "NEGLIGIBLE"

    @staticmethod
    def _rate_water(ws: float) -> str:
        if ws >= 4.0:  return "HIGH"
        if ws >= 3.0:  return "MEDIUM"
        if ws >= 1.5:  return "LOW"
        return "NEGLIGIBLE"

    @staticmethod
    def _rate_drought(mult: float) -> str:
        if mult >= 2.5:  return "HIGH"
        if mult >= 1.7:  return "MEDIUM"
        if mult >= 1.2:  return "LOW"
        return "NEGLIGIBLE"

    @staticmethod
    def _rate_overall(temp, slr, water, drought) -> str:
        scores = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NEGLIGIBLE": 0}
        vals = [scores[r] for r in [temp, slr, water, drought]]
        mx = max(vals)
        avg = sum(vals) / 4
        if mx == 3 or avg >= 2.0: return "HIGH"
        if mx == 2 or avg >= 1.2: return "MEDIUM"
        if mx == 1 or avg >= 0.5: return "LOW"
        return "NEGLIGIBLE"

    @staticmethod
    def _stranded_risk(elev, slr_mm, frac, book_m, ws, overall) -> str:
        """Stranded asset risk: inability to obtain insurance + chronic impairment."""
        impairment_m = book_m * frac
        if (elev < 3 and slr_mm > 200) or frac > 0.50 or impairment_m > 0.6 * book_m:
            return "CRITICAL"
        if (elev < 6 and slr_mm > 150) or frac > 0.25 or overall == "HIGH":
            return "HIGH"
        if frac > 0.05 or overall == "MEDIUM":
            return "MEDIUM"
        return "LOW"

    # ── Narrative builders ────────────────────────────────────────────────────

    @staticmethod
    def _build_findings(asset_name, region, sc, t50, s50, slr_frac, slr_m,
                        ws_base, ws_2050, dr_mult, opex_m, rev_m,
                        gross_var, strand50, uninsurable, elev) -> list[str]:
        fs = []
        fs.append(
            f"Under {sc}, {region} faces +{t50:.1f}°C by 2050 (above 1995–2014 baseline). "
            f"Annual energy and cooling OPEX uplift of ${opex_m:.1f}M represents a structural "
            f"increase in the cost base that cannot be fully passed through in competitive markets."
        )
        if s50 > 150 and elev < 15:
            fs.append(
                f"Sea-level rise of {s50:.0f}mm by 2050 relative to current baseline, combined "
                f"with the asset's {elev:.0f}m elevation, amplifies storm surge return periods. "
                f"A storm that previously occurred 1-in-50 years may become a 1-in-10 year event "
                f"by 2040 at this elevation."
            )
        if slr_frac > 0.05:
            fs.append(
                f"SLR impairment: {slr_frac*100:.0f}% of book value (${slr_m:.1f}M) is classified "
                f"as at material physical risk by 2050 under {sc}."
            )
        if ws_2050 >= 3.5:
            fs.append(
                f"Water stress escalates from {ws_base:.1f} to {ws_2050:.1f} (WRI Aqueduct scale 0–5) "
                f"by 2050. Process-water availability becomes a binding operational constraint, "
                f"likely requiring water recycling investment or allocation agreements."
            )
        if dr_mult >= 1.7:
            fs.append(
                f"Drought frequency multiplier {dr_mult:.1f}× by 2050: historical 1-in-10 year drought "
                f"becomes approximately 1-in-{max(1,int(10/dr_mult))} years. Supply chain disruption "
                f"revenue risk of ${rev_m:.1f}M/yr is material against earnings."
            )
        if uninsurable:
            fs.append(
                f"⚠ UNINSURABLE ASSET: SLR impairment exceeds 40% of book value (${slr_m:.1f}M). "
                f"Commercial underwriters are likely to withdraw flood/storm coverage by 2035–2040. "
                f"Under ESRS E1-9, this exposure of ${slr_m:.1f}M must be reported as 100% unmitigated "
                f"Gross VaR once insurance lapses."
            )
        if strand50 in {"HIGH", "CRITICAL"}:
            fs.append(
                f"Stranded asset risk is {strand50} by 2050. Investors should model managed retreat "
                f"or major adaptation CAPEX as the base case, not the downside scenario."
            )
        return fs

    @staticmethod
    def _build_esrs(asset_name, sc, t50, s50, slr_frac, slr_m,
                    gross_var, uninsurable, book_m, rev_m, strand50) -> list[str]:
        """
        Draft ESRS E1-9 disclosure data points for this asset.
        """
        ds = []
        ds.append(
            f"[ESRS E1-9 / E1-4] Scenario: {sc}. Physical climate horizon: short ≤2030, "
            f"medium ≤2040, long ≤2050. Assessment methodology: IPCC AR6 regional projections "
            f"cross-referenced with asset GPS coordinates and sector financial coefficients."
        )
        ds.append(
            f"[ESRS E1-9 §AR 9] Temperature warming of +{t50:.1f}°C above 1995–2014 baseline by 2050 "
            f"under {sc} has been applied to quantify energy OPEX uplift and heat-stress labour impacts."
        )
        if slr_frac > 0.0:
            ds.append(
                f"[ESRS E1-9 §AR 10] Sea-level rise of {s50:.0f}mm (50th percentile, {sc}) has been "
                f"applied to the asset book value of ${book_m:.0f}M. Estimated impairment: "
                f"${slr_m:.1f}M ({slr_frac*100:.0f}% of book value) under the long-term horizon."
            )
        if uninsurable:
            ds.append(
                f"[ESRS E1-9 §AR 11 — Coverage Gap] As of the assessment date, management confirms "
                f"that commercial flood and storm-surge insurance may become unavailable for this asset "
                f"under {sc} projections. The Gross Value at Risk of ${gross_var:.1f}M represents the "
                f"unmitigated exposure assuming zero third-party risk transfer."
            )
        ds.append(
            f"[ESRS E1-9 §AR 12 — Anticipated Financial Effects] Total cumulative Gross VaR over "
            f"the long-term horizon: ${gross_var:.1f}M. This includes 10-year cumulative OPEX uplift "
            f"and revenue risk plus one-time asset impairment. No insurance offset has been applied."
        )
        if strand50 in {"HIGH", "CRITICAL"}:
            ds.append(
                f"[ESRS E1-2 / E1-3 — Policies and Actions] Stranded asset risk is {strand50}. "
                f"Board is required to disclose one of: (a) Managed Retreat CAPEX plan, "
                f"(b) Self-insurance captive reserve, or (c) Adaptation CAPEX engineering plan "
                f"with costed timeline under ESRS E1-3."
            )
        return ds

    @staticmethod
    def _build_actions(temp_risk, slr_risk, water_risk, drought_risk, elev, ws_2050) -> list[str]:
        acts = []
        if temp_risk in {"HIGH", "MEDIUM"}:
            acts.append(
                "Conduct energy audit to identify electrification and efficiency measures that "
                "decouple production volume from cooling energy consumption."
            )
            acts.append(
                "Model heat-stress protocols: WBGT monitoring, scheduled rest periods, "
                "shift pattern redesign for outdoor roles to reduce WBGT-hour exposure."
            )
        if slr_risk in {"HIGH", "MEDIUM"}:
            if elev < 5:
                acts.append(
                    "Commission a site-specific coastal flood risk study including SLR + storm surge "
                    "compound event modelling at 2040 and 2050 return periods."
                )
                acts.append(
                    "Evaluate parametric flood insurance trigger (e.g., storm surge > 1.5m at "
                    "nearest tide gauge) to maintain liquidity cover while commercial insurance remains available."
                )
            else:
                acts.append(
                    "Monitor flood insurance market annually; establish self-insurance reserve "
                    "schedule if commercial rates exceed 3% of asset value per year."
                )
        if water_risk in {"HIGH", "MEDIUM"} or ws_2050 >= 3.0:
            acts.append(
                "Conduct water efficiency audit; target closed-loop recycling systems to reduce "
                "fresh-water withdrawal by ≥30% within 5 years. Secure long-term water allocation "
                "agreements with basin authorities before 2030."
            )
        if drought_risk in {"HIGH", "MEDIUM"}:
            acts.append(
                "Map top-10 suppliers against drought hazard zones; establish dual-sourcing "
                "or strategic inventory buffers for inputs most exposed to drought disruption."
            )
        if not acts:
            acts.append(
                "Chronic risk exposure is LOW to NEGLIGIBLE at this location under the assessed scenario. "
                "Include in routine 3-year climate risk review cycle."
            )
        return acts
