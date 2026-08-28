"""
Financial impact matrix — event type × sector → revenue, duration, capex impact.

Sources
-------
  Munich Re NatCatSERVICE (2000–2023 loss database)
  EM-DAT CRED International Disaster Database
  IATA Economics — airline disruption cost studies (2019, 2022)
  Cruise Lines International Association (CLIA) — port closure data
  World Steel Association — climate risk survey (2021)
  Cement Sustainability Initiative — production disruption analysis

All per-day revenue-at-risk figures are expressed as a fraction of
annual revenue ÷ 365 (i.e., daily revenue = 1.0).  So a coefficient of
0.80 means 80% of that asset's daily revenue is at risk on a peak impact day.

Asset sector codes
------------------
  "steel_mill"         : integrated blast furnace / EAF steel plant
  "port_terminal"      : general cargo / bulk port terminal
  "cruise_terminal"    : cruise ship home/turnaround port
  "airport"            : commercial passenger airport hub
  "cement_plant"       : dry-process cement / clinker plant
  "coal_mine"          : open-cut or underground coal mine
  "iron_ore_mine"      : open-pit iron ore mine
  "lng_terminal"       : LNG export / import terminal
  "office"             : office / headquarters (low physical exposure)
  "default"            : fallback for unknown sectors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .events import EventType, Severity


# ---------------------------------------------------------------------------
# Impact record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImpactRecord:
    """
    Financial impact parameters for a (event_type, severity, sector) triple.

    revenue_at_risk_fraction : 0–1 — fraction of daily revenue lost per day
    disruption_days          : expected operational disruption duration
    capex_trigger_prob       : 0–1 — probability this event triggers unplanned capex
    capex_fraction_of_revenue: expected unplanned capex as fraction of annual revenue
    ebitda_margin_hit_pp     : percentage-point EBITDA margin compression (sustained)
    notes                    : source / calibration comment
    """
    revenue_at_risk_fraction:   float
    disruption_days:            float
    capex_trigger_prob:         float
    capex_fraction_of_revenue:  float
    ebitda_margin_hit_pp:       float
    notes:                      str = ""


# Default / fallback
_DEFAULT = ImpactRecord(
    revenue_at_risk_fraction  = 0.30,
    disruption_days           = 2.0,
    capex_trigger_prob        = 0.05,
    capex_fraction_of_revenue = 0.002,
    ebitda_margin_hit_pp      = 0.5,
    notes = "default — no sector-specific calibration available",
)

# ---------------------------------------------------------------------------
# Lookup table
# Key: (EventType, Severity, sector_code)
# ---------------------------------------------------------------------------

_MATRIX: dict[tuple, ImpactRecord] = {

    # ── Tropical Cyclone ────────────────────────────────────────────────────

    (EventType.TROPICAL_CYCLONE, Severity.MODERATE, "cruise_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 0.60,
        disruption_days           = 2.0,
        capex_trigger_prob        = 0.05,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.3,
        notes = "TS-force: port closure 1-2 days; ships divert. CLIA 2022.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.SIGNIFICANT, "cruise_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 0.90,
        disruption_days           = 4.0,
        capex_trigger_prob        = 0.15,
        capex_fraction_of_revenue = 0.003,
        ebitda_margin_hit_pp      = 0.8,
        notes = "Cat 1: full port closure 3-5 days + rerouting costs. CLIA/Munich Re.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.SEVERE, "cruise_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 10.0,
        capex_trigger_prob        = 0.40,
        capex_fraction_of_revenue = 0.015,
        ebitda_margin_hit_pp      = 2.5,
        notes = "Cat 2-3: terminal damage likely; 7-14 day closure. EM-DAT/CLIA.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.EXTREME, "cruise_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 30.0,
        capex_trigger_prob        = 0.80,
        capex_fraction_of_revenue = 0.060,
        ebitda_margin_hit_pp      = 8.0,
        notes = "Cat 4-5: severe structural damage; 3-8 week closure. Munich Re NatCat.",
    ),

    (EventType.TROPICAL_CYCLONE, Severity.MODERATE, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 0.70,
        disruption_days           = 1.0,
        capex_trigger_prob        = 0.02,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.2,
        notes = "TS: ground stop ~12-24 hrs; ~$18M/day loss for major hub. IATA 2022.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.SIGNIFICANT, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 2.0,
        capex_trigger_prob        = 0.10,
        capex_fraction_of_revenue = 0.003,
        ebitda_margin_hit_pp      = 0.5,
        notes = "Cat 1: full ground stop 1-3 days. IATA disruption cost model.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.SEVERE, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 5.0,
        capex_trigger_prob        = 0.25,
        capex_fraction_of_revenue = 0.010,
        ebitda_margin_hit_pp      = 1.5,
        notes = "Cat 2-3: facility damage, 3-7 day closure. IATA/Munich Re.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.EXTREME, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 21.0,
        capex_trigger_prob        = 0.70,
        capex_fraction_of_revenue = 0.040,
        ebitda_margin_hit_pp      = 5.0,
        notes = "Cat 4-5: major structural damage, 2-4 week closure. EM-DAT.",
    ),

    (EventType.TROPICAL_CYCLONE, Severity.MODERATE, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.40,
        disruption_days           = 2.0,
        capex_trigger_prob        = 0.03,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.3,
        notes = "TS: partial ops reduction, workforce safety protocols. WSA 2021.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.SIGNIFICANT, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.80,
        disruption_days           = 4.0,
        capex_trigger_prob        = 0.12,
        capex_fraction_of_revenue = 0.005,
        ebitda_margin_hit_pp      = 1.0,
        notes = "Cat 1: shutdown + restart costs; rolling mill most vulnerable. WSA.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.SEVERE, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 12.0,
        capex_trigger_prob        = 0.35,
        capex_fraction_of_revenue = 0.025,
        ebitda_margin_hit_pp      = 3.0,
        notes = "Cat 2-3: BF reline risk, cooling system damage. EM-DAT/WSA.",
    ),

    (EventType.TROPICAL_CYCLONE, Severity.SIGNIFICANT, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 0.70,
        disruption_days           = 3.0,
        capex_trigger_prob        = 0.08,
        capex_fraction_of_revenue = 0.003,
        ebitda_margin_hit_pp      = 0.8,
        notes = "Cat 1: quarry flooding, kiln shutdown. CSI disruption survey.",
    ),
    (EventType.TROPICAL_CYCLONE, Severity.SEVERE, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 10.0,
        capex_trigger_prob        = 0.30,
        capex_fraction_of_revenue = 0.020,
        ebitda_margin_hit_pp      = 2.5,
        notes = "Cat 2-3: significant structural damage possible. EM-DAT.",
    ),

    # ── Extratropical Storm / Windstorm ─────────────────────────────────────

    (EventType.EXTRATROPICAL_STORM, Severity.MODERATE, "port_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 0.50,
        disruption_days           = 1.5,
        capex_trigger_prob        = 0.03,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.2,
        notes = "60-80 km/h winds: crane ops halted, 1-2 day backlog. Munich Re.",
    ),
    (EventType.EXTRATROPICAL_STORM, Severity.SIGNIFICANT, "port_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 0.85,
        disruption_days           = 2.0,
        capex_trigger_prob        = 0.08,
        capex_fraction_of_revenue = 0.003,
        ebitda_margin_hit_pp      = 0.5,
        notes = "80-100 km/h: port closure, equipment securing. Munich Re NatCat.",
    ),
    (EventType.EXTRATROPICAL_STORM, Severity.SEVERE, "port_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 4.0,
        capex_trigger_prob        = 0.20,
        capex_fraction_of_revenue = 0.008,
        ebitda_margin_hit_pp      = 1.2,
        notes = ">100 km/h: major closure + structural risk. EM-DAT European windstorms.",
    ),

    (EventType.EXTRATROPICAL_STORM, Severity.SIGNIFICANT, "cruise_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 0.70,
        disruption_days           = 1.5,
        capex_trigger_prob        = 0.05,
        capex_fraction_of_revenue = 0.002,
        ebitda_margin_hit_pp      = 0.4,
        notes = "European windstorm: 1-2 day port closure, itinerary adjustment.",
    ),
    (EventType.EXTRATROPICAL_STORM, Severity.SEVERE, "cruise_terminal"): ImpactRecord(
        revenue_at_risk_fraction  = 0.90,
        disruption_days           = 3.0,
        capex_trigger_prob        = 0.15,
        capex_fraction_of_revenue = 0.006,
        ebitda_margin_hit_pp      = 1.0,
        notes = "Major extratropical: 2-4 day closure. Based on 2013 St Jude storm data.",
    ),

    (EventType.EXTRATROPICAL_STORM, Severity.SIGNIFICANT, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 0.60,
        disruption_days           = 1.0,
        capex_trigger_prob        = 0.02,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.2,
        notes = "Strong windstorm: significant delays, some cancellations. IATA.",
    ),
    (EventType.EXTRATROPICAL_STORM, Severity.SEVERE, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 0.90,
        disruption_days           = 2.0,
        capex_trigger_prob        = 0.08,
        capex_fraction_of_revenue = 0.004,
        ebitda_margin_hit_pp      = 0.8,
        notes = "Major storm: ground stop, missed connection cascade. IATA 2022.",
    ),

    (EventType.EXTRATROPICAL_STORM, Severity.SIGNIFICANT, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.30,
        disruption_days           = 1.0,
        capex_trigger_prob        = 0.04,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.2,
        notes = "Wind: power reliability risk; most mill ops continue indoors.",
    ),

    # ── Heat Wave ───────────────────────────────────────────────────────────

    (EventType.HEAT_WAVE, Severity.SIGNIFICANT, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.15,
        disruption_days           = 5.0,
        capex_trigger_prob        = 0.02,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.8,
        notes = "35-40°C: outdoor labor -15-20%, cooling water stress. WSA 2021.",
    ),
    (EventType.HEAT_WAVE, Severity.SEVERE, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.25,
        disruption_days           = 7.0,
        capex_trigger_prob        = 0.05,
        capex_fraction_of_revenue = 0.003,
        ebitda_margin_hit_pp      = 1.5,
        notes = ">40°C: mandatory work stoppages, cooling system stress. WSA/IPCC AR6.",
    ),

    (EventType.HEAT_WAVE, Severity.SIGNIFICANT, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 0.10,
        disruption_days           = 5.0,
        capex_trigger_prob        = 0.02,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.6,
        notes = "35-40°C: quarry ops restricted; kiln efficiency -5%. CSI survey.",
    ),
    (EventType.HEAT_WAVE, Severity.SEVERE, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 0.20,
        disruption_days           = 7.0,
        capex_trigger_prob        = 0.04,
        capex_fraction_of_revenue = 0.002,
        ebitda_margin_hit_pp      = 1.2,
        notes = ">40°C: kiln clinker quality risk, labor stoppage risk. CSI.",
    ),

    (EventType.HEAT_WAVE, Severity.SIGNIFICANT, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 0.05,
        disruption_days           = 3.0,
        capex_trigger_prob        = 0.01,
        capex_fraction_of_revenue = 0.0005,
        ebitda_margin_hit_pp      = 0.1,
        notes = "Extreme heat: tarmac risk, reduced payload (density altitude). IATA.",
    ),
    (EventType.HEAT_WAVE, Severity.SEVERE, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 0.12,
        disruption_days           = 5.0,
        capex_trigger_prob        = 0.02,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.3,
        notes = ">45°C: Phoenix-2017 type event; density altitude cancellations. FAA data.",
    ),

    # ── Heavy Rain / Flood ──────────────────────────────────────────────────

    (EventType.HEAVY_RAIN_FLOOD, Severity.SIGNIFICANT, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.50,
        disruption_days           = 3.0,
        capex_trigger_prob        = 0.10,
        capex_fraction_of_revenue = 0.005,
        ebitda_margin_hit_pp      = 1.0,
        notes = "Flash flood: raw material stockpile exposure, access disruption. EM-DAT.",
    ),
    (EventType.HEAVY_RAIN_FLOOD, Severity.SEVERE, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.90,
        disruption_days           = 7.0,
        capex_trigger_prob        = 0.30,
        capex_fraction_of_revenue = 0.020,
        ebitda_margin_hit_pp      = 3.0,
        notes = "Major flood: plant inundation risk. Tata Steel Jamshedpur 2008 flood.",
    ),

    (EventType.HEAVY_RAIN_FLOOD, Severity.SIGNIFICANT, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 0.40,
        disruption_days           = 2.0,
        capex_trigger_prob        = 0.08,
        capex_fraction_of_revenue = 0.003,
        ebitda_margin_hit_pp      = 0.8,
        notes = "Flash flood: quarry access, conveyor belt disruption. CSI.",
    ),
    (EventType.HEAVY_RAIN_FLOOD, Severity.SEVERE, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 0.80,
        disruption_days           = 5.0,
        capex_trigger_prob        = 0.25,
        capex_fraction_of_revenue = 0.015,
        ebitda_margin_hit_pp      = 2.5,
        notes = "Major flood: quarry inundation, clinker pile washout. EM-DAT.",
    ),

    (EventType.HEAVY_RAIN_FLOOD, Severity.SIGNIFICANT, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 0.40,
        disruption_days           = 1.0,
        capex_trigger_prob        = 0.05,
        capex_fraction_of_revenue = 0.002,
        ebitda_margin_hit_pp      = 0.3,
        notes = "Heavy rain: runway closures, diversions. IATA 2022.",
    ),
    (EventType.HEAVY_RAIN_FLOOD, Severity.SEVERE, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 0.80,
        disruption_days           = 2.0,
        capex_trigger_prob        = 0.15,
        capex_fraction_of_revenue = 0.005,
        ebitda_margin_hit_pp      = 0.8,
        notes = "Major flood: terminal flooding, multi-day closure. EM-DAT.",
    ),

    # ── Drought ─────────────────────────────────────────────────────────────

    (EventType.DROUGHT, Severity.SIGNIFICANT, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.08,
        disruption_days           = 30.0,
        capex_trigger_prob        = 0.05,
        capex_fraction_of_revenue = 0.003,
        ebitda_margin_hit_pp      = 1.0,
        notes = "Water stress: cooling system constraint, BF coke rate up. WSA 2021.",
    ),
    (EventType.DROUGHT, Severity.SEVERE, "steel_mill"): ImpactRecord(
        revenue_at_risk_fraction  = 0.15,
        disruption_days           = 60.0,
        capex_trigger_prob        = 0.15,
        capex_fraction_of_revenue = 0.010,
        ebitda_margin_hit_pp      = 2.5,
        notes = "Severe water shortage: production curtailment mandatory. IPCC AR6 Ch12.",
    ),

    (EventType.DROUGHT, Severity.SIGNIFICANT, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 0.06,
        disruption_days           = 30.0,
        capex_trigger_prob        = 0.03,
        capex_fraction_of_revenue = 0.002,
        ebitda_margin_hit_pp      = 0.8,
        notes = "Water restriction: cooling + slurry process constraints. CSI.",
    ),
    (EventType.DROUGHT, Severity.SEVERE, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 0.12,
        disruption_days           = 60.0,
        capex_trigger_prob        = 0.10,
        capex_fraction_of_revenue = 0.006,
        ebitda_margin_hit_pp      = 1.8,
        notes = "Severe drought: mandatory water allocation cuts. EM-DAT/CSI.",
    ),

    # ── Wildfire Risk ────────────────────────────────────────────────────────

    (EventType.WILDFIRE_RISK, Severity.SIGNIFICANT, "cement_plant"): ImpactRecord(
        revenue_at_risk_fraction  = 0.20,
        disruption_days           = 3.0,
        capex_trigger_prob        = 0.05,
        capex_fraction_of_revenue = 0.002,
        ebitda_margin_hit_pp      = 0.5,
        notes = "Wildfire smoke/evacuation risk for open quarry operations. CSI.",
    ),
    (EventType.WILDFIRE_RISK, Severity.SIGNIFICANT, "airport"): ImpactRecord(
        revenue_at_risk_fraction  = 0.15,
        disruption_days           = 2.0,
        capex_trigger_prob        = 0.02,
        capex_fraction_of_revenue = 0.001,
        ebitda_margin_hit_pp      = 0.2,
        notes = "Smoke: IFR conditions, reduced capacity. LAX wildfire season data.",
    ),

    # ── Active Disaster (GDACS) ──────────────────────────────────────────────

    (EventType.ACTIVE_DISASTER, Severity.SEVERE, "default"): ImpactRecord(
        revenue_at_risk_fraction  = 0.80,
        disruption_days           = 5.0,
        capex_trigger_prob        = 0.20,
        capex_fraction_of_revenue = 0.010,
        ebitda_margin_hit_pp      = 2.0,
        notes = "GDACS confirmed active disaster. Sector-specific lookup unavailable.",
    ),
    (EventType.ACTIVE_DISASTER, Severity.EXTREME, "default"): ImpactRecord(
        revenue_at_risk_fraction  = 1.00,
        disruption_days           = 14.0,
        capex_trigger_prob        = 0.50,
        capex_fraction_of_revenue = 0.030,
        ebitda_margin_hit_pp      = 5.0,
        notes = "GDACS extreme disaster (tsunami, major cyclone).",
    ),
}


# ---------------------------------------------------------------------------
# Lookup function
# ---------------------------------------------------------------------------

def lookup(
    event_type: EventType,
    severity: Severity,
    sector: str,
) -> ImpactRecord:
    """
    Return the best-matching ImpactRecord for (event_type, severity, sector).

    Falls back gracefully:
      1. Exact match
      2. Severity −1 step with same event + sector
      3. "default" sector with exact event + severity
      4. Global _DEFAULT
    """
    key = (event_type, severity, sector)
    if key in _MATRIX:
        return _MATRIX[key]

    # Try one severity lower
    if severity.value > 0:
        lower = Severity(severity.value - 1)
        k2 = (event_type, lower, sector)
        if k2 in _MATRIX:
            rec = _MATRIX[k2]
            # Scale up by 25% per severity step above calibrated level
            steps = severity.value - lower.value
            scale = 1.25 ** steps
            return ImpactRecord(
                revenue_at_risk_fraction  = min(1.0, rec.revenue_at_risk_fraction * scale),
                disruption_days           = rec.disruption_days * scale,
                capex_trigger_prob        = min(0.95, rec.capex_trigger_prob * scale),
                capex_fraction_of_revenue = rec.capex_fraction_of_revenue * scale,
                ebitda_margin_hit_pp      = rec.ebitda_margin_hit_pp * scale,
                notes = f"{rec.notes} [extrapolated +{steps} severity step]",
            )

    # Default sector fallback
    k3 = (event_type, severity, "default")
    if k3 in _MATRIX:
        return _MATRIX[k3]

    return _DEFAULT
