"""
JRC Global Flood Depth-Damage Functions (Huizinga et al. 2017).

Expresses flood damage as a fraction of asset replacement cost value (RCV)
as a function of inundation depth, using log-logistic CDF curves calibrated
per asset class.

Reference
---------
Huizinga, J., De Moel, H., Szewczyk, W. (2017). Global flood depth-damage
  functions: Methodology and the database with guidelines.
  EUR 28552 EN. Publications Office of the European Union, Luxembourg.
  ISBN 978-92-79-67138-1. doi:10.2760/16510
  https://publications.jrc.ec.europa.eu/repository/handle/JRC105688

Curve form
----------
  damage_fraction = 1 / (1 + (depth / d50)^(-k))

where:
  depth : inundation depth in metres (0 = dry)
  d50   : depth at which 50% of maximum damage occurs (m) — from JRC Table 2
  k     : steepness shape parameter — higher k = more abrupt damage onset

Maximum damage (max_damage_usd_m2)
-----------------------------------
From JRC Table 1, inflated from EUR 2010 to USD 2024:
  EUR 2010 → EUR 2024: ×1.43 (ECB HICP)
  EUR 2024 → USD 2024: ×1.08 (ECB average 2024)
  Total multiplier: ×1.54

Asset class assignment
----------------------
Maps CRI sector keys to JRC asset classes. Industries that handle hazardous
materials (chemicals, oil & gas) are assigned a steeper curve (k=3.5) and
higher max_damage because secondary contamination and equipment failure occur
at lower depths and at greater cost.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DamageCurve:
    """Log-logistic depth-damage function for one asset class."""
    asset_class: str
    d50: float               # depth at 50% damage (m)
    k: float                 # steepness shape parameter
    max_damage_usd_m2: float # max damage per m² of floor area (2024 USD)
    description: str

    def damage_fraction(self, depth_m: float) -> float:
        """
        Fraction of asset RCV damaged at given inundation depth (0.0–1.0).
        At depth=0 returns 0; at depth → ∞ approaches 1.
        """
        if depth_m <= 0.0:
            return 0.0
        try:
            return 1.0 / (1.0 + (depth_m / self.d50) ** (-self.k))
        except (ZeroDivisionError, OverflowError):
            return 1.0 if depth_m > self.d50 else 0.0

    def damage_usd(self, depth_m: float, floor_area_m2: float) -> float:
        """
        USD loss at a given depth for an asset with known gross floor area.
        Requires floor_area_m2 — preferred when available from asset schema.
        """
        return self.damage_fraction(depth_m) * self.max_damage_usd_m2 * floor_area_m2

    def damage_from_rcv(self, depth_m: float, rcv_usd: float) -> float:
        """
        USD loss at a given depth using total replacement cost value.
        Use when floor area is unknown (typical case for portfolio-level data).
        """
        return self.damage_fraction(depth_m) * rcv_usd


# ── JRC asset class curves ────────────────────────────────────────────────────
#
# d50 and k derived from JRC 2017 Figure 3 curve fits (global averages).
# max_damage_usd_m2 converted from EUR 2010 Table 1 to USD 2024 (×1.54).

DAMAGE_CURVES: dict[str, DamageCurve] = {

    "industrial": DamageCurve(
        asset_class="industrial",
        d50=1.20,
        k=2.5,
        max_damage_usd_m2=800.0,     # JRC: ~520 EUR 2010 × 1.54 = ~800 USD 2024
        description=(
            "Industrial buildings — manufacturing plants, warehousing, food "
            "processing. Moderate resistance due to reinforced structure; "
            "machinery damage dominates at >0.5 m."
        ),
    ),

    "chemical": DamageCurve(
        asset_class="chemical",
        d50=0.40,
        k=3.5,
        max_damage_usd_m2=1850.0,    # Premium: high equipment cost + HAZMAT cleanup
        description=(
            "Chemical / petrochemical / hazmat facilities. Very steep damage "
            "onset: containment failure, reactor quenching, and toxic release "
            "occur at <0.5 m. HAZMAT remediation costs included in max damage."
        ),
    ),

    "commercial": DamageCurve(
        asset_class="commercial",
        d50=0.85,
        k=2.0,
        max_damage_usd_m2=1120.0,    # JRC: ~730 EUR 2010 × 1.54
        description=(
            "Commercial buildings — offices, retail, data centres, hotels. "
            "IT equipment and fit-out dominate losses at moderate depths."
        ),
    ),

    "residential": DamageCurve(
        asset_class="residential",
        d50=0.60,
        k=1.8,
        max_damage_usd_m2=570.0,     # JRC: ~370 EUR 2010 × 1.54
        description=(
            "Residential buildings. Low d50 reflects vulnerability of ground "
            "floor finishes, appliances, and insulation at shallow depths."
        ),
    ),

    "infrastructure": DamageCurve(
        asset_class="infrastructure",
        d50=2.10,
        k=2.8,
        max_damage_usd_m2=430.0,     # Roads, bridges, ports: structurally robust
        description=(
            "Roads, bridges, rail, ports, airports. Higher d50 reflects "
            "structural resilience; damage occurs from scouring and foundation "
            "undermining at significant depths."
        ),
    ),

    "energy": DamageCurve(
        asset_class="energy",
        d50=0.50,
        k=3.2,
        max_damage_usd_m2=960.0,     # JRC: ~620 EUR 2010 × 1.54; steep onset
        description=(
            "Power plants, substations, wind/solar installations. Steep "
            "curve: electrical equipment fails abruptly at low depths. "
            "Includes transformer replacement and downtime costs."
        ),
    ),

    "agriculture": DamageCurve(
        asset_class="agriculture",
        d50=0.25,
        k=1.5,
        max_damage_usd_m2=38.0,      # Per m² of farmland (JRC Table 1 crop losses)
        description=(
            "Agricultural land — crop loss, soil erosion, equipment damage. "
            "Very low d50: even shallow inundation destroys standing crops. "
            "Expressed per m² of total farm area."
        ),
    ),
}


# ── Sector → asset class mapping ─────────────────────────────────────────────

_SECTOR_TO_ASSET_CLASS: dict[str, str] = {
    # Hazardous — steep chemical curve
    "chemicals":       "chemical",
    "oil_gas":         "chemical",

    # Energy
    "energy":          "energy",
    "utilities":       "energy",

    # Industrial
    "industrials":     "industrial",
    "metals_mining":   "industrial",
    "mining":          "industrial",
    "cement":          "industrial",
    "food_beverage":   "industrial",
    "automotive":      "industrial",
    "beverages":       "industrial",

    # Infrastructure
    "construction":    "infrastructure",
    "transport":       "infrastructure",
    "shipping":        "infrastructure",
    "aviation":        "infrastructure",
    "telecom":         "infrastructure",

    # Agriculture
    "agriculture":     "agriculture",

    # Commercial
    "technology":      "commercial",
    "financials":      "commercial",
    "healthcare":      "commercial",
    "consumer":        "commercial",
    "real_estate":     "commercial",
    "media":           "commercial",

    # Default
    "default":         "industrial",
}


def get_damage_curve(sector_key: str) -> DamageCurve:
    """
    Return the JRC depth-damage curve for a given CRI sector key.

    Parameters
    ----------
    sector_key : str
        CRI internal sector key (e.g. "chemicals", "industrials", "energy").
        Uses "default" → "industrial" for unknown sectors.

    Returns
    -------
    DamageCurve
    """
    asset_class = _SECTOR_TO_ASSET_CLASS.get(sector_key, "industrial")
    return DAMAGE_CURVES[asset_class]


def flood_damage_fraction(depth_m: float, sector_key: str) -> float:
    """
    Convenience wrapper: damage fraction (0–1) at a given depth for a sector.

    Parameters
    ----------
    depth_m : float
        Inundation depth in metres.
    sector_key : str
        CRI sector key.

    Returns
    -------
    float
        Fraction of replacement cost value lost (0 = no damage, 1 = total loss).
    """
    return get_damage_curve(sector_key).damage_fraction(depth_m)


def flood_damage_usd(depth_m: float, sector_key: str, rcv_usd: float) -> float:
    """
    Convenience wrapper: USD loss at a given depth for an asset.

    Parameters
    ----------
    depth_m : float
        Inundation depth in metres.
    sector_key : str
        CRI sector key.
    rcv_usd : float
        Asset replacement cost value in USD.

    Returns
    -------
    float
        Expected loss in USD.
    """
    return get_damage_curve(sector_key).damage_from_rcv(depth_m, rcv_usd)
