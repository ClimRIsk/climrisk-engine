"""
Sector-Specific Physical Damage Chain Functions.

Each sector function takes a physical event, the asset's hazard profile
(with event multipliers already applied), the Asset object, and the company
financials, and returns a list of CostItem objects representing every
itemised financial consequence.

DESIGN PHILOSOPHY
─────────────────
Costs are computed bottom-up from physical quantities:
  1. Identify which hazards are materially elevated in the event
  2. Translate hazard severity into physical impact (e.g., % production halt,
     m³ of water shortfall, days of logistics closure)
  3. Multiply by unit costs to get USD amounts
  4. Add recovery and consequential costs

This produces the kind of granular output that goes into an insurance claim,
a board briefing, or a credit committee pack — not just a score.

EMPIRICAL SOURCES
─────────────────
• Swiss Re sigma No. 1/2023: Natural catastrophes
• Munich Re NatCatSERVICE economic loss data 2000–2022
• World Bank FATHOM Global Flood Model damage functions
• USACE Flood Damage Functions (HEC-FDA)
• WRI Aqueduct Water Risk Atlas — industrial water use data
• FAO AQUASTAT — crop water requirements
• IPCC AR6 WGII — sector-specific economic impacts (Ch5, 7, 8, 9)
• Beverage sector: AB InBev, Diageo, Coca-Cola CDP Water disclosures
• Mining sector: ICMM — water stewardship in mining
• Real estate: JLL Climate Risk and Real Estate research
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...data.schemas import CostCategory, CostItem

if TYPE_CHECKING:
    from ...data.schemas import Asset, Company
    from ..hazard_layers import AssetHazardProfile
    from .physical_events import PhysicalEvent


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _severity(profile: "AssetHazardProfile", hazard: str) -> float:
    """Return hazard severity (0–5) from profile, 0.0 if not applicable."""
    h = profile.hazards.get(hazard)
    if h is None or not h.applicable:
        return 0.0
    return min(5.0, h.severity_index)


def _asset_revenue(asset: "Asset", company: "Company") -> float:
    """Estimate per-asset annual revenue (USD).

    Uses carrying_value × 0.6 as revenue proxy if no direct figure —
    a conservative approximation for capital-intensive assets.
    Assets with carrying_value = 0 fall back to an equal share of company revenue.
    """
    if asset.carrying_value > 0:
        return asset.carrying_value * 0.6
    n = max(len(company.assets), 1)
    return company.financials.revenue / n


def _asset_ebitda(asset: "Asset", company: "Company") -> float:
    """Estimate per-asset EBITDA (USD)."""
    rev = _asset_revenue(asset, company)
    margin = company.financials.ebitda / max(company.financials.revenue, 1)
    return rev * margin


def _daily_revenue(asset: "Asset", company: "Company") -> float:
    return _asset_revenue(asset, company) / 365.0


def _severity_to_halt_fraction(severity: float) -> float:
    """Map 0–5 severity to fraction of production halted (0–1)."""
    # Piecewise: severity 0=0%, 2=15%, 3=40%, 4=70%, 5=100%
    if severity <= 0:
        return 0.0
    if severity <= 2:
        return severity * 0.075
    if severity <= 3:
        return 0.15 + (severity - 2) * 0.25
    if severity <= 4:
        return 0.40 + (severity - 3) * 0.30
    return min(1.0, 0.70 + (severity - 4) * 0.30)


def _months_to_days(months: int) -> int:
    return months * 30


# ---------------------------------------------------------------------------
# BEVERAGES sector damage chain
# ---------------------------------------------------------------------------
# Water-intensive manufacturing (4–10 L water per L beverage).
# Key value chain: water sourcing → ingredient procurement → brewing/bottling
# → warehousing → distribution.
# Sources: CDP Water Security questionnaires for AB InBev, Diageo, Heineken;
#          Beverage Industry Environmental Roundtable (BIER) water benchmarks.

def _beverages_costs(
    event: "PhysicalEvent",
    profile: "AssetHazardProfile",
    asset: "Asset",
    company: "Company",
) -> list[CostItem]:
    items: list[CostItem] = []
    rev = _asset_revenue(asset, company)
    ebitda = _asset_ebitda(asset, company)
    daily_rev = rev / 365.0
    duration_days = _months_to_days(event.duration_months)

    # ── Water stress / drought ───────────────────────────────────────────────
    ws = _severity(profile, "water_stress")
    drought = _severity(profile, "drought")
    water_sev = max(ws, drought)

    if water_sev >= 1.5:
        # Estimated water intensity: 7 L/L beverage (industry median)
        # Production volume in m³/year ≈ baseline_production × 0.007
        # Production is in hL; 1 hL = 0.1 m³; water intensity = 7 L/L = 0.7 m³/hL
        annual_water_m3 = asset.baseline_production * 0.7
        daily_water_m3 = annual_water_m3 / 365.0

        # Allocation cut fraction (severity → cut)
        alloc_cut = _severity_to_halt_fraction(water_sev) * 0.8
        shortfall_m3_per_day = daily_water_m3 * alloc_cut

        # Emergency water trucking premium (USD/m³ above baseline piped)
        # Baseline municipal: ~USD 1.50/m³; emergency trucking: USD 8–15/m³
        trucking_premium = 7.0 + (water_sev - 1.5) * 1.5  # USD/m³
        trucking_days = int(duration_days * 0.7)
        trucking_cost = shortfall_m3_per_day * trucking_premium * trucking_days

        trucking_cost_m = trucking_cost / 1_000_000  # convert raw USD → USD millions
        if trucking_cost_m > 0:
            items.append(CostItem(
                category=CostCategory.EMERGENCY_RESPONSE,
                description=(
                    f"Emergency water trucking — {shortfall_m3_per_day:.0f} m³/day "
                    f"deficit × USD {trucking_premium:.1f}/m³ premium × {trucking_days} days"
                ),
                amount_usd=round(trucking_cost_m, 4),
                duration_note=f"{trucking_days} days at {shortfall_m3_per_day:.0f} m³/day",
                confidence="medium",
                source_assumption=(
                    "BIER Water Use Benchmarking: 7 L water/L beverage. "
                    "Emergency trucking premium USD 7–12/m³ (Cape Town 2018 precedent)."
                ),
            ))

        # Production curtailment from water shortage
        prod_halt_frac = _severity_to_halt_fraction(water_sev) * 0.6
        prod_loss_usd = daily_rev * prod_halt_frac * duration_days * 0.4  # EBITDA-margin portion
        if prod_loss_usd > 0:
            items.append(CostItem(
                category=CostCategory.PRODUCTION_LOSS,
                description=(
                    f"Production curtailment from water allocation cut "
                    f"({prod_halt_frac*100:.0f}% capacity reduction × {duration_days} days)"
                ),
                amount_usd=round(prod_loss_usd),
                duration_note=f"{duration_days} days × {prod_halt_frac*100:.0f}% curtailment",
                confidence="medium",
                source_assumption=(
                    "Beverage plant requires minimum 40% water allocation to operate. "
                    "Loss = curtailed volume × EBITDA margin."
                ),
            ))

        # Agricultural input cost escalation (barley, hops, sugar, fruit)
        # Drought raises grain prices 15–40% depending on regional exposure
        input_cost_increase = (water_sev / 5.0) * 0.32 * rev * 0.15  # input ≈ 15% of revenue
        if input_cost_increase > 0:
            items.append(CostItem(
                category=CostCategory.SUPPLY_CHAIN,
                description=(
                    f"Agricultural input price escalation — barley, hops, sugar "
                    f"({water_sev/5*32:.0f}% above contract due to regional crop stress)"
                ),
                amount_usd=round(input_cost_increase),
                confidence="medium",
                source_assumption=(
                    "USDA NASS: 1% yield loss in key grain → 0.3–0.5% input price increase. "
                    "Agricultural inputs ≈ 15% of beverage COGS."
                ),
            ))

        # Water infrastructure emergency capex (temporary storage, recycling)
        if water_sev >= 3.0:
            infra_capex = rev * 0.015 * (water_sev / 5.0)
            items.append(CostItem(
                category=CostCategory.RECOVERY_CAPEX,
                description=(
                    "Emergency water storage and recycling infrastructure "
                    "(temporary tanks, grey-water recycling, on-site treatment)"
                ),
                amount_usd=round(infra_capex),
                confidence="low",
                source_assumption=(
                    "Industry benchmark: emergency water resilience capex = "
                    "1–3% of asset revenue for severe water stress events."
                ),
            ))

    # ── Flood damage ─────────────────────────────────────────────────────────
    flood = max(_severity(profile, "flood_riverine"), _severity(profile, "flood_coastal"))

    if flood >= 1.5:
        # Warehouse and distribution centre damage
        # Assume warehouse replacement cost ≈ 20% of carrying_value
        wh_value = max(asset.carrying_value * 0.20, rev * 0.08)
        # Depth-based damage fraction (USACE damage function)
        flood_damage_frac = min(0.80, (flood / 5.0) ** 1.5)
        wh_damage = wh_value * flood_damage_frac
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                f"Warehouse / distribution hub structural and fit-out damage "
                f"(flood depth severity {flood:.1f}/5.0)"
            ),
            amount_usd=round(wh_damage),
            confidence="medium",
            source_assumption=(
                "USACE HEC-FDA depth-damage functions for commercial warehouse: "
                "damage = 10–80% of replacement value at 0.5–2m+ depth."
            ),
        ))

        # Inventory write-off (finished goods + raw materials in warehouse)
        inventory_value = rev * 0.06  # ~6 weeks of inventory
        inventory_loss = inventory_value * min(0.90, flood_damage_frac * 1.2)
        items.append(CostItem(
            category=CostCategory.INVENTORY_LOSS,
            description=(
                "Finished goods and raw material inventory write-off "
                "(flood inundation of primary storage)"
            ),
            amount_usd=round(inventory_loss),
            confidence="medium",
            source_assumption=(
                "Beverage industry holds ~6 weeks COGS as warehouse inventory. "
                "Flood loss fraction proportional to inundation depth."
            ),
        ))

        # Production facility equipment damage
        equip_value = max(asset.carrying_value * 0.50, rev * 0.20)
        equip_damage = equip_value * flood_damage_frac * 0.6  # not all equipment equally exposed
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                "Brewing/bottling equipment damage — electrical systems, "
                "control panels, conveyor belts, refrigeration units"
            ),
            amount_usd=round(equip_damage),
            confidence="low",
            source_assumption=(
                "Brewing equipment replacement cost ≈ 50% of asset carrying value. "
                "Flood damage rate 20–70% depending on inundation depth and duration."
            ),
        ))

        # Distribution disruption — lost sales during logistics shutdown
        dist_halt_days = int(10 + flood * 12)  # 10–70 days
        dist_loss = daily_rev * dist_halt_days * 0.85  # 85% of revenue is lost during halt
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Distribution network shutdown — road/rail damage prevents "
                f"product delivery for {dist_halt_days} days"
            ),
            amount_usd=round(dist_loss),
            duration_note=f"{dist_halt_days} days distribution halt",
            confidence="medium",
            source_assumption=(
                "2011 Thai floods precedent: beverage and FMCG distribution "
                "shutdown 3–8 weeks for severity-comparable events."
            ),
        ))

        # Emergency response: temporary logistics, flood defence
        emergency = rev * 0.012 * (flood / 5.0)
        items.append(CostItem(
            category=CostCategory.EMERGENCY_RESPONSE,
            description=(
                "Emergency logistics rerouting, temporary storage rental, "
                "flood defence deployment (sandbags, pumping)"
            ),
            amount_usd=round(emergency),
            confidence="medium",
            source_assumption="Industry benchmark: emergency response 1–2% of revenue.",
        ))

    # ── Heat stress ───────────────────────────────────────────────────────────
    heat = _severity(profile, "heat_stress")
    if heat >= 2.0:
        # Cooling energy cost premium
        cooling_extra = rev * 0.008 * (heat / 5.0) * (event.duration_months / 12)
        items.append(CostItem(
            category=CostCategory.ENERGY_UTILITY,
            description=(
                f"Refrigeration and cooling energy surcharge "
                f"(ambient temperature +{heat:.0f}σ above normal)"
            ),
            amount_usd=round(cooling_extra),
            confidence="high",
            source_assumption=(
                "Beverage production: refrigeration is 30–40% of energy cost. "
                "High-heat periods increase cooling load 15–35%."
            ),
        ))

        # Labour productivity loss (outdoor logistics, delivery)
        labour_loss = rev * 0.005 * _severity_to_halt_fraction(heat)
        items.append(CostItem(
            category=CostCategory.LABOUR,
            description=(
                "Labour productivity loss — outdoor loading/delivery operations "
                "constrained by heat stress protocols (WBGT > 28°C limits)"
            ),
            amount_usd=round(labour_loss),
            confidence="medium",
            source_assumption=(
                "ILO 2019: productivity loss 20–50% for outdoor labour at "
                "WBGT >28°C. Delivery/logistics ≈ 10% of operating cost."
            ),
        ))

    # ── Cyclone ───────────────────────────────────────────────────────────────
    cyclone = _severity(profile, "cyclone")
    if cyclone >= 2.0:
        # Roof and building envelope damage
        struct_value = max(asset.carrying_value * 0.30, rev * 0.12)
        struct_damage = struct_value * (cyclone / 5.0) ** 2
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                f"Roof, cladding, and building envelope damage "
                f"(Category {int(cyclone)} wind severity)"
            ),
            amount_usd=round(struct_damage),
            confidence="medium",
            source_assumption="Swiss Re: wind damage = 10–60% of building value at Cat 2–5.",
        ))

        # Power outage — refrigeration failure → product spoilage
        outage_days = int(cyclone * 6)
        spoilage_loss = daily_rev * outage_days * 0.30  # 30% of daily revenue lost to spoilage
        items.append(CostItem(
            category=CostCategory.INVENTORY_LOSS,
            description=(
                f"Product spoilage during power outage ({outage_days} days) — "
                "temperature-sensitive beverages in distribution"
            ),
            amount_usd=round(spoilage_loss),
            duration_note=f"{outage_days} days power outage",
            confidence="medium",
            source_assumption=(
                "Cyclone-driven PSPS (pre-emptive power shutoff) averages "
                "5–30 days for Cat 2–5 events. Perishable beverage loss ≈ 30% of daily sales."
            ),
        ))

    # ── Insurance and consequential ───────────────────────────────────────────
    total_direct = sum(i.amount_usd for i in items
                       if i.category in (CostCategory.PHYSICAL_DAMAGE,
                                         CostCategory.INVENTORY_LOSS))
    if total_direct > 50_000:
        deductible = min(total_direct * 0.10, rev * 0.02)
        items.append(CostItem(
            category=CostCategory.INSURANCE,
            description=(
                "Insurance deductible on property and business interruption claim"
            ),
            amount_usd=round(deductible),
            confidence="high",
            source_assumption=(
                "Typical beverage sector BI policy deductible: "
                "10% of claim value or 2% of revenue, whichever is lower."
            ),
        ))

    return items


# ---------------------------------------------------------------------------
# AGRICULTURE sector damage chain
# ---------------------------------------------------------------------------
# Direct yield loss is the primary impact path.
# Sources: FAO (food price volatility & losses), IPCC AR6 Ch5 (food),
#          USDA Economic Research Service crop loss data.

def _agriculture_costs(
    event: "PhysicalEvent",
    profile: "AssetHazardProfile",
    asset: "Asset",
    company: "Company",
) -> list[CostItem]:
    items: list[CostItem] = []
    rev = _asset_revenue(asset, company)
    daily_rev = rev / 365.0
    duration_days = _months_to_days(event.duration_months)

    # ── Drought / water stress — crop yield loss ─────────────────────────────
    drought = _severity(profile, "drought")
    ws = _severity(profile, "water_stress")
    drought_sev = max(drought, ws)

    if drought_sev >= 1.0:
        # Yield loss fraction: calibrated to FAO crop loss curves
        # Cereal: 5% loss per unit severity; severe (4+) = 50–80% loss
        yield_loss_frac = min(0.90, (drought_sev / 5.0) ** 1.3 * 0.85)
        crop_loss_usd = rev * yield_loss_frac
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Crop yield loss from drought stress "
                f"({yield_loss_frac*100:.0f}% of annual harvest affected)"
            ),
            amount_usd=round(crop_loss_usd),
            confidence="medium",
            source_assumption=(
                "IPCC AR6 Ch5: cereal yield loss 5–9% per °C warming. "
                "FAO drought damage functions for rainfed systems. "
                f"Severity {drought_sev:.1f}/5.0 → {yield_loss_frac*100:.0f}% yield loss."
            ),
        ))

        # Irrigation cost escalation (emergency groundwater pumping)
        # Assume irrigation covers 60% of production at baseline
        irrigation_area_ha = asset.baseline_production / 5.0  # rough: 5 t/ha cereal
        pump_cost_per_ha = 60 + (drought_sev ** 1.5) * 40  # USD/ha above normal
        irrigation_extra = irrigation_area_ha * pump_cost_per_ha / 1_000_000  # → USD millions
        items.append(CostItem(
            category=CostCategory.ENERGY_UTILITY,
            description=(
                f"Emergency irrigation — increased groundwater pumping "
                f"(USD {pump_cost_per_ha:.0f}/ha premium × {irrigation_area_ha:.0f} ha)"
            ),
            amount_usd=round(irrigation_extra),
            confidence="low",
            source_assumption=(
                "FAO AQUASTAT: emergency drought pumping premium USD 60–200/ha "
                "above gravity/surface irrigation baseline."
            ),
        ))

        # Replanting costs if crop fails completely (severity > 3.5)
        if drought_sev >= 3.5:
            replant_area = irrigation_area_ha * (drought_sev - 3.5) / 1.5
            replant_cost = replant_area * 350 / 1_000_000  # USD/ha → USD millions
            items.append(CostItem(
                category=CostCategory.RECOVERY_CAPEX,
                description=(
                    f"Crop replanting after complete failure "
                    f"({replant_area:.0f} ha × USD 350/ha seed + preparation)"
                ),
                amount_usd=round(replant_cost),
                confidence="low",
                source_assumption=(
                    "Industry benchmark: replanting cost USD 200–600/ha "
                    "depending on crop type and soil remediation required."
                ),
            ))

    # ── Flood — standing crop and field damage ─────────────────────────────
    flood = _severity(profile, "flood_riverine")

    if flood >= 1.5:
        # Standing crop loss (inundation > 3 days = total loss for most crops)
        flood_area_frac = min(1.0, (flood / 5.0) ** 1.2 * 0.80)
        crop_flood_loss = rev * flood_area_frac
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Standing crop loss — inundation of {flood_area_frac*100:.0f}% "
                f"of cropped area (>72h submersion = total loss)"
            ),
            amount_usd=round(crop_flood_loss),
            confidence="medium",
            source_assumption=(
                "FAO: most annual crops suffer >90% yield loss after "
                "72-hour submersion during vegetative/flowering stage."
            ),
        ))

        # Soil erosion and degradation remediation
        total_area_ha = asset.baseline_production / 5.0
        eroded_ha = total_area_ha * flood_area_frac * 0.40
        soil_remed_cost = eroded_ha * 280 / 1_000_000  # USD/ha → USD millions
        items.append(CostItem(
            category=CostCategory.RECOVERY_CAPEX,
            description=(
                f"Soil erosion remediation — {eroded_ha:.0f} ha affected "
                f"(topsoil loss, channel repair, contour restoration)"
            ),
            amount_usd=round(soil_remed_cost),
            confidence="low",
            source_assumption=(
                "USDA NRCS: post-flood soil remediation USD 150–500/ha "
                "depending on erosion severity and slope."
            ),
        ))

        # Farm equipment damage (tractors, harvesters, irrigation pivots)
        equip_value = max(asset.carrying_value * 0.25, rev * 0.10)
        equip_damage = equip_value * (flood / 5.0) * 0.50
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                "Farm equipment damage — tractors, harvesters, irrigation systems "
                "submerged or displaced by floodwaters"
            ),
            amount_usd=round(equip_damage),
            confidence="low",
            source_assumption=(
                "EM-DAT: agricultural equipment loss = 20–60% of replacement "
                "value in moderate to major flood events."
            ),
        ))

        # Storage silo / grain store flood damage
        storage_value = rev * 0.05
        storage_damage = storage_value * min(0.90, flood / 5.0 * 1.1)
        if storage_damage > 0:
            items.append(CostItem(
                category=CostCategory.PHYSICAL_DAMAGE,
                description=(
                    "Grain storage silo and cold-store flood damage "
                    "(structural + stored product contamination)"
                ),
                amount_usd=round(storage_damage),
                confidence="medium",
                source_assumption=(
                    "Cereal stored in flooded silos: 100% contamination loss "
                    "if inundation exceeds aeration floor level."
                ),
            ))

    # ── Heat stress — yield reduction and livestock ──────────────────────────
    heat = _severity(profile, "heat_stress")
    if heat >= 2.0:
        heat_yield_loss_frac = (heat / 5.0) ** 1.5 * 0.35
        heat_loss_usd = rev * heat_yield_loss_frac
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Heat stress crop yield reduction — grain filling impaired, "
                f"quality downgrade ({heat_yield_loss_frac*100:.0f}% effective yield loss)"
            ),
            amount_usd=round(heat_loss_usd),
            confidence="medium",
            source_assumption=(
                "IPCC AR6 Ch5: every 1°C increase reduces wheat yield 6%, "
                "maize 7.4%, rice 3.2%. Extrapolated to severity scale."
            ),
        ))

        # Livestock heat stress (if mixed farming)
        livestock_revenue_proxy = rev * 0.20  # estimate 20% livestock
        livestock_loss = livestock_revenue_proxy * (heat / 5.0) * 0.25
        if livestock_loss > 0:
            items.append(CostItem(
                category=CostCategory.PRODUCTION_LOSS,
                description=(
                    "Livestock heat stress — reduced milk yield, weight gain, "
                    "and increased mortality during heat event"
                ),
                amount_usd=round(livestock_loss),
                confidence="low",
                source_assumption=(
                    "USDA: heat stress costs US livestock industry USD 2.4 bn/year. "
                    "Estimated 10–30% productivity loss above WBGT threshold."
                ),
            ))

    # ── Late-season frost ────────────────────────────────────────────────────
    frost = _severity(profile, "frost_crop_damage")
    if frost >= 2.0:
        frost_loss_frac = min(1.0, (frost / 5.0) ** 0.8 * 0.95)
        frost_loss = rev * frost_loss_frac
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Late-season frost crop damage — {frost_loss_frac*100:.0f}% "
                f"of flowering/bud-burst stage affected (irrecoverable)"
            ),
            amount_usd=round(frost_loss),
            confidence="high",
            source_assumption=(
                "2021 French viticulture frost: 40–80% loss in Bordeaux, Cognac. "
                "Loss depends on crop stage at frost onset."
            ),
        ))

    return items


# ---------------------------------------------------------------------------
# MINING / EXTRACTIVES sector damage chain
# ---------------------------------------------------------------------------
# Key value chain: pit/underground extraction → crushing/processing →
# tailings management → haul road → port/rail logistics.
# Sources: ICMM Water Stewardship; NRC Canada mining disruption data;
#          Australian Bureau of Statistics mining sector water use.

def _mining_costs(
    event: "PhysicalEvent",
    profile: "AssetHazardProfile",
    asset: "Asset",
    company: "Company",
) -> list[CostItem]:
    items: list[CostItem] = []
    rev = _asset_revenue(asset, company)
    daily_rev = rev / 365.0
    carrying = max(asset.carrying_value, rev * 1.5)  # mines are capital intensive
    duration_days = _months_to_days(event.duration_months)

    # ── Flood — pit flooding and infrastructure ───────────────────────────────
    flood = max(_severity(profile, "flood_riverine"), _severity(profile, "flood_coastal"))

    if flood >= 1.5:
        # Production halt during flooding and dewatering
        halt_days = int(15 + (flood ** 2) * 18)  # 15–270 days depending on severity
        production_halt_loss = daily_rev * halt_days * 0.75  # 75% EBITDA equivalent
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Open-pit / underground mine production halt — pit flooding, "
                f"dewatering and safety inspection required ({halt_days} days)"
            ),
            amount_usd=round(production_halt_loss),
            duration_note=f"{halt_days} days × USD {daily_rev*0.75:,.0f}/day EBITDA",
            confidence="medium",
            source_assumption=(
                "ICMM: major flood events cause 30–180+ day production halts. "
                "Revenue lost during halt = daily revenue × EBITDA margin proxy 0.75."
            ),
        ))

        # Pit dewatering cost (large pumps, pipe installation)
        dewater_cost = max(0.5, flood * 2.5)  # USD millions: 0.5M–12.5M
        items.append(CostItem(
            category=CostCategory.EMERGENCY_RESPONSE,
            description=(
                f"Pit dewatering operation — pump hire, pipe installation, "
                f"power supply ({halt_days} days of continuous pumping)"
            ),
            amount_usd=round(dewater_cost),
            confidence="medium",
            source_assumption=(
                "Industry benchmarks: dewatering costs USD 50k–500k/day "
                "for large open pit depending on inflow rate and pit depth."
            ),
        ))

        # Haul road and infrastructure damage
        road_damage = carrying * 0.04 * (flood / 5.0)
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                "Haul road, access road, and site infrastructure damage — "
                "erosion, embankment failure, bridge/culvert washout"
            ),
            amount_usd=round(road_damage),
            confidence="low",
            source_assumption=(
                "NRC Canada: haul road repair after major flood = 2–6% of "
                "mine replacement cost. Varies by road length and gradient."
            ),
        ))

        # Tailings dam risk management (elevated risk, monitoring + reinforcement)
        if flood >= 3.0:
            tailings_safety = max(1.0, carrying * 0.015)  # USD millions floor: 1.0 = USD 1M
            items.append(CostItem(
                category=CostCategory.EMERGENCY_RESPONSE,
                description=(
                    "Tailings dam emergency reinforcement and heightened "
                    "monitoring during flood event (regulatory requirement)"
                ),
                amount_usd=round(tailings_safety),
                confidence="medium",
                source_assumption=(
                    "ICMM Global Tailings Standard: emergency monitoring and "
                    "reinforcement triggered at >1-in-100-year inflow events."
                ),
            ))

        # Rail/port logistics shutdown (for export mines)
        logistics_halt_days = int(halt_days * 0.6)
        if logistics_halt_days > 0:
            logistics_loss = daily_rev * logistics_halt_days * 0.20  # 20% daily revenue from logistics
            items.append(CostItem(
                category=CostCategory.SUPPLY_CHAIN,
                description=(
                    f"Export logistics suspension — rail or port damaged, "
                    f"product stockpiled at mine site ({logistics_halt_days} days)"
                ),
                amount_usd=round(logistics_loss),
                duration_note=f"{logistics_halt_days} days logistics suspension",
                confidence="low",
                source_assumption=(
                    "2011 QLD floods: coal export rail shutdown 4–12 weeks. "
                    "Stockpile financing and demurrage costs approx 20% of revenue/day."
                ),
            ))

    # ── Water stress — processing restrictions ───────────────────────────────
    ws = max(_severity(profile, "water_stress"), _severity(profile, "drought"))

    if ws >= 2.0:
        # Processing water curtailment (copper processing: 0.5–1.5 m³/tonne)
        # Water cut → proportional production cut
        water_cut_frac = _severity_to_halt_fraction(ws) * 0.7
        processing_loss = daily_rev * water_cut_frac * duration_days * 0.60
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Ore processing curtailment from water allocation restriction "
                f"({water_cut_frac*100:.0f}% throughput reduction × {duration_days} days)"
            ),
            amount_usd=round(processing_loss),
            duration_note=f"{duration_days} days × {water_cut_frac*100:.0f}% reduction",
            confidence="medium",
            source_assumption=(
                "ICMM: mining operations require 0.5–5 m³/tonne ore processed. "
                "Water restriction directly throttles mill throughput."
            ),
        ))

        # Alternative water sourcing (groundwater license, water purchase)
        if ws >= 3.0:
            alt_water_cost = rev * 0.008 * (ws / 5.0)
            items.append(CostItem(
                category=CostCategory.ENERGY_UTILITY,
                description=(
                    "Alternative water sourcing — emergency groundwater licences, "
                    "water purchase from third parties, tanker supply"
                ),
                amount_usd=round(alt_water_cost),
                confidence="low",
                source_assumption=(
                    "Atacama copper mines: emergency water trucking + licence "
                    "costs = 0.5–1.5% of annual revenue in severe drought years."
                ),
            ))

        # Water recycling capex (medium-term investment triggered by crisis)
        if ws >= 3.5:
            recycling_capex = max(2.0, rev * 0.02)  # USD millions floor: 2.0 = USD 2M
            items.append(CostItem(
                category=CostCategory.RECOVERY_CAPEX,
                description=(
                    "Water recycling and treatment plant installation — "
                    "closed-loop processing circuit to reduce intake dependency"
                ),
                amount_usd=round(recycling_capex),
                confidence="low",
                source_assumption=(
                    "Industry: water recycling capex USD 2–20M for mid-size mine. "
                    "Payback 3–5 years under USD 5/m³ supply cost."
                ),
            ))

    # ── Cyclone ───────────────────────────────────────────────────────────────
    cyclone = _severity(profile, "cyclone")
    if cyclone >= 2.0:
        # Surface infrastructure (buildings, conveyors, electrical)
        surface_damage = carrying * 0.06 * (cyclone / 5.0) ** 1.5
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                f"Surface infrastructure damage — buildings, conveyor systems, "
                f"electrical switchrooms, process plant (Cat {int(cyclone)} wind)"
            ),
            amount_usd=round(surface_damage),
            confidence="medium",
            source_assumption=(
                "Swiss Re: mining/industrial wind damage = 3–12% of replacement "
                "asset value at Category 3–5 cyclone. Tropical mine precedent."
            ),
        ))

        # Production halt (evacuation + post-cyclone inspection)
        cyclone_halt_days = int(7 + cyclone * 15)
        cyclone_prod_loss = daily_rev * cyclone_halt_days * 0.80
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Production halt during evacuation, cyclone passage, and "
                f"post-event structural inspection ({cyclone_halt_days} days)"
            ),
            amount_usd=round(cyclone_prod_loss),
            duration_note=f"{cyclone_halt_days} days",
            confidence="high",
            source_assumption=(
                "Cyclone Yasi 2011: QLD mining operations halted 7–21 days. "
                "Regulatory requirement for structural clearance before restart."
            ),
        ))

    # ── Heat stress — labour and equipment ───────────────────────────────────
    heat = _severity(profile, "heat_stress")
    if heat >= 2.5:
        # Outdoor labour productivity loss (mandatory heat work limits)
        labour_cost = rev * 0.12  # labour ≈ 12% of mining revenue
        labour_loss = labour_cost * _severity_to_halt_fraction(heat) * 0.40
        items.append(CostItem(
            category=CostCategory.LABOUR,
            description=(
                f"Outdoor labour productivity loss — heat work-limit protocols "
                f"(mandatory rest periods, wet-bulb > 30°C threshold)"
            ),
            amount_usd=round(labour_loss),
            confidence="medium",
            source_assumption=(
                "Safe Work Australia: mandatory work-rest ratios at WBGT > 28°C. "
                "40% productivity reduction for outdoor mining at severe heat levels."
            ),
        ))

        # Diesel consumption increase (equipment cooling loads)
        diesel_extra = rev * 0.004 * (heat / 5.0)
        items.append(CostItem(
            category=CostCategory.ENERGY_UTILITY,
            description=(
                "Diesel fuel cost increase — equipment cooling loads, "
                "air-conditioning for haul trucks and surface buildings"
            ),
            amount_usd=round(diesel_extra),
            confidence="low",
            source_assumption=(
                "Mining equipment diesel: cooling load increases 8–15% "
                "per 10°C ambient temperature rise above 30°C."
            ),
        ))

    return items


# ---------------------------------------------------------------------------
# REAL ESTATE sector damage chain
# ---------------------------------------------------------------------------
# Key impacts: structural damage, tenant displacement (vacancy), capex.
# Sources: JLL Climate Risk research; Swiss Re property damage database;
#          USACE depth-damage functions; CoreLogic extreme weather loss data.

def _real_estate_costs(
    event: "PhysicalEvent",
    profile: "AssetHazardProfile",
    asset: "Asset",
    company: "Company",
) -> list[CostItem]:
    items: list[CostItem] = []
    # For RE: asset_value = carrying_value (the property market value, in USD millions)
    # Annual rental income = production × unit_cost / 1M (sqm × USD/sqm/year)
    asset_value = asset.carrying_value
    annual_rental_income = (
        asset.baseline_production * asset.baseline_unit_cost / 1_000_000
        if asset.baseline_production > 0 and asset.baseline_unit_cost > 0
        else _asset_revenue(asset, company) * 0.06   # 6% yield fallback
    )
    daily_rental = annual_rental_income / 365.0
    duration_days = _months_to_days(event.duration_months)

    # ── Flood structural damage ───────────────────────────────────────────────
    flood = max(_severity(profile, "flood_riverine"),
                _severity(profile, "flood_coastal"),
                _severity(profile, "saltwater_intrusion") * 0.8)

    if flood >= 1.5:
        # USACE depth-damage function (commercial):
        # 0.5m → 10-20%; 1m → 25-40%; 2m → 45-65%; >3m → 60-80%
        # Approximate via: damage_frac = (sev/5)^0.9 × 0.75
        damage_frac = min(0.80, (flood / 5.0) ** 0.9 * 0.75)
        structural_damage = asset_value * damage_frac * 0.40  # 40% of value is structural
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                f"Structural flood damage — foundations, ground floor systems, "
                f"electrical, HVAC, lift wells ({damage_frac*100:.0f}% damage severity)"
            ),
            amount_usd=round(structural_damage),
            confidence="medium",
            source_assumption=(
                "USACE HEC-FDA commercial building depth-damage functions. "
                "Ground floor structure + MEP = 40% of building replacement cost."
            ),
        ))

        # Tenant fit-out and contents damage
        fitout_value = asset_value * 0.15  # tenant fit-out ≈ 15% of building value
        fitout_damage = fitout_value * damage_frac
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                "Tenant fit-out damage — workstations, server rooms, "
                "specialist fit-out, contents submerged or contaminated"
            ),
            amount_usd=round(fitout_damage),
            confidence="medium",
            source_assumption=(
                "JLL: commercial tenant fit-out = 10–20% of building value. "
                "Flood-damaged fit-out is typically written off and replaced."
            ),
        ))

        # Tenant displacement — vacancy loss during repair
        repair_months = math.ceil(damage_frac * 18)  # up to 18 months for severe damage
        vacancy_loss = daily_rental * repair_months * 30
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=(
                f"Rental income loss during repair — tenant displacement "
                f"and building closure ({repair_months} months)"
            ),
            amount_usd=round(vacancy_loss),
            duration_note=f"{repair_months} months vacancy",
            confidence="medium",
            source_assumption=(
                "Post-flood building closure: 2–18 months for moderate to severe "
                "structural damage. Rent ceases during uninhabitable period."
            ),
        ))

        # Renovation and reinstatement capex
        renovation_capex = structural_damage * 1.3  # reinstatement > damage cost
        items.append(CostItem(
            category=CostCategory.RECOVERY_CAPEX,
            description=(
                "Building reinstatement capex — structural repair, MEP "
                "replacement, flood resilience upgrades during restoration"
            ),
            amount_usd=round(renovation_capex),
            confidence="low",
            source_assumption=(
                "Reinstatement cost typically 120–140% of damage assessment "
                "due to wet work, specialist trades, and building code upgrades."
            ),
        ))

        # Insurance excess and premium increase
        if structural_damage > 0.1:  # > USD 100K (0.1 USD millions)
            excess = min(structural_damage * 0.08, asset_value * 0.01)
            items.append(CostItem(
                category=CostCategory.INSURANCE,
                description="Flood insurance excess and anticipated premium uplift (3-year)",
                amount_usd=round(excess + annual_rental_income * 0.015),
                confidence="medium",
                source_assumption=(
                    "Post-major-flood: BI and property insurance premiums "
                    "increase 20–60% for 3 renewal cycles. Standard excess 5–10%."
                ),
            ))

    # ── Wildfire ──────────────────────────────────────────────────────────────
    wildfire = _severity(profile, "wildfire")
    if wildfire >= 2.5:
        # Structural fire damage (partial or total loss)
        fire_damage_frac = min(1.0, (wildfire / 5.0) ** 1.2 * 0.90)
        fire_structural = asset_value * fire_damage_frac * 0.60
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                f"Wildfire structural damage — partial to full building loss "
                f"({fire_damage_frac*100:.0f}% affected based on perimeter proximity)"
            ),
            amount_usd=round(fire_structural),
            confidence="low",
            source_assumption=(
                "CoreLogic: properties within wildfire perimeter: "
                "50–100% total loss. 10km buffer: 5–30% damage rate."
            ),
        ))

        # Smoke damage remediation (even if structure survives)
        smoke_remed = asset_value * 0.03 * (wildfire / 5.0)
        items.append(CostItem(
            category=CostCategory.RECOVERY_CAPEX,
            description=(
                "Smoke and ash remediation — HVAC deep clean, surface "
                "decontamination, air quality testing and certification"
            ),
            amount_usd=round(smoke_remed),
            confidence="medium",
            source_assumption=(
                "Industry: smoke remediation for commercial building "
                "= 1–4% of replacement value. Required for re-occupancy."
            ),
        ))

        # Vacancy during remediation
        fire_vacancy_months = max(3, int(fire_damage_frac * 24))
        fire_vacancy_loss = daily_rental * fire_vacancy_months * 30
        items.append(CostItem(
            category=CostCategory.PRODUCTION_LOSS,
            description=f"Rental income loss during fire damage remediation ({fire_vacancy_months} months)",
            amount_usd=round(fire_vacancy_loss),
            duration_note=f"{fire_vacancy_months} months",
            confidence="medium",
            source_assumption=(
                "Post-wildfire commercial vacancy: 3–24 months depending on "
                "damage severity and contractor availability."
            ),
        ))

    # ── Heat stress — HVAC retrofit and energy ────────────────────────────────
    heat = _severity(profile, "heat_stress")
    if heat >= 2.0:
        # HVAC upgrade capex (tenants demand better cooling, code requirements)
        # Use stated floor area (baseline_production in sqm for RE assets)
        # Fallback: asset_value in USD millions × 1M / USD 3k/sqm replacement cost
        gross_floor_area_sqm = max(
            1_000,
            asset.baseline_production if asset.baseline_production > 0
            else asset_value * 1_000_000 / 3_000
        )
        hvac_capex_per_sqm = 25 + heat * 12  # USD/sqm for HVAC upgrade
        hvac_capex = gross_floor_area_sqm * hvac_capex_per_sqm / 1_000_000  # → USD millions
        items.append(CostItem(
            category=CostCategory.RECOVERY_CAPEX,
            description=(
                f"HVAC system upgrade — enhanced cooling capacity for "
                f"extreme heat resilience ({gross_floor_area_sqm:.0f} sqm × "
                f"USD {hvac_capex_per_sqm:.0f}/sqm)"
            ),
            amount_usd=round(hvac_capex),
            confidence="medium",
            source_assumption=(
                "JLL / CBRE: heat resilience HVAC upgrade USD 25–100/sqm "
                "for commercial office. Green Star / NABERS compliance driver."
            ),
        ))

        # Utility cost increase during heat event
        cooling_util = annual_rental_income * 0.04 * (heat / 5.0)
        items.append(CostItem(
            category=CostCategory.ENERGY_UTILITY,
            description=(
                "Cooling utility cost increase — peak electricity demand "
                "surcharge during heat dome period"
            ),
            amount_usd=round(cooling_util),
            confidence="high",
            source_assumption=(
                "Commercial HVAC electricity = 35–50% of building energy. "
                "Heat dome increases cooling demand 30–60%."
            ),
        ))

        # Tenant comfort risk (churn if cooling inadequate)
        if heat >= 3.5:
            tenant_churn_loss = annual_rental_income * 0.08
            items.append(CostItem(
                category=CostCategory.CONSEQUENTIAL,
                description=(
                    "Tenant lease non-renewal risk from inadequate heat resilience "
                    "(estimated 8% lease value at risk)"
                ),
                amount_usd=round(tenant_churn_loss),
                confidence="low",
                source_assumption=(
                    "JLL Tenant Survey 2022: 35% of commercial tenants would "
                    "not renew lease if building fails to meet heat comfort standards."
                ),
            ))

    # ── Cyclone ───────────────────────────────────────────────────────────────
    cyclone = _severity(profile, "cyclone")
    if cyclone >= 2.0:
        # Roof and envelope damage (primary wind damage)
        roof_value = asset_value * 0.12
        roof_damage = roof_value * (cyclone / 5.0) ** 1.8
        items.append(CostItem(
            category=CostCategory.PHYSICAL_DAMAGE,
            description=(
                f"Roof, glazing and building envelope damage "
                f"(Category {int(cyclone)} wind event)"
            ),
            amount_usd=round(roof_damage),
            confidence="medium",
            source_assumption=(
                "Swiss Re: commercial roof damage = 5–40% of building value "
                "at Cat 2–5. Roof = 12% of replacement cost."
            ),
        ))

        # Tenant disruption and temporary relocation
        temp_reloc_months = int(2 + cyclone * 1.5)
        temp_reloc_cost = annual_rental_income * 0.06 * (temp_reloc_months / 12)
        items.append(CostItem(
            category=CostCategory.EMERGENCY_RESPONSE,
            description=(
                f"Temporary tenant relocation assistance "
                f"({temp_reloc_months} months × temporary space subsidy)"
            ),
            amount_usd=round(temp_reloc_cost),
            duration_note=f"{temp_reloc_months} months",
            confidence="low",
            source_assumption=(
                "Industry practice: landlord-funded temporary relocation "
                "= 30–60% of displaced tenants' rent for 2–8 months."
            ),
        ))

    return items


# ---------------------------------------------------------------------------
# Sector dispatch
# ---------------------------------------------------------------------------

def _food_costs(event, profile, asset, company) -> list[CostItem]:
    """Food manufacturing — broadly similar to agriculture upstream + beverages downstream."""
    ag = _agriculture_costs(event, profile, asset, company)
    bev = _beverages_costs(event, profile, asset, company)
    # Food manufacturing: blend of supply-side (ag inputs) and production-side (bev plant)
    blended = ag + [c for c in bev if c.category in (
        CostCategory.PRODUCTION_LOSS,
        CostCategory.PHYSICAL_DAMAGE,
        CostCategory.SUPPLY_CHAIN,
        CostCategory.ENERGY_UTILITY,
    )]
    return blended


def _chemicals_manufacturing_costs(event, profile, asset, company) -> list[CostItem]:
    """Chemicals / general manufacturing — similar to mining for water + flood."""
    return _mining_costs(event, profile, asset, company)


# Public dispatch map: Commodity → cost function
from ...data.schemas import Commodity  # noqa: E402 — import here to avoid circular

SECTOR_CHAIN: dict[str, object] = {
    Commodity.BEVERAGES.value:           _beverages_costs,
    Commodity.FOOD.value:                _food_costs,
    Commodity.AGRICULTURE.value:         _agriculture_costs,
    Commodity.IRON_ORE.value:            _mining_costs,
    Commodity.COPPER.value:              _mining_costs,
    Commodity.ALUMINIUM.value:           _mining_costs,
    Commodity.COAL_THERMAL.value:        _mining_costs,
    Commodity.COAL_METALLURGICAL.value:  _mining_costs,
    Commodity.CRUDE_OIL.value:           _mining_costs,
    Commodity.NATURAL_GAS.value:         _mining_costs,
    Commodity.CEMENT.value:              _mining_costs,
    Commodity.CHEMICALS.value:           _chemicals_manufacturing_costs,
    Commodity.MANUFACTURING.value:       _chemicals_manufacturing_costs,
    Commodity.REAL_ESTATE.value:         _real_estate_costs,
    Commodity.ELECTRICITY.value:         _mining_costs,   # power plant — infra-focused
    Commodity.RETAIL.value:              _beverages_costs,  # store network — similar to bev
    Commodity.FINANCIAL_SERVICES.value:  _real_estate_costs,  # real-estate backed
    Commodity.REFINED_PRODUCTS.value:    _mining_costs,
}


def get_sector_chain(commodity: str):
    """Return the cost function for a commodity, falling back to mining (infra)."""
    return SECTOR_CHAIN.get(commodity, _mining_costs)
