"""
Carbon-market position: EU ETS compliance balance and carbon-credit quality screen.

EU ETS position (per installation and in total)
  Compliance balance = verified emissions − (free allocation received + allowances held/purchased for the year).
  A shortfall must be covered by surrendering EUAs by 30 September of the following year (from the 2024
  compliance year) — the cost is valued at the EUA price supplied (default: the World Bank Carbon Pricing
  Dashboard's latest EU ETS price in the engine, in USD). Excess-emissions penalty is €100/t indexed to inflation, and
  the allowances must still be surrendered (Directive 2003/87/EC art. 16).
  Forward view: free allocation for CBAM sectors (cement, iron & steel, aluminium, fertilisers, hydrogen,
  electricity) is reduced by the CBAM factor 2026–2034 (Regulation (EU) 2023/956 / ETS Directive art. 10a(1a));
  an additional annual allocation decline (benchmark updates, cross-sectoral correction) is a user input.

Carbon-credit screen
  Rules applied to each credit lot, with the reason for every flag:
  • Core Carbon Principles (ICVCM) label present or not — as declared from the registry record.
  • Removal vs avoidance/reduction, and durability (nature-based vs geological/mineral storage).
  • Vintage age.
  • Corresponding adjustment (needed for use towards another country's NDC / CORSIA Phase 1 eligibility).
  • Usage limits: SBTi — credits cannot count towards near- or long-term targets; only permanent removals may
    neutralise residual emissions at net zero; other credits are beyond-value-chain mitigation.
    EU Directive 2024/825 (Empowering Consumers for the Green Transition), applying from 27 September 2026:
    product claims of neutral/reduced/positive impact based on offsetting are banned in the EU.
    ESRS E1-7: credits are disclosed separately from GHG emissions and targets.
"""
from __future__ import annotations

from typing import Optional

CBAM_SECTORS = {"cement", "iron_steel", "steel", "aluminium", "fertilisers", "fertilizers", "hydrogen", "electricity"}
REMOVAL_TYPES = {"afforestation", "reforestation", "arr", "biochar", "beccs", "daccs", "dac", "enhanced_weathering",
                 "soil_carbon", "blue_carbon_restoration", "mineralisation"}
DURABLE_REMOVALS = {"biochar", "beccs", "daccs", "dac", "enhanced_weathering", "mineralisation"}


class MarketError(ValueError):
    pass


def _eua_price_default() -> Optional[float]:
    try:
        from .finance.policy import _wb
        W = _wb()
        ins = W["instruments"] if "instruments" in W else W
        eu = next(i for i in ins if i.get("id") == "ETS_EU")
        prices = [p for p in eu["prices"] if p]
        return prices[-1] if prices else None
    except Exception:
        return None


def ets_position(installations: list, compliance_year: int, eua_price: Optional[float] = None, currency: str = "EUR",
                 projection_to: int = 2034, emissions_change_pct_per_year: float = 0.0,
                 allocation_decline_pct_per_year: float = 0.0) -> dict:
    """installations: [{name, sector, verified_emissions_t, free_allocation_t, allowances_held_t}]"""
    if not installations:
        raise MarketError("give at least one installation")
    from .finance.policy import EU_CBAM_FREE_SHARE
    price = eua_price if eua_price is not None else _eua_price_default()
    currency = currency if eua_price is not None else "USD"
    rows, tot = [], {"verified": 0.0, "free": 0.0, "held": 0.0}
    for ins in installations:
        v, f, h = (float(ins.get(k, 0) or 0) for k in ("verified_emissions_t", "free_allocation_t", "allowances_held_t"))
        if min(v, f, h) < 0:
            raise MarketError(f"{ins.get('name')}: values must be ≥ 0")
        bal = v - f - h
        sector = (ins.get("sector") or "").lower().replace(" ", "_").replace("&", "")
        cbam = sector in CBAM_SECTORS
        proj = []
        for y in range(compliance_year + 1, projection_to + 1):
            k = y - compliance_year
            e = v * (1 + emissions_change_pct_per_year / 100) ** k
            fa = f * (1 - allocation_decline_pct_per_year / 100) ** k
            if cbam:
                fa *= float(EU_CBAM_FREE_SHARE.get(y, EU_CBAM_FREE_SHARE.get(max(EU_CBAM_FREE_SHARE), 0.0)))
            short = max(0.0, e - fa)
            proj.append({"year": y, "emissions_t": round(e), "free_allocation_t": round(fa), "shortfall_t": round(short),
                         "cost_at_price": round(short * price) if price else None})
        rows.append({"name": ins.get("name"), "sector": ins.get("sector"), "cbam_sector": cbam,
                     "verified_emissions_t": v, "free_allocation_t": f, "allowances_held_t": h,
                     "balance_t": round(bal), "position": "short" if bal > 0 else "long",
                     "cost_to_cover": round(max(0.0, bal) * price) if price else None, "projection": proj})
        tot["verified"] += v
        tot["free"] += f
        tot["held"] += h
    bal = tot["verified"] - tot["free"] - tot["held"]
    return {"compliance_year": compliance_year, "surrender_deadline": f"30 September {compliance_year + 1}",
            "eua_price": price, "currency": currency, "price_source": "user-supplied" if eua_price is not None else
            "World Bank Carbon Pricing Dashboard, latest EU ETS price in USD/t — supply today's EUA price in EUR for a live figure",
            "totals": {"verified_emissions_t": tot["verified"], "free_allocation_t": tot["free"],
                       "allowances_held_t": tot["held"], "balance_t": round(bal),
                       "cost_to_cover": round(max(0.0, bal) * price) if price else None},
            "installations": rows,
            "notes": ["Excess-emissions penalty: €100/t CO2 (inflation-indexed) and the allowances must still be surrendered.",
                      "CBAM sectors' free allocation is multiplied by the CBAM factor from 2026 (2.5 % cut) to 2034 (100 %).",
                      "Allocation also changes with benchmark updates and the cross-sectoral correction — set "
                      "allocation_decline_pct_per_year if known."]}


def screen_credits(lots: list, current_year: int = 2026, max_vintage_age: int = 5) -> dict:
    """lots: [{id, registry, project_type, quantity_t, vintage, ccp_label, corresponding_adjustment, retired, intended_use}]
    intended_use: target | neutralisation | bvcm | product_claim | compliance"""
    out, tot = [], {"quantity_t": 0.0, "ccp": 0.0, "removals": 0.0, "durable_removals": 0.0}
    for l in lots:
        q = float(l.get("quantity_t", 0) or 0)
        if q < 0:
            raise MarketError(f"{l.get('id')}: quantity must be ≥ 0")
        pt = (l.get("project_type") or "").lower().replace(" ", "_").replace("-", "_")
        removal = pt in REMOVAL_TYPES
        durable = pt in DURABLE_REMOVALS
        age = current_year - int(l["vintage"]) if l.get("vintage") else None
        use = (l.get("intended_use") or "bvcm").lower()
        flags, ok_uses = [], ["bvcm"]
        if l.get("quantity_t") is None:
            flags.append("quantity missing in the record: left out of the totals, not counted as zero")
        if not l.get("ccp_label"):
            flags.append("no ICVCM Core Carbon Principles label declared for this lot")
        if age is not None and age > max_vintage_age:
            flags.append(f"vintage {l['vintage']} is {age} years old (> {max_vintage_age})")
        if removal and not durable:
            flags.append("nature-based removal — reversal risk; check buffer pool and monitoring period")
        if removal:
            ok_uses.append("neutralisation")
        if l.get("corresponding_adjustment"):
            ok_uses.append("compliance (Article 6 / CORSIA, subject to scheme rules)")
        if use == "target":
            flags.append("credits cannot count towards SBTi near- or long-term targets (report as BVCM instead)")
        if use == "product_claim":
            flags.append("EU Directive 2024/825: offset-based product neutrality claims banned in the EU from 27 Sep 2026")
        if use == "neutralisation" and not removal:
            flags.append("avoidance/reduction credit — cannot neutralise residual emissions under SBTi net-zero")
        if not l.get("retired"):
            flags.append("not yet retired — a claim requires retirement on the registry in the company's name")
        cer = None
        from .carbon_registry import assess_cer, is_cdm_cer
        if is_cdm_cer(l):
            cer = assess_cer(l)
            flags.append("CDM CER: " + "; ".join(f"{p['pathway']}: {p['status'].replace('_', ' ')}" for p in cer["pathways"]) + " (see cer.pathways for the reasons)")
        tot["quantity_t"] += q
        tot["ccp"] += q if l.get("ccp_label") else 0
        tot["removals"] += q if removal else 0
        tot["durable_removals"] += q if durable else 0
        out.append({**l, "removal": removal, "durable": durable, "vintage_age": age, "suitable_uses": ok_uses, "cer": cer,
                    "flags": flags, "status": "ok" if not flags else ("caution" if all(
                        f.startswith(("nature-based", "no ICVCM", "not yet retired", "CDM CER", "quantity missing")) for f in flags) else "issue")})
    Q = tot["quantity_t"] or 1
    return {"lots": out, "totals": {k: round(v, 1) for k, v in tot.items()},
            "shares": {"ccp_labelled": round(tot["ccp"] / Q, 3), "removals": round(tot["removals"] / Q, 3),
                       "durable_removals": round(tot["durable_removals"] / Q, 3)},
            "disclosure": "ESRS E1-7: report credits cancelled in the year and planned, by type and standard, "
                          "separately from Scope 1–3 emissions and from progress towards GHG targets.",
            "rules": ["SBTi Corporate Net-Zero Standard: no credits towards targets; permanent removals only for "
                      "neutralising residual emissions", "EU Directive 2024/825 (ECGT) from 27 Sep 2026",
                      "ICVCM Core Carbon Principles label as declared by the registry record",
                      "CDM CERs: Article 6.4 first-NDC conditions (decision 3/CMA.3 annex, para 75) and ICAO CORSIA Eligible Emissions Units (Oct 2025), see carbon_registry.py"]}
