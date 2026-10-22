"""
ESRS E1 (climate change) disclosure gap assessment.

Takes what a company has actually provided and reports, requirement by requirement,
what is provided, what is only an engine estimate, and what is missing.

What this is
  A coverage check against a checklist of the ESRS E1 disclosure requirements (Commission Delegated
  Regulation (EU) 2023/2772, Annex I, ESRS E1) plus the three cross-cutting items E1 refers to
  (GOV-3, IRO-1, SBM-3). Requirement codes and the E1-1 paragraph-16 letters follow the standard;
  finer paragraph numbers and the full EFRAG data-point list are NOT encoded.

What this is not
  - Not a compliance opinion, not assurance, not legal advice.
  - It does not decide whether CSRD applies to a company, nor any phase-in or omission relief. The
    scope and timing of CSRD have been amended by EU legislation; confirm applicability separately.
  - It does not judge the quality of what was provided. A provided text only has to be non-empty;
    a provided number only has to be a finite non-negative number.
  - It does not produce digital (XBRL) tagging.

Null is not zero: a value of 0 is a valid provided number; None or an empty string is missing.
Engine estimates never count as "provided". They are shown as `proxy_only`, labelled as modelled
and not company-reported, because ESRS asks for the company's own measured figures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# kind: text | number | bool | list
# when: (other_item_id, required_value) - applicable only if that item was provided with that value
# proxy: key under which the engine may supply a modelled stand-in


@dataclass(frozen=True)
class Item:
    id: str
    req: str
    title: str
    kind: str
    unit: str = ""
    when: Optional[tuple] = None
    proxy: Optional[str] = None


CHECKLIST: tuple[Item, ...] = (
    # ── cross-cutting items that ESRS E1 refers to ──────────────────────────────────────────────
    Item("E1.GOV-3.incentives", "E1.GOV-3", "Whether and how climate-related considerations are in incentive schemes for the administrative, management and supervisory bodies", "text"),
    Item("E1.IRO-1.process", "E1.IRO-1", "Process to identify and assess climate-related impacts, risks and opportunities", "text"),
    Item("E1.IRO-1.scenario_analysis", "E1.IRO-1", "Climate scenario analysis used for physical and transition risk, incl. a high-emission scenario", "text", proxy="scenario_analysis"),
    Item("E1.SBM-3.risks_and_resilience", "E1.SBM-3", "Material climate-related physical and transition risks and opportunities, and resilience of strategy", "text", proxy="scenario_analysis"),
    # ── E1-1 transition plan ────────────────────────────────────────────────────────────────────
    Item("E1-1.has_plan", "E1-1", "Whether a transition plan for climate change mitigation has been adopted (or if not, whether and when)", "bool"),
    Item("E1-1.16a", "E1-1", "How GHG reduction targets are compatible with limiting global warming to 1.5 C", "text"),
    Item("E1-1.16b", "E1-1", "Decarbonisation levers identified and key actions", "text"),
    Item("E1-1.16c", "E1-1", "Significant CapEx / OpEx needed to implement the plan", "number", "currency", proxy="transition_capex"),
    Item("E1-1.16d", "E1-1", "Potential locked-in GHG emissions from key assets and products", "number", "tCO2e"),
    Item("E1-1.16e", "E1-1", "Objectives or plans for aligning economic activities with the EU Taxonomy (CapEx / OpEx)", "text"),
    Item("E1-1.16f", "E1-1", "Significant CapEx for coal, oil and gas related economic activities", "number", "currency"),
    Item("E1-1.16g", "E1-1", "Whether the undertaking is excluded from the EU Paris-aligned Benchmarks", "bool"),
    Item("E1-1.16h", "E1-1", "How the plan is embedded in and aligned with business strategy and financial planning", "text"),
    Item("E1-1.16i", "E1-1", "Whether the plan is approved by the administrative, management and supervisory bodies", "bool"),
    Item("E1-1.16j", "E1-1", "Progress in implementing the plan", "text"),
    # ── E1-2 .. E1-4 ────────────────────────────────────────────────────────────────────────────
    Item("E1-2.policy_mitigation", "E1-2", "Policies for climate change mitigation", "text"),
    Item("E1-2.policy_adaptation", "E1-2", "Policies for climate change adaptation", "text"),
    Item("E1-3.actions", "E1-3", "Actions taken and planned for mitigation and adaptation", "text"),
    Item("E1-3.resources", "E1-3", "Resources (CapEx / OpEx) allocated to those actions", "number", "currency", proxy="adaptation_capex"),
    Item("E1-4.ghg_targets", "E1-4", "GHG emission reduction targets (scope, base year, target year, reduction) ", "list"),
    Item("E1-4.adaptation_targets", "E1-4", "Targets for climate change adaptation", "text"),
    # ── E1-5 energy ─────────────────────────────────────────────────────────────────────────────
    Item("E1-5.total_energy", "E1-5", "Total energy consumption", "number", "MWh"),
    Item("E1-5.fossil_energy", "E1-5", "Energy consumption from fossil sources", "number", "MWh"),
    Item("E1-5.nuclear_energy", "E1-5", "Energy consumption from nuclear sources", "number", "MWh"),
    Item("E1-5.renewable_energy", "E1-5", "Energy consumption from renewable sources", "number", "MWh"),
    Item("E1-5.energy_intensity", "E1-5", "Energy intensity per net revenue (activities in high climate impact sectors)", "number", "MWh per currency unit"),
    # ── E1-6 gross GHG ──────────────────────────────────────────────────────────────────────────
    Item("E1-6.scope1", "E1-6", "Gross Scope 1 GHG emissions", "number", "tCO2e", proxy="scope1"),
    Item("E1-6.scope2_location", "E1-6", "Gross Scope 2 GHG emissions, location-based", "number", "tCO2e", proxy="scope2"),
    Item("E1-6.scope2_market", "E1-6", "Gross Scope 2 GHG emissions, market-based", "number", "tCO2e"),
    Item("E1-6.scope3_total", "E1-6", "Gross Scope 3 GHG emissions (significant categories)", "number", "tCO2e", proxy="scope3"),
    Item("E1-6.total_ghg", "E1-6", "Total GHG emissions", "number", "tCO2e"),
    Item("E1-6.ghg_intensity", "E1-6", "GHG intensity per net revenue", "number", "tCO2e per currency unit"),
    # ── E1-7 removals and credits ───────────────────────────────────────────────────────────────
    Item("E1-7.removals", "E1-7", "GHG removals and storage from own operations and value chain", "number", "tCO2e"),
    Item("E1-7.credits_cancelled", "E1-7", "GHG emission reductions from carbon credits verified and cancelled in the reporting year", "number", "tCO2e"),
    # ── E1-8 internal carbon pricing ────────────────────────────────────────────────────────────
    Item("E1-8.uses_price", "E1-8", "Whether an internal carbon pricing scheme is applied", "bool"),
    Item("E1-8.price", "E1-8", "Internal carbon price applied", "number", "currency per tCO2e", when=("E1-8.uses_price", True)),
    # ── E1-9 anticipated financial effects ──────────────────────────────────────────────────────
    Item("E1-9.physical_assets_at_risk", "E1-9", "Carrying amount of assets at material physical risk", "number", "currency"),
    Item("E1-9.physical_loss", "E1-9", "Anticipated financial effect of material physical risks", "number", "currency", proxy="physical_loss"),
    Item("E1-9.transition_assets_at_risk", "E1-9", "Carrying amount of assets at material transition risk", "number", "currency"),
    Item("E1-9.transition_effect", "E1-9", "Anticipated financial effect of material transition risks", "number", "currency", proxy="transition_effect"),
    Item("E1-9.opportunities", "E1-9", "Potential climate-related opportunities (expected cost savings, market size)", "text"),
)

BY_ID = {i.id: i for i in CHECKLIST}
STATUSES = ("provided", "invalid", "proxy_only", "missing", "not_applicable")


def _check(item: Item, value: Any) -> tuple[str, str]:
    """Returns (status, note) for a user-provided value."""
    if value is None:
        return "missing", ""
    if item.kind == "text":
        if isinstance(value, str) and value.strip():
            return "provided", ""
        return ("missing", "") if isinstance(value, str) else ("invalid", "expected text")
    if item.kind == "bool":
        return ("provided", "") if isinstance(value, bool) else ("invalid", "expected true or false")
    if item.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "invalid", "expected a number"
        if not math.isfinite(value) or value < 0:
            return "invalid", "expected a finite number, zero or above"
        return "provided", ""
    if item.kind == "list":
        if not isinstance(value, (list, tuple)):
            return "invalid", "expected a list"
        if not value:
            return "missing", ""
        return "provided", ""
    return "invalid", "unknown kind"


def assess_esrs_e1(
    provided: Optional[dict] = None,
    engine_proxies: Optional[dict] = None,
    e1_material: Optional[bool] = None,
    e1_not_material_explanation: Optional[str] = None,
) -> dict:
    """
    provided        item id -> the company's own value. Unknown ids are reported, not ignored.
    engine_proxies  proxy key -> modelled value from this engine (never counted as provided).
    e1_material     the company's own conclusion from its materiality assessment, or None if not stated.
    """
    provided = dict(provided or {})
    proxies = dict(engine_proxies or {})
    unknown = sorted(k for k in provided if k not in BY_ID)

    rows = []
    for it in CHECKLIST:
        if it.when is not None:
            dep_id, need = it.when
            dep_status, _ = _check(BY_ID[dep_id], provided.get(dep_id))
            if dep_status == "provided" and provided.get(dep_id) != need:
                rows.append({"id": it.id, "requirement": it.req, "title": it.title, "status": "not_applicable",
                             "note": f"applies only if {dep_id} is {need}"})
                continue
        status, note = _check(it, provided.get(it.id))
        row = {"id": it.id, "requirement": it.req, "title": it.title, "status": status, "note": note}
        if it.unit:
            row["unit"] = it.unit
        if status == "provided":
            row["value"] = provided[it.id]
        elif status == "missing" and it.proxy and proxies.get(it.proxy) is not None:
            row["status"] = "proxy_only"
            row["engine_estimate"] = proxies[it.proxy]
            row["note"] = "engine estimate only (modelled, not company-reported); ESRS needs the company's own figure"
        rows.append(row)

    applicable = [r for r in rows if r["status"] != "not_applicable"]
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    by_req: dict[str, dict] = {}
    for r in applicable:
        d = by_req.setdefault(r["requirement"], {"applicable": 0, "provided": 0, "proxy_only": 0, "missing": 0, "invalid": 0})
        d["applicable"] += 1
        d[r["status"]] += 1
    for d in by_req.values():
        d["coverage"] = round(d["provided"] / d["applicable"], 3)

    warnings = []
    if e1_material is None:
        warnings.append("Materiality of E1 not stated. ESRS 1 requires the company's own materiality assessment; "
                        "a requirement can be omitted only if E1 is concluded not material and that is explained.")
    elif e1_material is False and not (e1_not_material_explanation or "").strip():
        warnings.append("E1 is stated as not material but no explanation is given. An explanation of the conclusion is required.")
    if unknown:
        warnings.append(f"Unrecognised item ids ignored in the coverage: {', '.join(unknown)}.")
    if counts["invalid"]:
        warnings.append(f"{counts['invalid']} provided value(s) failed the format check and are counted as gaps.")

    n = len(applicable)
    return {
        "framework": "ESRS E1 (Delegated Regulation (EU) 2023/2772), requirement-level checklist",
        "e1_material": e1_material,
        "summary": {
            "items_applicable": n,
            "provided": counts["provided"],
            "proxy_only": counts["proxy_only"],
            "missing": counts["missing"],
            "invalid": counts["invalid"],
            "not_applicable": counts["not_applicable"],
            "coverage": round(counts["provided"] / n, 3) if n else None,
            "coverage_meaning": "share of checklist items the company has supplied in a valid format; "
                                "not a compliance score and not a measure of quality",
        },
        "by_requirement": by_req,
        "items": rows,
        "gaps": [r["id"] for r in applicable if r["status"] in ("missing", "invalid", "proxy_only")],
        "warnings": warnings,
        "limits": [
            "Checklist is at requirement level (E1-1 to E1-9 and the cross-cutting GOV-3, IRO-1, SBM-3). Requirement codes and the "
            "E1-1 paragraph-16 letters follow the standard; finer paragraph numbers and the full EFRAG data-point list are not encoded.",
            "Whether CSRD applies to the company, and any phase-in or omission relief, are not assessed. Scope and timing have been "
            "amended by EU legislation, so confirm applicability separately.",
            "Content is checked for presence and format only, not for quality, consistency or auditability.",
            "Engine estimates are shown for orientation and never counted as provided.",
            "Other topical standards (E2 to E5, S1 to S4, G1) and ESRS 2 general disclosures are out of scope here.",
        ],
    }
