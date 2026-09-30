"""
CSRD double materiality assessment — IRO (impacts, risks, opportunities) register.

Follows ESRS 1 chapter 3 (paras 43–51) and EFRAG IG 1 (Materiality Assessment Implementation Guidance):
  Impact materiality — severity of the impact (scale, scope and, for negative impacts, irremediable character)
                       and, for potential impacts, likelihood. For potential negative human-rights impacts
                       severity takes precedence over likelihood (ESRS 1 para 45).
  Financial materiality — likelihood of occurrence and potential magnitude of financial effects.
A matter is material if it is material from either perspective.

ESRS does not prescribe a scoring formula; the one used here is stated in every output and the threshold is
the company's to set (default 3 on a 1–5 scale). Stakeholder ratings are aggregated and shown alongside the
company's own scores — they inform, they do not override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

ESRS_TOPICS = {
    "E1": "Climate change", "E2": "Pollution", "E3": "Water and marine resources",
    "E4": "Biodiversity and ecosystems", "E5": "Resource use and circular economy",
    "S1": "Own workforce", "S2": "Workers in the value chain", "S3": "Affected communities",
    "S4": "Consumers and end-users", "G1": "Business conduct",
}
TYPES = ("negative_impact", "positive_impact", "risk", "opportunity")


@dataclass
class IRO:
    id: str
    topic: str                          # ESRS topical standard code, e.g. "E1"
    subtopic: str
    type: str                           # negative_impact | positive_impact | risk | opportunity
    description: str
    actual: bool = False                # actual (True) or potential impact
    human_rights: bool = False
    value_chain: str = "own_operations" # own_operations | upstream | downstream
    time_horizon: str = "medium"        # short (<1y) | medium (1–5y) | long (>5y) — ESRS 1 para 77
    scale: Optional[int] = None         # 1–5 (impacts)
    scope: Optional[int] = None         # 1–5 (impacts)
    irremediability: Optional[int] = None  # 1–5 (negative impacts)
    likelihood: Optional[int] = None    # 1–5
    magnitude: Optional[int] = None     # 1–5 (risks/opportunities: size of financial effect)
    evidence: str = ""
    stakeholder_ratings: list = field(default_factory=list)   # [{"group": "investors", "rating": 1–5}]


class MaterialityError(ValueError):
    pass


def _chk(v, name, iro):
    if v is None or not 1 <= int(v) <= 5:
        raise MaterialityError(f"{iro.id}: {name} must be an integer 1–5")
    return int(v)


def score(iro: IRO) -> dict:
    if iro.topic not in ESRS_TOPICS:
        raise MaterialityError(f"{iro.id}: topic must be one of {', '.join(ESRS_TOPICS)}")
    if iro.type not in TYPES:
        raise MaterialityError(f"{iro.id}: type must be one of {', '.join(TYPES)}")
    out = {"impact_score": None, "financial_score": None, "working": ""}
    if iro.type.endswith("impact"):
        sc, sp = _chk(iro.scale, "scale", iro), _chk(iro.scope, "scope", iro)
        parts = [sc, sp]
        if iro.type == "negative_impact":
            parts.append(_chk(iro.irremediability, "irremediability", iro))
        severity = mean(parts)
        if iro.actual or (iro.type == "negative_impact" and iro.human_rights):
            s = severity
            out["working"] = (f"severity mean({', '.join(map(str, parts))}) = {severity:.2f}"
                              + ("; actual impact" if iro.actual else "; human-rights impact: severity takes precedence"))
        else:
            lk = _chk(iro.likelihood, "likelihood", iro)
            s = (severity * lk) ** 0.5
            out["working"] = f"√(severity {severity:.2f} × likelihood {lk})"
        out["impact_score"] = round(s, 2)
        out["severity"] = round(severity, 2)
    else:
        mg, lk = _chk(iro.magnitude, "magnitude", iro), _chk(iro.likelihood, "likelihood", iro)
        out["financial_score"] = round((mg * lk) ** 0.5, 2)
        out["working"] = f"√(magnitude {mg} × likelihood {lk})"
    if iro.stakeholder_ratings:
        rs = [float(r["rating"]) for r in iro.stakeholder_ratings if 1 <= float(r.get("rating", 0)) <= 5]
        out["stakeholder_mean"] = round(mean(rs), 2) if rs else None
        out["stakeholder_groups"] = sorted({r.get("group", "") for r in iro.stakeholder_ratings})
    return out


def assess(iros: list, threshold: float = 3.0) -> dict:
    if not 1 <= threshold <= 5:
        raise MaterialityError("threshold must be between 1 and 5")
    ids = [i.id for i in iros]
    if len(ids) != len(set(ids)):
        raise MaterialityError("IRO ids must be unique")
    rows = []
    for i in iros:
        s = score(i)
        mat_i = s["impact_score"] is not None and s["impact_score"] >= threshold
        mat_f = s["financial_score"] is not None and s["financial_score"] >= threshold
        rows.append({"id": i.id, "topic": i.topic, "topic_name": ESRS_TOPICS[i.topic], "subtopic": i.subtopic,
                     "type": i.type, "description": i.description, "actual": i.actual, "value_chain": i.value_chain,
                     "time_horizon": i.time_horizon, **s, "material_impact": mat_i, "material_financial": mat_f,
                     "material": mat_i or mat_f, "evidence": i.evidence,
                     "divergence": (s.get("stakeholder_mean") is not None and
                                    abs((s["impact_score"] or s["financial_score"]) - s["stakeholder_mean"]) >= 1.5)})
    topics = {}
    for r in rows:
        t = topics.setdefault(r["topic"], {"topic": r["topic"], "name": r["topic_name"], "material": False,
                                           "impact": False, "financial": False, "iros": []})
        t["iros"].append(r["id"])
        t["material"] |= r["material"]
        t["impact"] |= r["material_impact"]
        t["financial"] |= r["material_financial"]
    checks = []
    if "E1" not in topics:
        checks.append({"level": "warning", "message": "No climate (E1) IROs assessed — if E1 is not material, ESRS 2 "
                                                      "IRO-2 requires a detailed explanation of that conclusion"})
    for r in rows:
        if not r["evidence"]:
            checks.append({"level": "warning", "message": f"{r['id']}: no evidence recorded for the assessment"})
        if r["divergence"]:
            checks.append({"level": "info", "message": f"{r['id']}: stakeholder view differs from the company score by ≥ 1.5"})
    return {
        "threshold": threshold,
        "method": "Impacts: severity = mean(scale, scope[, irremediability]); potential impacts √(severity × likelihood); "
                  "actual and human-rights impacts use severity. Financial: √(magnitude × likelihood). All 1–5. "
                  "Material if either score ≥ threshold (ESRS 1 §43–51; EFRAG IG 1).",
        "iros": rows, "topics": sorted(topics.values(), key=lambda t: t["topic"]),
        "material_topics": [t["topic"] for t in sorted(topics.values(), key=lambda t: t["topic"]) if t["material"]],
        "matrix": [{"id": r["id"], "x_financial": r["financial_score"], "y_impact": r["impact_score"]} for r in rows],
        "checks": checks,
        "disclosures": {"ESRS 2 IRO-1": "description of the process (this register + method + stakeholder input)",
                        "ESRS 2 IRO-2": "list of disclosure requirements complied with — for each material topic",
                        "ESRS 2 SBM-3": "material IROs and their interaction with strategy (from the material rows)"},
    }


def suggest_e1(inventory_totals_t: Optional[dict] = None, physical_aal_pct: Optional[float] = None,
               carbon_price_usd_t: Optional[float] = None) -> list:
    """Starter E1 IROs pre-filled from engine results — the company still scores and evidences them."""
    s = []
    if inventory_totals_t:
        tot = inventory_totals_t.get("total_market") or inventory_totals_t.get("total_location")
        s.append({"id": "E1-GHG", "topic": "E1", "subtopic": "Climate change mitigation", "type": "negative_impact",
                  "actual": True, "description": f"GHG emissions of {tot:,.0f} t CO2e (Scope 1+2+3, market-based)",
                  "evidence": "ClimRisk GHG inventory"})
    if physical_aal_pct is not None:
        s.append({"id": "E1-PHYS", "topic": "E1", "subtopic": "Climate change adaptation", "type": "risk",
                  "description": f"Physical damage to sites — average annual loss {physical_aal_pct:.2f}% of asset value",
                  "evidence": "ClimRisk Risk Analyst"})
    if carbon_price_usd_t:
        s.append({"id": "E1-PRICE", "topic": "E1", "subtopic": "Climate change mitigation", "type": "risk",
                  "description": f"Carbon pricing — ${carbon_price_usd_t:.0f}/t applies at operating sites today",
                  "evidence": "World Bank Carbon Pricing Dashboard via ClimRisk"})
    s.append({"id": "E1-ENERGY", "topic": "E1", "subtopic": "Energy", "type": "opportunity",
              "description": "Energy efficiency and renewable electricity cost savings", "evidence": ""})
    return s


def from_dict(d: dict) -> IRO:
    return IRO(**d)
