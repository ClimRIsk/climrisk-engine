"""LLM micro-agent layer — generates human-readable outputs from engine data.

Pillar 3: Hybrid AI Integration.
Uses Anthropic Claude API (ANTHROPIC_API_KEY env variable).
Gracefully falls back to template-based output if no key is configured.

Design principle: the LLM is the communicator, the engine is the calculator.
Every number in the output comes from the engine — the LLM only arranges prose.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Article type config ───────────────────────────────────────────────────────

_ARTICLE_CONFIG = {
    "linkedin": {
        "label": "LinkedIn Article",
        "format": (
            "a LinkedIn article suitable for a financial/risk professional audience. "
            "Use short paragraphs (2-3 sentences each). Include a compelling opening hook. "
            "Bold 2-3 key statistics using **bold**. End with a clear takeaway or call-to-action. "
            "No bullet lists — prose only."
        ),
        "max_words": 800,
    },
    "investor_memo": {
        "label": "Investor Memo",
        "format": (
            "a formal investor memo with sections: Executive Summary, Risk Assessment, "
            "Financial Impact, Recommended Actions, Disclosure Implications. "
            "Use precise financial language. Include all key metrics from the engine output."
        ),
        "max_words": 1200,
    },
    "case_study": {
        "label": "Case Study",
        "format": (
            "a case study narrative describing the company's climate risk profile. "
            "Structure: Company Context → Climate Exposure → Financial Impact → "
            "Strategic Response → Outcome/Recommendation. Use third-person tone."
        ),
        "max_words": 900,
    },
    "regulatory_commentary": {
        "label": "Regulatory Commentary",
        "format": (
            "a regulatory-style commentary suitable for TCFD/IFRS S2 supplementary disclosure. "
            "Reference the scenario analysis (NZE, Current Policies), quantify material risks, "
            "and describe risk management processes. Formal register, no marketing language."
        ),
        "max_words": 1000,
    },
}

_AUDIENCE_TONE = {
    "professional": "Use precise financial and climate risk terminology. Assume the reader is a CRO or CFO.",
    "executive":    "Use plain English for a board or C-suite audience. Translate jargon into business impact.",
    "accessible":   "Write for an informed general audience. Avoid acronyms without explanation.",
}

_SECTION_PROMPTS = {
    "executive_summary":  "Write a 3-paragraph executive summary of the climate risk assessment.",
    "physical_risk":      "Write a detailed physical risk section covering hazard exposure, EAL, and adaptation requirements.",
    "transition_risk":    "Write a transition risk section covering carbon cost exposure, EBITDA compression, and stranded asset risk.",
    "financial_impact":   "Write a financial impact section translating physical and transition risks into balance sheet and P&L terms.",
    "recommendations":    "Write a recommendations section with specific, prioritised actions for the risk manager.",
}


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> Optional[str]:
    """Call Anthropic Claude API. Returns None if key not set or call fails."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set — using template fallback")
        return None

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return msg.content[0].text if msg.content else None
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


# ── Data extraction helpers ───────────────────────────────────────────────────

def _extract_key_numbers(brief: Dict[str, Any]) -> List[str]:
    """Pull the most important numbers from a DecisionBrief dict for citation."""
    fi = brief.get("financial_impact", {})
    sig = brief.get("signal", {})
    numbers = []
    if fi.get("expected_annual_loss_usd_m"):
        numbers.append(f"EAL: ${fi['expected_annual_loss_usd_m']:.1f}M/year")
    if fi.get("worst_case_1pct_usd_m"):
        numbers.append(f"CVaR₉₉: ${fi['worst_case_1pct_usd_m']:.1f}M")
    if fi.get("ebitda_impact_pct"):
        numbers.append(f"EBITDA impact: {fi['ebitda_impact_pct']:.1f}%")
    if fi.get("stranded_asset_value_usd_m"):
        numbers.append(f"Stranded assets: ${fi['stranded_asset_value_usd_m']:.1f}M")
    if fi.get("npv_haircut_pct"):
        numbers.append(f"NPV haircut: {fi['npv_haircut_pct']:.1f}%")
    if fi.get("credit_spread_widening_bps"):
        numbers.append(f"Credit spread: +{fi['credit_spread_widening_bps']:.0f} bps")
    if sig.get("action"):
        numbers.append(f"Signal: {sig['action']}")
    for d in brief.get("top_drivers", [])[:3]:
        numbers.append(f"{d.get('factor','')}: {d.get('contribution_pct',0):.0f}% of EAL")
    return numbers


def _brief_to_context(brief: Dict[str, Any]) -> str:
    """Serialise a DecisionBrief to a compact, LLM-readable context block."""
    fi = brief.get("financial_impact", {})
    sig = brief.get("signal", {})
    drivers = brief.get("top_drivers", [])
    actions = brief.get("recommended_actions", [])

    lines = [
        f"COMPANY: {brief.get('company_name', brief.get('company_id', 'Unknown'))}",
        f"SECTOR: {brief.get('sector', '–')}",
        f"SCENARIO: {brief.get('scenario_label', 'Current Policies')}",
        f"HORIZON: {brief.get('horizon_year', 2035)}",
        "",
        "=== RISK SIGNAL ===",
        f"Action: {sig.get('action', '–')}",
        f"Confidence: {sig.get('confidence', '–')}",
        f"Urgency: {sig.get('urgency', '–')}",
        f"Rationale: {sig.get('rationale', '–')}",
        "",
        "=== FINANCIAL IMPACT ===",
        f"Expected Annual Loss (EAL): ${fi.get('expected_annual_loss_usd_m', 0):.2f}M/year",
        f"Worst-Case CVaR₉₉: ${fi.get('worst_case_1pct_usd_m', 0):.2f}M",
        f"EBITDA Impact: {fi.get('ebitda_impact_pct', 0):.2f}%",
        f"Revenue at Risk: {fi.get('revenue_at_risk_pct', 0):.2f}%",
        f"Stranded Asset Value: ${fi.get('stranded_asset_value_usd_m', 0):.2f}M",
        f"Credit Spread Widening: {fi.get('credit_spread_widening_bps', 0):.0f} bps",
        f"WACC Uplift: {fi.get('wacc_uplift_bps', 0):.0f} bps",
        f"NPV Haircut: {fi.get('npv_haircut_pct', 0):.1f}%",
        "",
        "=== TOP RISK DRIVERS ===",
    ]
    for d in drivers[:5]:
        lines.append(
            f"  • {d.get('factor','?')} [{d.get('category','?')}]: "
            f"{d.get('contribution_pct',0):.0f}% of EAL | Severity: {d.get('severity','?')}"
        )
    lines.append("")
    lines.append("=== RECOMMENDED ACTIONS ===")
    for a in actions[:5]:
        cost = f" | Cost: ${a.get('cost_usd_m','?')}M" if a.get("cost_usd_m") else ""
        dl = f" | Deadline: {a['deadline']}" if a.get("deadline") else ""
        lines.append(f"  {a.get('priority','?')}. {a.get('action','?')}{cost}{dl}")

    cl = brief.get("compliance_layer") or {}
    if cl:
        lines += [
            "",
            "=== COMPLIANCE ===",
            f"ISSB S2 material: {cl.get('issb_s2_material', False)}",
            f"CSRD required: {cl.get('csrd_required', False)}",
        ]
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_article(
    brief: Dict[str, Any],
    article_type: str = "linkedin",
    tone: str = "professional",
    word_count: int = 600,
    focus: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a grounded article from a DecisionBrief dict.

    Returns a dict with keys: content, word_count_actual, key_numbers_used.
    All numbers in the content come exclusively from `brief`.
    """
    cfg = _ARTICLE_CONFIG.get(article_type, _ARTICLE_CONFIG["linkedin"])
    tone_instruction = _AUDIENCE_TONE.get(tone, _AUDIENCE_TONE["professional"])
    context = _brief_to_context(brief)
    key_numbers = _extract_key_numbers(brief)
    company = brief.get("company_name") or brief.get("company_id", "the company")
    signal = (brief.get("signal") or {}).get("action", "WATCH")

    system_prompt = (
        "You are ClimRisk's institutional content engine. You transform deterministic "
        "climate risk engine outputs into high-quality written content for finance professionals.\n\n"
        "CRITICAL RULES:\n"
        "1. Every number you include MUST come from the ENGINE OUTPUT below. Never invent figures.\n"
        "2. If a number is not in the engine output, describe the factor qualitatively — do not estimate.\n"
        "3. The signal (HOLD/WATCH/REDUCE/EXIT) must be referenced accurately.\n"
        "4. Do not add disclaimers or caveats within the article text — those are added separately.\n"
        f"5. Tone instruction: {tone_instruction}"
    )

    focus_line = f"\nFocus especially on {focus} risk factors." if focus else ""
    user_prompt = (
        f"Write {cfg['format']}\n\n"
        f"Target length: approximately {min(word_count, cfg['max_words'])} words.{focus_line}\n\n"
        f"ENGINE OUTPUT (use only these numbers):\n{context}\n\n"
        f"Write the {cfg['label']} now:"
    )

    content = _call_llm(system_prompt, user_prompt, max_tokens=min(word_count * 2, 2000))

    if content is None:
        # Template fallback — no LLM
        fi = brief.get("financial_impact", {})
        sig = brief.get("signal", {})
        drivers_txt = ", ".join(
            d.get("factor", "?") for d in brief.get("top_drivers", [])[:3]
        )
        content = _template_article(
            company=company,
            signal=signal,
            eal=fi.get("expected_annual_loss_usd_m", 0),
            cvar=fi.get("worst_case_1pct_usd_m", 0),
            ebitda_pct=fi.get("ebitda_impact_pct", 0),
            stranded=fi.get("stranded_asset_value_usd_m", 0),
            npv_haircut=fi.get("npv_haircut_pct", 0),
            drivers_txt=drivers_txt,
            sector=brief.get("sector", ""),
            horizon=brief.get("horizon_year", 2035),
            rationale=sig.get("rationale", ""),
            article_type=article_type,
        )

    actual_words = len(re.findall(r"\w+", content))
    return {
        "content": content,
        "word_count_actual": actual_words,
        "key_numbers_used": key_numbers,
    }


def generate_narrative(
    brief: Dict[str, Any],
    section: str,
    audience: str = "risk_manager",
) -> Dict[str, Any]:
    """Generate one narrative section from a DecisionBrief dict."""
    section_prompt = _SECTION_PROMPTS.get(
        section, f"Write a {section.replace('_', ' ')} section for this climate risk assessment."
    )
    tone_instruction = _AUDIENCE_TONE.get(audience, _AUDIENCE_TONE["professional"])
    context = _brief_to_context(brief)
    fi = brief.get("financial_impact", {})

    system_prompt = (
        "You are a climate risk analyst producing institutional-grade written assessments. "
        "Use only the numbers from the engine output. Do not invent or estimate figures. "
        f"Audience tone: {tone_instruction}"
    )
    user_prompt = (
        f"{section_prompt}\n\n"
        f"ENGINE OUTPUT:\n{context}\n\n"
        "Write 3–5 paragraphs. Be specific and quantitative where the data allows."
    )

    content = _call_llm(system_prompt, user_prompt, max_tokens=800)
    if content is None:
        content = _template_narrative_section(section, brief)

    key_metrics = {
        "eal_usd_m": fi.get("expected_annual_loss_usd_m"),
        "ebitda_impact_pct": fi.get("ebitda_impact_pct"),
        "signal": (brief.get("signal") or {}).get("action"),
    }
    return {"content": content, "key_metrics": key_metrics}


def generate_executive_summary(brief: Dict[str, Any]) -> str:
    """Convenience wrapper — one-paragraph executive summary."""
    result = generate_narrative(brief, section="executive_summary", audience="executive")
    return result["content"]


# ── Template fallbacks (no API key) ──────────────────────────────────────────

def _template_article(
    company: str, signal: str, eal: float, cvar: float, ebitda_pct: float,
    stranded: float, npv_haircut: float, drivers_txt: str, sector: str,
    horizon: int, rationale: str, article_type: str,
) -> str:
    signal_verb = {
        "EXIT": "exit or significantly reduce",
        "REDUCE": "reduce",
        "WATCH": "closely monitor",
        "HOLD": "maintain with enhanced monitoring of",
    }.get(signal, "review")

    if article_type == "linkedin":
        return (
            f"Climate risk is no longer a distant scenario — for {company}, it's a balance "
            f"sheet reality today.\n\n"
            f"Our CRI Engine assessment signals **{signal}** on {company}'s climate exposure, "
            f"with a {rationale.lower() or 'material risk profile requiring attention'}.\n\n"
            f"The numbers: **${eal:.1f}M expected annual loss** under Current Policies "
            f"({ebitda_pct:.1f}% of EBITDA), rising to a **${cvar:.1f}M tail loss** at CVaR₉₉. "
            f"Under a Net Zero scenario, the stranded asset exposure reaches **${stranded:.1f}M**, "
            f"with an NPV haircut of {npv_haircut:.1f}%.\n\n"
            f"The primary drivers — {drivers_txt} — combine physical hazard exposure with "
            f"transition-driven EBITDA compression that accelerates post-2030.\n\n"
            f"Risk managers should {signal_verb} this exposure by {horizon}. "
            f"The data is deterministic, physics-based, and audit-ready. "
            f"The question is whether your portfolio framework is ready to act on it.\n\n"
            f"#ClimateRisk #FinancialRisk #ESG #IFRS S2 #ClimRisk"
        )
    else:
        return (
            f"CLIMATE RISK ASSESSMENT: {company.upper()}\n"
            f"Sector: {sector} | Horizon: {horizon} | Signal: {signal}\n\n"
            f"EXECUTIVE SUMMARY\n"
            f"{company} faces material climate financial risk with an Expected Annual Loss "
            f"of ${eal:.1f}M ({ebitda_pct:.1f}% of EBITDA) under Current Policies. "
            f"Tail risk (CVaR₉₉) reaches ${cvar:.1f}M. Stranded asset exposure under NZE "
            f"is ${stranded:.1f}M, representing an NPV haircut of {npv_haircut:.1f}%.\n\n"
            f"PRIMARY RISK DRIVERS\n{drivers_txt}\n\n"
            f"RECOMMENDATION\nSignal: {signal}. {rationale}\n"
            f"Risk managers are advised to {signal_verb} this exposure ahead of {horizon} "
            f"given the combination of physical hazard acceleration and transition policy tightening."
        )


def _template_narrative_section(section: str, brief: Dict[str, Any]) -> str:
    fi = brief.get("financial_impact", {})
    company = brief.get("company_name") or brief.get("company_id", "the company")
    signal = (brief.get("signal") or {}).get("action", "WATCH")
    eal = fi.get("expected_annual_loss_usd_m", 0)
    ebitda_pct = fi.get("ebitda_impact_pct", 0)

    if section == "executive_summary":
        return (
            f"The CRI Engine assessment for {company} produces a {signal} signal, reflecting "
            f"a material climate risk profile that warrants prompt attention from risk managers "
            f"and credit committees. The Expected Annual Loss under Current Policies is "
            f"${eal:.1f}M, representing {ebitda_pct:.1f}% of EBITDA — a threshold that "
            f"exceeds most institutional materiality screens.\n\n"
            f"The risk is driven by a combination of accelerating physical hazard exposure "
            f"and transition-driven EBITDA compression, both of which intensify toward the "
            f"2035 horizon. The recommended course of action and supporting financial "
            f"quantification are detailed in the sections below."
        )

    if section == "physical_risk":
        ph = brief.get("top_drivers", [])
        physical_drivers = [d for d in ph if d.get("category") == "physical"]
        top_phys = physical_drivers[0] if physical_drivers else {}
        return (
            f"Physical climate hazards present a quantifiable and growing operational risk "
            f"for {company}. The dominant physical exposure is "
            f"{top_phys.get('factor', 'multi-hazard compound risk')}, contributing "
            f"{top_phys.get('contribution_pct', 0):.0f}% of the total Expected Annual Loss "
            f"under the Current Policies scenario.\n\n"
            f"Asset-level damage curves (log-logistic CDF, IPCC AR6-calibrated) indicate "
            f"that peak physical losses will concentrate in the 2030–2035 window, driven by "
            f"compound event amplification under SSP3-7.0. The adaptation capital required "
            f"to maintain current production levels is included in the recommended actions."
        )

    return (
        f"This section covers the {section.replace('_', ' ')} for {company}. "
        f"The CRI Engine signal is {signal} with an EAL of ${eal:.1f}M "
        f"({ebitda_pct:.1f}% EBITDA impact)."
    )
