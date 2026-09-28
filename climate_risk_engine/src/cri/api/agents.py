"""
ClimRisk Agentic AI Module
==========================
Four capabilities:
  1. Portfolio monitoring   — scheduled re-assessments + email alerts
  2. Web research agent     — yfinance + news augmentation before scoring
  3. Conversational AI      — Groq LLM (free) with tool-calling
  4. Autonomous batch       — full portfolio analysis → email delivery

Environment variables needed (add to Render / .env):
  GROQ_API_KEY        — free at console.groq.com (llama-3.3-70b)
  GMAIL_USER          — Gmail address for sending alerts
  GMAIL_APP_PASSWORD  — Gmail App Password (not your login password)
  ENGINE_BASE_URL     — defaults to https://climrisk-github-io.onrender.com
"""

from __future__ import annotations

import json
import logging
import os
import re
import smtplib
from collections import defaultdict
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = "llama-3.3-70b-versatile"
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

GMAIL_USER     = os.getenv("GMAIL_USER", "")
GMAIL_APP_PASS = os.getenv("GMAIL_APP_PASSWORD", "")

ENGINE_BASE    = os.getenv("ENGINE_BASE_URL", "https://climrisk-github-io.onrender.com")

WATCHLIST_PATH = Path("watchlist.json")
SCORES_PATH    = Path("watchlist_scores.json")

ALERT_THRESHOLD = 8.0   # CRI score points — alert if delta exceeds this

# ─── Router ───────────────────────────────────────────────────────────────────
router = APIRouter(tags=["agents"])

# ─── Pydantic models ──────────────────────────────────────────────────────────
class WatchlistAddRequest(BaseModel):
    company_name: str
    sector:       str = "Other"
    alert_email:  str = ""
    notes:        str = ""

class ChatRequest(BaseModel):
    message:         str
    conversation_id: str = "default"
    api_key:         str = ""

class BatchRequest(BaseModel):
    companies:    List[str]
    email:        str = ""
    report_label: str = "Portfolio Analysis"

class ResearchRequest(BaseModel):
    company_name: str


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION 1 — PORTFOLIO MONITORING + EMAIL ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

def _load_watchlist() -> List[dict]:
    if WATCHLIST_PATH.exists():
        try:
            return json.loads(WATCHLIST_PATH.read_text())
        except Exception:
            return []
    return []

def _save_watchlist(wl: List[dict]) -> None:
    WATCHLIST_PATH.write_text(json.dumps(wl, indent=2))

def _load_scores() -> Dict[str, dict]:
    if SCORES_PATH.exists():
        try:
            return json.loads(SCORES_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_scores(scores: Dict[str, dict]) -> None:
    SCORES_PATH.write_text(json.dumps(scores, indent=2))


@router.post("/watchlist/add")
def watchlist_add(req: WatchlistAddRequest) -> dict:
    """Add a company to the monitoring watchlist."""
    wl = _load_watchlist()
    existing = [w for w in wl if w["company_name"].lower() == req.company_name.lower()]
    if existing:
        return {"status": "already_exists", "company": req.company_name}
    wl.append({
        "company_name": req.company_name,
        "sector":       req.sector,
        "alert_email":  req.alert_email,
        "notes":        req.notes,
        "added_at":     datetime.now(timezone.utc).isoformat(),
    })
    _save_watchlist(wl)
    return {"status": "added", "company": req.company_name, "total": len(wl)}


@router.get("/watchlist")
def watchlist_get() -> dict:
    """Return current watchlist with latest cached scores."""
    wl     = _load_watchlist()
    scores = _load_scores()
    for entry in wl:
        entry["last_score"] = scores.get(entry["company_name"], {})
    return {"watchlist": wl, "count": len(wl)}


@router.delete("/watchlist/{company_name}")
def watchlist_remove(company_name: str) -> dict:
    wl = _load_watchlist()
    before = len(wl)
    wl = [w for w in wl if w["company_name"].lower() != company_name.lower()]
    _save_watchlist(wl)
    return {"status": "removed" if len(wl) < before else "not_found", "remaining": len(wl)}


@router.post("/watchlist/monitor")
async def watchlist_monitor_now(background_tasks: BackgroundTasks) -> dict:
    """Trigger a manual monitoring run (also called by daily scheduler)."""
    wl = _load_watchlist()
    if not wl:
        return {"status": "empty_watchlist"}
    background_tasks.add_task(_run_monitoring_cycle, wl)
    return {"status": "monitoring_started", "companies": len(wl)}


async def _run_monitoring_cycle(wl: List[dict]) -> None:
    """Core monitoring loop — re-assess all watchlist companies, email alerts."""
    scores  = _load_scores()
    alerts  = []
    updated = {}

    async with httpx.AsyncClient(timeout=90.0) as client:
        for entry in wl:
            name = entry["company_name"]
            try:
                r = await client.post(f"{ENGINE_BASE}/agent/assess", json={
                    "company_name":    name,
                    "sector_hint":     (entry.get("sector", "other"))
                                       .lower().replace(" ", "_"),
                    "assessment_scope": "standard",
                })
                if not r.is_success:
                    continue
                data = r.json()

                new_composite = data.get("composite_score") or (
                    ((data.get("physical",  {}).get("score") or 0) +
                     (data.get("transition",{}).get("score") or 0)) / 2
                )
                new_composite = round(new_composite, 1)

                prev = scores.get(name, {})
                prev_composite = prev.get("composite_score", None)

                snapshot = {
                    "composite_score": new_composite,
                    "cri_rating":      data.get("cri_rating"),
                    "physical_score":  (data.get("physical",  {}) or {}).get("score"),
                    "transition_score":(data.get("transition",{}) or {}).get("score"),
                    "checked_at":      datetime.now(timezone.utc).isoformat(),
                }
                updated[name] = snapshot

                if prev_composite is not None:
                    delta = abs(new_composite - prev_composite)
                    if delta >= ALERT_THRESHOLD:
                        direction = "↑ increased" if new_composite > prev_composite else "↓ decreased"
                        alerts.append({
                            "company":     name,
                            "prev":        prev_composite,
                            "new":         new_composite,
                            "delta":       round(delta, 1),
                            "direction":   direction,
                            "rating":      data.get("cri_rating"),
                            "alert_email": entry.get("alert_email", ""),
                        })
            except Exception as exc:
                logger.warning("Monitoring failed for %s: %s", name, exc)

    # Merge updated scores
    scores.update(updated)
    _save_scores(scores)

    # Send email alerts
    for alert in alerts:
        _send_alert_email(alert)

    logger.info("Monitoring cycle complete: %d assessed, %d alerts", len(updated), len(alerts))


def _send_alert_email(alert: dict) -> None:
    """Send a climate risk alert email via Gmail SMTP."""
    if not (GMAIL_USER and GMAIL_APP_PASS):
        logger.warning("Email not configured — skipping alert for %s", alert["company"])
        return

    recipient = alert.get("alert_email") or GMAIL_USER
    subject   = (
        f"⚠️ ClimRisk Alert: {alert['company']} risk score "
        f"{alert['direction']} by {alert['delta']} pts"
    )
    body = f"""ClimRisk Portfolio Monitoring Alert
{'='*50}

Company:      {alert['company']}
Previous CRI: {alert['prev']}
New CRI:      {alert['new']}  ({alert['direction']} by {alert['delta']} points)
CRI Rating:   {alert.get('rating', 'N/A')}
Checked:      {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

This alert was triggered because the composite climate risk score
changed by {alert['delta']} points (threshold: {ALERT_THRESHOLD} pts).

Log in to your ClimRisk dashboard to view the full updated assessment:
https://climrisk.io/dashboard.html?auth=CRI2026

---
ClimRisk Intelligence | hello@climrisk.io | climrisk.io
To adjust alert thresholds, update your watchlist settings in the dashboard.
"""
    try:
        msg = MIMEMultipart()
        msg["From"]    = GMAIL_USER
        msg["To"]      = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_APP_PASS)
            smtp.sendmail(GMAIL_USER, recipient, msg.as_string())
        logger.info("Alert email sent to %s for %s", recipient, alert["company"])
    except Exception as exc:
        logger.error("Failed to send alert email: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION 2 — WEB RESEARCH AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/research")
async def research_company(req: ResearchRequest) -> dict:
    """Gather public data on a company: financials, news, ESG signals."""
    return await _research_company(req.company_name)


async def _research_company(company_name: str) -> dict:
    """Fetch yfinance data + recent news headlines for a company."""
    result: dict[str, Any] = {
        "company":   company_name,
        "financials": {},
        "news":       [],
        "esg_signals": [],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── yfinance via REST (no SDK needed — uses Yahoo Finance query API) ───
    ticker_guess = re.sub(r"[^A-Z]", "", company_name.upper())[:5]
    try:
        async with httpx.AsyncClient(timeout=15.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            # Yahoo Finance search
            search_r = await client.get(
                "https://query1.finance.yahoo.com/v1/finance/search",
                params={"q": company_name, "quotesCount": 1, "newsCount": 5},
            )
            if search_r.is_success:
                sdata = search_r.json()
                quotes = sdata.get("quotes", [])
                news   = sdata.get("news",   [])

                if quotes:
                    sym = quotes[0].get("symbol", "")
                    result["ticker"] = sym
                    result["exchange"] = quotes[0].get("exchange", "")

                    # Fetch summary detail
                    summary_r = await client.get(
                        f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{sym}",
                        params={"modules": "summaryProfile,financialData,esgScores"},
                    )
                    if summary_r.is_success:
                        qdata = summary_r.json().get("quoteSummary", {}).get("result", [{}])[0]

                        profile = qdata.get("summaryProfile", {})
                        fin     = qdata.get("financialData",  {})
                        esg     = qdata.get("esgScores",      {})

                        result["financials"] = {
                            "sector":       profile.get("sector"),
                            "industry":     profile.get("industry"),
                            "country":      profile.get("country"),
                            "employees":    profile.get("fullTimeEmployees"),
                            "revenue_bn":   round(fin.get("totalRevenue",  {}).get("raw", 0) / 1e9, 2),
                            "ebitda_bn":    round(fin.get("ebitda",        {}).get("raw", 0) / 1e9, 2),
                            "debt_to_eq":   fin.get("debtToEquity",       {}).get("raw"),
                        }
                        if esg:
                            result["esg_signals"] = {
                                "total_esg":       esg.get("totalEsg",       {}).get("raw"),
                                "environment":     esg.get("environmentScore",{}).get("raw"),
                                "social":          esg.get("socialScore",    {}).get("raw"),
                                "governance":      esg.get("governanceScore",{}).get("raw"),
                                "controversy":     esg.get("highestControversy"),
                                "esg_performance": esg.get("esgPerformance"),
                            }

                # News headlines
                result["news"] = [
                    {
                        "title":     n.get("title"),
                        "publisher": n.get("publisher"),
                        "link":      n.get("link"),
                        "published": datetime.fromtimestamp(
                            n.get("providerPublishTime", 0), tz=timezone.utc
                        ).strftime("%Y-%m-%d"),
                    }
                    for n in news[:5]
                ]

    except Exception as exc:
        logger.warning("Research fetch failed for %s: %s", company_name, exc)
        result["error"] = str(exc)

    return result


@router.post("/assess/augmented")
async def assess_augmented(company_name: str, sector_hint: str = "other") -> dict:
    """Research a company first, then run a climate assessment with richer context."""
    research = await _research_company(company_name)

    # Derive better sector from Yahoo Finance if available
    yf_sector = (research.get("financials", {}) or {}).get("sector", "")
    if yf_sector and sector_hint in ("other", ""):
        sector_hint = yf_sector.lower().replace(" ", "_")

    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(f"{ENGINE_BASE}/agent/assess", json={
            "company_name":     company_name,
            "sector_hint":      sector_hint,
            "assessment_scope": "standard",
        })
        if not r.is_success:
            raise HTTPException(r.status_code, "Engine assessment failed")
        assessment = r.json()

    return {
        "assessment": assessment,
        "research":   research,
        "augmented":  True,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION 3 — CONVERSATIONAL AI (Groq LLM, free tier)
# ═══════════════════════════════════════════════════════════════════════════════

# In-memory conversation store (keyed by conversation_id)
_conversations: Dict[str, List[dict]] = defaultdict(list)

SYSTEM_PROMPT = """You are ClimRisk AI, an expert climate financial risk analyst embedded in the ClimRisk Intelligence platform.

You assist institutional investors, asset managers, banks, and ESG analysts with:
- Climate risk assessments (physical + transition risk)
- NGFS scenario analysis (Net Zero 2050, Delayed Transition, Current Policies)
- Portfolio climate risk monitoring
- TCFD / ISSB S2 / CSRD disclosure support

When asked about a specific company, ALWAYS call assess_company to get real data — never make up scores.
When comparing companies, call compare_companies to run them in parallel.
When a user asks about their portfolio or watchlist, call get_watchlist.
When asked for news or recent developments, call research_company.

Be concise, professional, and quantitative. Cite specific scores and scenarios.
Format numbers as: "CRI Score: 62/100 | Rating: B+ | Physical: -8.3% NPV | Transition: -14.1% NPV"
"""

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "assess_company",
            "description": "Run a full climate risk assessment on any public company. Returns CRI rating, physical risk score, transition risk score, and NPV impacts under 3 NGFS scenarios.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "Full company name (e.g. 'Shell plc', 'Tesco', 'HSBC')"},
                    "sector_hint":  {"type": "string", "description": "Industry sector (optional): energy, financials, consumer_staples, etc."},
                },
                "required": ["company_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies",
            "description": "Compare climate risk profiles of 2–5 companies side by side. Runs parallel assessments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "companies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 2–5 company names",
                    },
                },
                "required": ["companies"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_watchlist",
            "description": "Return the current portfolio watchlist with latest cached risk scores.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research_company",
            "description": "Fetch recent news headlines, financial data, and ESG signals for a company from public sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"},
                },
                "required": ["company_name"],
            },
        },
    },
]


async def _groq_call(messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
    """Call Groq API (OpenAI-compatible)."""
    payload: dict[str, Any] = {
        "model":    GROQ_MODEL,
        "messages": messages,
        "max_tokens": 1024,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json=payload,
        )
        if not r.is_success:
            raise HTTPException(502, f"Groq API error {r.status_code}: {r.text[:200]}")
        return r.json()


async def _execute_tool(name: str, args: dict) -> Any:
    """Execute a tool called by the LLM."""
    if name == "assess_company":
        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(f"{ENGINE_BASE}/agent/assess", json={
                "company_name":     args["company_name"],
                "sector_hint":      args.get("sector_hint", "other"),
                "assessment_scope": "standard",
            })
            return r.json() if r.is_success else {"error": f"Engine {r.status_code}"}

    elif name == "compare_companies":
        results: dict[str, Any] = {}
        companies = args.get("companies", [])[:5]
        async with httpx.AsyncClient(timeout=90.0) as client:
            for company in companies:
                r = await client.post(f"{ENGINE_BASE}/agent/assess", json={
                    "company_name": company, "assessment_scope": "standard",
                })
                results[company] = r.json() if r.is_success else {"error": str(r.status_code)}
        return results

    elif name == "get_watchlist":
        wl     = _load_watchlist()
        scores = _load_scores()
        for entry in wl:
            entry["last_score"] = scores.get(entry["company_name"], {})
        return {"watchlist": wl}

    elif name == "research_company":
        return await _research_company(args.get("company_name", ""))

    return {"error": f"Unknown tool: {name}"}


@router.post("/chat")
async def agent_chat(req: ChatRequest) -> dict:
    """
    Conversational climate risk AI powered by Groq (llama-3.3-70b, free tier).
    Requires GROQ_API_KEY env var.
    """
    if not GROQ_API_KEY:
        raise HTTPException(503, detail={
            "error": "Conversational AI not configured",
            "hint":  "Set GROQ_API_KEY env var. Get a free key at console.groq.com"
        })

    conv_id  = req.conversation_id or "default"
    messages = _conversations[conv_id]

    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    messages.append({"role": "user", "content": req.message})

    # Agentic loop — up to 6 tool calls per turn
    for _ in range(6):
        response = await _groq_call(messages, tools=AGENT_TOOLS)
        choice   = response["choices"][0]
        msg      = choice["message"]

        # Strip None fields for cleanliness
        msg = {k: v for k, v in msg.items() if v is not None}
        messages.append(msg)

        if choice.get("finish_reason") == "tool_calls":
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                fn_name  = tc["function"]["name"]
                fn_args  = json.loads(tc["function"]["arguments"] or "{}")
                logger.info("Agent calling tool: %s(%s)", fn_name, list(fn_args.keys()))
                result = await _execute_tool(fn_name, fn_args)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "name":         fn_name,
                    "content":      json.dumps(result, default=str),
                })
        else:
            # Final text response
            _conversations[conv_id] = messages[-40:]  # keep last 40 messages
            return {
                "reply":           msg.get("content", ""),
                "conversation_id": conv_id,
                "model":           GROQ_MODEL,
            }

    return {
        "reply":           "I reached the analysis limit for this turn. Please try a simpler question or start a new conversation.",
        "conversation_id": conv_id,
    }


@router.delete("/chat/{conversation_id}")
def chat_reset(conversation_id: str) -> dict:
    """Clear a conversation's history."""
    _conversations.pop(conversation_id, None)
    return {"status": "cleared", "conversation_id": conversation_id}


# ═══════════════════════════════════════════════════════════════════════════════
#  OPTION 4 — AUTONOMOUS BATCH PORTFOLIO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/portfolio/batch")
async def portfolio_batch(req: BatchRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Autonomously assess every company in the list, then email a summary pack.
    Runs in the background — returns immediately with a job ID.
    """
    if not req.companies:
        raise HTTPException(400, "No companies provided")

    job_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    background_tasks.add_task(
        _run_batch_analysis, job_id, req.companies, req.email, req.report_label
    )
    return {
        "status":    "started",
        "job_id":    job_id,
        "companies": len(req.companies),
        "message":   f"Assessing {len(req.companies)} companies autonomously. "
                     f"{'Results will be emailed to ' + req.email if req.email else 'Results saved to watchlist scores.'}",
    }


async def _run_batch_analysis(
    job_id: str, companies: List[str], email: str, label: str
) -> None:
    """Background task: assess all companies, compile summary, email results."""
    results    = []
    errors     = []
    start_time = datetime.now(timezone.utc)

    async with httpx.AsyncClient(timeout=90.0) as client:
        for company in companies:
            try:
                r = await client.post(f"{ENGINE_BASE}/agent/assess", json={
                    "company_name":     company,
                    "assessment_scope": "standard",
                })
                if r.is_success:
                    data = r.json()
                    results.append({
                        "company":          company,
                        "cri_rating":       data.get("cri_rating"),
                        "cri_rating_label": data.get("cri_rating_label"),
                        "composite_score":  data.get("composite_score"),
                        "physical_score":   (data.get("physical",   {}) or {}).get("score"),
                        "transition_score": (data.get("transition", {}) or {}).get("score"),
                        "summary":          data.get("summary", "")[:300],
                        "status":           "success",
                    })
                else:
                    errors.append({"company": company, "error": f"HTTP {r.status_code}"})
            except Exception as exc:
                errors.append({"company": company, "error": str(exc)})

    duration  = (datetime.now(timezone.utc) - start_time).seconds
    # Sort by composite score descending (highest risk first)
    results.sort(key=lambda x: x.get("composite_score") or 0, reverse=True)

    if email:
        _send_batch_email(job_id, label, results, errors, duration, email)

    logger.info("Batch %s complete: %d ok, %d errors, %ds", job_id, len(results), len(errors), duration)


def _send_batch_email(
    job_id: str, label: str, results: List[dict], errors: List[dict],
    duration: int, recipient: str
) -> None:
    """Email the batch analysis summary as a plain-text report."""
    if not (GMAIL_USER and GMAIL_APP_PASS):
        logger.warning("Email not configured — batch results not emailed")
        return

    lines = [
        f"ClimRisk Intelligence — {label}",
        f"Job ID: {job_id}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Duration: {duration}s  |  Assessed: {len(results)}  |  Errors: {len(errors)}",
        "",
        "=" * 70,
        f"{'COMPANY':<30} {'RATING':<8} {'SCORE':>6} {'PHYS':>6} {'TRANS':>6}",
        "=" * 70,
    ]
    for r in results:
        lines.append(
            f"{r['company']:<30} "
            f"{(r.get('cri_rating') or 'N/A'):<8} "
            f"{str(r.get('composite_score') or '')[:5]:>6} "
            f"{str(r.get('physical_score') or '')[:5]:>6} "
            f"{str(r.get('transition_score') or '')[:5]:>6}"
        )

    if errors:
        lines += ["", "ERRORS:", *[f"  {e['company']}: {e['error']}" for e in errors]]

    lines += [
        "",
        "=" * 70,
        "Risk ranking: higher scores indicate greater climate financial risk exposure.",
        "Log in to view full scenario projections: https://climrisk.io/dashboard.html?auth=CRI2026",
        "",
        "ClimRisk Intelligence | hello@climrisk.io | climrisk.io",
    ]

    body = "\n".join(lines)

    try:
        msg = MIMEMultipart()
        msg["From"]    = GMAIL_USER
        msg["To"]      = recipient
        msg["Subject"] = f"ClimRisk Batch Analysis — {label} ({len(results)} companies)"
        msg.attach(MIMEText(body, "plain"))

        # Attach JSON results for programmatic use
        attachment = MIMEBase("application", "octet-stream")
        attachment.set_payload(json.dumps(results, indent=2).encode())
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", f"attachment; filename={job_id}_results.json")
        msg.attach(attachment)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_APP_PASS)
            smtp.sendmail(GMAIL_USER, recipient, msg.as_string())
        logger.info("Batch email sent to %s", recipient)
    except Exception as exc:
        logger.error("Failed to send batch email: %s", exc)
