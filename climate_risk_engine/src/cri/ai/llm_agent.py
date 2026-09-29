"""
ai/llm_agent.py — ClimRisk LLM Orchestrator

A true reasoning agent that decides WHICH tools to call, calls them against
the live engine, then synthesises the results into a structured answer.

Architecture
────────────
  User message
       │
       ▼
  [Groq LLaMA-3.3-70b]  ← system prompt + tool schemas
       │  tool_calls=[...]
       ▼
  execute_tools(calls)  ← direct Python calls, no HTTP
       │  tool_results
       ▼
  [Groq LLaMA-3.3-70b]  ← synthesise final answer
       │
       ▼
  AgentResponse(narrative, tool_calls_made, raw_tool_results, tokens_used)

Fallback chain
──────────────
  Groq (fast, cheap)  →  Anthropic Claude Haiku  →  template string

Groq model selection
────────────────────
  CLIMRISK_GROQ_MODEL env var  (default: llama-3.3-70b-versatile)
  Other good options: llama-3.1-8b-instant (faster), mixtral-8x7b-32768

Tool call limit: MAX_TOOL_ROUNDS (default 3).  The agent can chain tools
across rounds — e.g. assess_company → analyze_disclosure → compare — up
to the limit.  After that it synthesises from whatever it has.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_GROQ_MODEL   = os.environ.get("CLIMRISK_GROQ_MODEL", "llama-3.3-70b-versatile")
_MAX_ROUNDS   = int(os.environ.get("CLIMRISK_AGENT_MAX_ROUNDS", "3"))
_MAX_TOKENS   = int(os.environ.get("CLIMRISK_AGENT_MAX_TOKENS", "1500"))


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are ClimRisk Intelligence — an institutional-grade climate financial risk \
analyst powered by the CRI Engine. You have access to tools that assess \
climate risk for any company, analyse climate disclosures, retrieve live \
hazard data, and explain NGFS scenarios.

BEHAVIOUR RULES:
1. Always call tools before answering factual questions about specific companies, \
   locations, tickers, or disclosure texts. Do not make up risk numbers.
2. You may call multiple tools in sequence — assess_company first, then \
   get_location_hazard for the HQ coordinates if relevant.
3. When comparing companies, always call compare_companies (one call) rather than \
   two separate assess_company calls.
4. After tool calls, synthesise a clear, quantitative answer. Lead with the \
   risk signal (LOW / MODERATE / HIGH / CRITICAL), then the headline numbers \
   (composite score, VaR, EBITDA impact), then the primary drivers.
5. Always cite which NGFS scenario a number comes from.
6. Keep the response concise — 150–300 words unless the user asks for detail.
7. If a tool returns an error, say so and explain what data would be needed.
8. Never invent carbon prices, loss figures, or ratings. Only use tool outputs.
9. For the financial climate risk of a specific plant/site (coordinates) or of a \
   company's physical assets — physical hazards, the carbon-pricing policies that \
   apply to it, NGFS transition projections and the loss distribution — call \
   run_risk_analyst. Report its loss figures as present values 2026–2050 in USD, \
   name the scenario, and give its problem → action pairs.
10. When the user message already contains computed Risk Analyst results, answer \
   from those numbers without calling tools, and say when a question goes beyond them.
"""


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class AgentResponse:
    narrative:         str                      # Final prose answer for the user
    tool_calls_made:   list[str]                # Names of tools actually called
    raw_tool_results:  dict[str, Any]           # Tool name → result dict
    tokens_used:       int                      # Total Groq tokens consumed
    model_used:        str                      # Which model produced the answer
    rounds:            int                      # How many tool-call rounds were used
    error:             Optional[str]   = None   # Set if the whole agent failed
    fallback_used:     bool            = False  # True if Groq was unavailable


# ── Groq client ───────────────────────────────────────────────────────────────
def _get_groq_client():
    """Return a Groq client or None if key not set / groq not installed."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.info("GROQ_API_KEY not set")
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        logger.warning("groq package not installed — pip install groq")
        return None


# ── Anthropic fallback ────────────────────────────────────────────────────────
def _groq_unavailable_fallback(user_message: str) -> AgentResponse:
    """
    When Groq is unavailable, try Anthropic Claude Haiku (narrator.py path).
    If that also fails, return a template string that at least explains what
    the user would need to do.
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        f"{user_message}\n\n"
                        "[Note: Tool calling is temporarily unavailable. "
                        "Answer from general knowledge where possible, and note "
                        "that specific numbers require the ClimRisk engine tools.]"
                    )
                }],
            )
            text = msg.content[0].text if msg.content else "Unable to generate response."
            return AgentResponse(
                narrative=text,
                tool_calls_made=[],
                raw_tool_results={},
                tokens_used=msg.usage.input_tokens + msg.usage.output_tokens,
                model_used="claude-haiku-4-5-20251001",
                rounds=0,
                fallback_used=True,
            )
        except Exception as exc:
            logger.warning("Anthropic fallback failed: %s", exc)

    # Last resort template
    return AgentResponse(
        narrative=(
            "The ClimRisk AI engine requires a GROQ_API_KEY to be configured on "
            "the server. The parametric assessment tools (physical risk, transition "
            "risk, Monte Carlo VaR) are all available — they just need an LLM to "
            "orchestrate them. Please contact your administrator to enable AI mode."
        ),
        tool_calls_made=[],
        raw_tool_results={},
        tokens_used=0,
        model_used="template",
        rounds=0,
        fallback_used=True,
        error="No LLM API key configured",
    )


# ── Core agent loop ───────────────────────────────────────────────────────────
def run_agent(
    user_message: str,
    context: Optional[dict] = None,
    max_rounds: int = _MAX_ROUNDS,
) -> AgentResponse:
    """
    Run the LLM orchestration loop.

    Parameters
    ----------
    user_message : str
        The natural-language query from the user.
    context : dict, optional
        Extra context to inject into the first message (e.g. current company
        being viewed in the dashboard: {"viewing": "Shell", "sector": "oil_gas"}).
    max_rounds : int
        Maximum tool-call rounds before forcing synthesis.

    Returns
    -------
    AgentResponse
    """
    from .tools import get_all_definitions, execute_tool

    client = _get_groq_client()
    if client is None:
        return _groq_unavailable_fallback(user_message)

    tool_definitions = get_all_definitions()

    # Build opening user message
    if context:
        ctx_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
        full_message = f"[Dashboard context]\n{ctx_lines}\n\nUser question: {user_message}"
    else:
        full_message = user_message

    messages = [{"role": "user", "content": full_message}]

    tool_calls_made: list[str]     = []
    raw_tool_results: dict         = {}
    total_tokens                    = 0
    rounds                          = 0

    # ── Agentic loop ──────────────────────────────────────────────────────────
    for round_num in range(max_rounds + 1):
        rounds = round_num

        try:
            response = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
                tools=tool_definitions if round_num < max_rounds else None,
                tool_choice="auto" if round_num < max_rounds else None,
                max_tokens=_MAX_TOKENS,
                temperature=0.1,   # Low temp — factual, deterministic outputs
            )
        except Exception as exc:
            logger.error("Groq API error on round %d: %s", round_num, exc)
            return _groq_unavailable_fallback(user_message)

        choice = response.choices[0]
        total_tokens += (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

        # Finished — no more tool calls
        if choice.finish_reason == "stop" or not getattr(choice.message, "tool_calls", None):
            narrative = choice.message.content or "No response generated."
            return AgentResponse(
                narrative=narrative,
                tool_calls_made=tool_calls_made,
                raw_tool_results=raw_tool_results,
                tokens_used=total_tokens,
                model_used=_GROQ_MODEL,
                rounds=rounds,
            )

        # ── Process tool calls ────────────────────────────────────────────────
        tool_calls = choice.message.tool_calls

        # Append the assistant message with tool_calls
        messages.append({
            "role": "assistant",
            "content": choice.message.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in tool_calls
            ]
        })

        # Execute each tool call
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            logger.info("[agent] round=%d calling tool=%s args=%s",
                        round_num, tool_name, list(tool_args.keys()))

            t0     = time.time()
            result = execute_tool(tool_name, tool_args)
            elapsed = round(time.time() - t0, 2)

            tool_calls_made.append(tool_name)
            raw_tool_results[tool_name] = result

            # Inject tool result back into message history
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result, default=str),
            })

            logger.info("[agent] tool=%s elapsed=%.2fs keys=%s",
                        tool_name, elapsed, list(result.keys()))

    # Reached max rounds — force final synthesis without tools
    try:
        messages.append({
            "role": "user",
            "content": (
                "You have reached the tool call limit. "
                "Please synthesise your findings into a final answer now."
            )
        })
        final = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
            max_tokens=_MAX_TOKENS,
            temperature=0.1,
        )
        total_tokens += (
            final.usage.prompt_tokens + final.usage.completion_tokens
        )
        narrative = final.choices[0].message.content or "Assessment complete."
    except Exception as exc:
        logger.error("Final synthesis failed: %s", exc)
        narrative = (
            f"Assessment complete. Tools called: {', '.join(tool_calls_made)}. "
            f"See raw_tool_results for the data."
        )

    return AgentResponse(
        narrative=narrative,
        tool_calls_made=tool_calls_made,
        raw_tool_results=raw_tool_results,
        tokens_used=total_tokens,
        model_used=_GROQ_MODEL,
        rounds=rounds,
    )


# ── Convenience: single-company decision brief ────────────────────────────────
def get_decision_brief(
    company_name: str,
    sector: str,
    jurisdiction: str,
    revenue_usd_m: float = 5000.0,
) -> AgentResponse:
    """
    Convenience wrapper that runs a focused single-company assessment and
    returns a board-ready decision brief narrative.
    """
    message = (
        f"Produce a concise board-level decision brief for {company_name}, "
        f"a {sector} company headquartered in {jurisdiction} with approximately "
        f"${revenue_usd_m:.0f}M annual revenue. "
        f"Include: risk signal, composite CRI score, top physical and transition "
        f"risk drivers, Monte Carlo VaR, and three specific recommended actions."
    )
    return run_agent(
        user_message=message,
        context={
            "company": company_name,
            "sector": sector,
            "jurisdiction": jurisdiction,
            "revenue_usd_m": revenue_usd_m,
            "mode": "decision_brief",
        }
    )
