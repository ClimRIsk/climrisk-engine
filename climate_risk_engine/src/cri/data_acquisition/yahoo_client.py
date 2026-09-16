"""
yahoo_client.py — Market data via Yahoo Finance (yfinance).

Fetches for listed companies:
  • Market capitalisation (USD M)
  • Enterprise value (USD M)
  • Revenue TTM (USD M) — backup to EDGAR
  • EBITDA (USD M)
  • Total debt (USD M)
  • Beta

Dependency: yfinance (pip install yfinance).
Falls back gracefully if the package is not installed or ticker is unknown.

yfinance uses the Yahoo Finance API which is public (unofficial).
No API key required, but heavy use may be rate-limited.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .provenance import ConfidenceTier, DataGap, ProvenanceField, today_iso

logger = logging.getLogger(__name__)


@dataclass
class YahooResult:
    ticker:           Optional[str]             = None
    market_cap_usd_m: Optional[ProvenanceField] = None
    ev_usd_m:         Optional[ProvenanceField] = None
    revenue_usd_m:    Optional[ProvenanceField] = None
    ebitda_usd_m:     Optional[ProvenanceField] = None
    total_debt_usd_m: Optional[ProvenanceField] = None
    beta:             Optional[ProvenanceField] = None
    data_gaps:        list[DataGap]            = field(default_factory=list)

    def to_dict(self) -> dict:
        def _pf(f):
            return f.to_dict() if f else None
        return {
            "ticker":           self.ticker,
            "market_cap_usd_m": _pf(self.market_cap_usd_m),
            "ev_usd_m":         _pf(self.ev_usd_m),
            "revenue_usd_m":    _pf(self.revenue_usd_m),
            "ebitda_usd_m":     _pf(self.ebitda_usd_m),
            "total_debt_usd_m": _pf(self.total_debt_usd_m),
            "beta":             _pf(self.beta),
            "data_gaps":        [
                {"field": g.field_name, "reason": g.reason, "action": g.suggested_action}
                for g in self.data_gaps
            ],
        }


def _m(raw: Optional[float]) -> Optional[float]:
    """Convert raw USD to USD millions; None passthrough."""
    if raw is None:
        return None
    return round(raw / 1_000_000, 2)


def _pf_from(value, label: str, ticker: str, notes: str = "") -> Optional[ProvenanceField]:
    if value is None:
        return None
    return ProvenanceField(
        value=value,
        source=f"Yahoo Finance ({ticker})",
        url=f"https://finance.yahoo.com/quote/{ticker}",
        retrieval_date=today_iso(),
        confidence_tier=ConfidenceTier.REPORTED,
        notes=notes or "Market data as of retrieval date",
    )


async def fetch_yahoo_data(
    ticker: str,
) -> YahooResult:
    """
    Fetch market data from Yahoo Finance for a given ticker.

    Parameters
    ----------
    ticker : e.g. "XOM", "SHEL", "BPAQF"
    """
    result = YahooResult(ticker=ticker.upper())

    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        result.data_gaps.append(DataGap(
            field_name="market_data",
            reason="yfinance not installed (pip install yfinance)",
            impact="No market cap / EV data from Yahoo Finance",
            suggested_action="Install yfinance or provide market cap and EV manually",
        ))
        return result

    try:
        import asyncio
        # yfinance is synchronous — run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: yf.Ticker(ticker).info)

        if not info or info.get("regularMarketPrice") is None:
            result.data_gaps.append(DataGap(
                field_name="ticker",
                reason=f"Yahoo Finance returned no data for ticker '{ticker}'",
                impact="Market data unavailable",
                suggested_action=(
                    "Verify ticker symbol is correct and company is listed on a major exchange"
                ),
            ))
            return result

        mc  = _m(info.get("marketCap"))
        ev  = _m(info.get("enterpriseValue"))
        rev = _m(info.get("totalRevenue"))
        ebi = _m(info.get("ebitda"))
        dbt = _m(info.get("totalDebt"))
        bet = info.get("beta")

        result.market_cap_usd_m = _pf_from(mc,  "Market Cap (USD M)",    ticker)
        result.ev_usd_m         = _pf_from(ev,  "Enterprise Value (USD M)", ticker)
        result.revenue_usd_m    = _pf_from(rev, "Revenue TTM (USD M)",    ticker, "TTM = trailing 12 months")
        result.ebitda_usd_m     = _pf_from(ebi, "EBITDA TTM (USD M)",     ticker, "TTM = trailing 12 months")
        result.total_debt_usd_m = _pf_from(dbt, "Total Debt (USD M)",     ticker)
        result.beta             = _pf_from(bet, "Beta",                   ticker, "5-year monthly vs S&P500")

        # Record gaps for None fields
        for field_name, val in [
            ("market_cap_usd_m", mc),
            ("ev_usd_m",         ev),
            ("revenue_usd_m",    rev),
        ]:
            if val is None:
                result.data_gaps.append(DataGap(
                    field_name=field_name,
                    reason=f"Yahoo Finance returned None for {field_name} on {ticker}",
                    impact="Financial metric unavailable from market data",
                    suggested_action="Provide value manually or check ticker symbol",
                ))

    except Exception as exc:
        logger.warning("Yahoo Finance fetch failed for '%s': %s", ticker, exc)
        result.data_gaps.append(DataGap(
            field_name="market_data",
            reason=str(exc),
            impact="All Yahoo Finance data unavailable",
            suggested_action="Check ticker or provide financial data manually",
        ))

    return result


async def find_ticker_from_name(company_name: str) -> Optional[str]:
    """
    Best-effort ticker lookup from company name via yfinance search.
    Returns None if not found.
    """
    try:
        import yfinance as yf
        import asyncio
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: yf.Search(company_name, max_results=3).quotes
        )
        if results:
            return results[0].get("symbol")
    except Exception:
        pass
    return None
