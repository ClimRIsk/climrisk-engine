"""HuggingFace ClimateRiskSense AI — NLP disclosure document analyzer.
══════════════════════════════════════════════════════════════════════════════
Source: MANSOORahme67/climaterisksense-ai (HuggingFace Spaces, Running)
  https://huggingface.co/spaces/MANSOORahme67/climaterisksense-ai

What it provides
────────────────
  • Climate event detection in financial text (flood, wildfire, drought, etc.)
  • Named entity recognition (companies, locations, assets mentioned)
  • Relation extraction (company → affected by → hazard)
  • Sentiment analysis (risk-negative vs. opportunity-positive framing)

Use case in ClimRisk
────────────────────
  • CSRD / TCFD disclosure analysis tab — upload a company annual report,
    get structured climate risk events extracted automatically.
  • Screening uploaded PDFs for physical and transition risk signals before
    running the full engine simulation.

Usage
─────
    from cri.connectors.hf_nlp_analyzer import ClimateNLPAnalyzer

    analyzer = ClimateNLPAnalyzer()

    result = analyzer.analyze_text(
        text = "ExxonMobil expects rising sea levels to increase flooding risk
                at its Gulf Coast refinery assets by 2035 ..."
    )
    # → {
    #     'events':    [{'type': 'flooding', 'confidence': 0.91, ...}],
    #     'entities':  [{'name': 'ExxonMobil', 'type': 'company'}, ...],
    #     'sentiment': {'label': 'risk', 'score': 0.78},
    #     'risk_signals': ['physical_risk', 'asset_exposure'],
    #     'source': 'hf_climaterisksense'
    #   }

    # One-liner: just get a risk signal summary
    signals = analyzer.extract_risk_signals(text)
    # → ['physical_risk', 'transition_risk', 'regulatory']
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_HF_SPACE_URL  = "https://MANSOORahme67-climaterisksense-ai.hf.space"
_API_ENDPOINT  = f"{_HF_SPACE_URL}/api/predict"
_TIMEOUT_S     = 60
_MAX_RETRIES   = 3
_RETRY_DELAY_S = 20

# Keyword-based fallback classifier (runs when Space is unavailable)
_PHYSICAL_KEYWORDS = [
    "flood", "wildfire", "hurricane", "drought", "heat stress",
    "sea level", "storm surge", "extreme weather", "precipitation",
    "temperature increase", "water scarcity", "coastal erosion",
    "permafrost", "tropical cyclone", "heatwave",
]
_TRANSITION_KEYWORDS = [
    "carbon price", "carbon tax", "emission", "stranded asset",
    "regulatory", "low-carbon", "net zero", "decarbonisation", "CBAM",
    "cap and trade", "renewable energy", "fossil fuel", "NGFS",
    "transition pathway", "scope 1", "scope 2", "scope 3",
]
_OPPORTUNITY_KEYWORDS = [
    "green bond", "clean energy", "energy efficiency", "climate solution",
    "sustainable", "ESG", "adaptation", "resilience", "electrification",
]


def _keyword_classify(text: str) -> Dict[str, Any]:
    """Fallback when the HF Space is unavailable — keyword-based heuristic."""
    lower = text.lower()
    phy_hits  = [k for k in _PHYSICAL_KEYWORDS  if k in lower]
    tra_hits  = [k for k in _TRANSITION_KEYWORDS if k in lower]
    opp_hits  = [k for k in _OPPORTUNITY_KEYWORDS if k in lower]

    signals: List[str] = []
    if phy_hits:  signals.append("physical_risk")
    if tra_hits:  signals.append("transition_risk")
    if opp_hits:  signals.append("green_opportunity")
    if not signals: signals.append("no_signal_detected")

    risk_count = len(phy_hits) + len(tra_hits)
    opp_count  = len(opp_hits)
    sentiment_label = "risk" if risk_count >= opp_count else "opportunity"
    sentiment_score = min(1.0, max(0.0, (risk_count - opp_count) / max(1, risk_count + opp_count + 1) + 0.5))

    events = []
    for k in phy_hits[:5]:
        events.append({"type": k.replace(" ", "_"), "confidence": 0.70, "source": "keyword"})
    for k in tra_hits[:5]:
        events.append({"type": k.replace(" ", "_"), "confidence": 0.65, "source": "keyword"})

    return {
        "events":       events,
        "entities":     [],
        "sentiment":    {"label": sentiment_label, "score": round(sentiment_score, 2)},
        "risk_signals": signals,
        "source":       "hf_climaterisksense_fallback",
        "method":       "keyword_classifier",
    }


class ClimateNLPAnalyzer:
    """NLP client for the MANSOORahme67/climaterisksense-ai HuggingFace Space."""

    def __init__(self, base_url: str = _HF_SPACE_URL, timeout: int = _TIMEOUT_S):
        self._base     = base_url.rstrip("/")
        self._endpoint = f"{self._base}/api/predict"
        self._timeout  = timeout

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_text(self, text: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Analyze a climate disclosure text passage.

        Truncates to max_chars to avoid Gradio payload limits.

        Returns dict with:
            events       list of {type, confidence, text_span}
            entities     list of {name, type}  (company/location/asset)
            sentiment    {label: risk|opportunity|neutral, score: 0-1}
            risk_signals list[str]  from: physical_risk, transition_risk,
                                          regulatory, green_opportunity, no_signal_detected
            source       str       'hf_climaterisksense' or '_fallback'
        """
        excerpt = text[:max_chars]
        payload = {"data": [excerpt]}
        raw     = self._call(payload)
        if raw is None:
            return _keyword_classify(text)
        return self._parse(raw, text)

    def extract_risk_signals(self, text: str) -> List[str]:
        """Quick extraction — just returns the risk_signals list."""
        return self.analyze_text(text).get("risk_signals", ["no_signal_detected"])

    def sentiment(self, text: str) -> Dict[str, Any]:
        """Quick sentiment call — returns {label, score}."""
        return self.analyze_text(text).get("sentiment", {"label": "neutral", "score": 0.5})

    def batch_analyze(self, passages: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple passages, each independently."""
        return [self.analyze_text(p) for p in passages]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call(self, payload: Dict) -> Optional[Dict]:
        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            self._endpoint,
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (503, 502) and attempt < _MAX_RETRIES:
                    log.info("ClimateRiskSense sleeping (HTTP %s), waiting %ss …", e.code, _RETRY_DELAY_S)
                    time.sleep(_RETRY_DELAY_S)
                else:
                    log.warning("ClimateRiskSense HTTP %s — falling back to keyword classifier.", e.code)
                    return None
            except Exception as exc:
                log.warning("ClimateRiskSense request error: %s — using keyword fallback.", exc)
                return None
        return None

    @staticmethod
    def _parse(raw: Dict, original_text: str) -> Dict[str, Any]:
        """Extract structured fields from the Space's raw API response."""
        try:
            data = raw.get("data", [raw])
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    pass
            if isinstance(data, dict):
                return {
                    "events":       data.get("events", []),
                    "entities":     data.get("entities", []),
                    "sentiment":    data.get("sentiment", {"label": "neutral", "score": 0.5}),
                    "risk_signals": data.get("risk_signals", []),
                    "source":       "hf_climaterisksense",
                }
        except Exception as exc:
            log.debug("Could not parse ClimateRiskSense response: %s", exc)

        # Graceful fallback
        return _keyword_classify(original_text)


# ── Module-level singleton ────────────────────────────────────────────────────
_analyzer: Optional[ClimateNLPAnalyzer] = None


def get_analyzer() -> ClimateNLPAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ClimateNLPAnalyzer()
    return _analyzer
