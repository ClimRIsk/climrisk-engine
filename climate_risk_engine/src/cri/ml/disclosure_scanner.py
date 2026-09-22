"""
ClimateBERT Disclosure Intelligence Scanner
──────────────────────────────────────────────────────────────────────────────
Reads company ESG reports, sustainability disclosures, annual report sections,
or any free text and extracts structured climate risk intelligence using four
ClimateBERT classifiers from Hugging Face:

  climatebert/distilroberta-base-climate-detector      → is text climate-related?
  climatebert/distilroberta-base-climate-sentiment      → risk / neutral / opportunity
  climatebert/netzero-reduction-detector                → net-zero / reduction mention?
  climatebert/distilroberta-base-climate-specificity    → specific vs. vague claim?

Output
──────
DisclosureResult:
  evidence_quotes          List of scored, typed quotes for the dashboard
  commitment_score         0–1  (0 = no commitments, 1 = highly specific net-zero)
  commitment_specificity   0–1
  risk_mention_count       int
  opportunity_count        int
  itr_adjustment_c         float (degrees to ADD to ITR; negative = better)
  credibility_gap_flag     bool  (high claim specificity vs. low concrete action)
  ai_confidence            0–1
  model_used               str

References
──────────
Webersinke et al. 2021 — ClimateBERT: A Pretrained Language Model for
  Climate-Related Text (https://arxiv.org/abs/2110.12010)
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    from transformers import pipeline, Pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# ── Evidence quote type ───────────────────────────────────────────────────────

@dataclass
class EvidenceQuote:
    text: str                       # original sentence (truncated to 200 chars)
    quote_type: str                 # "physical_risk" | "commitment" | "opportunity" | "vague_claim"
    confidence: float               # 0–1
    sentiment: str                  # "risk" | "neutral" | "opportunity"
    is_specific: bool               # True if ClimateBERT specificity says "specific"
    is_netzero: bool                # True if net-zero / reduction mention detected


@dataclass
class DisclosureResult:
    evidence_quotes: list[EvidenceQuote]
    commitment_score: float                  # 0–1
    commitment_specificity: float            # 0–1
    risk_mention_count: int
    opportunity_count: int
    itr_adjustment_c: float                  # degrees to ADD to computed ITR (>0 = worse)
    credibility_gap_flag: bool               # high claims, low specificity
    ai_confidence: float                     # overall confidence in the analysis
    model_used: str                          # "climatebert" | "keyword_fallback"
    text_chars_scanned: int
    sentences_classified: int


# ── Keyword fallback (no transformers) ───────────────────────────────────────

_RISK_KEYWORDS = [
    "physical risk", "climate risk", "flood risk", "heat stress", "water scarcity",
    "sea level rise", "wildfire", "hurricane", "cyclone", "extreme weather",
    "stranded asset", "transition risk", "carbon price", "regulatory risk",
    "litigation risk", "supply chain disruption", "asset impairment",
]
_COMMITMENT_KEYWORDS = [
    "net zero", "net-zero", "carbon neutral", "paris agreement", "1.5°c",
    "scope 1", "scope 2", "scope 3", "science based target", "sbti",
    "carbon reduction", "decarbonisation", "decarbonization", "renewable energy",
    "emissions target", "ghg reduction",
]
_VAGUE_KEYWORDS = [
    "committed to", "we believe", "working towards", "aspire to", "aim to",
    "we intend", "continue to", "we support", "recognise", "recognize the importance",
]
_SPECIFIC_KEYWORDS = [
    "%", "by 2030", "by 2035", "by 2040", "by 2050", "mt co2", "tco2",
    "megawatt", "mw", "usd", "$", "€", "£", "billion", "million",
    "science based targets", "sbti validated", "third-party verified",
]


def _keyword_scan(sentences: list[str]) -> DisclosureResult:
    """Rule-based fallback when transformers is not available."""
    quotes: list[EvidenceQuote] = []
    risk_count = opp_count = commitment_count = specific_count = 0

    for sent in sentences:
        low = sent.lower()
        is_risk       = any(k in low for k in _RISK_KEYWORDS)
        is_commit     = any(k in low for k in _COMMITMENT_KEYWORDS)
        is_vague      = any(k in low for k in _VAGUE_KEYWORDS)
        is_specific   = any(k in low for k in _SPECIFIC_KEYWORDS)
        is_netzero    = any(k in low for k in ["net zero", "net-zero", "carbon neutral"])

        if not (is_risk or is_commit):
            continue

        if is_risk:
            risk_count += 1
            qtype = "physical_risk"
            sentiment = "risk"
        elif is_commit:
            commitment_count += 1
            qtype = "commitment" if is_specific else "vague_claim"
            sentiment = "neutral"
        else:
            continue

        if is_specific:
            specific_count += 1

        quotes.append(EvidenceQuote(
            text=textwrap.shorten(sent, width=200, placeholder="…"),
            quote_type=qtype,
            confidence=0.72 if is_specific else 0.55,
            sentiment=sentiment,
            is_specific=is_specific,
            is_netzero=is_netzero,
        ))

    total = risk_count + commitment_count
    commitment_score = min(1.0, commitment_count / max(total, 1)) if commitment_count else 0.0
    specificity     = specific_count / max(commitment_count, 1)
    credibility_gap = commitment_score > 0.4 and specificity < 0.3

    # ITR adjustment: many vague commitments with no specifics → worse
    itr_adj = 0.0
    if commitment_score > 0.3 and specificity > 0.5:
        itr_adj = -0.15          # specific commitments improve ITR
    elif commitment_score > 0.3 and specificity < 0.2:
        itr_adj = +0.10          # vague claims, penalise slightly

    return DisclosureResult(
        evidence_quotes=quotes[:20],
        commitment_score=round(commitment_score, 3),
        commitment_specificity=round(specificity, 3),
        risk_mention_count=risk_count,
        opportunity_count=opp_count,
        itr_adjustment_c=round(itr_adj, 3),
        credibility_gap_flag=credibility_gap,
        ai_confidence=0.60,
        model_used="keyword_fallback",
        text_chars_scanned=sum(len(s) for s in sentences),
        sentences_classified=len(sentences),
    )


# ── ClimateBERT scanner ───────────────────────────────────────────────────────

class DisclosureScanner:
    """
    Load ClimateBERT models once and reuse across calls.

    Usage
    ─────
    scanner = DisclosureScanner()

    # Feed any free text (ESG report, 10-K section, news article)
    result = scanner.scan(text, company_name="Shell PLC")
    """

    # HuggingFace model IDs
    _DETECTOR_MODEL    = "climatebert/distilroberta-base-climate-detector"
    _SENTIMENT_MODEL   = "climatebert/distilroberta-base-climate-sentiment"
    _NETZERO_MODEL     = "climatebert/netzero-reduction-detector"
    _SPECIFICITY_MODEL = "climatebert/distilroberta-base-climate-specificity"

    def __init__(self, device: int = -1):
        """
        Args:
            device: -1 for CPU (default); 0+ for GPU index.
                    CPU is fine for demo throughput (~0.5s per batch).
        """
        self._device = device
        self._detector:    Optional[Pipeline] = None
        self._sentiment:   Optional[Pipeline] = None
        self._netzero:     Optional[Pipeline] = None
        self._specificity: Optional[Pipeline] = None
        self._loaded = False

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return
        if not _HAS_TRANSFORMERS:
            self._loaded = True
            return
        try:
            self._detector    = pipeline("text-classification", model=self._DETECTOR_MODEL,    device=self._device)
            self._sentiment   = pipeline("text-classification", model=self._SENTIMENT_MODEL,   device=self._device)
            self._netzero     = pipeline("text-classification", model=self._NETZERO_MODEL,     device=self._device)
            self._specificity = pipeline("text-classification", model=self._SPECIFICITY_MODEL, device=self._device)
        except Exception:
            # Model download failed (no internet, etc.) — fall back to keyword mode
            self._detector = None
        self._loaded = True

    @property
    def _models_ready(self) -> bool:
        self._load()
        return self._detector is not None

    # ── Text splitting ─────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str, max_chars: int = 512) -> list[str]:
        """Split text into sentence-sized chunks ≤ max_chars for the model."""
        # Basic sentence splitter (no nltk dependency)
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks: list[str] = []
        buf = ""
        for s in raw:
            s = s.strip()
            if not s:
                continue
            if len(buf) + len(s) + 1 <= max_chars:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    chunks.append(buf)
                buf = s[:max_chars]
        if buf:
            chunks.append(buf)
        return chunks

    # ── Batch inference ────────────────────────────────────────────────────────

    def _run_batch(self, pipe: Pipeline, texts: list[str]) -> list[dict]:
        """Run pipeline in batches of 32 for memory efficiency."""
        results = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(pipe(batch, truncation=True, max_length=512))
        return results

    # ── Main scan ─────────────────────────────────────────────────────────────

    def scan(self, text: str, company_name: str = "") -> DisclosureResult:
        """
        Scan free text and return structured disclosure intelligence.

        Args:
            text:         Any company disclosure text (ESG report, 10-K, news)
            company_name: Optional — used for logging only

        Returns:
            DisclosureResult with evidence quotes and risk adjustments
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return self._empty_result()

        if not self._models_ready:
            # Keyword scan needs individual sentences, not 512-char chunks.
            # Re-split the raw text so mixed risk+commitment chunks don't
            # collapse to risk-only due to if/elif priority.
            fine_sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
            return _keyword_scan(fine_sents if fine_sents else sentences)

        # ── Step 1: Filter to climate-related sentences ───────────────────────
        det_results = self._run_batch(self._detector, sentences)
        climate_sents = [
            s for s, r in zip(sentences, det_results)
            if r["label"].lower() == "yes" and r["score"] >= 0.65
        ]
        if not climate_sents:
            # Model filtered everything; fall back to keyword scan on all
            return _keyword_scan(sentences)

        # ── Step 2: Sentiment (risk / neutral / opportunity) ──────────────────
        sent_results  = self._run_batch(self._sentiment,   climate_sents)
        nz_results    = self._run_batch(self._netzero,     climate_sents)
        spec_results  = self._run_batch(self._specificity, climate_sents)

        # ── Step 3: Build evidence quotes ─────────────────────────────────────
        quotes: list[EvidenceQuote] = []
        risk_count = opp_count = commit_count = specific_count = 0

        for sent, sr, nzr, specr in zip(climate_sents, sent_results, nz_results, spec_results):
            sentiment_label = sr["label"].lower()    # "risk" | "neutral" | "opportunity"
            is_netzero      = nzr["label"].lower() == "yes" and nzr["score"] >= 0.60
            is_specific     = specr["label"].lower() == "specific" and specr["score"] >= 0.60
            confidence      = round((sr["score"] + specr["score"]) / 2, 3)

            if sentiment_label == "risk":
                risk_count += 1
                qtype = "physical_risk"
            elif is_netzero or sentiment_label == "neutral":
                commit_count += 1
                qtype = "commitment" if is_specific else "vague_claim"
            else:
                opp_count += 1
                qtype = "opportunity"

            if is_specific:
                specific_count += 1

            quotes.append(EvidenceQuote(
                text=textwrap.shorten(sent, width=220, placeholder="…"),
                quote_type=qtype,
                confidence=confidence,
                sentiment=sentiment_label,
                is_specific=is_specific,
                is_netzero=is_netzero,
            ))

        # ── Step 4: Aggregate scores ──────────────────────────────────────────
        total = max(len(climate_sents), 1)
        commitment_score = round(commit_count / total, 3)
        specificity      = round(specific_count / max(commit_count, 1), 3)
        credibility_gap  = commitment_score > 0.35 and specificity < 0.25

        # ITR adjustment heuristic
        # High risk mentions + low commitment → worse ITR
        # High specific commitments → better ITR
        risk_ratio   = risk_count / total
        commit_ratio = commit_count / total
        itr_adj = 0.0
        if commit_ratio > 0.3 and specificity > 0.5:
            itr_adj = -0.20         # strong specific commitments
        elif commit_ratio > 0.2 and specificity > 0.3:
            itr_adj = -0.10         # moderate commitments
        elif risk_ratio > 0.4 and commit_ratio < 0.15:
            itr_adj = +0.15         # lots of risk disclosure, no commitments → worse
        elif credibility_gap:
            itr_adj = +0.08         # vague claims, penalise

        ai_confidence = round(
            sum(q.confidence for q in quotes) / max(len(quotes), 1), 3
        )

        return DisclosureResult(
            evidence_quotes=quotes[:25],        # top 25 for dashboard
            commitment_score=commitment_score,
            commitment_specificity=specificity,
            risk_mention_count=risk_count,
            opportunity_count=opp_count,
            itr_adjustment_c=round(itr_adj, 3),
            credibility_gap_flag=credibility_gap,
            ai_confidence=ai_confidence,
            model_used="climatebert",
            text_chars_scanned=len(text),
            sentences_classified=len(climate_sents),
        )

    @staticmethod
    def _empty_result() -> DisclosureResult:
        return DisclosureResult(
            evidence_quotes=[],
            commitment_score=0.0,
            commitment_specificity=0.0,
            risk_mention_count=0,
            opportunity_count=0,
            itr_adjustment_c=0.0,
            credibility_gap_flag=False,
            ai_confidence=0.0,
            model_used="none",
            text_chars_scanned=0,
            sentences_classified=0,
        )

    # ── Convenience: scan a list of documents ─────────────────────────────────

    def scan_documents(self, docs: list[str], company_name: str = "") -> DisclosureResult:
        """Merge multiple documents (e.g., 10-K + ESG report) into one scan."""
        combined = "\n\n".join(docs)
        return self.scan(combined, company_name=company_name)
