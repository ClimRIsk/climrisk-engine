"""
CRI Client PDF Report Generator
────────────────────────────────────────────────────────────────────────────────
Generates a 4-page branded PDF climate risk assessment report for a company,
suitable for delivery to an investment committee, lender, or regulator.

Pages
─────
1. Executive Summary  — CRI rating, traffic-light scores, key findings
2. Scenario Trajectory — NZE / Delayed / Current Policies score curve (text table)
3. Emissions & Disclosure — Scope 1/2/3 estimates, commitment score, greenwashing flag
4. Methodology & Data Sources

Uses reportlab only (already in requirements.txt for Render).
No torch, no transformers, no external network calls.
"""

from __future__ import annotations

import io
import datetime
from typing import Optional


# ── Graceful import ───────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


# ── Brand palette ─────────────────────────────────────────────────────────────
_NAVY    = colors.HexColor("#0D2340")
_TEAL    = colors.HexColor("#1B9B8E")
_AMBER   = colors.HexColor("#F59E0B")
_RED     = colors.HexColor("#DC2626")
_GREEN   = colors.HexColor("#16A34A")
_LIGHT   = colors.HexColor("#F8FAFC")
_MID     = colors.HexColor("#E2E8F0")
_DARK    = colors.HexColor("#1E293B")
_WHITE   = colors.white

# Rating → colour
_RATING_COLOR = {
    "A": _GREEN, "A+": _GREEN, "A-": _GREEN,
    "B": _TEAL,  "B+": _TEAL, "B-": _TEAL,
    "C": _AMBER, "C+": _AMBER, "C-": _AMBER,
    "D": _RED,   "D+": _RED,  "D-": _RED,
    "E": _RED,
}


def _rating_color(rating: str) -> object:
    return _RATING_COLOR.get(str(rating).upper(), _AMBER)


def generate_client_pdf(
    company_name: str,
    sector: str,
    rating: str,
    rating_label: str,
    summary: str,
    physical_label: str,
    transition_label: str,
    financial_label: str,
    # Trajectory data (years 2025–2050, 26 points)
    nze_scores:     Optional[list[float]] = None,
    delayed_scores: Optional[list[float]] = None,
    cp_scores:      Optional[list[float]] = None,
    # Emissions
    scope1_mt: Optional[float] = None,
    scope2_mt: Optional[float] = None,
    scope3_mt: Optional[float] = None,
    # Disclosure
    commitment_score: Optional[float] = None,
    commitment_specificity: Optional[float] = None,
    credibility_gap: bool = False,
    itr_adjustment: Optional[float] = None,
    evidence_quotes: Optional[list[dict]] = None,
    # Metadata
    reporting_year: int = 2025,
    analyst_name: str = "ClimRisk Intelligence",
) -> bytes:
    """Return a PDF as raw bytes.

    Args
    ────
    All score/label fields come directly from the /ratings and /runs API responses.
    nze_scores / delayed_scores / cp_scores — lists of 26 yearly CRI scores (2025–2050).
    evidence_quotes — list of {"text": str, "quote_type": str} dicts from DisclosureResult.

    Returns
    ───────
    Raw PDF bytes — caller writes to file or streams as HTTP response.
    """
    if not _HAS_REPORTLAB:
        raise RuntimeError(
            "reportlab is not installed. Add 'reportlab>=4.1' to requirements.txt."
        )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"CRI Report — {company_name}",
        author="ClimRisk Intelligence",
        subject="Climate Financial Risk Assessment",
    )

    styles = getSampleStyleSheet()

    def _style(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=styles[parent], **kw)

    H1 = _style("H1", fontSize=20, textColor=_NAVY, spaceAfter=4, leading=24, fontName="Helvetica-Bold")
    H2 = _style("H2", fontSize=13, textColor=_NAVY, spaceAfter=3, spaceBefore=10, leading=16, fontName="Helvetica-Bold")
    H3 = _style("H3", fontSize=10, textColor=_DARK, spaceAfter=2, spaceBefore=6, leading=13, fontName="Helvetica-Bold")
    BODY = _style("BODY", fontSize=9, textColor=_DARK, leading=13, spaceAfter=4)
    SMALL = _style("SMALL", fontSize=7.5, textColor=colors.HexColor("#64748B"), leading=11)
    CAPTION = _style("CAPTION", fontSize=8, textColor=colors.HexColor("#64748B"), leading=11, italics=1)
    QUOTE = _style("QUOTE", fontSize=8.5, textColor=_DARK, leading=12, leftIndent=8, borderPad=4,
                   borderColor=_TEAL, borderWidth=1.5, borderRadius=2)

    story = []

    # ─── Page 1: Executive Summary ────────────────────────────────────────────

    # Header banner (simulated with a wide table)
    banner_data = [[
        Paragraph(f"<b>Climate Risk Intelligence</b><br/><font size=8 color='#94A3B8'>Confidential Assessment Report</font>", H2),
        Paragraph(f"<font size=8 color='#94A3B8'>Generated {datetime.date.today().strftime('%d %b %Y')} · {reporting_year} Reporting Year</font>",
                  _style("RIGHT", fontSize=8, textColor=colors.HexColor("#94A3B8"), alignment=TA_RIGHT)),
    ]]
    banner = Table(banner_data, colWidths=[120 * mm, 55 * mm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, -1), _WHITE),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (0, -1), 10),
        ("RIGHTPADDING",  (-1, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(banner)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph(company_name, H1))
    story.append(Paragraph(f"{sector} · Climate Financial Risk Assessment", BODY))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_TEAL, spaceAfter=6))

    # Rating + pillar card
    rc = _rating_color(rating)
    rating_data = [
        [
            Paragraph(f"<b>CRI Rating</b>", _style("RH", fontSize=9, textColor=_WHITE, fontName="Helvetica-Bold")),
            Paragraph(f"<b>Physical Risk</b>", _style("PH", fontSize=9, textColor=_WHITE, fontName="Helvetica-Bold")),
            Paragraph(f"<b>Transition Risk</b>", _style("TH", fontSize=9, textColor=_WHITE, fontName="Helvetica-Bold")),
            Paragraph(f"<b>Financial Impact</b>", _style("FH", fontSize=9, textColor=_WHITE, fontName="Helvetica-Bold")),
        ],
        [
            Paragraph(f"<b>{rating}</b>", _style("RV", fontSize=36, textColor=_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)),
            Paragraph(physical_label, _style("PV", fontSize=11, textColor=_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)),
            Paragraph(transition_label, _style("TV", fontSize=11, textColor=_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)),
            Paragraph(financial_label, _style("FV", fontSize=11, textColor=_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)),
        ],
        [
            Paragraph(rating_label, _style("RL", fontSize=7, textColor=_WHITE, alignment=TA_CENTER)),
            Paragraph("", SMALL),
            Paragraph("", SMALL),
            Paragraph("", SMALL),
        ],
    ]
    colw = [38 * mm, 48 * mm, 48 * mm, 41 * mm]
    rtbl = Table(rating_data, colWidths=colw, rowHeights=[8 * mm, 16 * mm, 6 * mm])
    rtbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), rc),
        ("BACKGROUND", (1, 0), (-1, -1), _NAVY),
        ("TEXTCOLOR",  (0, 0), (-1, -1), _WHITE),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEAFTER",  (0, 0), (-2, -1), 0.5, colors.HexColor("#334155")),
        ("ROUNDEDCORNERS", [3]),
    ]))
    story.append(rtbl)
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Key Findings", H2))
    story.append(Paragraph(summary, BODY))
    story.append(Spacer(1, 4 * mm))

    # Evidence quotes (top 3 commitment/risk quotes)
    if evidence_quotes:
        story.append(Paragraph("Selected Disclosure Evidence", H3))
        for q in evidence_quotes[:3]:
            qtype = q.get("quote_type", "")
            icon = "⚠" if "risk" in qtype else "✓"
            story.append(Paragraph(f'{icon} "{q.get("text", "")}"', QUOTE))
            story.append(Spacer(1, 2 * mm))

    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "This report is produced by ClimRisk Intelligence using NGFS Phase 4 scenarios, "
        "IPCC AR6 physical hazard data, and proprietary ML-enhanced disclosure analysis. "
        "It does not constitute investment advice.",
        CAPTION,
    ))

    story.append(PageBreak())

    # ─── Page 2: Scenario Trajectory ─────────────────────────────────────────

    story.append(Paragraph("CRI Score Trajectory — 2025 to 2050", H2))
    story.append(Paragraph(
        "The table below shows the projected composite Climate Risk Index (CRI) score "
        "(0–100, higher = greater risk) under each NGFS Phase 4 scenario. Scores are "
        "computed via the GradientBoosting trajectory model trained on NGFS macro-financial "
        "pathways and CLIMADA sector loss data.",
        BODY,
    ))
    story.append(Spacer(1, 3 * mm))

    # Build trajectory table (every 5 years)
    milestones = [0, 5, 10, 15, 20, 25]  # index offsets from 2025
    milestone_years = [2025 + i for i in milestones]

    def _fmt_score(scores, idx):
        if scores and idx < len(scores):
            return f"{scores[idx]:.1f}"
        return "—"

    traj_header = ["Year", "NZE 2050", "Delayed Transition", "Current Policies"]
    traj_rows = [traj_header]
    for i, yr in zip(milestones, milestone_years):
        traj_rows.append([
            str(yr),
            _fmt_score(nze_scores, i),
            _fmt_score(delayed_scores, i),
            _fmt_score(cp_scores, i),
        ])

    traj_col_w = [25 * mm, 45 * mm, 55 * mm, 50 * mm]
    traj_tbl = Table(traj_rows, colWidths=traj_col_w)
    traj_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT, _WHITE]),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",        (0, 0), (-1, -1), 0.4, _MID),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(traj_tbl)
    story.append(Spacer(1, 4 * mm))

    # Divergence narrative
    if nze_scores and cp_scores:
        divergence_2050 = abs(cp_scores[-1] - nze_scores[-1])
        story.append(Paragraph(
            f"<b>Scenario divergence by 2050:</b> {divergence_2050:.1f} CRI points "
            f"separate the Net Zero (NZE) and Current Policies (CP) pathways. "
            f"This represents the maximum avoidable risk premium from timely decarbonisation "
            f"policy. Institutional lenders and equity investors are increasingly pricing "
            f"this spread into cost-of-capital and asset valuations.",
            BODY,
        ))

    story.append(PageBreak())

    # ─── Page 3: Emissions & Disclosure ──────────────────────────────────────

    story.append(Paragraph("Emissions Profile", H2))

    em_rows = [["Scope", "Estimate (Mt CO₂e/yr)", "Source"]]
    em_rows.append(["Scope 1 (direct)", f"{scope1_mt:.2f}" if scope1_mt else "—", "Revenue-based intensity imputation"])
    em_rows.append(["Scope 2 (energy)", f"{scope2_mt:.2f}" if scope2_mt else "—", "Grid emission factor × energy use"])
    em_rows.append(["Scope 3 (value chain)", f"{scope3_mt:.2f}" if scope3_mt else "—", "Sector supply-chain multiplier"])

    em_tbl = Table(em_rows, colWidths=[48 * mm, 55 * mm, 72 * mm])
    em_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), _NAVY),
        ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT, _WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.4, _MID),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(em_tbl)
    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph("Disclosure Quality Analysis", H2))
    if commitment_score is not None:
        cs_pct = round(commitment_score * 100)
        sp_pct = round((commitment_specificity or 0) * 100)
        gap_flag = "⚠ Credibility gap detected" if credibility_gap else "✓ No credibility gap"
        itr_txt = f"{itr_adjustment:+.2f}°C adjustment" if itr_adjustment is not None else "neutral"

        disc_rows = [
            ["Metric", "Value", "Interpretation"],
            ["Commitment Score", f"{cs_pct}%", "Share of climate-positive disclosure"],
            ["Commitment Specificity", f"{sp_pct}%", "% of commitments with quantified targets"],
            ["Credibility Gap", gap_flag, "High claims vs. low concrete action"],
            ["ITR Adjustment", itr_txt, "Implied Temperature Rise delta from disclosures"],
        ]
        disc_tbl = Table(disc_rows, colWidths=[55 * mm, 40 * mm, 80 * mm])
        disc_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), _TEAL),
            ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT, _WHITE]),
            ("GRID",        (0, 0), (-1, -1), 0.4, _MID),
            ("TOPPADDING",  (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(disc_tbl)

    story.append(PageBreak())

    # ─── Page 4: Methodology ─────────────────────────────────────────────────

    story.append(Paragraph("Methodology & Data Sources", H2))

    methodology_text = """
<b>Physical Risk.</b> Asset-level hazard profiles are computed from a five-layer data pipeline:
WRI Aqueduct 4.0 regional baselines → OpenTopoData elevation → NASA POWER climate normals →
CMIP6 / IPCC AR6 SSP projections → live event data (NASA FIRMS wildfires, GDACS floods/cyclones).
Physical loss costs are expressed in USD millions consistent with asset carrying values.

<b>Transition Risk.</b> Carbon cost trajectories follow NGFS Phase 4 (2023) scenario pathways
(Net Zero 2050, Delayed Transition, Current Policies). Commodity demand curves use IEA WEO 2023-aligned
data from Our World in Data. EBITDA compression is computed via a DCF working-capital model.

<b>ML Trajectory Model.</b> A GradientBoostingRegressor (sklearn) is trained on 150,000 synthetic
company-years generated from NGFS macro-financial parameters and CLIMADA v3 sector loss data.
Feature importance: base CRI score (91.5%), emission intensity (6.1%), carbon scenario (1.2%).

<b>Disclosure Intelligence.</b> Free text from ESG reports and annual filings is processed by four
ClimateBERT classifiers (Webersinke et al. 2021) or a keyword-based fallback: climate detection,
sentiment classification (risk/neutral/opportunity), net-zero/reduction detection, and specificity scoring.

<b>Emissions Estimation.</b> Scope 1/2/3 emissions are estimated via an XGBoost model trained on
OWID and WRI sector intensity benchmarks when company-reported figures are unavailable.
"""
    story.append(Paragraph(methodology_text, BODY))

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Primary Data Sources", H3))

    sources = [
        ["Source", "Coverage", "URL"],
        ["NGFS Phase 4 (2023)", "Climate scenarios & carbon pricing", "ngfs.net"],
        ["IPCC AR6 (2021)", "Warming trajectories & hazard projections", "ipcc.ch"],
        ["WRI Aqueduct 4.0", "Water stress & flood risk baselines", "aqueduct.wri.org"],
        ["EM-DAT (1990–2024)", "Historical disaster loss calibration", "emdat.be"],
        ["NASA NEX-GDDP", "Temperature & precipitation projections", "nccs.nasa.gov"],
        ["CLIMADA v3.3.2", "Sector damage functions", "github.com/CLIMADA-project"],
        ["ClimateBERT (2021)", "NLP disclosure classification", "huggingface.co/climatebert"],
    ]
    src_tbl = Table(sources, colWidths=[48 * mm, 70 * mm, 57 * mm])
    src_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), _DARK),
        ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT, _WHITE]),
        ("GRID",        (0, 0), (-1, -1), 0.3, _MID),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(src_tbl)

    story.append(Spacer(1, 6 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_MID))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"Report produced by {analyst_name} · CRI Engine v0.4.0 · "
        f"© ClimRisk Intelligence {datetime.date.today().year}. "
        "This document is confidential and intended solely for the named recipient. "
        "It does not constitute investment advice, an offer to sell, or a solicitation "
        "to purchase any financial instrument.",
        SMALL,
    ))

    doc.build(story)
    return buf.getvalue()
