"""
cri/story.py — the engine's story format: one document model, four renderers, and the chart helpers.

A story is what the ClimRisk research pieces look like: a kicker, a title, a standfirst, then sections that move from
what happened, to what the engine found, to what it means, to how it was measured. Every claim carries a source marker
that becomes a numbered reference list; every figure is drawn from engine numbers.

Model (plain dicts, JSON-safe)
  {kicker, title, standfirst, byline, date, brand{name, accent, logo_b64, footer, client},
   blocks:[{type: h2|h3|p|list|quote|table|figure|callout|facts, ...}], sources:{key: {title, url, note}}}

Inline markers
  **bold**            bold text
  [[S:key]]           citation: replaced by [n] in order of first use; key must exist in `sources`
Renderers: html(), markdown(), pdf(), linkedin().
"""
from __future__ import annotations

import base64
import html as _h
import io
import re
from datetime import date
from typing import Optional

PALETTE = ["#0ea5e9", "#f97316", "#22c55e", "#a855f7", "#ef4444", "#eab308", "#14b8a6", "#64748b"]
INK, MUTED, RULE = "#0f172a", "#64748b", "#e2e8f0"


# ── construction helpers ─────────────────────────────────────────────────────

class Story:
    def __init__(self, kicker: str, title: str, standfirst: str = "", byline: str = "ClimRisk engine", brand: Optional[dict] = None,
                 when: Optional[str] = None):
        self.d = {"kicker": kicker, "title": title, "standfirst": standfirst, "byline": byline, "date": when or date.today().isoformat(),
                  "brand": brand or {}, "blocks": [], "sources": {}}

    # blocks
    def h2(self, t): self.d["blocks"].append({"type": "h2", "text": t}); return self
    def h3(self, t): self.d["blocks"].append({"type": "h3", "text": t}); return self
    def p(self, t): self.d["blocks"].append({"type": "p", "text": t}); return self
    def ul(self, items): self.d["blocks"].append({"type": "list", "items": list(items)}); return self
    def quote(self, t, src=""): self.d["blocks"].append({"type": "quote", "text": t, "source": src}); return self
    def callout(self, t, tone="info"): self.d["blocks"].append({"type": "callout", "text": t, "tone": tone}); return self
    def table(self, head, rows, caption=""): self.d["blocks"].append({"type": "table", "head": head, "rows": rows, "caption": caption}); return self
    def facts(self, items): self.d["blocks"].append({"type": "facts", "items": [{"value": v, "label": l} for v, l in items]}); return self
    def figure(self, png: bytes, caption: str = "", alt: str = ""):
        self.d["blocks"].append({"type": "figure", "png_b64": base64.b64encode(png).decode(), "caption": caption, "alt": alt or caption}); return self
    def source(self, key, title, url="", note=""): self.d["sources"][key] = {"title": title, "url": url, "note": note}; return self
    def section(self, key: str, title: str): self.d["blocks"].append({"type": "h2", "text": title, "section": key}); return self
    def build(self) -> dict: return self.d


def pick_sections(story: dict, keep: Optional[list]) -> dict:
    """Keep only the named sections (blocks between h2 markers); None keeps all. The header and sources are never dropped."""
    if not keep:
        return story
    out, on, kept = [], True, set(keep)
    for b in story["blocks"]:
        if b["type"] == "h2":
            on = (b.get("section") in kept) or b.get("section") is None
        if on:
            out.append(b)
    s = dict(story); s["blocks"] = out
    return s


def sections_of(story: dict) -> list:
    return [{"key": b["section"], "title": b["text"]} for b in story["blocks"] if b["type"] == "h2" and b.get("section")]


# ── inline text ──────────────────────────────────────────────────────────────

_CITE = re.compile(r"\[\[S:([A-Za-z0-9_\-\.]+)\]\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _number_sources(story: dict) -> tuple:
    order = []
    def scan(t):
        for k in _CITE.findall(t or ""):
            if k in story["sources"] and k not in order:
                order.append(k)
    scan(story.get("standfirst", ""))
    for b in story["blocks"]:
        for f in ("text", "caption", "source"):
            scan(b.get(f, ""))
        for it in b.get("items", []) or []:
            scan(it if isinstance(it, str) else it.get("label", ""))
        for r in b.get("rows", []) or []:
            for c in r:
                scan(str(c))
    return {k: i + 1 for i, k in enumerate(order)}, order


def _inline(t: str, num: dict, mode: str) -> str:
    t = t or ""
    if mode == "html":
        t = _h.escape(t)
        t = _BOLD.sub(r"<strong>\1</strong>", t)
        return _CITE.sub(lambda m: f'<sup class="c"><a href="#s{num[m.group(1)]}">[{num[m.group(1)]}]</a></sup>' if m.group(1) in num else "", t)
    if mode == "md":
        return _CITE.sub(lambda m: f"[{num[m.group(1)]}]" if m.group(1) in num else "", t)
    t = _BOLD.sub(r"\1", t)                                    # plain
    return _CITE.sub(lambda m: f"[{num[m.group(1)]}]" if m.group(1) in num else "", t)


# ── HTML ─────────────────────────────────────────────────────────────────────

_CSS = """
:root{--accent:%(accent)s;--ink:#0f172a;--muted:#64748b;--rule:#e2e8f0}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:17px/1.65 Georgia,'Iowan Old Style','Times New Roman',serif}
.wrap{max-width:760px;margin:0 auto;padding:36px 22px 80px}
.top{display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--rule);padding-bottom:14px;margin-bottom:26px;font:600 13px/1.2 -apple-system,'Segoe UI',sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.top img{height:30px}.top b{color:var(--accent);letter-spacing:.2em;font-size:15px}
.kicker{font:700 12px/1 -apple-system,'Segoe UI',sans-serif;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:12px}
h1{font:800 38px/1.12 -apple-system,'Segoe UI',sans-serif;margin:0 0 14px;letter-spacing:-.01em}
.stand{font:400 20px/1.5 -apple-system,'Segoe UI',sans-serif;color:#334155;margin:0 0 16px}
.by{font:13px -apple-system,'Segoe UI',sans-serif;color:var(--muted);margin-bottom:26px}
h2{font:800 24px/1.25 -apple-system,'Segoe UI',sans-serif;margin:38px 0 10px}h3{font:700 18px -apple-system,'Segoe UI',sans-serif;margin:26px 0 6px}
p{margin:0 0 16px}ul{margin:0 0 16px 22px;padding:0}li{margin-bottom:6px}
sup.c a{color:var(--accent);text-decoration:none;font:600 11px -apple-system,sans-serif}
blockquote{margin:22px 0;padding:14px 18px;border-left:4px solid var(--accent);background:#f8fafc;font-size:16px}
blockquote .src{display:block;font:700 11px -apple-system,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}
figure{margin:26px 0}figure img{width:100%%;height:auto;border:1px solid var(--rule);border-radius:6px}
figcaption{font:italic 13px/1.5 Georgia,serif;color:var(--muted);margin-top:6px}
table{width:100%%;border-collapse:collapse;margin:20px 0;font:14px -apple-system,'Segoe UI',sans-serif}
th{text-align:left;border-bottom:2px solid var(--ink);padding:8px 10px;font-weight:700}td{padding:8px 10px;border-bottom:1px solid var(--rule)}
.cap{font:italic 13px Georgia,serif;color:var(--muted);margin:-12px 0 18px}
.call{margin:22px 0;padding:14px 18px;border-radius:8px;background:#f0f9ff;border:1px solid #bae6fd;font:15px/1.55 -apple-system,'Segoe UI',sans-serif}
.call.warn{background:#fffbeb;border-color:#fde68a}.call.bad{background:#fef2f2;border-color:#fecaca}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:22px 0}
.fact{border:1px solid var(--rule);border-radius:10px;padding:12px 14px}.fact b{display:block;font:800 26px/1.1 -apple-system,sans-serif;color:var(--accent)}
.fact span{font:12.5px/1.4 -apple-system,sans-serif;color:var(--muted)}
ol.src{font:13.5px/1.55 -apple-system,'Segoe UI',sans-serif;color:#334155;padding-left:20px}ol.src a{color:var(--accent)}
.foot{margin-top:46px;padding-top:16px;border-top:1px solid var(--rule);font:12.5px/1.6 -apple-system,sans-serif;color:var(--muted)}
@media print{.wrap{padding:0}h2{break-after:avoid}figure,table,blockquote,.fact{break-inside:avoid}}
"""


def html(story: dict) -> str:
    num, order = _number_sources(story)
    b = story.get("brand") or {}
    accent = b.get("accent") or "#0ea5e9"
    I = lambda t: _inline(t, num, "html")                      # noqa: E731
    out = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{_h.escape(story['title'])}</title><style>{_CSS % {'accent': accent}}</style></head><body><div class='wrap'>"]
    logo = f"<img alt='' src='data:image/png;base64,{b['logo_b64']}'>" if b.get("logo_b64") else f"<b>{_h.escape(b.get('name') or 'CLIMRISK')}</b>"
    out.append(f"<div class='top'>{logo}<span>{_h.escape(b.get('client') and 'Prepared for ' + b['client'] or 'Climate financial risk')}</span></div>")
    out.append(f"<div class='kicker'>{_h.escape(story.get('kicker', ''))}</div><h1>{_h.escape(story['title'])}</h1>")
    if story.get("standfirst"):
        out.append(f"<p class='stand'>{I(story['standfirst'])}</p>")
    words = sum(len(re.findall(r"\w+", x.get("text", "") or "")) for x in story["blocks"])
    out.append(f"<div class='by'>{_h.escape(story.get('byline', ''))} · {_h.escape(story.get('date', ''))} · {max(1, round(words / 220))} min read</div>")
    for x in story["blocks"]:
        t = x["type"]
        if t == "h2": out.append(f"<h2>{_h.escape(x['text'])}</h2>")
        elif t == "h3": out.append(f"<h3>{_h.escape(x['text'])}</h3>")
        elif t == "p": out.append(f"<p>{I(x['text'])}</p>")
        elif t == "list": out.append("<ul>" + "".join(f"<li>{I(i)}</li>" for i in x["items"]) + "</ul>")
        elif t == "quote": out.append(f"<blockquote><span class='src'>{I(x.get('source', ''))}</span>{I(x['text'])}</blockquote>")
        elif t == "callout": out.append(f"<div class='call {x.get('tone', 'info') if x.get('tone') != 'info' else ''}'>{I(x['text'])}</div>")
        elif t == "facts": out.append("<div class='facts'>" + "".join(f"<div class='fact'><b>{_h.escape(str(i['value']))}</b><span>{I(i['label'])}</span></div>" for i in x["items"]) + "</div>")
        elif t == "figure": out.append(f"<figure><img alt='{_h.escape(x.get('alt', ''))}' src='data:image/png;base64,{x['png_b64']}'><figcaption>{I(x.get('caption', ''))}</figcaption></figure>")
        elif t == "table":
            out.append("<table><tr>" + "".join(f"<th>{I(str(c))}</th>" for c in x["head"]) + "</tr>" +
                       "".join("<tr>" + "".join(f"<td>{I(str(c))}</td>" for c in r) + "</tr>" for r in x["rows"]) + "</table>")
            if x.get("caption"): out.append(f"<div class='cap'>{I(x['caption'])}</div>")
    if order:
        out.append("<h2>Sources</h2><ol class='src'>")
        for k in order:
            s = story["sources"][k]; i = num[k]
            link = f" <a href='{_h.escape(s['url'])}'>{_h.escape(s['url'])}</a>" if s.get("url") else ""
            out.append(f"<li id='s{i}'>{_h.escape(s['title'])}{(' — ' + _h.escape(s['note'])) if s.get('note') else ''}{link}</li>")
        out.append("</ol>")
    out.append(f"<div class='foot'>{_h.escape(b.get('footer') or 'Produced by the ClimRisk engine from the sources listed. Figures are model outputs with the limitations stated above; they are not forecasts or investment advice.')}</div>")
    out.append("</div></body></html>")
    return "".join(out)


# ── Markdown / LinkedIn ──────────────────────────────────────────────────────

def markdown(story: dict) -> str:
    num, order = _number_sources(story)
    I = lambda t: _inline(t, num, "md")                        # noqa: E731
    L = [f"# {story['title']}", "", f"*{story.get('kicker', '')}*", ""]
    if story.get("standfirst"): L += [I(story["standfirst"]), ""]
    for x in story["blocks"]:
        t = x["type"]
        if t == "h2": L += [f"## {x['text']}", ""]
        elif t == "h3": L += [f"### {x['text']}", ""]
        elif t == "p": L += [I(x["text"]), ""]
        elif t == "list": L += [f"- {I(i)}" for i in x["items"]] + [""]
        elif t == "quote": L += [f"> {I(x['text'])}", f"> — {I(x.get('source', ''))}", ""]
        elif t == "callout": L += [f"> **Note:** {I(x['text'])}", ""]
        elif t == "facts": L += [f"- **{i['value']}** — {I(i['label'])}" for i in x["items"]] + [""]
        elif t == "figure": L += [f"*[Figure] {I(x.get('caption', ''))}*", ""]
        elif t == "table":
            L += ["| " + " | ".join(map(str, x["head"])) + " |", "|" + "---|" * len(x["head"])] + ["| " + " | ".join(I(str(c)) for c in r) + " |" for r in x["rows"]] + [""]
    if order:
        L += ["## Sources", ""] + [f"{num[k]}. {story['sources'][k]['title']}" + (f" — {story['sources'][k]['url']}" if story["sources"][k].get("url") else "") for k in order]
    return "\n".join(L)


def linkedin(story: dict, hook: str, points: list, closing: str = "", tags: Optional[list] = None, link: str = "", limit: int = 2900) -> str:
    """A LinkedIn post: a hook line, short numbered findings, a closing line, tags. Plain text (LinkedIn does not render markdown)."""
    num, _ = _number_sources(story)
    P = lambda t: _inline(t, {}, "plain").replace("[S:", "")   # noqa: E731
    lines = [P(hook), ""]
    for i, pt in enumerate(points[:5], 1):
        lines += [f"{i}. {P(pt)}", ""]
    if closing: lines += [P(closing), ""]
    if link: lines += [link, ""]
    tags = tags or ["#ClimateRisk", "#PhysicalRisk", "#TransitionRisk", "#ClimateFinance"]
    lines.append(" ".join(tags))
    txt = "\n".join(lines).strip()
    return txt if len(txt) <= limit else txt[:limit - 1].rsplit("\n", 1)[0] + "\n…"


# ── PDF ──────────────────────────────────────────────────────────────────────

def pdf(story: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as RLImage, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    num, order = _number_sources(story)
    b = story.get("brand") or {}
    acc = colors.HexColor(b.get("accent") or "#0ea5e9")
    ss = getSampleStyleSheet()
    Bd = ParagraphStyle("b", parent=ss["Normal"], fontName="Times-Roman", fontSize=10.6, leading=15.5, spaceAfter=7)
    K = ParagraphStyle("k", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=8, textColor=acc, spaceAfter=6)
    T = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=24, leading=28, alignment=0, spaceAfter=8)
    SF = ParagraphStyle("sf", parent=Bd, fontName="Helvetica", fontSize=12.5, leading=17.5, textColor=colors.HexColor("#334155"))
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=15, leading=19, spaceBefore=14, spaceAfter=6, keepWithNext=1)
    H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=9, spaceAfter=3, keepWithNext=1)
    CP = ParagraphStyle("cp", parent=Bd, fontName="Times-Italic", fontSize=8.6, leading=11.5, textColor=colors.HexColor("#64748b"))
    SM = ParagraphStyle("sm", parent=Bd, fontName="Helvetica", fontSize=8.2, leading=11.2, spaceAfter=3)
    W = A4[0] - 36 * mm

    def esc(t):
        t = _inline(t, num, "plain")
        t = _h.escape(t)
        return t

    def pp(t, st=Bd):
        t = _h.escape(_BOLD.sub(lambda m: "\x01" + m.group(1) + "\x02", _CITE.sub(lambda m: f"[{num[m.group(1)]}]" if m.group(1) in num else "", t or "")))
        t = t.replace("\x01", "<b>").replace("\x02", "</b>")
        return Paragraph(t, st)
    fl = [pp(story.get("kicker", "").upper(), K), Paragraph(_h.escape(story["title"]), T)]
    if story.get("standfirst"): fl.append(pp(story["standfirst"], SF))
    fl.append(Paragraph(_h.escape(f"{story.get('byline', '')} · {story.get('date', '')}" + (f" · prepared for {b['client']}" if b.get("client") else "")), CP))
    fl.append(Spacer(1, 6))
    for x in story["blocks"]:
        t = x["type"]
        if t == "h2": fl.append(Paragraph(_h.escape(x["text"]), H2))
        elif t == "h3": fl.append(Paragraph(_h.escape(x["text"]), H3))
        elif t == "p": fl.append(pp(x["text"]))
        elif t == "list":
            for i in x["items"]: fl.append(pp("• " + i, Bd))
        elif t == "quote": fl.append(KeepTogether([pp(x.get("source", "").upper(), CP), pp(x["text"], ParagraphStyle("q", parent=Bd, leftIndent=10, textColor=colors.HexColor("#334155")))]))
        elif t == "callout": fl.append(Table([[pp(x["text"], SM)]], colWidths=[W], style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f9ff")), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#bae6fd")), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)])))
        elif t == "facts":
            VS = ParagraphStyle("fv", parent=SM, fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=colors.HexColor(b.get("accent") or "#0ea5e9"))
            LS = ParagraphStyle("fl", parent=SM, fontSize=7.8, leading=10)
            cells = [[[Paragraph(_h.escape(str(i["value"])), VS), Paragraph(_h.escape(_inline(i["label"], num, "plain")), LS)] for i in x["items"][:4]]]
            if cells[0]:
                fl.append(Table(cells, colWidths=[W / len(cells[0])] * len(cells[0]), style=TableStyle([("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")), ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)])))
                fl.append(Spacer(1, 6))
        elif t == "figure":
            raw = base64.b64decode(x["png_b64"])
            from PIL import Image
            iw, ih = Image.open(io.BytesIO(raw)).size
            wpt = W; hpt = W * ih / iw
            if hpt > 260: hpt = 260; wpt = hpt * iw / ih
            fl.append(KeepTogether([RLImage(io.BytesIO(raw), width=wpt, height=hpt), pp(x.get("caption", ""), CP)]))
        elif t == "table":
            data = [[Paragraph(f"<b>{_h.escape(_inline(str(c), num, 'plain'))}</b>", SM) for c in x["head"]]] + [[Paragraph(_h.escape(_inline(str(c), num, "plain")), SM) for c in r] for r in x["rows"]]
            cw = [W / len(x["head"])] * len(x["head"])
            fl.append(Table(data, colWidths=cw, repeatRows=1, style=TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#e2e8f0")), ("VALIGN", (0, 0), (-1, -1), "TOP")])))
            if x.get("caption"): fl.append(pp(x["caption"], CP))
    if order:
        fl.append(Paragraph("Sources", H2))
        for k in order:
            s = story["sources"][k]
            fl.append(Paragraph(_h.escape(f"{num[k]}. {s['title']}" + (f" — {s['note']}" if s.get("note") else "") + (f" {s['url']}" if s.get("url") else "")), SM))
    fl.append(Spacer(1, 8)); fl.append(Paragraph(_h.escape(b.get("footer") or "Produced by the ClimRisk engine from the sources listed. Model outputs with the limitations stated; not forecasts or investment advice."), CP))
    buf = io.BytesIO()

    def deco(c, d):
        c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.HexColor("#94a3b8"))
        c.drawString(18 * mm, 10 * mm, f"{b.get('name') or 'ClimRisk'} · {story['title'][:80]}"); c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {d.page}"); c.restoreState()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm, title=story["title"], author=story.get("byline", "")).build(fl, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()


# ── charts (matplotlib → PNG bytes) ──────────────────────────────────────────

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#94a3b8", "axes.labelcolor": "#334155", "xtick.color": "#475569", "ytick.color": "#475569"})
    return plt


def _png(fig) -> bytes:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", facecolor="white"); buf.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return buf.getvalue()


def chart_bars(labels, values, title="", unit="", color=None, errors=None, horizontal=False, size=(7.4, 3.6), annotate=True) -> bytes:
    plt = _mpl(); fig, ax = plt.subplots(figsize=size)
    cols = color or [PALETTE[0]] * len(values)
    cols = cols if isinstance(cols, list) else [cols] * len(values)
    if horizontal:
        bars = ax.barh(labels, values, color=cols, xerr=errors, ecolor="#94a3b8", capsize=3); ax.invert_yaxis()
    else:
        bars = ax.bar(labels, values, color=cols, yerr=errors, ecolor="#94a3b8", capsize=3)
    if annotate:
        for bt, v in zip(bars, values):
            if horizontal: ax.text(bt.get_width(), bt.get_y() + bt.get_height() / 2, f" {v:,.3g}{unit}", va="center", fontsize=9, color="#334155")
            else: ax.text(bt.get_x() + bt.get_width() / 2, bt.get_height(), f"{v:,.3g}{unit}", ha="center", va="bottom", fontsize=9, color="#334155")
    if title: ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.margins(y=0.12); fig.tight_layout()
    return _png(fig)


def chart_lines(x, series: dict, title="", ylabel="", size=(7.4, 3.6), bands: Optional[dict] = None) -> bytes:
    plt = _mpl(); fig, ax = plt.subplots(figsize=size)
    for i, (k, y) in enumerate(series.items()):
        ax.plot(x, y, label=k, color=PALETTE[i % len(PALETTE)], lw=2)
        if bands and k in bands:
            lo, hi = bands[k]; ax.fill_between(x, lo, hi, color=PALETTE[i % len(PALETTE)], alpha=0.15, lw=0)
    ax.set_ylabel(ylabel)
    if len(series) > 1: ax.legend(frameon=False, fontsize=9)
    if title: ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return _png(fig)


def chart_events(years, values, title="", ylabel="", highlight: Optional[dict] = None, size=(7.4, 3.4), threshold: Optional[tuple] = None) -> bytes:
    """Event record as stems: year on x, intensity on y; highlight={year: label}."""
    plt = _mpl(); fig, ax = plt.subplots(figsize=size)
    ax.vlines(years, 0, values, color="#94a3b8", lw=1.6)
    ax.scatter(years, values, s=26, color=PALETTE[0], zorder=3)
    for yy, lab in (highlight or {}).items():
        if yy in years:
            v = values[list(years).index(yy)]; ax.scatter([yy], [v], s=60, color=PALETTE[1], zorder=4); ax.annotate(lab, (yy, v), textcoords="offset points", xytext=(4, 6), fontsize=8.5, color="#334155")
    if threshold: ax.axhline(threshold[0], color="#ef4444", lw=1, ls="--"); ax.text(min(years), threshold[0], " " + threshold[1], va="bottom", fontsize=8.5, color="#ef4444")
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return _png(fig)


def chart_scatter_map(lons, lats, sizes, colors_v, labels=None, title="", cbar="", size=(7.4, 4.4), land: Optional[list] = None, cmap="YlOrRd") -> bytes:
    plt = _mpl(); fig, ax = plt.subplots(figsize=size)
    for ring in land or []:
        xs, ys = zip(*ring); ax.fill(xs, ys, color="#eef2f7", lw=0.3, ec="#cbd5e1")
    sc = ax.scatter(lons, lats, s=sizes, c=colors_v, cmap=cmap, edgecolor="#475569", lw=0.5, zorder=3, alpha=0.9)
    if labels:
        for x, y, t in zip(lons, lats, labels):
            if t: ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color="#334155")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label(cbar)
    ax.set_aspect(1 / max(0.2, abs(__import__("math").cos(__import__("math").radians(sum(lats) / len(lats))))))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    if title: ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return _png(fig)


def chart_card(headline: str, numbers: list, footer: str = "", size=(7.4, 4.2), accent="#0ea5e9") -> bytes:
    """The 'event in six numbers' graphic: numbers = [(value, label)]."""
    plt = _mpl(); fig = plt.figure(figsize=size); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#0b1220"))
    ax.text(0.04, 0.9, headline, color="white", fontsize=15, fontweight="bold", va="top", wrap=True)
    n = len(numbers[:6]); cols = 3; rows = (n + cols - 1) // cols
    for i, (v, l) in enumerate(numbers[:6]):
        r, c = divmod(i, cols); x = 0.04 + c * 0.32; y = 0.66 - r * 0.3
        ax.text(x, y, str(v), color=accent, fontsize=22, fontweight="bold", va="top")
        ax.text(x, y - 0.1, l, color="#cbd5e1", fontsize=8.5, va="top", wrap=True)
    ax.text(0.04, 0.04, footer or "ClimRisk engine", color="#64748b", fontsize=8)
    return _png(fig)
