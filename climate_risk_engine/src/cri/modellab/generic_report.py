"""One report builder for every hazard study. A study returns a plain dict; this turns it into the engine's story format."""
from __future__ import annotations

from .. import story as S
from .report import _tone, _verification, _notebook, _basis, _next


def build_story(r: dict, basis: list = None) -> dict:
    st = S.Story(f'ClimRisk Model Lab · {r["label"]}', r["title"], r["standfirst"], byline="Written and run by the ClimRisk engine")
    st.callout(f'Model status: **{r["status"]}**. ' + ("Every number below comes from a model that passed its verification tests." if r["ok"] else "Verification failed; do not use these results."), _tone(r["status"]))
    if r.get("headline"):
        st.facts(r["headline"])
    for b in r["blocks"]:
        k = b[0]
        if k == "h2":
            st.section(b[1], b[2])
        elif k == "p":
            st.p(b[1])
        elif k == "ul":
            st.ul(b[1])
        elif k == "fig":
            st.figure(b[1], b[2])
        elif k == "table":
            st.table(b[1], b[2], b[3] if len(b) > 3 else "")
        elif k == "callout":
            st.callout(b[1], b[2] if len(b) > 2 else "info")
        elif k == "basis":
            _basis(st, basis or [])
    if r["verification"]:
        st.section("verification", "Is the model solved correctly?")
        _verification(st, r["verification"])
    if r.get("sensitivity"):
        st.section("sensitivity", "What the answer depends on")
        st.figure(S.chart_bars([x["label"] for x in r["sensitivity"]], [x["swing"] for x in r["sensitivity"]], title=f'How far the answer moves across each input\'s plausible range ({r.get("sens_unit", "")})', horizontal=True),
                  "One input varied at a time, the rest held at their base values.")
    st.section("assumptions", "Assumptions and limits")
    st.ul(r["assumptions"] + r["limitations"])
    _next(st, r)
    st.section("notebook", "Lab notebook")
    _notebook(st, r["notebook"])
    return st.build()


class Notebook:
    def __init__(self, progress=None):
        import time
        self.rows = []; self.t0 = time.time(); self.progress = progress

    def __call__(self, step, title, detail, **kw):
        import time
        self.rows.append({"step": step, "title": title, "detail": detail, "t": round(time.time() - self.t0, 1), **kw})
        if self.progress:
            try:
                self.progress(f"{step}: {title}")
            except Exception:                                         # noqa: BLE001
                pass
