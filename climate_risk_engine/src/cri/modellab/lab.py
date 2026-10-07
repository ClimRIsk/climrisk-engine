"""The Model Lab front door: parse a goal, run the right study, write the report, and say honestly what was and was not established."""
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Callable, Optional

DOMAINS = {
    "flood": {"label": "Flood inundation (rain-driven, 2-D shallow-water)", "needs": "a place (coordinates or name)", "validates_with": "observed flood depths or extents"},
    "air": {"label": "Air quality (PM2.5 and AQI, physical box model)", "needs": "a city (coordinates or name)", "validates_with": "ground-station PM2.5 series"},
}

_FLOOD = re.compile(r"\b(flood\w*|inundat\w*|storm\s*water|pluvial|rain(?:fall)?[\s\-]*runoff|waterlogg\w*|drainage)\b", re.I)
_AIR = re.compile(r"\b(air[\s\-]*quality|aqi|pm\s*2\.?5|pm\s*10|particulate\w*|smog|pollution|air pollution)\b", re.I)
_MODEL = re.compile(r"\b(model|simulat\w*|build|construct|develop|calibrat\w*|validat\w*|phd|thesis|study)\b", re.I)


def detect_domain(text: str) -> Optional[str]:
    f, a = bool(_FLOOD.search(text)), bool(_AIR.search(text))
    if a and not f:
        return "air"
    if f and not a:
        return "flood"
    if f and a:
        return "air" if _AIR.search(text).start() < _FLOOD.search(text).start() else "flood"
    return None


def is_model_request(text: str) -> bool:
    return bool(_MODEL.search(text)) and detect_domain(text) is not None


def lab_dir() -> Path:
    """Same folder the chat serves files from (/resident/file/<name>)."""
    from ..resident import memory as M
    d = Path(M.db_path()).parent / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


_BASIS_Q = {
    "flood": ["shallow water equations", "Manning formula", "curve number runoff", "Gumbel distribution return period"],
    "air": ["atmospheric dispersion modeling Gaussian plume", "box model air pollution", "air quality index PM2.5", "planetary boundary layer"],
}


def physics_basis(domain: str, learn: bool = True) -> list:
    """Retrieve what the engine has read about each governing relation, so the report can cite it. Silent on failure."""
    out = []
    try:
        from ..resident import knowledge as K
    except Exception:                                                   # noqa: BLE001
        return out
    seen = set()
    for i, q in enumerate(_BASIS_Q.get(domain, [])):
        try:
            ps = K.search(q, 1)
            if (not ps or ps[0]["coverage"] < 0.5) and learn and not os.environ.get("CRI_STORY_OFFLINE"):
                K.learn_topic(q)
                ps = K.search(q, 1)
            if not ps or ps[0]["coverage"] < 0.5 or ps[0]["url"] in seen:
                continue
            p = ps[0]
            seen.add(p["url"])
            sent = re.split(r"(?<=[.!?])\s+", p["text"].strip())[0]
            sent = " ".join(sent.split()[:30]) + ("…" if len(sent.split()) > 30 else "")
            out.append({"key": f"b{i}", "title": p["title"], "url": p["url"], "text": f'{p["title"]}: {sent}'})
        except Exception:                                               # noqa: BLE001
            continue
    return out


def _numbers(t: str) -> set:
    return {x.replace(",", "") for x in re.findall(r"\d[\d,]*\.?\d*", t)}


def narrate(domain: str, result: dict, default: str) -> str:
    """Plain-language summary written by the local model from the computed numbers only; falls back to the template text."""
    try:
        from ..resident import answer as A
        ls = A.llm_status()
        if not ls["available"]:
            return default
        import json, urllib.request
        facts = default + " Verification tests: " + ", ".join(f'{v["name"]} {"passed" if v["ok"] else "FAILED"}' for v in result["verification"]) + ". Status: " + result["status"] + "."
        sysmsg = ("You write for a climate-risk manager. Using ONLY the facts given, write 90 to 130 words of plain prose: what was built, what it found, how far it can be trusted, and the one thing that would most improve it. "
                  "Never introduce a number that is not in the facts. No lists, no headings.")
        body = json.dumps({"model": ls["model"], "stream": False, "keep_alive": "30m", "options": {"temperature": 0.2, "num_ctx": 4096, "num_predict": 260},
                           "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": facts}]}).encode()
        req = urllib.request.Request(A._OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            txt = json.loads(r.read()).get("message", {}).get("content", "").strip()
        allowed = {n.lstrip("0") or "0" for n in _numbers(facts)}
        used = {n.lstrip("0") or "0" for n in _numbers(txt)}
        if len(txt.split()) < 40 or not used <= allowed:
            return default
        return txt
    except Exception:                                                   # noqa: BLE001
        return default


def run_study(domain: str, lat: float, lon: float, name: str, options: Optional[dict] = None, progress: Optional[Callable] = None, write: bool = True, use_llm: bool = False) -> dict:
    from .. import story as S
    from . import air_study, flood_study, report
    opts = dict(options or {})
    if domain == "flood":
        r = flood_study.run(lat, lon, name, progress=progress, **opts)
        story_fn = report.flood_story
    elif domain == "air":
        r = air_study.run(lat, lon, name, progress=progress, **opts)
        story_fn = report.air_story
    else:
        raise ValueError(f"unknown domain {domain!r}; available: {', '.join(DOMAINS)}")
    basis = physics_basis(domain)
    st0 = story_fn(r, "", basis)
    default = st0["standfirst"]
    text = narrate(domain, r, default) if use_llm else default
    st = story_fn(r, text, basis) if text != default else st0
    files = []
    if write:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()[:40] or domain
        base = f"modellab-{domain}_{slug}_{uuid.uuid4().hex[:8]}"
        d = lab_dir()
        (d / f"{base}.html").write_text(S.html(st), encoding="utf-8")
        files.append({"label": "Open the study (HTML)", "url": f"/resident/file/{base}.html"})
        try:
            (d / f"{base}.pdf").write_bytes(S.pdf(st))
            files.append({"label": "Download PDF", "url": f"/resident/file/{base}.pdf"})
        except Exception:                                               # noqa: BLE001
            pass
    return {"result": r, "story": st, "files": files, "written_by": "language model" if text != default else "template", "basis": basis}


def summary(r: dict) -> list:
    """Rows for a chat table: what was established and what was not."""
    rows = [["Status", r["status"]]]
    nv = sum(1 for v in r["verification"] if v["ok"])
    rows.append(["Verification", f'{nv} of {len(r["verification"])} tests passed'])
    if r["domain"] == "flood":
        o = r["out"]
        rows += [["Peak depth at the site", f'{o["site_depth_m"]:.2f} m'], ["Time under water", f'{o["wet_hours"]:.1f} h'], ["Area deeper than 30 cm", f'{o["flooded_area_km2"]:.2f} km²'],
                 ["Design storm", f'{r["return_period"]}-year, {r["storm_mm"]:.0f} mm in {r["storm_hours"]:g} h']]
        if r["sensitivity"]:
            rows.append(["Biggest uncertainty", r["sensitivity"][0]["label"]])
        rows.append(["Calibrated / validated", "yes" if r["validation"]["done"] else "no: no observations supplied"])
    else:
        v = r["validation"]
        rows += [["Chosen structure", f'{r["structure"]}: {r["structures"][r["structure"]]["label"]}'], ["Unseen-data correlation", f'{v["r"]:.2f}'], ["Unseen-data RMSE", f'{v["rmse"]:.1f} µg/m³'],
                 ["Skill vs mean daily cycle", f'{v["skill_vs_clim"]:+.0%}']]
        if "aqi_category_hit" in v:
            rows.append(["AQI category right", f'{v["aqi_category_hit"]:.0%} of {v["aqi_days"]} unseen days'])
        if r["sensitivity"]:
            rows.append(["Biggest uncertainty", r["sensitivity"][0]["label"]])
    return rows
