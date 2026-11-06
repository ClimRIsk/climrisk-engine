"""CEA CO2 Baseline Database for the Indian Power Sector: seed data, live version monitor, guarded ingestion.

What CEA publishes (Central Electricity Authority, Ministry of Power):
  * one all-India grid emission factor per fiscal year (weighted average incl. renewables), plus operating, build and combined margins
    for CDM baselines. There are no state-wise or utility-wise factors. CO2 only, on net generation.
  * each release is an XLSX database and a PDF user guide, listed with a version number on
    https://cea.nic.in/cdm-co2-baseline-view?mappingId=234&lang=en (the page is server-rendered; links are /api/cdm-report-version/view-pdf/en/<id>).

What this module does:
  seed       the numbers of Version 22.0 (August 2026), transcribed and cross-checked (scripts/build_cea_baseline.py).
  monitor    sync() reads the listing at most once a day, finds the newest version and compares it with the version the engine holds.
  ingest     if a newer user guide is listed, its PDF is downloaded and the Annexure I table and Table S are parsed. Nothing is accepted
             unless every check passes (weighted average = CO2 / generation for every year, combined margin = mean of OM and BM, newest
             year later than the seed, at least ten years). A parse that fails any check leaves the seed in place and says so.
  honesty    status() always states which version the numbers are from and whether a newer one is known, and what the monitor could not do.
Nothing here is guessed: if the listing cannot be read the status says "unverified", it does not say "current".
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Optional

DATA = Path(__file__).resolve().parents[1] / "data" / "cea_baseline.json"
BASE = "https://cea.nic.in"
LISTING_URL = BASE + "/cdm-co2-baseline-view?mappingId=234&lang=en"
_LINK = re.compile(r"/api/cdm-report-version/view-pdf/en/(\d+)")
_UA = {"User-Agent": "ClimRisk-engine/1.0 (CEA baseline monitor; contact shrinivashdkannan@gmail.com)"}
CHECK_EVERY_S = 24 * 3600                       # the listing is read at most once a day
UNVERIFIED_AFTER_DAYS = 7                       # a last good check older than this no longer supports "current"
MAX_PDF_BYTES = 40_000_000
_LOCK = threading.RLock()
_BG = {"thread": None}
MONTHS = "January February March April May June July August September October November December".split()


# ── files ────────────────────────────────────────────────────────────────────

def _home() -> Path:
    return Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))


def _state_path() -> Path:
    return _home() / "cea_state.json"


def _overlay_path() -> Path:
    return _home() / "cea_overlay.json"


def _read(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _write(p: Path, d: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(p)
    except OSError:
        pass                                                          # a read-only home must not break the engine


def seed() -> dict:
    return json.loads(DATA.read_text())


def _vt(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or ""))


def current() -> dict:
    """The data in use: the seed, or a newer release that was ingested and passed every check."""
    d = seed()
    ov = _read(_overlay_path())
    if ov and _vt(ov.get("version")) > _vt(d["version"]):
        d["weighted_average"] = {**d["weighted_average"], **ov["weighted_average"]}
        d["margins"] = {**d["margins"], **ov.get("margins", {})}
        for k in ("version", "published", "latest_fy", "user_guide_url", "database_url"):
            if ov.get(k):
                d[k] = ov[k]
        d["source"] = "ingested"
        d["ingested_at"] = ov.get("ingested_at")
        d["ingest_sha256"] = ov.get("source_sha256")
    else:
        d["source"] = "seed"
    return d


# ── the listing ──────────────────────────────────────────────────────────────

class _Listing(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.versions: list = []
        self._cur: Optional[dict] = None
        self._label = ""
        self._a: Optional[dict] = None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            m = _LINK.search(dict(attrs).get("href") or "")
            if m and self._cur is not None:
                self._a = {"id": int(m.group(1)), "text": ""}

    def handle_data(self, data):
        t = " ".join(data.split())
        if not t:
            return
        if self._a is not None:
            self._a["text"] += " " + t
            return
        m = re.fullmatch(r"(?:Version\s+(\d+(?:\.\d+)?)|Draft Version)", t, re.I)
        if m:
            self._cur = {"version": m.group(1) or "draft", "database": None, "user_guide": None}
            self.versions.append(self._cur)
            self._label = ""
        else:
            self._label = t

    def handle_endtag(self, tag):
        if tag == "a" and self._a is not None and self._cur is not None:
            kind = (re.match(r"[A-Za-z]+", self._a["text"].strip()) or [""])[0].upper()
            role = "user_guide" if "user guide" in self._label.lower() else "database"
            if self._cur.get(role) is None:
                self._cur[role] = {"id": self._a["id"], "kind": kind, "url": f"{BASE}/api/cdm-report-version/view-pdf/en/{self._a['id']}",
                                   "label": self._label[:140]}
            self._a = None


def parse_listing(html: str) -> list:
    """Versions on the listing page, newest first as CEA shows them. Empty list if the layout is not recognised."""
    p = _Listing()
    p.feed(html or "")
    return [v for v in p.versions if v["version"] != "draft" and (v["database"] or v["user_guide"])]


def newest(listing: list) -> Optional[dict]:
    return max(listing, key=lambda v: _vt(v["version"]), default=None)


def _get(url: str, timeout: float = 25.0, limit: int = 3_000_000) -> bytes:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(limit)


def fetch_listing() -> str:
    raw = _get(LISTING_URL).decode("utf-8", "replace")
    if "Baseline Carbon Dioxide" not in raw:
        raise ValueError("the page did not look like the CEA listing")
    return raw


# ── reading a user guide ─────────────────────────────────────────────────────

_NUM = r"(\d[\d,]*\.?\d*)"
_ROW = re.compile(r"(20\d\d-\d\d)\*?\s+" + r"\s+".join([_NUM] * 5) + r"(?=\s|$)", re.M)


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def pdf_text(data: bytes) -> str:
    """Plain text of a PDF with pdftotext (if installed) or pypdf. Raises RuntimeError when neither is available."""
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "guide.pdf"
        f.write_bytes(data)
        try:
            out = subprocess.run(["pdftotext", "-layout", str(f), "-"], capture_output=True, timeout=60)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.decode("utf-8", "replace")
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            import pypdf
            return "\n".join((pg.extract_text() or "") for pg in pypdf.PdfReader(str(f)).pages)
        except ImportError as e:
            raise RuntimeError("no PDF reader: install pypdf (pip install pypdf) or poppler's pdftotext") from e


def parse_user_guide(text: str, seed_latest_fy: str = "2025-26") -> dict:
    """Annexure I and Table S from a user-guide text, validated. Returns {ok, ...}; ok is True only if every check passes."""
    why = []
    rows = {}
    for m in _ROW.finditer(text):                        # six numbers per fiscal year: CO2, conventional, RE, total, factor
        rows[m.group(1)] = [_num(m.group(i)) for i in range(2, 7)]
    series = {}
    for fy, (co2, conv, re_, tot, f) in rows.items():
        if tot <= 0 or abs(co2 / tot - f) >= 0.0006:
            why.append(f"{fy}: CO2/generation = {co2 / tot if tot else 0:.4f}, table says {f}")
            continue
        series[fy] = {"co2_mt": co2, "net_generation_conventional_bu": conv, "re_generation_bu": re_, "total_net_generation_bu": tot,
                      "adjustment_bu_captive_and_cross_border": round(tot - conv - re_, 2) if int(fy[:4]) >= 2022 else 0.0,
                      "weighted_average_tco2_per_mwh": f}
    if len(series) < 10:
        why.append(f"only {len(series)} valid fiscal-year rows found (need at least 10)")
    latest = max(series, default=None)
    if latest and latest <= seed_latest_fy:
        why.append(f"newest year in the table is {latest}, not later than the seed's {seed_latest_fy}")
    vm = re.search(r"Version\s+(\d+\.\d+)", text[:4000])
    pm = re.search(r"(" + "|".join(MONTHS) + r")\s+(20\d\d)", text[:4000])
    if not vm:
        why.append("version number not found on the cover page")
    margins = {}
    ts = re.search(r"Average\s+OM\s+BM\s+CM\s+" + r"\s+".join([r"(0\.\d{3})"] * 4), text)
    if ts and latest:
        avg, om, bm, cm = (float(ts.group(i)) for i in range(1, 5))
        if abs(avg - series.get(latest, {}).get("weighted_average_tco2_per_mwh", -1)) > 1e-9:
            why.append(f"Table S average {avg} differs from Annexure I {latest}")
        elif abs((om + bm) / 2 - cm) >= 0.0011:
            why.append(f"Table S combined margin {cm} is not the mean of OM {om} and BM {bm}")
        else:
            margins[latest] = {"om": om, "bm": bm, "cm": cm}
    elif latest:
        why.append("Table S (average, OM, BM, CM) not found")
    ok = not why
    return {"ok": ok, "problems": why, "version": vm.group(1) if vm else None, "published": f"{pm.group(1)} {pm.group(2)}" if pm else None,
            "latest_fy": latest, "weighted_average": series if ok else {}, "margins": margins if ok else {}, "rows_seen": len(rows)}


# ── the monitor ──────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state() -> dict:
    return _read(_state_path()) or {}


def sync(force: bool = False, fetch: Optional[Callable[[], str]] = None, download: Optional[Callable[[str], bytes]] = None) -> dict:
    """Read the CEA listing (at most once a day unless forced); ingest a newer user guide if one is listed. Never raises; returns status()."""
    if os.environ.get("CRI_STORY_OFFLINE", "").strip() not in ("", "0") and fetch is None:
        return status()
    with _LOCK:
        st = state()
        last = _parse_ts(st.get("last_attempt"))
        if not force and last and time.time() - last < CHECK_EVERY_S:
            return status()
        st["last_attempt"] = _now()
        try:
            listing = parse_listing((fetch or fetch_listing)())
            top = newest(listing)
            if not top:
                raise ValueError("no versions recognised: CEA may have changed the page layout")
            st.update(checked_at=_now(), last_error=None, latest_listed=top["version"], listing_top=listing[:3])
        except Exception as e:                                       # noqa: BLE001
            st["last_error"] = f"{type(e).__name__}: {str(e)[:160]}"
            _write(_state_path(), st)
            return status()
        held = current()["version"]
        if _vt(st["latest_listed"]) > _vt(held):
            st["newer_seen"] = st.get("newer_seen") if st.get("newer_seen_version") == st["latest_listed"] else _now()
            st["newer_seen_version"] = st["latest_listed"]
            guide = top.get("user_guide")
            if guide and guide.get("kind") == "PDF" and st.get("ingest_tried") != st["latest_listed"]:
                st["ingest_tried"] = st["latest_listed"]
                try:
                    data = (download or (lambda u: _get(u, 120.0, MAX_PDF_BYTES)))(guide["url"])
                    res = parse_user_guide(pdf_text(data), seed()["latest_fy"])
                    st["ingest_result"] = {k: res[k] for k in ("ok", "problems", "version", "published", "latest_fy", "rows_seen")}
                    if res["ok"] and res["version"] == st["latest_listed"]:
                        _write(_overlay_path(), {"version": res["version"], "published": res["published"], "latest_fy": res["latest_fy"],
                                                 "weighted_average": res["weighted_average"], "margins": res["margins"],
                                                 "user_guide_url": guide["url"], "database_url": (top.get("database") or {}).get("url"),
                                                 "source_sha256": hashlib.sha256(data).hexdigest(), "ingested_at": _now()})
                    elif res["ok"]:
                        st["ingest_result"]["problems"] = [f"guide says Version {res['version']} but the listing entry is {st['latest_listed']}"]
                except Exception as e:                               # noqa: BLE001
                    st["ingest_result"] = {"ok": False, "problems": [f"{type(e).__name__}: {str(e)[:160]}"]}
        _write(_state_path(), st)
    return status()


def _parse_ts(s: Optional[str]) -> Optional[float]:
    try:
        return datetime.fromisoformat(s).timestamp() if s else None
    except ValueError:
        return None


def status() -> dict:
    """Which CEA version the engine uses, and whether anything newer is known. 'current' is only said after a recent successful read of the listing."""
    d, st = current(), state()
    checked = _parse_ts(st.get("checked_at"))
    age = None if checked is None else round((time.time() - checked) / 86400, 1)
    listed = st.get("latest_listed")
    newer = bool(listed and _vt(listed) > _vt(d["version"]))
    if newer:
        ir = st.get("ingest_result") or {}
        fresh = "newer_available"
        msg = (f"CEA has published Version {listed}; the engine's numbers are from Version {d['version']} ({d['published']}). "
               + ("The new user guide was fetched but not accepted: " + "; ".join(ir.get("problems", []))[:240] if ir and not ir.get("ok") else
                  "The new tables have not been loaded yet."))
    elif st.get("last_error") and checked is not None:
        fresh, msg = "unverified", (f"Using Version {d['version']} ({d['published']}). The latest attempt to read the CEA listing failed ({st['last_error']}); "
                                    f"the last good read was {st['checked_at'][:10]}, so I cannot confirm it is still the newest.")
    elif checked is None:
        fresh, msg = "unverified", f"Using Version {d['version']} ({d['published']}). The CEA listing has not been read from this machine yet, so I cannot say it is the newest."
    elif age is not None and age > UNVERIFIED_AFTER_DAYS:
        fresh, msg = "unverified", f"Using Version {d['version']} ({d['published']}). The CEA listing was last read {age:.0f} days ago, so I cannot confirm it is still the newest."
    else:
        fresh, msg = "current", f"Version {d['version']} ({d['published']}) is the newest on the CEA listing as of {st['checked_at'][:10]}."
    return {"version": d["version"], "published": d["published"], "latest_fy": d["latest_fy"], "source": d["source"], "freshness": fresh,
            "message": msg, "checked_at": st.get("checked_at"), "age_days": age, "latest_listed": listed, "last_error": st.get("last_error"),
            "newer_seen": st.get("newer_seen"), "ingest_result": st.get("ingest_result"), "listing_url": LISTING_URL,
            "user_guide_url": d["user_guide_url"], "database_url": d["database_url"]}


def start_background(interval_s: int = 6 * 3600) -> bool:
    """While the app is live: re-read the CEA listing on a schedule (sync() itself limits real requests to one a day). Idempotent."""
    with _LOCK:
        t = _BG["thread"]
        if t and t.is_alive():
            return False

        def loop():
            time.sleep(20)                                           # let the app finish starting
            while True:
                try:
                    sync()
                except Exception:                                    # noqa: BLE001
                    pass
                time.sleep(interval_s)
        t = threading.Thread(target=loop, name="cea-monitor", daemon=True)
        _BG["thread"] = t
        t.start()
    return True


# ── factors ──────────────────────────────────────────────────────────────────

KINDS = {"average": "weighted_average", "weighted_average": "weighted_average", "om": "om", "bm": "bm", "cm": "cm"}


def factor(fy: Optional[str] = None, kind: str = "weighted_average") -> dict:
    """An all-India grid factor with its provenance. fy like '2024-25' (default: the latest published). kind: weighted_average, om, bm or cm."""
    d = current()
    k = KINDS.get((kind or "").lower())
    if not k:
        raise KeyError(f"unknown kind '{kind}': use weighted_average, om, bm or cm")
    fy = fy or d["latest_fy"]
    if k == "weighted_average":
        row = d["weighted_average"].get(fy)
        v = row["weighted_average_tco2_per_mwh"] if row else None
    else:
        row = d["margins"].get(fy)
        v = row[k] if row else None
    if v is None:
        have = sorted(d["weighted_average"] if k == "weighted_average" else d["margins"])
        raise KeyError(f"CEA Version {d['version']} has no {k} for FY {fy}; it has {have[0]} to {have[-1]}")
    note = []
    if k != "weighted_average" and int(fy[:4]) < 2023:
        note.append("margins before FY2023-24 do not account for renewable generation in the build margin")
    if int(fy[:4]) < 2022:
        note.append("before FY2022-23 totals are not adjusted for cross-border transfers and captive injection")
    return {"fy": fy, "kind": k, "tco2_per_mwh": v, "g_per_kwh": round(v * 1000, 1), "version": d["version"], "published": d["published"],
            "is_latest": fy == d["latest_fy"], "basis": "CO2 only, net generation, all-India grid, fiscal year (Apr-Mar)",
            "source": f"CEA CO2 Baseline Database v{d['version']} ({d['published']})", "url": d["user_guide_url"], "notes": note}


# ── what the resident agent knows ────────────────────────────────────────────

def knowledge_text() -> str:
    """The database as passages for the agent's index ('== heading ==' lines become passage labels). Rebuilt whenever the data or status change."""
    d, s = current(), status()
    wa = d["weighted_average"]
    yrs = sorted(wa)
    series = "; ".join(f"FY{y} {wa[y]['weighted_average_tco2_per_mwh']:.3f}" for y in yrs)
    mg = d["margins"]
    mseries = "; ".join(f"FY{y} OM {mg[y]['om']:.3f}, BM {mg[y]['bm']:.3f}, CM {mg[y]['cm']:.3f}" for y in sorted(mg))
    last = yrs[-1]
    rev = d["revision_history"][-3:]
    return "\n\n".join([
        f"== CEA CO2 Baseline Database: what it is ==\nThe CO2 Baseline Database for the Indian Power Sector is published by the Central Electricity Authority (CEA), Ministry of Power, Government of India, "
        f"for CDM baselines. {d['scope_statement']} Each release is an XLSX database and a PDF user guide. Current version in the engine: {d['version']} ({d['published']}), covering FY {yrs[0]} to FY {last}.",
        f"== CEA all-India grid emission factor, weighted average ==\nThe weighted average emission factor of the Indian grid including renewable generation, in tCO2/MWh (equal to kg CO2/kWh), by fiscal year: {series}. "
        f"The FY{last} value is {wa[last]['weighted_average_tco2_per_mwh']:.3f} tCO2/MWh, i.e. {wa[last]['weighted_average_tco2_per_mwh'] * 1000:.0f} g CO2/kWh. {d['definitions']['weighted_average']}",
        f"== CEA operating margin, build margin and combined margin ==\nOM, BM and CM of the Indian grid (tCO2/MWh): {mseries}. {d['definitions']['om']} {d['definitions']['bm']} {d['definitions']['cm']} "
        "CDM methodologies (ACM0002 for large grid-connected renewables) require the combined margin; the weighted average is used by some small-scale methodologies such as AMS-I.D.",
        "== CEA state-wise emission factors ==\nCEA publishes one emission factor for the single Indian grid. All regional grids were integrated in December 2013, so there are no state-wise or utility-wise factors in the CEA CO2 Baseline Database. "
        "A state figure such as one for Tamil Nadu or Andhra Pradesh is not a CEA value. For a plant buying from a specific supplier, use the supplier's own factor or the national CEA factor and say which.",
        "== Using CEA factors in a GHG inventory or product footprint ==\n" + " ".join(d["method_notes"]) +
        " For Scope 2 location-based reporting under the GHG Protocol, the CEA weighted average for the matching fiscal year is a national-authority factor for India. "
        "For a product carbon footprint under ISO 14067 the grid factor should include upstream emissions and non-CO2 gases, which CEA's CO2-only factor does not; the engine's lifecycle grid factor comes from Ember and a CEA value can be supplied as an override with that limitation stated.",
        "== CEA database versions and updates ==\nCEA updates the database annually and archives earlier versions. Latest revisions: " + " ".join(f"Version {r['version']} ({r['published']}): {r['changes']}." for r in rev) +
        f" Listing page: {LISTING_URL}",
        "== CEA database: status in this engine ==\n" + s["message"] + (f" Last read of the CEA listing: {s['checked_at']}." if s.get("checked_at") else ""),
    ])


def learn() -> dict:
    """Sync with CEA, then (re)index the database into the resident agent's knowledge. Safe to call every cycle."""
    from ..resident import knowledge as K
    st = sync()
    text = knowledge_text()
    sig = hashlib.sha256(text.encode()).hexdigest()[:16]
    url = LISTING_URL + "#engine"
    if K._due(url, 10 ** 9) or state().get("learned_sig") != sig:
        n = K.learn_text(url, "CEA CO2 Baseline Database for the Indian Power Sector (engine copy)", text, kind="primary")
        s2 = state()
        s2["learned_sig"] = sig
        _write(_state_path(), s2)
        return {"passages": n, "status": st["freshness"]}
    return {"passages": 0, "status": st["freshness"]}


_CEA_Q = re.compile(r"\b(cea|central electricity authority|cdm[\s\-]*co2|co2 baseline database|baseline (?:carbon )?(?:dioxide )?(?:emission )?database|"
                    r"(?:india|indian)\b.{0,30}\b(?:grid|power)\b.{0,30}\b(?:emission )?factor|(?:grid|emission) factor.{0,40}\b(?:india|indian)\b)", re.I)


def matches(text: str) -> bool:
    return bool(_CEA_Q.search(text or ""))


def answer(text: str) -> Optional[str]:
    """A direct, sourced answer to a question about CEA's CO2 Baseline Database; None if the question is not about it."""
    if not matches(text):
        return None
    t = text.lower()
    d, s = current(), status()
    yrs = sorted(d["weighted_average"])
    fy = None
    m = re.search(r"(20\d\d)\s*[-/–]\s*(\d{2,4})", text)
    if m:
        fy = f"{m.group(1)}-{m.group(2)[-2:]}"
    kind = "weighted_average"
    for k, pat in (("cm", r"\bcm\b|combined margin"), ("bm", r"\bbm\b|build margin"), ("om", r"\bom\b|operating margin")):
        if re.search(pat, t):
            kind = k
            break
    cite = f"Source: CEA CO2 Baseline Database v{d['version']} ({d['published']}), {d['user_guide_url']}. "
    if re.search(r"state|tamil|andhra|karnataka|maharashtra|gujarat|utility|discom|regional", t) and not fy:
        return ("CEA publishes one grid emission factor for the whole Indian grid; all regional grids were integrated in December 2013, and there are no state-wise or utility-wise factors in its CO2 Baseline Database. "
                f"The latest all-India weighted average is {d['weighted_average'][yrs[-1]]['weighted_average_tco2_per_mwh']:.3f} tCO2/MWh (FY{yrs[-1]}). "
                "If someone quotes a state figure, it is not a CEA value; use the supplier's own factor if you have one. " + cite + s["message"])
    if re.search(r"\b(new|newest|latest|current|up to date|updated?|published|version|release)\b", t) and not fy and kind == "weighted_average" and not re.search(r"factor|emission", t):
        return f"{s['message']} The listing is at {LISTING_URL}. " + (f"Checked {s['checked_at'][:10]}. " if s.get("checked_at") else "") + (f"Last monitor error: {s['last_error']}." if s.get("last_error") else "")
    try:
        f = factor(fy, kind)
    except KeyError as e:
        return str(e).strip("'\"") + ". " + cite
    label = {"weighted_average": "weighted average grid emission factor (including renewables)", "om": "operating margin", "bm": "build margin", "cm": "combined margin"}[f["kind"]]
    extra = ""
    if f["kind"] == "weighted_average":
        py = int(f["fy"][:4]) - 1
        prev = d["weighted_average"].get(f"{py}-{(py + 1) % 100:02d}")
        if prev:
            extra = f" The previous year was {prev['weighted_average_tco2_per_mwh']:.3f}."
    return (f"CEA's all-India {label} for FY{f['fy']} is {f['tco2_per_mwh']:.3f} tCO2/MWh ({f['g_per_kwh']:.0f} g CO2/kWh).{extra} It is CO2 only, on net generation, for fiscal years (April to March). "
            + (" ".join(f["notes"]) + ". " if f["notes"] else "") + cite + s["message"])
