"""Edition monitor for every external source the engine's numbers come from.

Each source has an adapter with three parts: what the engine HOLDS (read from its own data file), a DETECTOR that reads the publisher's
release page and says what the newest edition is, and the rule that decides whether the newest is newer than what is held.
CEA's database (connectors/cea_baseline.py) is also loaded from here, with its own guarded ingestion.

What this does and does not do:
  * It tells the engine, the agent and the user when a publisher has released something newer than the engine holds. It does NOT load the
    numbers of UK DESNZ, US eGRID, AIB or Ember by itself: those are large workbooks rebuilt by scripts/build_carbon_factors.py and checked by a person.
    The same holds for the national grid tables (Australia, Canada, Germany, China, Japan), rebuilt by scripts/build_national_grids.py.
  * A source with no page to read (China's MEE) carries a `manual` reason and is reported "unverified" rather than guessed.
  * A source is "current" only after a successful read of its release page within the last 7 days. Otherwise "unverified". A page whose layout
    is not recognised is an error ("layout changed"), never a silent "current".
  * Real pages were read on 6 Nov 2026 to write the detectors; the first live run on each machine confirms them (status shows the error if not).
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from . import cea_baseline as CEA

CHECK_EVERY_S = 24 * 3600
UNVERIFIED_AFTER_DAYS = 7
_UA = {"User-Agent": "ClimRisk-engine/1.0 (source monitor; contact shrinivashdkannan@gmail.com)"}
_LOCK = threading.RLock()
_BG = {"thread": None}
MONTHS = "January February March April May June July August September October November December".split()
_DATE = re.compile(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})\b")


# ── helpers ──────────────────────────────────────────────────────────────────

def _home() -> Path:
    return Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))


def _state_path() -> Path:
    return _home() / "source_state.json"


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _write(p: Path, d: dict) -> None:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(p)
    except OSError:
        pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ts(s: Optional[str]) -> Optional[float]:
    try:
        return datetime.fromisoformat(s).timestamp() if s else None
    except ValueError:
        return None


def _text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html or "")
    return " ".join(_html.unescape(re.sub(r"(?s)<[^>]+>", " ", html)).split())


def _iso(d: str) -> Optional[str]:
    """'31 July 2026' -> '2026-07-31'; '6/12/2025' (US) -> '2025-06-12'; already-ISO strings pass through."""
    m = _DATE.search(d or "")
    if m:
        return f"{m.group(3)}-{MONTHS.index(m.group(2)) + 1:02d}-{int(m.group(1)):02d}"
    m = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*", d or "")
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.match(r"(\d{4}-\d{2}-\d{2})", d or "")
    return m.group(1) if m else None


def _days_between(a: str, b: str) -> int:
    return (datetime.fromisoformat(a) - datetime.fromisoformat(b)).days


# ── what the engine holds ────────────────────────────────────────────────────

def _factors_meta() -> dict:
    from ..carbon import factors as F
    return F.lib()


def held_desnz() -> dict:
    d = _factors_meta()
    m = re.search(r"Reporting (\d{4})", d["desnz"]["source"])
    y = int(m.group(1)) if m else 0
    return {"edition": [y], "label": f"UK DESNZ GHG conversion factors {y}", "date": _iso(d["version"])}


def held_egrid() -> dict:
    d = _factors_meta()
    m = re.search(r"eGRID(\d{4})(?: rev(\d+))?", d["electricity"]["us_egrid"]["source"])
    y, r = (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)
    return {"edition": [y, r], "label": f"eGRID{y}" + (f" rev{r}" if r else ""), "date": _iso(d["version"])}


def held_aib() -> dict:
    d = _factors_meta()
    a = d["electricity"]["europe_aib"]
    m = re.search(r"published (\d{1,2} \w+ \d{4})", a["source"])
    pub = _iso(m.group(1)) if m else None
    return {"edition": [int(a["year"]), pub or ""], "label": f"AIB European Residual Mix {a['year']}", "date": pub}


def held_ember() -> dict:
    d = _factors_meta()
    return {"edition": [_iso(d["version"]) or ""], "label": f"Ember yearly electricity data as built {d['version']}", "date": _iso(d["version"])}


def _held_national(iso: str) -> Callable[[], dict]:
    def held() -> dict:
        from ..carbon import national_grids as NG
        c = NG.data()["countries"][iso]
        return {"edition": list(c["edition"]), "label": f"{c['name']}: {c['label']}", "date": _iso(c.get("published", ""))}
    return held


# ── detectors: publisher page -> newest edition ──────────────────────────────

def detect_desnz(html: str) -> dict:
    found: dict = {}
    for m in re.finditer(r"/publications/greenhouse-gas-reporting-conversion-factors-(\d{4})", html or ""):
        y = int(m.group(1))
        d = _DATE.search(_text(html[m.end():m.end() + 700]))
        found.setdefault(y, _iso(d.group(0)) if d else None)
    if not found:
        raise ValueError("layout changed: no conversion-factor publications listed")
    y = max(found)
    return {"edition": [y], "label": f"UK DESNZ GHG conversion factors {y}", "published": found[y],
            "link": f"https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-{y}"}


def detect_egrid(html: str) -> dict:
    t = _text(html)
    m = re.search(r"eGRID with (\d{4}) Data", t)
    if not m:
        raise ValueError("layout changed: no 'eGRID with <year> Data' heading")
    y = int(m.group(1))
    revs = [(int(n), _iso(d)) for n, d in re.findall(r"Revision (\d+) Released:\s*([\d/]+)", t)]
    r, rdate = max(revs, default=(0, None))
    first = re.search(r"Released:\s*([\d/]+)", t)
    return {"edition": [y, r], "label": f"eGRID{y}" + (f" rev{r}" if r else ""), "published": rdate or (_iso(first.group(1)) if first else None),
            "link": "https://www.epa.gov/egrid/detailed-data"}


def detect_aib(html: str) -> dict:
    years = [int(y) for y in re.findall(r"/facts/european-residual-mix/(20\d\d)\b", html or "")]
    if not years:
        raise ValueError("layout changed: no residual-mix years listed")
    y = max(years)
    t = _text(html)
    m = re.search(r"Residual Mixes for the calendar year (\d{4}).{0,40}?Version\s+[\d.]+,\s*(\d{4}-\d{2}-\d{2})", t)
    pub = m.group(2) if m and int(m.group(1)) == y else None
    return {"edition": [y, pub or ""], "label": f"AIB European Residual Mix {y}", "published": pub,
            "link": "https://www.aib-net.org/facts/european-residual-mix"}


def detect_ember(html: str) -> dict:
    m = (re.search(r'article:modified_time"?\s+content="([^"]+)"', html or "") or re.search(r'content="([^"]+)"\s+(?:property|name)="article:modified_time"', html or ""))
    if not m or not _iso(m.group(1)):
        raise ValueError("layout changed: no modified date on the page")
    d = _iso(m.group(1))
    return {"edition": [d], "label": f"Ember yearly electricity data, page updated {d}", "published": d,
            "link": "https://ember-energy.org/data/yearly-electricity-data/"}


def detect_nga(html: str) -> dict:
    """Australia: the DCCEEW publication index links one page per edition, .../national-greenhouse-accounts-factors-<year>."""
    years = [int(y) for y in re.findall(r"/publications/national-greenhouse-accounts-factors-(20\d\d)\b", html or "")]
    if not years:
        raise ValueError("layout changed: no NGA Factors editions listed")
    y = max(years)
    return {"edition": [y], "label": f"NGA Factors {y}", "published": None,
            "link": f"https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors-{y}"}


def detect_eccc(html: str) -> dict:
    """Canada: the open.canada.ca record of the inventory carries 'Date Published' and 'Temporal Coverage' (last year covered)."""
    t = _text(html)
    pub = re.search(r"Date Published:\s*(\d{4}-\d{2}-\d{2})", t)
    cov = re.search(r"Temporal Coverage:\s*\d{4}-\d{2}-\d{2} to (\d{4})-\d{2}-\d{2}", t)
    if not pub or not cov:
        raise ValueError("layout changed: no 'Date Published' / 'Temporal Coverage' on the record")
    y = int(cov.group(1))
    return {"edition": [y, pub.group(1)], "label": f"NIR (data to {y})", "published": pub.group(1),
            "link": "https://open.canada.ca/data/en/dataset/779c7bcf-4982-47eb-af1b-a33618a05e5b"}


def detect_uba(html: str) -> dict:
    """Germany: the UBA topic page lists the publication 'Entwicklung der spezifischen Treibhausgas-Emissionen des deutschen Strommix in den Jahren 1990 - <year>'."""
    years = [int(y) for y in re.findall(r"Strommix in den Jahren 1990\s*[-–]\s*(20\d\d)", _text(html))]
    if not years:
        raise ValueError("layout changed: no 'Strommix in den Jahren 1990 - <year>' publication listed")
    y = max(years)
    return {"edition": [y], "label": f"UBA electricity-mix emission factors 1990-{y}", "published": None,
            "link": "https://www.umweltbundesamt.de/themen/luft/emissionen-von-luftschadstoffen/spezifische-emissionsfaktoren-fuer-den-deutschen"}


def detect_moe(html: str) -> dict:
    """Japan: the SHK page links r<Reiwa year>_denki_coefficient[_rev<n>].pdf; the newest (year, revision) is the current electricity supplier list."""
    found = [(int(a), int(b or 0)) for a, b in re.findall(r"/calc/r(\d{2})_denki_coefficient(?:_rev(\d+))?\.pdf", html or "")]
    if not found:
        raise ValueError("layout changed: no electricity supplier coefficient files listed")
    r, rev = max(found)
    return {"edition": [r, rev], "label": f"SHK electricity supplier list R{r}" + (f" rev{rev}" if rev else ""), "published": None,
            "link": "https://policies.env.go.jp/earth/ghg-santeikohyo/calc.html"}


# ── registry ─────────────────────────────────────────────────────────────────

@dataclass
class Source:
    id: str
    name: str
    publisher: str
    url: str
    held: Callable[[], dict]
    detect: Optional[Callable[[str], dict]] = None
    files: tuple = ()
    rolling_days: Optional[int] = None            # rolling releases: "newer" only if published this many days after the held copy
    rebuild: str = ""
    custom: bool = False                          # delegated to its own connector (CEA)
    note: str = ""
    manual: str = ""                              # non-empty: no page the monitor can read; the text says why and the status stays "unverified"


def _cea_held() -> dict:
    d = CEA.current()
    return {"edition": list(CEA._vt(d["version"])), "label": f"CEA CO2 Baseline Database Version {d['version']} ({d['published']})", "date": d.get("retrieved")}


SOURCES: list = [
    Source("desnz", "UK DESNZ conversion factors", "Department for Energy Security and Net Zero (UK)", "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting",
           held_desnz, detect_desnz, files=("carbon_factors.json",), rebuild="scripts/build_carbon_factors.py",
           note="Annual; published June to October. Feeds UK Scope 1/2/3 factors and the UK grid factor."),
    Source("egrid", "US EPA eGRID", "US Environmental Protection Agency", "https://www.epa.gov/egrid/detailed-data",
           held_egrid, detect_egrid, files=("carbon_factors.json",), rebuild="scripts/build_carbon_factors.py",
           note="Annual, with revisions. Feeds US subregion and state grid factors and grid losses."),
    Source("aib", "AIB European Residual Mix", "Association of Issuing Bodies", "https://www.aib-net.org/facts/european-residual-mix",
           held_aib, detect_aib, files=("carbon_factors.json",), rebuild="scripts/build_carbon_factors.py",
           note="Annual, published about May. Feeds European production and residual-mix factors for market-based Scope 2."),
    Source("ember", "Ember yearly electricity data", "Ember (energy think tank)", "https://ember-energy.org/data/yearly-electricity-data/",
           held_ember, detect_ember, files=("carbon_factors.json", "lca_factors.json"), rolling_days=90, rebuild="scripts/build_carbon_factors.py and scripts/build_lca_data.py",
           note="Rolling, updated twice a month. Fallback grid factor and lifecycle grid factors for product footprints. CSV layout changed in July 2026; rebuild scripts must be checked against it."),
    Source("cea", "CEA CO2 Baseline Database (India)", "Central Electricity Authority, India", CEA.LISTING_URL, _cea_held, None, files=("cea_baseline.json",),
           custom=True, note="Annual. Loaded automatically when a newer user guide passes every check (connectors/cea_baseline.py)."),
    Source("nga", "Australia NGA Factors", "DCCEEW, Australian Government", "https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors",
           _held_national("AUS"), detect_nga, files=("national_grids.json",), rebuild="scripts/build_national_grids.py",
           note="Annual, mid-year. State and grid Scope 2/3 electricity factors and the national residual mix (AR5)."),
    Source("eccc", "Canada National Inventory Report, electricity intensity", "Environment and Climate Change Canada", "https://open.canada.ca/data/en/dataset/779c7bcf-4982-47eb-af1b-a33618a05e5b",
           _held_national("CAN"), detect_eccc, files=("national_grids.json",), rebuild="scripts/build_national_grids.py",
           note="Annual, April. Provincial and territorial generation and consumption intensity (Annex 7). The latest year is preliminary."),
    Source("uba", "Germany UBA electricity-mix emission factors", "Umweltbundesamt (German Environment Agency)", "https://www.umweltbundesamt.de/themen/luft/emissionen-von-luftschadstoffen/spezifische-emissionsfaktoren-fuer-den-deutschen",
           _held_national("DEU"), detect_uba, files=("national_grids.json",), rebuild="scripts/build_national_grids.py",
           note="Annual, March. CO2 and GHG factors of the German mix, with and without upstream; the newest year is an estimate."),
    Source("mee", "China electricity CO2 emission factors", "Ministry of Ecology and Environment with the National Bureau of Statistics", "https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/t20251231_1139517.html",
           _held_national("CHN"), None, files=("national_grids.json",), rebuild="scripts/build_national_grids.py",
           manual="MEE posts each year's factors as a separate announcement on a page I could not read as a list (the announcement index is built by script). New data years appear about two years late (2023 data came out on 31 Dec 2025)",
           note="Irregular, roughly annual. National, regional and provincial average CO2 factors plus a market-based variant."),
    Source("moe", "Japan SHK electricity supplier emission factors", "Ministry of the Environment and METI, Japan", "https://policies.env.go.jp/earth/ghg-santeikohyo/calc.html",
           _held_national("JPN"), detect_moe, files=("national_grids.json",), rebuild="scripts/build_national_grids.py",
           note="Annual (January) with revisions through the summer. Supplier-specific; the engine holds only the substitute value."),
]
BY_ID = {s.id: s for s in SOURCES}


def register(src: Source) -> None:
    """Add or replace a source (used by country and standards packs)."""
    for i, s in enumerate(SOURCES):
        if s.id == src.id:
            SOURCES[i] = src
            break
    else:
        SOURCES.append(src)
    BY_ID[src.id] = src


# ── fetching and checking ────────────────────────────────────────────────────

def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read(3_000_000).decode(r.headers.get_content_charset() or "utf-8", "replace")


def _is_newer(src: Source, latest: dict, held: dict) -> bool:
    if src.rolling_days is not None:
        a, b = latest.get("published"), held.get("date")
        return bool(a and b and _days_between(a, b) > src.rolling_days)
    return list(latest["edition"]) > list(held["edition"])


def sync(force: bool = False, fetch: Optional[Callable[[str], str]] = None, only: Optional[str] = None) -> dict:
    """Read each publisher's release page (each at most once a day unless forced). One source failing never stops the others. Returns status()."""
    if os.environ.get("CRI_STORY_OFFLINE", "").strip() not in ("", "0") and fetch is None:
        return status()
    for s in SOURCES:
        if only and s.id != only:
            continue
        try:
            if s.manual:
                continue
            if s.custom:
                if s.id == "cea":
                    CEA.sync(force=force, **({"fetch": (lambda: fetch(s.url))} if fetch else {}))
                continue
            with _LOCK:
                st = _read(_state_path())
                e = st.setdefault(s.id, {})
                last = _ts(e.get("last_attempt"))
                if not force and last and time.time() - last < CHECK_EVERY_S:
                    continue
                e["last_attempt"] = _now()
                try:
                    latest = s.detect((fetch or _fetch)(s.url))
                    e.update(checked_at=_now(), last_error=None, latest=latest)
                    if _is_newer(s, latest, s.held()):
                        if e.get("newer_edition") != latest["edition"]:
                            e["newer_seen"] = _now()
                        e["newer_edition"] = latest["edition"]
                except Exception as ex:                                  # noqa: BLE001
                    e["last_error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
                _write(_state_path(), st)
        except Exception:                                                # noqa: BLE001
            continue
    return status()


def _one(s: Source, st: dict) -> dict:
    held = s.held()
    if s.id == "cea":
        c = CEA.status()
        return {"id": s.id, "name": s.name, "publisher": s.publisher, "url": s.url, "held": held, "latest": {"label": f"Version {c['latest_listed']}"} if c.get("latest_listed") else None,
                "freshness": c["freshness"], "message": c["message"], "checked_at": c["checked_at"], "age_days": c["age_days"], "last_error": c["last_error"],
                "files": list(s.files), "note": s.note, "rebuild": s.rebuild}
    if s.manual:
        return {"id": s.id, "name": s.name, "publisher": s.publisher, "url": s.url, "held": held, "latest": None, "freshness": "unverified",
                "message": f"The engine holds {held['label']}. {s.manual}, so I cannot say automatically whether it is the newest; check {s.url} (or {s.publisher}'s announcements) by hand.",
                "checked_at": None, "age_days": None, "last_error": None, "files": list(s.files), "note": s.note, "rebuild": s.rebuild, "manual": True}
    e = st.get(s.id, {})
    checked = _ts(e.get("checked_at"))
    age = None if checked is None else round((time.time() - checked) / 86400, 1)
    latest = e.get("latest")
    newer = bool(latest and _is_newer(s, latest, held))
    if newer:
        fresh = "newer_available"
        msg = (f"{s.publisher} has published {latest['label']}" + (f" ({latest['published']})" if latest.get("published") else "") +
               f"; the engine holds {held['label']}. The new numbers are not loaded yet" + (f": rebuild with {s.rebuild} and check the result." if s.rebuild else "."))
    elif e.get("last_error") and checked is not None:
        fresh, msg = "unverified", (f"The engine holds {held['label']}. The latest attempt to read {s.publisher}'s release page failed ({e['last_error']}); "
                                    f"the last good read was {e['checked_at'][:10]}, so I cannot confirm it is still the newest.")
    elif checked is None:
        fresh, msg = "unverified", f"The engine holds {held['label']}. {s.publisher}'s release page has not been read from this machine yet, so I cannot say it is the newest."
    elif age is not None and age > UNVERIFIED_AFTER_DAYS:
        fresh, msg = "unverified", f"The engine holds {held['label']}. The release page was last read {age:.0f} days ago, so I cannot confirm it is still the newest."
    elif s.rolling_days is not None:
        fresh, msg = "current", (f"{s.name} is a rolling release. The engine's copy is dated {held['date']}; the page was last updated {latest.get('published')}, within the {s.rolling_days}-day window I allow. "
                                 "Checked " + e["checked_at"][:10] + ".")
    else:
        fresh, msg = "current", f"{held['label']} is the newest listed by {s.publisher} as of {e['checked_at'][:10]}."
    return {"id": s.id, "name": s.name, "publisher": s.publisher, "url": s.url, "held": held, "latest": latest, "freshness": fresh, "message": msg,
            "checked_at": e.get("checked_at"), "age_days": age, "last_error": e.get("last_error"), "files": list(s.files), "note": s.note, "rebuild": s.rebuild}


def status(source_id: Optional[str] = None) -> dict:
    st = _read(_state_path())
    rows = [_one(s, st) for s in SOURCES if not source_id or s.id == source_id]
    return {"as_of": _now(), "sources": rows,
            "summary": {k: sum(r["freshness"] == k for r in rows) for k in ("current", "newer_available", "unverified")},
            "note": "'current' only after a successful read of the publisher's release page within 7 days. DESNZ, eGRID, AIB, Ember and the national grid tables (Australia, Canada, Germany, China, Japan) are detected, not auto-loaded; "
                    "CEA is loaded when it passes every check. China's publisher has no page I can read automatically, so it stays 'unverified' by design."}


def for_file(name: str) -> list:
    """Monitor rows for sources that feed a data file (used by the provenance registry)."""
    st = _read(_state_path())
    return [_one(s, st) for s in SOURCES if name in s.files]


def start_background(interval_s: int = 6 * 3600) -> bool:
    """While the app is live: re-check every source on a schedule (each source is still read at most once a day). Idempotent."""
    with _LOCK:
        t = _BG["thread"]
        if t and t.is_alive():
            return False

        def loop():
            time.sleep(20)
            while True:
                try:
                    sync()
                except Exception:                                        # noqa: BLE001
                    pass
                time.sleep(interval_s)
        t = threading.Thread(target=loop, name="source-monitor", daemon=True)
        _BG["thread"] = t
        t.start()
    return True


# ── the agent ────────────────────────────────────────────────────────────────

def knowledge_text() -> str:
    rows = status()["sources"]
    parts = ["== Data sources and their editions ==\nThe engine's emission factors come from named publishers, each with an edition. " +
             " ".join(f"{r['name']}: {r['message']}" for r in rows)]
    for r in rows:
        parts.append(f"== {r['name']} ==\n{r['publisher']}. {r['note']} Held: {r['held']['label']}. {r['message']} Release page: {r['url']}")
    return "\n\n".join(parts)


def learn() -> dict:
    """Sync every source, then index the status into the resident agent's knowledge when it changed. Also keeps the CEA data current."""
    from ..resident import knowledge as K
    st = sync()
    text = knowledge_text() + "\n\n" + CEA.knowledge_text()
    sig = hashlib.sha256(text.encode()).hexdigest()[:16]
    url = "engine://data-sources"
    stt = _read(_state_path())
    if K._due(url, 10 ** 9) or stt.get("_learned_sig") != sig:
        n = K.learn_text(url, "Data sources behind the engine's emission factors, with editions and status", text, kind="engine")
        stt = _read(_state_path())
        stt["_learned_sig"] = sig
        _write(_state_path(), stt)
        return {"passages": n, "summary": st["summary"]}
    return {"passages": 0, "summary": st["summary"]}


_Q = re.compile(r"\b(desnz|defra|conversion factors?|egrid|residual mix|\baib\b|ember|data sources?|emission factors?.{0,40}(?:up to date|newer|latest|current|edition|version|source|from where)|"
                r"(?:up to date|newer|latest|current|edition|version).{0,40}emission factors?|which (?:data|factors|sources|editions))\b", re.I)
_ALIASES = {"desnz": ("desnz", "defra", "uk conversion", "uk government conversion", "uk factors"), "egrid": ("egrid", "epa"), "aib": ("residual mix", "aib"), "ember": ("ember",), "cea": ("cea", "india"),
            "nga": ("nga factors", "nga ", "australia", "dcceew"), "eccc": ("eccc", "canada", "canadian"), "uba": ("umweltbundesamt", "germany", "german "),
            "mee": ("china", "chinese", "ministry of ecology"), "moe": ("japan", "shk")}


# a country source is asked about only when the question joins a country, a grid/emission-factor word and a currency word ("is the Australian grid factor up to date")
_COUNTRY = r"(?:australia\w*|nga factors?|dcceew|canad\w*|eccc|german\w*|umweltbundesamt|china|chinese|japan\w*|shk)"
_Q2 = re.compile(_COUNTRY + r".{0,60}(?:grid|electricity|emission factors?).{0,70}\b(?:up to date|newer|latest|current|edition|version|published|release)\b|"
                 r"\b(?:up to date|newer|latest|current|edition|version|published|release)\b.{0,70}(?:grid|electricity|emission factors?).{0,60}" + _COUNTRY, re.I)


def matches(text: str) -> bool:
    return bool(_Q.search(text or "") or _Q2.search(text or ""))


def answer(text: str) -> Optional[str]:
    if not matches(text):
        return None
    low = text.lower()
    st = status()
    want = [i for i, al in _ALIASES.items() if any(a in low for a in al)]
    rows = [r for r in st["sources"] if r["id"] in want] or st["sources"]
    head = "" if want else (f"The engine's factors come from {len(rows)} monitored sources. Now: " + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in st["summary"].items() if v) + ". ")
    body = " ".join(f"{r['name']}: {r['message']}" for r in rows)
    return head + body + " " + (rows[0]["url"] if len(rows) == 1 else "")
