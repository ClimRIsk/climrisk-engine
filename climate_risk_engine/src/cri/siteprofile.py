"""
cri/siteprofile.py — every hazard, for any coordinate, by default.

    profile(lat, lon)  →  one entry per hazard, same contract everywhere.

Status of each hazard (the only four values):
  modelled     a physical or published model ran on data for this place
  constructed  the primary model could not run, so the value was BUILT from a fallback source or a proxy
               (e.g. terrain position instead of a flood map); lower confidence, and the entry says how
  none_found   the engine established that the hazard is absent here (e.g. no cyclone exposure, site above the coastal
               cut-off). This is a finding with a value of 0, not a gap.
  null         no data and no way to construct it. value is None and `reason` says what was tried.
               NULL IS NOT ZERO: a null hazard has not been assessed and must never be read as low risk.

Every entry also carries `attempts` (each rung tried, in order, and why it passed or failed), `method`, `sources`
and, where the model reports one, an evidence grade. Entries stream out through `progress` as they finish.

3D twin: if a twin built by `asset_twin.build_twin` is supplied (or one is cached for the place) its event history and
routed flood scenarios are reused where the direct sources fail, and it is reported under `history`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

SEV = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "CRITICAL"]
STATUSES = ("modelled", "constructed", "none_found", "null")


def _entry(key: str, label: str, status: str, *, value=None, unit: str = "", severity: Optional[str] = None, metrics: Optional[dict] = None,
           method: str = "", grade: Optional[str] = None, sources: Optional[list] = None, attempts: Optional[list] = None,
           reason: Optional[str] = None, family: str = "", caveat: Optional[str] = None, elapsed_s: Optional[float] = None) -> dict:
    assert status in STATUSES, status
    if status == "null":
        value, severity = None, None                         # null stays null
    if status == "none_found":
        value, severity = (0.0 if value is None else value), "NEGLIGIBLE"
    return {"key": key, "label": label, "family": family, "status": status, "value": value, "unit": unit, "severity": severity,
            "metrics": metrics or {}, "method": method, "evidence_grade": grade, "sources": sources or [], "attempts": attempts or [],
            "reason": reason, "caveat": caveat, "elapsed_s": elapsed_s}


def _null(key, label, family, attempts, reason):
    return _entry(key, label, "null", attempts=attempts, reason=reason, family=family)


class Ctx:
    """Shared, lazily computed inputs so several hazards reuse one fetch."""

    def __init__(self, lat: float, lon: float, name: str, twin: Optional[dict] = None):
        self.lat, self.lon, self.name, self.twin = lat, lon, name, twin
        self._v: dict = {}
        self._locks: dict = {}
        self._g = threading.Lock()

    def once(self, key: str, fn: Callable):
        with self._g:
            lk = self._locks.setdefault(key, threading.Lock())
        with lk:
            if key not in self._v:
                try:
                    self._v[key] = ("ok", fn())
                except Exception as e:                       # noqa: BLE001
                    self._v[key] = ("err", f"{type(e).__name__}: {str(e)[:160]}")
            return self._v[key]


# ── fast, mostly offline ──────────────────────────────────────────────────────────

def _h_cyclone(c: Ctx) -> dict:
    from .analysis import asset_intelligence as ai
    KEY, LAB = "cyclone", "Tropical cyclone wind"
    try:
        f = ai._tropical_cyclone(c.lat, c.lon)
    except Exception as e:                                   # noqa: BLE001
        return _null(KEY, LAB, "wind", [{"rung": "IBTrACS + Holland model", "ok": False, "why": str(e)[:160]}], "Cyclone track data could not be read.")
    ev = {e["metric"]: e["value"] for e in f["evidence"]}
    n200 = ev.get("Cyclones passing within 200 km since 1980") or 0
    rp100 = (f.get("detail", {}).get("rp_wind_ms") or {}).get(100) or (f.get("detail", {}).get("rp_wind_ms") or {}).get("100") or 0.0
    att = [{"rung": "IBTrACS 1980–2025 + Holland wind model", "ok": True}]
    src = ["NOAA IBTrACS v04r01"]
    if f["severity"] == "NEGLIGIBLE" and not n200 and rp100 < 17.5:
        return _entry(KEY, LAB, "none_found", value=0.0, unit="m/s (1-in-100-yr wind)", metrics={"storms_within_200km_since_1980": n200},
                      method="No cyclone has passed within 200 km since 1980 and the 1-in-100-year wind is below tropical-storm force.",
                      sources=src, attempts=att, family="wind", reason="No cyclone exposure found.")
    return _entry(KEY, LAB, "modelled", value=float(rp100), unit="m/s (1-in-100-yr 1-min wind, open terrain)", severity=f["severity"],
                  metrics={"evidence": {e["metric"]: e["value"] for e in f["evidence"]}, "top_events": f.get("detail", {}).get("top_events")},
                  method="Cyclone track set with a Holland wind profile, validated against station winds.", sources=src, attempts=att,
                  family="wind", caveat=f.get("caveat"))


def _flood_maps(c: Ctx):
    from .finance.flood_hazard import site_flood
    return site_flood(c.lat, c.lon, 250.0, None, None)


def _terrain_rank(c: Ctx):
    from .analysis.flood_event_model import site_score
    from .terrain.hand import hand_metrics
    hm = hand_metrics(c.lat, c.lon, radius_km=4.0)
    ts = site_score(hm)
    if not ts:
        raise RuntimeError("terrain position model unavailable")
    return ts, hm


def _depth_sev(d: float) -> str:
    return SEV[0 if d < 0.05 else 1 if d < 0.3 else 2 if d < 1.0 else 3 if d < 2.0 else 4]


def _rank_sev(r: float) -> str:
    return "MODERATE" if r >= 0.8 else "LOW" if r >= 0.6 else "NEGLIGIBLE"          # constructed evidence is capped at MODERATE


def _h_river(c: Ctx) -> dict:
    KEY, LAB, FAM = "river_flood", "River flood", "flood"
    att = []
    st, fl = c.once("flood_maps", lambda: _flood_maps(c))
    if st == "ok":
        riv = fl.get("river") or {}
        if riv.get("available") and not riv.get("footprint_on_water") and riv.get("depths"):
            d100 = riv["depths"].get("100") or []
            mean = float(sum(d100) / len(d100)) if d100 else 0.0
            wet = float(sum(1 for x in d100 if x > 0.05) / len(d100)) if d100 else 0.0
            att.append({"rung": "JRC CEMS-GloFAS river flood maps", "ok": True})
            return _entry(KEY, LAB, "modelled", value=round(mean, 2), unit="m (1-in-100-yr mean depth over the footprint)", severity=_depth_sev(mean),
                          metrics={"wet_share_rp100": round(wet, 2), "return_periods": riv.get("rps")}, method="JRC river flood hazard maps (undefended, 90 m).",
                          sources=[riv.get("source") or "JRC CEMS-GloFAS v2.1"], attempts=att, family=FAM)
        why = "site footprint lies on the mapped water body" if riv.get("footprint_on_water") else (riv.get("reason") or riv.get("note") or "no river flood map here")
        att.append({"rung": "JRC CEMS-GloFAS river flood maps", "ok": False, "why": str(why)[:160]})
    else:
        att.append({"rung": "JRC CEMS-GloFAS river flood maps", "ok": False, "why": fl})
    # construct: terrain position trained on observed flood extents
    s2, v2 = c.once("terrain_rank", lambda: _terrain_rank(c))
    if s2 == "ok":
        ts, hm = v2
        att.append({"rung": "terrain position model (trained on observed flood extents)", "ok": True})
        return _entry(KEY, LAB, "constructed", value=ts["rank"], unit="terrain rank 0–1 (not a depth)", severity=_rank_sev(ts["rank"]),
                      metrics={"hand_m_1km2": ts["hand_m_1.0"], "elev_percentile_in_window": ts["elev_percentile"], "heldout_auc": ts["heldout_auc"]},
                      method="No flood map: built from height above nearest drainage and relative elevation, fitted on 72 observed flood events.",
                      sources=["AWS terrain tiles", "Copernicus EMS observed flood extents (training)"], attempts=att, family=FAM,
                      caveat="Ranks how flood-like the terrain position is; it is not a depth or a return period.")
    att.append({"rung": "terrain position model (trained on observed flood extents)", "ok": False, "why": v2})
    # 3D twin routed scenarios
    tw = (c.twin or {}).get("scenarios", {}).get("flood") if c.twin else None
    if tw:
        att.append({"rung": "3D twin flood scenarios", "ok": True})
        return _entry(KEY, LAB, "constructed", value=None, unit="", severity=None, metrics={"scenarios": [s.get("name") for s in tw][:6]},
                      method="Reused routed flood scenarios from the 3D twin.", attempts=att, family=FAM, sources=["3D twin"])
    return _null(KEY, LAB, FAM, att, "No flood map and no terrain data for this place; river flood could not be assessed.")


def _h_coastal(c: Ctx) -> dict:
    KEY, LAB, FAM = "coastal_flood", "Coastal flood and sea-level rise", "flood"
    att = []
    st, fl = c.once("flood_maps", lambda: _flood_maps(c))
    if st != "ok":
        att.append({"rung": "Deltares coastal flood maps", "ok": False, "why": fl})
    else:
        cst = fl.get("coastal") or {}
        if cst.get("available") and cst.get("depths"):
            d = cst["depths"]
            def m(k):
                v = d.get(k) or []
                return float(sum(v) / len(v)) if v else 0.0
            d18, d50 = m("2018_100"), m("2050_100")
            att.append({"rung": "Deltares coastal flood maps", "ok": True})
            return _entry(KEY, LAB, "modelled", value=round(d50, 2), unit="m (1-in-100-yr mean depth, 2050)", severity=_depth_sev(d50),
                          metrics={"depth_2018_m": round(d18, 2), "depth_2050_m": round(d50, 2)}, method="Deltares global coastal flood maps incl. sea-level rise.",
                          sources=[cst.get("source") or "Deltares Global Flood Maps"], attempts=att, family=FAM)
        why = str(cst.get("reason") or "no coastal map here")
        att.append({"rung": "Deltares coastal flood maps", "ok": False, "why": why[:160]})
        if "above 30 m" in why:
            return _entry(KEY, LAB, "none_found", value=0.0, unit="m", metrics={"reason": why}, method="Site is above the 30 m coastal cut-off.",
                          attempts=att, family=FAM, reason=why)
    return _null(KEY, LAB, FAM, att, "No coastal flood map for this place. The storm-surge and sea-level studies below may still give a result.")


def _water_users(c: Ctx) -> str:
    try:
        from .analysis import local_air as LA
        k = [x["kind"] for x in LA.declared(c.lat, c.lon) if x["kind"] in ("thermal", "cement")]
    except Exception:                                        # noqa: BLE001
        return ""
    return (" Declared nearby industrial water users (" + ", ".join(sorted(set(k))) + ") draw on the same basin: thermal power and cement are large cooling and process water users, "
            "so local stress can be worse than the basin score. Their withdrawals were not found in any open database and are not in this number.") if k else ""


def _h_water(c: Ctx) -> dict:
    from .analysis import asset_intelligence as ai
    KEY, LAB = "water_stress", "Water stress and drought (basin scale)"
    try:
        f = ai._water_stress(c.lat, c.lon)
    except Exception as e:                                   # noqa: BLE001
        return _null(KEY, LAB, "water", [{"rung": "WRI Aqueduct 4.0", "ok": False, "why": str(e)[:160]}], "Aqueduct data could not be read.")
    if f is None:
        return _null(KEY, LAB, "water", [{"rung": "WRI Aqueduct 4.0", "ok": False, "why": "outside basin coverage"}],
                     "Point is outside Aqueduct basin coverage (open water, or no unit within about 15 km).")
    return _entry(KEY, LAB, "modelled", value=f["detail"]["indicator_score"], unit="Aqueduct score 0–5", severity=f["severity"],
                  metrics={"path_bau": f["detail"]["path_bau"], "evidence": {e["metric"]: e["value"] for e in f["evidence"]}},
                  method="WRI Aqueduct 4.0, basin scale; not site supply.", sources=["WRI Aqueduct 4.0 (CC BY 4.0)"], attempts=[{"rung": "WRI Aqueduct 4.0", "ok": True}],
                  family="water", caveat=f["caveat"] + _water_users(c))



def _h_air_local(c: Ctx) -> dict:
    """Air quality from the measured CAMS record plus a source screen of the emitters known or declared near the site."""
    from .analysis import local_air as LA
    KEY, LAB = "air_local", "Air quality (measured regional record + nearby sources)"
    a = LA.assess(c.lat, c.lon)
    if a["status"] == "null":
        return _null(KEY, LAB, "air", [{"rung": "CAMS record + ERA5 wind", "ok": False, "why": a["reason"]}], a["reason"])
    rec = a["record"]
    cav = [rec["resolution"]]
    v = a.get("validation") or {}
    if v:
        cav.append(f'Checked against 65 ground monitors (2023): daily correlation about {v["daily_correlation_median"]}, annual mean within 30% at {round(v["annual_mean_within_30pct_of_monitor_share"] * 100)}% of held-out sites, '
                   f'median model/monitor ratio {v["polluted_sites_median_cams_over_monitor"]} where monitors read 20 or more µg/m3 (the model tends to read low in polluted air; treat this as a lower bound there). {v["caveat"]}')
    if a.get("assumption_note"):
        cav.append(a["assumption_note"])
    if a.get("unverified_sources"):
        cav.append("Sources declared by the user and not verifiable in any open database: " + ", ".join(a["unverified_sources"]) + ".")
    cav.append("Not modelled: " + "; ".join(a["not_modelled"]) + ".")
    return _entry(KEY, LAB, a["status"], value=a["annual_pm25_total_central"], unit="µg/m3 annual PM2.5 (regional record + central source screen)", severity=a["severity"],
                  metrics={k: a[k] for k in ("record", "sources", "increment", "thresholds", "osm", "research")},
                  method="CAMS hourly record at the site plus a Gaussian-plume screen (Briggs urban, hourly ERA5 wind) of each known or declared source; emissions from capacity and documented factors, in three scenarios.",
                  grade="C", sources=["CAMS via Open-Meteo air-quality API", "ERA5 via Open-Meteo archive", "WHO 2021 air quality guidelines", "India NAAQS 2009"],
                  attempts=[{"rung": "CAMS record + ERA5 wind", "ok": True}, {"rung": "source screen", "ok": bool(a["sources"]), "why": None if a["sources"] else "no source known or declared"}],
                  family="air", caveat=" ".join(cav))


# ── site analysis (terrain, imagery, land cover) — one call feeds several hazards ─────

_SITE_MAP = {"flood": ("flood_terrain_radar", "Flood from terrain and observed water history", "flood"),
             "debris_flow": ("debris_flow", "Debris flow", "slope"), "landslide": ("landslide_slope", "Landslide / slope instability", "slope"),
             "wildfire": ("wildfire_fuel", "Wildfire fuel exposure", "fire"), "urban_heat": ("urban_heat_landcover", "Urban heat (land-cover screening)", "heat")}


def _site_assess(c: Ctx):
    from .analysis import asset_intelligence as ai
    return ai.assess_asset(c.lat, c.lon, c.name, "industrial", include_visuals=False, sar_timeout_s=60.0)


def _site_entry(c: Ctx, haz: str) -> dict:
    KEY, LAB, FAM = _SITE_MAP[haz]
    st, a = c.once("site_assess", lambda: _site_assess(c))
    if st != "ok":
        return _null(KEY, LAB, FAM, [{"rung": "site analysis (terrain, Sentinel, land cover)", "ok": False, "why": a}], "The site analysis could not run for this place.")
    needs_terrain = haz in ("flood", "debris_flow", "landslide", "wildfire")
    if needs_terrain and "terrain" in (a.get("errors") or {}):
        return _null(KEY, LAB, FAM, [{"rung": "site analysis", "ok": False, "why": str(a["errors"]["terrain"])[:160]}],
                     "Terrain did not load, so this hazard could not be assessed.")
    f = next((x for x in a["findings"] if x["hazard"] == haz), None)
    if f is None:
        return _null(KEY, LAB, FAM, [{"rung": "site analysis", "ok": False, "why": "no finding returned"}], "The analysis returned no result for this hazard.")
    basis = f.get("basis", "proxy")
    status = "modelled" if basis == "modelled" else "constructed"
    return _entry(KEY, LAB, status, value=f.get("score"), unit="points 0–6", severity=f["severity"],
                  metrics={"evidence": {e["metric"]: e["value"] for e in f["evidence"][:12]}}, method=f.get("basis_note", ""),
                  sources=a.get("sources", [])[:4], attempts=[{"rung": "site analysis (terrain, Sentinel, land cover)", "ok": True}], family=FAM, caveat=f.get("caveat"))


# ── Model Lab studies ───────────────────────────────────────────────────────────────

_LAB = {  # key → (label, family, ladder of option sets, options always passed)
    "heat": ("Extreme heat (WBGT extreme values)", "heat", [{"bootstrap": 100}]),
    "rain": ("Extreme rainfall", "flood", [{"bootstrap": 100}]),
    "wind": ("Extreme wind (all causes)", "wind", [{"bootstrap": 100}]),
    "fire": ("Fire weather (Canadian FWI)", "fire", [{"bootstrap": 100}]),
    "drought": ("Drought (soil water balance)", "water", [{}]),
    "river": ("River flow (rainfall-runoff)", "flood", [{}]),
    "earthquake": ("Earthquake shaking", "seismic", [{}, {"radius_km": 600}]),
    "surge": ("Storm surge and extreme sea level", "flood", [{}]),
    "sealevel": ("Sea-level rise and shoreline retreat", "flood", [{}]),
    "landslide": ("Landslide (infinite-slope physics)", "slope", [{}]),
    "air": ("Air quality (PM2.5)", "air", [{}]),
    "flood": ("Flood inundation (rain-driven 2-D)", "flood", [{"radius_km": 2.0}]),
}
_ABSENT = re.compile(r"no (?:tropical cyclone|river|coast|water body|sea)\b|not (?:exposed|applicable)|\binland\b|no exposure|too far from the (?:sea|coast)|no sea within|does not reach", re.I)
_DATA = re.compile(r"too little data|too sparse|too few|not enough|insufficient|could not (?:read|fetch|load)|no data|unavailable|catalog", re.I)


def _num(s) -> Optional[float]:
    m = re.match(r"\s*[+-]?\d+(?:\.\d+)?", str(s))
    return float(m.group(0)) if m else None


def _lab_map(key: str, r: dict, attempt_note: str) -> tuple:
    """Returns (entry or None, classification) from a Model Lab result."""
    label, fam, _ = _LAB[key]
    status_txt = str(r.get("status") or "")
    if r.get("kind") == "no_model" or status_txt.startswith("no model"):
        why = (r.get("limitations") or [status_txt])[0] if r.get("limitations") else status_txt
        txt = status_txt + " " + str(why)
        if not _DATA.search(txt) and _ABSENT.search(txt):             # data shortage is checked first: sparse data is NOT absence
            return _entry(key, label, "none_found", value=0.0, metrics={"headline": r.get("headline")}, method=str(why)[:200], attempts=[], family=fam,
                          reason=str(why)[:200], sources=["Model Lab applicability check"]), "absent"
        return None, ("data" if _DATA.search(txt) else "other") + "|" + txt[:200]
    head = r.get("headline") or []
    grade = next((h[0] for h in head if "evidence grade" in str(h[1]).lower()), None)
    # the primary value must be a HAZARD level (a return level, a peak, an unstable share), never a validation statistic
    good = re.compile(r"year|return|pga|unstable|peak|depth|level", re.I)
    bad = re.compile(r"correlation|nash|kling|skill|detected|false|rmse|grade|trend|uniformity|\bp =|drift", re.I)
    cand = [h for h in head if good.search(str(h[1])) and not bad.search(str(h[1])) and _num(h[0]) is not None]
    val_item = next((h for h in cand if "100" in str(h[1])), None) or (cand[0] if cand else None)
    val = _num(val_item[0]) if val_item else None
    unit = (re.sub(r"^\s*[+-]?[\d.]+\s*", "", str(val_item[0])) + " " + str(val_item[1])).strip() if val_item else ""
    summ = []
    try:
        from .modellab import lab as L
        summ = L.summary(r)
    except Exception:                                        # noqa: BLE001
        pass
    if val is None:
        row = next((x for x in summ if re.search(r"peak depth", str(x[0]), re.I) and _num(x[1]) is not None), None)
        if row:
            tail = re.sub(r"^[\s+-]*[\d.]+", "", str(row[1])).strip()
            val, unit = _num(row[1]), "%s (%s)" % (row[0], tail)
    caveat = None
    status = "modelled"
    if key == "heat":
        caveat = "Outdoor WBGT in the sun (Liljegren) from ERA5. ERA5 inputs ran about 0.7 °C below airport station inputs (about 1.1 °C on the hottest hours, 20 sites), so peaks are slightly understated; screening values, not safe-work thresholds."
    if key == "drought":
        caveat = "Against the US Drought Monitor (26 counties, unseen years) the model ranks drought weeks above normal weeks with ROC area about 0.70 (0.5 = no skill) and catches only about a fifth of drought weeks at its default threshold; weaker where irrigated; untested outside the US."
    if val is None and status == "modelled":
        caveat = ((caveat + " ") if caveat else "") + "Modelled and validated, but this study reports series and skill, not a single hazard level: see metrics."
    if r.get("skilful") is False:
        status, grade = "constructed", "no skill"
        caveat = ((caveat + " ") if caveat else "") + "Validation shows no skill over simple baselines for this place: treat the value as uninformative."
    return _entry(key, label, status, value=val, unit=unit, metrics={"headline": head, "summary": summ, "status_text": status_txt},
                  method=str(r.get("standfirst") or status_txt)[:300], grade=grade, sources=[r.get("label") or label], attempts=[], family=fam,
                  caveat=caveat), "ok"


def _h_lab(c: Ctx, key: str) -> dict:
    from .modellab import hazards as H
    label, fam, ladder = _LAB[key]
    att = []
    last_cls = ""
    for opts in ladder:
        rung = f"Model Lab {key} study" + (f" with {opts}" if opts else "")
        t = time.time()
        try:
            if key in ("air", "flood"):
                from .modellab import lab as L
                r = L.run_study(key, c.lat, c.lon, c.name, dict(opts), write=False)["result"]
            else:
                r = H.run_generic(key, c.lat, c.lon, c.name, dict(opts))
        except Exception as e:                               # noqa: BLE001
            att.append({"rung": rung, "ok": False, "why": f"{type(e).__name__}: {str(e)[:140]}"})
            last_cls = "error"
            continue
        ent, cls = _lab_map(key, r, rung)
        if ent is not None:
            att.append({"rung": rung, "ok": True, "seconds": round(time.time() - t, 1)})
            ent["attempts"] = att
            if len(att) > 1 and ent["status"] == "modelled":
                ent["status"] = "constructed"                # first rung failed; a widened or alternative rung produced it
                ent["method"] = "Widened search after the first attempt had too little data. " + ent["method"]
            return ent
        att.append({"rung": rung, "ok": False, "why": cls.split("|", 1)[-1]})
        last_cls = cls
        if cls.startswith("other"):
            break
    return _null(key, label, fam, att, "Not enough data to build this model for this place, even after widening the search." if "data" in last_cls
                 else "The model could not be built here: " + (att[-1]["why"] if att else "unknown"))


# ── registry and runner ─────────────────────────────────────────────────────────────

def registry() -> list:
    reg = [("cyclone", "fast", _h_cyclone), ("river_flood", "fast", _h_river), ("coastal_flood", "fast", _h_coastal), ("water_stress", "fast", _h_water), ("air_local", "fast", _h_air_local)]
    reg += [(_SITE_MAP[h][0], "site", (lambda c, h=h: _site_entry(c, h))) for h in _SITE_MAP]
    reg += [(f"lab:{k}", "deep", (lambda c, k=k: _h_lab(c, k))) for k in _LAB]
    return reg


def hazard_keys() -> list:
    return [k.replace("lab:", "") for k, _, _ in registry()]


def _history(c: Ctx) -> dict:
    out: dict = {"storms": None, "twin": None, "note": "Events from the data the engine holds; not a complete record of everything that has happened here."}
    try:
        from . import asset_twin as AT
        out["storms"] = AT.storm_history(c.lat, c.lon)
    except Exception as e:                                   # noqa: BLE001
        out["storms_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    if c.twin and c.twin.get("history"):
        out["twin"] = {k: v for k, v in c.twin["history"].items() if k in ("windstorms", "observed", "incidents")}
    try:
        from .analysis import flood_events
        out["flood_events"] = flood_events.near(c.lat, c.lon)
    except Exception:                                        # noqa: BLE001
        out["flood_events"] = None
    return out



def _twin_dir() -> Path:
    home = Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))
    d = home / "twins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_twin(lat: float, lon: float, twin: dict) -> None:
    """Keep the parts of a built 3D twin that the profile can reuse (event history and routed flood/wind scenarios)."""
    keep = {"history": twin.get("history"), "scenarios": twin.get("scenarios"), "built": time.time(), "lat": lat, "lon": lon}
    (_twin_dir() / f"{lat:.3f}_{lon:.3f}.json").write_text(json.dumps(keep, default=str))


def cached_twin(lat: float, lon: float, max_age_s: float = 30 * 86400) -> Optional[dict]:
    """A twin built within ~1 km of this point in the last 30 days, or None."""
    best = None
    for f in _twin_dir().glob("*.json"):
        try:
            d = json.loads(f.read_text())
            if abs(d["lat"] - lat) <= 0.01 and abs(d["lon"] - lon) <= 0.01 and time.time() - d["built"] < max_age_s:
                best = d
        except Exception:                                    # noqa: BLE001
            continue
    return best


def _cache_path(lat: float, lon: float) -> Path:
    home = Path(os.environ.get("CRI_HOME") or (Path.home() / ".climrisk"))
    d = home / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{lat:.3f}_{lon:.3f}.json"


def profile(lat: float, lon: float, name: str = "Site", twin: Optional[dict] = None, only: Optional[list] = None, budget_s: float = 600.0,
            workers: int = 6, progress: Optional[Callable] = None, use_cache: bool = True, max_age_s: float = 7 * 86400,
            registry_override: Optional[list] = None) -> dict:
    """Run every hazard for the coordinate. Never raises for a hazard failure: a failure becomes a null entry with its reason."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("invalid coordinates")
    cp = _cache_path(lat, lon)
    if use_cache and only is None and registry_override is None and cp.exists() and time.time() - cp.stat().st_mtime < max_age_s:
        try:
            cached = json.loads(cp.read_text())
            cached["from_cache"] = True
            return cached
        except Exception:                                    # noqa: BLE001
            pass
    t0 = time.time()
    c = Ctx(lat, lon, name, twin)
    reg = registry_override if registry_override is not None else registry()
    if only:
        want = {o.replace("lab:", "") for o in only}
        reg = [r for r in reg if r[0].replace("lab:", "") in want]
    results: dict = {}
    order = {"fast": 0, "site": 1, "deep": 2}
    reg = sorted(reg, key=lambda r: order.get(r[1], 3))
    ex = ThreadPoolExecutor(max_workers=workers)
    futs = {ex.submit(lambda fn=fn, k=k: _timed(fn, c, k)): (k, tier) for k, tier, fn in reg}
    try:
        for f in as_completed(futs, timeout=budget_s):
            k, _ = futs[f]
            ent = f.result()
            results[k] = ent
            if progress:
                try:
                    progress(k, ent)
                except Exception:                            # noqa: BLE001
                    pass
    except Exception:                                        # noqa: BLE001  (timeout: whatever is still running is null)
        pass
    for f, (k, _) in futs.items():
        if k not in results:
            results[k] = _null(k.replace("lab:", ""), k, "", [{"rung": "run", "ok": False, "why": f"not finished in {budget_s:.0f} s"}],
                               "Did not finish within the time budget; run again (finished work is cached).")
    ex.shutdown(wait=False, cancel_futures=True)
    hz = {k.replace("lab:", ""): v for k, v in results.items()}
    counts = {s: sum(1 for e in hz.values() if e["status"] == s) for s in STATUSES}
    sevs = [e["severity"] for e in hz.values() if e["severity"] in SEV and e["status"] in ("modelled", "constructed")]
    out = {"location": {"lat": lat, "lon": lon, "name": name}, "hazards": hz, "counts": counts,
           "top_severity": max(sevs, key=SEV.index) if sevs else None,
           "null_hazards": sorted(k for k, e in hz.items() if e["status"] == "null"),
           "history": _history(c) if only is None else None,
           "rules": ["null means not assessed: no data and no way to construct it. It is never zero risk.",
                     "none_found means the engine established the hazard is absent here (value 0).",
                     "constructed means the primary model could not run and the value was built from a fallback; lower confidence."],
           "elapsed_s": round(time.time() - t0, 1), "from_cache": False, "used_3d_twin": bool(twin)}
    if only is None and registry_override is None and counts["null"] < len(hz):
        try:
            cp.write_text(json.dumps(out, default=str))
        except Exception:                                    # noqa: BLE001
            pass
    return out


def _timed(fn: Callable, c: Ctx, key: str) -> dict:
    t = time.time()
    try:
        e = fn(c)
    except Exception as ex_:                                 # noqa: BLE001
        e = _null(key.replace("lab:", ""), key, "", [{"rung": "run", "ok": False, "why": f"{type(ex_).__name__}: {str(ex_)[:160]}"}], "The hazard run failed.")
    e["elapsed_s"] = round(time.time() - t, 1)
    return e


def summarise(p: dict) -> str:
    """Plain-language summary that keeps null, none_found and constructed apart. Used by the resident agent."""
    hz = p["hazards"]
    def names(st):
        return [e["label"] for e in hz.values() if e["status"] == st]
    parts = []
    mod = [e for e in hz.values() if e["status"] in ("modelled", "constructed") and e["severity"] in ("MODERATE", "HIGH", "CRITICAL")]
    mod.sort(key=lambda e: -SEV.index(e["severity"]))
    if mod:
        parts.append("Material: " + "; ".join(f"{e['label']} {e['severity']}" + (" (constructed)" if e["status"] == "constructed" else "") for e in mod) + ".")
    else:
        parts.append("No hazard with a severity scale reached MODERATE on the data that loaded.")
    if names("none_found"):
        parts.append("Established absent here: " + ", ".join(names("none_found")) + ".")
    if names("constructed"):
        parts.append("Built from a fallback (lower confidence): " + ", ".join(names("constructed")) + ".")
    if names("null"):
        parts.append("NULL, not assessed (no data and nothing to construct it from; this is not low risk): " + ", ".join(names("null")) + ".")
    c = p["counts"]
    parts.append(f"{c['modelled']} modelled, {c['constructed']} constructed, {c['none_found']} none found, {c['null']} null.")
    return " ".join(parts)
