"""
Port resolution that shows its work and asks when unsure.

Why: the country of each port decides whether EU ETS / FuelEU apply, and free-text geocoding is unreliable for ports. Measured against the live OpenStreetMap geocoder on 28 Oct 2026:
"port of Shanghai" returned a massage shop in the UK, "port of Hamburg" a restaurant, "port of Mumbai" a government office and "port of Santos" a harbour in the Philippines. Searching the place name
("Shanghai", "Hamburg") returned the right city every time, but "Valencia" and "Newcastle" each return cities in several countries, and "Santos" also returns a village in Hungary.

Rules:
  1. Search the place name (with the country if you gave one: "Valencia, Spain"). Port-feature hits ("port of X") are used only when the hit is a real harbour/port feature, never a shop, office or building.
  2. Keep place candidates (cities, towns, administrative areas) and port features; drop streets, buildings, amenities. One candidate per country; weak ones (importance < 20% of the best) are dropped.
  3. If one country remains: resolved. If several remain, ask for port evidence (is there a harbour/port feature within 15 km?). Exactly one with evidence: resolved and says why. Otherwise: AMBIGUOUS, the caller must ask. Nothing is guessed.
  4. Every resolved port comes back with the feature matched, its country, its coordinates and whether port evidence was found, so the reply can show it. No evidence is a warning, not a block.
"""
from __future__ import annotations

from typing import Callable, Optional

PORT_FEATURES = {("landuse", "harbour"), ("landuse", "port"), ("harbour", "yes"), ("man_made", "pier"), ("man_made", "quay"), ("amenity", "ferry_terminal"), ("waterway", "dock"), ("seamark:type", "harbour")}
PLACE_TYPES = {"city", "town", "municipality", "administrative", "village", "island"}
WEAK = 0.2
EVIDENCE_RADIUS_KM = 15
DOMINANCE = 1.5            # top place must be this many times more prominent than the runner-up from another country to be chosen without asking


def _kind(c: dict) -> Optional[str]:
    cl, ty = (c.get("class") or "").lower(), (c.get("type") or "").lower()
    if (cl, ty) in PORT_FEATURES or cl == "harbour":
        return "port"
    if cl in ("place", "boundary") and ty in PLACE_TYPES:
        return "place"
    return None


def _name_hit(c: dict, name: str) -> bool:
    return name.lower() in (c.get("name") or "").lower()


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _view(c: dict, evidence: Optional[bool]) -> dict:
    k = _kind(c)
    return {"name": c.get("name"), "lat": float(c["lat"]), "lon": float(c["lon"]), "country_code": (c.get("country_code") or "").upper() or None, "country": c.get("country"),
            "feature": f"{c.get('class')}/{c.get('type')}", "is_port_feature": k == "port", "port_evidence": evidence, "source": "OpenStreetMap Nominatim"}


def resolve(name: str, country_hint: Optional[str], search: Callable[[str], list], evidence: Optional[Callable[[float, float], Optional[bool]]] = None) -> dict:
    """search(query) -> [{lat, lon, name, country_code, country, class, type, importance}]; evidence(lat, lon) -> True/False/None (None = could not check).
    Returns {status: ok|ambiguous|not_found, port, candidates, notes}."""
    notes = []
    queries = [f"{name}, {country_hint}" if country_hint else name, f"port of {name}" + (f", {country_hint}" if country_hint else "")]
    pool = []
    for q in queries:
        for c in search(q) or []:
            if c.get("lat") is None:
                continue
            k = _kind(c)
            if k == "port" and not _name_hit(c, name) and "port" not in (c.get("name") or "").lower():
                continue
            if k is None:
                continue
            pool.append(c)
    if country_hint:
        h = _norm(country_hint)
        filt = [c for c in pool if _norm(c.get("country")) == h or _norm(c.get("country_code")) == h]
        if filt:
            pool = filt
        elif pool:
            notes.append(f"No candidate matched the country '{country_hint}'; showing the best matches anywhere.")
    if not pool:
        return {"status": "not_found", "port": None, "candidates": [], "notes": notes}
    top_imp = max((c.get("importance") or 0) for c in pool) or 1e-9
    by_cc = {}
    for c in sorted(pool, key=lambda c: (-(1 if _kind(c) == "port" else 0), -(c.get("importance") or 0))):
        cc = (c.get("country_code") or "").lower()
        if cc not in by_cc and ((c.get("importance") or 0) >= WEAK * top_imp or _kind(c) == "port"):
            by_cc[cc] = c
    cands = list(by_cc.values())
    ev = {}
    def check(c):
        k = id(c)
        if k not in ev:
            ev[k] = evidence(float(c["lat"]), float(c["lon"])) if evidence else None
        return ev[k]
    if len(cands) == 1:
        c = cands[0]
        e = check(c) if _kind(c) == "place" else True
        if e is False:
            notes.append(f"No harbour or port feature was found within {EVIDENCE_RADIUS_KM} km of this match: it may not be a seaport. Check the country and name.")
        elif e is None and _kind(c) == "place":
            notes.append("Matched a place (city/town); port evidence could not be checked. The coordinates may be inland of the actual quay.")
        return {"status": "ok", "port": _view(c, e), "candidates": [], "notes": notes}
    ordered = sorted(cands, key=lambda c: -(c.get("importance") or 0))
    if len(ordered) >= 2 and (ordered[1].get("importance") or 0) > 0 and (ordered[0].get("importance") or 0) >= DOMINANCE * (ordered[1].get("importance") or 0) and _kind(ordered[0]) == "place":
        c = ordered[0]
        e = check(c)
        if e is not False:
            notes.append(f"Several countries share this name; chose {c.get('country')} as the clearly most prominent place of that name (OpenStreetMap importance {c.get('importance'):.2f} against {ordered[1].get('importance'):.2f} for {ordered[1].get('country')}). "
                         f"Write '{name}, {ordered[1].get('country')}' if you meant that one.")
            return {"status": "ok", "port": _view(c, e), "candidates": [], "notes": notes}
    flags = {id(c): (True if _kind(c) == "port" else check(c)) for c in cands}
    with_ev = [c for c in cands if flags[id(c)] is True]
    if len(with_ev) == 1 and any(flags[id(c)] is False for c in cands if c is not with_ev[0]):
        c = with_ev[0]
        notes.append(f"Several countries share this name; chose {c.get('country')} because it is the only one with a harbour or port feature within {EVIDENCE_RADIUS_KM} km.")
        return {"status": "ok", "port": _view(c, True), "candidates": [], "notes": notes}
    return {"status": "ambiguous", "port": None, "candidates": [_view(c, flags[id(c)]) for c in cands[:5]], "notes": notes}
