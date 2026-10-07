"""
cri/resident/autopilot.py — the resident agent working on its own, inside the engine, from the moment the app opens.

No outside AI service and no key: every step is an engine call. A background thread starts with the engine (the desktop app starts
the engine when it opens) and runs a cycle shortly after start-up and then every CRI_AUTOPILOT_MIN minutes (default 45).

Each cycle
  1  research   scan GDACS / USGS / NOAA CPC and map events onto the asset register (vicinity, never "damaged")
  2  triage     for the top event: numbers on the largest asset in range from the risk analyst
  3  sites      for every watched site that a live event can reach: build the 3D twin, run the solver, verify protection and
                run the scenario lab — the full site workflow, unattended
  4  watchlist  re-assess watched companies, compare with the last baseline, flag moves of 10% or more
  5  drafts     a LinkedIn draft for the top event (never published: publishing needs an explicit approval in the UI)
  6  briefing   one daily summary

Everything it finds lands in the inbox (/resident/inbox) and is shown when the dock or the dashboard chat opens.
Switch off with CRI_AUTOPILOT=0.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import threading
import time
import uuid
from typing import Optional

from . import memory as M

_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
_STOP = threading.Event()
MAX_INBOX = 120


def enabled() -> bool:
    return os.environ.get("CRI_AUTOPILOT", "1").strip().lower() not in ("0", "false", "off", "no")


def interval_s() -> float:
    try:
        return max(60.0, float(os.environ.get("CRI_AUTOPILOT_MIN", "45")) * 60.0)
    except ValueError:
        return 45 * 60.0


# ── inbox ────────────────────────────────────────────────────────────────────

def inbox_add(kind: str, title: str, detail: str = "", blocks: Optional[list] = None, actions: Optional[list] = None, key: Optional[str] = None) -> Optional[dict]:
    box = M.get_state("inbox", [])
    if key and any(i.get("key") == key for i in box):
        return None
    it = {"id": uuid.uuid4().hex[:8], "ts": time.time(), "kind": kind, "title": title, "detail": detail, "blocks": blocks or [], "actions": actions or [], "read": False, "key": key}
    box.insert(0, it)
    M.set_state("inbox", box[:MAX_INBOX])
    return it


def inbox(unread_only: bool = False) -> list:
    box = M.get_state("inbox", [])
    return [i for i in box if not i.get("read")] if unread_only else box


def mark_read(ids: Optional[list] = None) -> int:
    box = M.get_state("inbox", []); n = 0
    for i in box:
        if not i.get("read") and (ids is None or i["id"] in ids):
            i["read"] = True; n += 1
    M.set_state("inbox", box)
    return n


# ── watched sites ────────────────────────────────────────────────────────────

def register_site(lat: float, lon: float, name: str, sector: Optional[str] = None) -> dict:
    sites = M.get_state("sites", [])
    for s in sites:
        if abs(s["lat"] - lat) < 0.002 and abs(s["lon"] - lon) < 0.002:
            return s
    s = {"lat": round(float(lat), 5), "lon": round(float(lon), 5), "name": name or f"{lat:.3f},{lon:.3f}", "sector": sector, "added": time.time(), "last_twin": None}
    sites.append(s)
    M.set_state("sites", sites[-25:])
    return s


def sites() -> list:
    return M.get_state("sites", [])


def remove_site(name: str) -> None:
    M.set_state("sites", [s for s in sites() if s["name"].lower() != name.lower()])


# ── helpers ──────────────────────────────────────────────────────────────────

def _hav(lat0, lon0, lat1, lon1) -> float:
    p = math.pi / 180
    a = math.sin((lat1 - lat0) * p / 2) ** 2 + math.cos(lat0 * p) * math.cos(lat1 * p) * math.sin((lon1 - lon0) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))


def _usd(x) -> str:
    if x is None:
        return "n/a"
    a = abs(x)
    return f"${x / 1e9:,.1f}bn" if a >= 1e9 else f"${x / 1e6:,.1f}m" if a >= 1e6 else f"${x / 1e3:,.0f}k"


def _blk(kind: str, **kw) -> dict:
    return {"type": kind, **kw}


# ── the steps ────────────────────────────────────────────────────────────────

def _step_research(log: list) -> Optional[dict]:
    from . import research as R
    sc = R.scan()
    M.set_state("last_scan", sc)
    f = sc["feeds"]
    feeds = ", ".join(k + " " + str(v).split(" ")[0] for k, v in f.items())
    log.append(f'research: {len(sc["events"])} GDACS events; feeds {feeds}')
    if all(str(v).startswith("failed") for v in f.values()):
        inbox_add("note", "Research feeds unreachable", "GDACS, USGS and NOAA could not be reached from this machine. I will retry on the next cycle; nothing was invented in their place.", key=f"offline-{dt.date.today()}")
    ev = R.pick_story_event(sc)
    for e in [x for x in sc["events"] if x["near"]["n"] > 0][:3]:
        near = e["near"]
        rows = [[a["name"], a["country"], f'{a["dist_km"]:.0f} km', f'{(a["emissions_t"] or 0) / 1e6:.2f} Mt', f'{a["rp100_ms"]:.0f} m/s', a["screen"]] for a in near["top"][:5]]
        inbox_add("event", f'{e["title"]} ({e["alert"]}): {near["n"]} register assets in range',
                  f'{e["type"] == "TC" and "Tropical cyclone" or "Flood"} in {e.get("country") or "the region"}; {near["n"]} assets within {near["radius_km"]:.0f} km emit {near["emissions_t"] / 1e6:.1f} Mt CO₂e a year. Exposure in the vicinity, not damage.',
                  [_blk("table", head=["Asset", "Country", "From centre", "CO₂e/yr", "1-in-100 wind", "Screen"], rows=rows)],
                  [{"type": "open", "label": "Open 3D twin at event centre", "url": f'/asset-twin/dashboard?lat={e["lat"]:.4f}&lon={e["lon"]:.4f}&auto=1&name=' + e["title"].replace(" ", "%20")}],
                  key=f'event-{e["id"]}-{e["alert"]}')
    return ev and {"sc": sc, "ev": ev}


def _step_triage(ctx: Optional[dict], log: list) -> None:
    if not ctx or not ctx["ev"]["near"]["top"]:
        return
    from ..finance import analyst as AN
    ev = ctx["ev"]; a = ev["near"]["top"][0]
    r = AN.analyse_asset(a["lat"], a["lon"], n=500)
    ph = r["financial"]["physical"]; cy = r["physical"]["cyclone"]
    d = r["financial"]["scenarios"].get("Delayed transition") or next(iter(r["financial"]["scenarios"].values()))
    log.append(f'triage: {a["name"]} assessed')
    inbox_add("triage", f'Largest asset near {ev["title"]}: {a["name"]}',
              f'{a["name"]} ({a["country"]}), {a["dist_km"]:.0f} km from the event centre. Cyclone screening {cy["severity"]}, 1-in-100 wind {cy["rp100_ms"]:.0f} m/s ({cy["category_100yr"]}); modelled average annual physical damage {ph["aal_pct"]:.2f}% of value. '
              f'Expected climate loss to 2050 under a delayed transition: {_usd(d["total_npv"]["mean"])} (95th percentile {_usd(d["total_npv"]["p95"])}); value basis: {r["asset"]["value"]["basis"]}.',
              [], [{"type": "say", "label": "Run the 3D twin and scenario lab here", "text": f'protect the site at {a["lat"]:.4f}, {a["lon"]:.4f}'}],
              key=f'triage-{ev["id"]}-{a["i"]}')


def _step_sites(ctx: Optional[dict], log: list) -> None:
    ss = sites()
    if not ss or not ctx:
        return
    from . import agent as A
    sc = ctx["sc"]
    for s in ss:
        hit = None
        for e in sc["events"]:
            reach = max(e["near"].get("radius_km", 100.0) * 2.0, 200.0)
            d = _hav(s["lat"], s["lon"], e["lat"], e["lon"])
            if d <= reach and (hit is None or d < hit[0]):
                hit = (d, e)
        if not hit:
            continue
        d, e = hit
        key = f'site-{s["name"]}-{e["id"]}'
        if any(i.get("key") == key for i in inbox()):
            continue
        ag = A.get_agent()
        tw = ag.twin_builder(lat=s["lat"], lon=s["lon"], name=s["name"], sector=s.get("sector"))
        an = ag.analyser(tw)
        lab = ag.lab(tw, an) if an["findings"] else None
        fnd = an["findings"]
        log.append(f'sites: {s["name"]} twin + scenario lab ({len(fnd)} findings)')
        blocks = []
        if fnd:
            blocks.append(_blk("list", items=[f["title"] for f in fnd[:6]]))
        if lab and lab.get("solutions"):
            blocks.append(_blk("table", head=["Solution", "Resolves base", "Flood margin", "Wind margin"],
                               rows=[[x["name"], "yes" if x["resolves_base"] else "no", f'+{x["flood_margin_m"]} m' if x["flood_margin_m"] is not None else "–", f'+{x["wind_margin_pct"]}%' if x["wind_margin_pct"] is not None else "–"] for x in lab["solutions"][:5]]))
            blocks.append(_blk("note", text=lab["recommendation"]))
        inbox_add("site", f'{s["name"]}: {e["title"]} is {d:.0f} km away — twin re-run',
                  f'{e["title"]} ({e["alert"]}) is within reach of {s["name"]}. I rebuilt the 3D twin, ran the hydraulic solver and the scenario lab without being asked: {len(fnd)} finding(s).',
                  blocks, [{"type": "open", "label": "Open the 3D twin", "url": f'/asset-twin/dashboard?lat={s["lat"]:.5f}&lon={s["lon"]:.5f}&auto=1&name=' + s["name"].replace(" ", "%20")},
                           {"type": "open", "label": "Open the physical story report", "url": f'/story-reports/dashboard?type=physical&lat={s["lat"]:.5f}&lon={s["lon"]:.5f}&name=' + s["name"].replace(" ", "%20") + "&auto=1"}],
                  key=key)
        break                                                    # one full twin per cycle keeps the machine responsive


def _step_watchlist(log: list) -> None:
    try:
        from ..api.agents import _load_watchlist
        wl = _load_watchlist()
    except Exception:                                            # noqa: BLE001
        wl = []
    if not wl:
        return
    from ..data.asset_register import search
    from ..finance import analyst as AN
    base = M.get_state("watch_baseline", {})
    done = 0
    for w in wl:
        name = w["company_name"]
        last = base.get(name) or {}
        if time.time() - last.get("ts", 0) < 20 * 3600 or done >= 3:
            continue
        res = search(name, limit=10)
        if not res:
            base[name] = {"ts": time.time(), "mean": None}
            continue
        keys = [r["key"] for r in res if r["match"] == "group"] or [res[0]["key"]]
        r = AN.analyse_company(keys, top_n=8, n=500)
        d = r["scenarios"].get("Delayed transition") or next(iter(r["scenarios"].values()))
        mean = d["total_npv"]["mean"]; val = r.get("portfolio_value_mode")
        done += 1
        log.append(f"watchlist: {name} reassessed")
        if last.get("mean"):
            chg = (mean - last["mean"]) / last["mean"]
            if abs(chg) >= 0.10:
                inbox_add("watch", f'{name}: expected loss moved {chg:+.0%} since the last check',
                          f'Delayed-transition expected loss {_usd(last["mean"])} → {_usd(mean)} (95th percentile {_usd(d["total_npv"]["p95"])}). {r["n_assets"]} register assets.', [], [], key=f'watch-{name}-{dt.date.today()}')
        else:
            inbox_add("watch", f'{name}: baseline recorded',
                      f'{r["n_assets"]} register assets, value about {_usd(val)}. Expected loss to 2050 under a delayed transition {_usd(mean)} (95th percentile {_usd(d["total_npv"]["p95"])}). I will tell you when it moves by 10% or more.',
                      [], [{"type": "say", "label": "Write the financial story", "text": f"write the financial report for {name}"}], key=f"watchbase-{name}")
        base[name] = {"ts": time.time(), "mean": mean}
    M.set_state("watch_baseline", base)


def _step_draft(ctx: Optional[dict], log: list) -> None:
    if not ctx:
        return
    from . import research as R
    ev, sc = ctx["ev"], ctx["sc"]
    if any((p.get("meta") or {}).get("event") == ev["id"] for p in [M.get_post(x["id"]) for x in M.posts(30)] if p):
        return
    story = R.build_story(sc, ev)
    post = R.post_from_story(story, ev, sc)
    pid = M.add_post(post["title"], post["text"], post["image"], {"from": "autopilot", "event": ev["id"]})
    M.set_state("pending_post", pid)
    log.append("drafts: LinkedIn post drafted")
    inbox_add("draft", f'Draft LinkedIn post ready: {ev["title"]}', "Written from the research scan. Nothing is published until you approve it.",
              [_blk("post", id=pid, text=post["text"], has_image=True, image_url=f"/resident/post/{pid}/image")],
              [{"type": "publish", "label": "Approve & publish to LinkedIn", "post_id": pid}, {"type": "copy", "label": "Copy text", "post_id": pid}], key=f'draft-{ev["id"]}')


def _step_briefing(log: list) -> None:
    key = f"brief-{dt.date.today()}"
    if any(i.get("key") == key for i in inbox()):
        return
    sc = M.get_state("last_scan") or {"events": []}
    n_near = sum(1 for e in sc["events"] if e["near"]["n"] > 0)
    enso = (sc.get("enso") or {}).get("text")
    inbox_add("briefing", f'Daily briefing · {dt.date.today().strftime("%d %B")}',
              f'{len(sc["events"])} live hazard events; {n_near} have register assets in range. ' + (f"ENSO: {enso}. " if enso else "") +
              f'{len(sites())} watched site(s), {len(M.get_state("watch_baseline", {}))} watched company(ies), {len(M.posts(100))} post draft(s) on file.', [], [], key=key)


# ── the cycle ────────────────────────────────────────────────────────────────

def status() -> dict:
    s = M.get_state("autopilot", {}) or {}
    return {"enabled": enabled(), "running": _LOCK.locked(), "interval_min": round(interval_s() / 60), "last_run": s.get("last_run"), "next_run": s.get("next_run"), "cycles": s.get("cycles", 0),
            "log": s.get("log", [])[-12:], "unread": len(inbox(True)), "sites": len(sites()), "thread_alive": bool(_THREAD and _THREAD.is_alive())}


def run_cycle(reason: str = "scheduled") -> dict:
    if not _LOCK.acquire(blocking=False):
        return {"ok": False, "note": "a cycle is already running"}
    log: list = []
    try:
        t0 = time.time()
        ctx = None
        for name, fn in (("research", lambda: _step_research(log)),):
            try:
                ctx = fn()
            except Exception as e:                               # noqa: BLE001
                log.append(f"{name} failed: {type(e).__name__}: {str(e)[:100]}")
        for name, fn in (("triage", lambda: _step_triage(ctx, log)), ("sites", lambda: _step_sites(ctx, log)), ("watchlist", lambda: _step_watchlist(log)),
                         ("drafts", lambda: _step_draft(ctx, log)), ("briefing", lambda: _step_briefing(log))):
            try:
                fn()
            except Exception as e:                               # noqa: BLE001
                log.append(f"{name} failed: {type(e).__name__}: {str(e)[:100]}")
        s = M.get_state("autopilot", {}) or {}
        now = time.time()
        s.update({"last_run": now, "next_run": now + interval_s(), "cycles": s.get("cycles", 0) + 1, "log": (s.get("log", []) + [f'{dt.datetime.utcnow():%H:%M} {reason}: ' + " | ".join(log)])[-30:]})
        M.set_state("autopilot", s)
        M.add_run("autopilot", {"reason": reason}, log, True, f'cycle in {time.time() - t0:.0f}s: ' + "; ".join(log)[:300])
        return {"ok": True, "elapsed_s": round(time.time() - t0, 1), "log": log}
    finally:
        _LOCK.release()


def _loop() -> None:
    if _STOP.wait(float(os.environ.get("CRI_AUTOPILOT_DELAY_S", "12"))):
        return
    last = (M.get_state("autopilot", {}) or {}).get("last_run") or 0
    reason = "app opened"
    if time.time() - last < 600:                                 # opened twice within ten minutes: do not repeat the scan
        reason = None
    while not _STOP.is_set():
        try:
            if reason:
                run_cycle(reason)
        except Exception:                                        # noqa: BLE001
            pass
        reason = "scheduled"
        if _STOP.wait(interval_s()):
            break


def start() -> bool:
    """Idempotent; called from the engine's start-up hook."""
    global _THREAD
    if not enabled() or (_THREAD and _THREAD.is_alive()):
        return False
    _STOP.clear()
    _THREAD = threading.Thread(target=_loop, name="cri-autopilot", daemon=True)
    _THREAD.start()
    return True


def stop() -> None:
    _STOP.set()
