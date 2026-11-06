"""Status of the standards and rules the engine's methods follow (GHG Protocol, ISO 14067, IFRS S2, ESRS, CBAM, PCAF).

The facts live in data/standards_watch.json (built and checked by scripts/build_standards_watch.py, each read from the publisher's own page).
The live part is in connectors/source_monitor.py: every item has one or more monitors that read the publisher's page and say whether it now
reads differently from what the engine recorded. This module joins the two: what the standard's status is, what the engine implements, and
whether the publisher's page has moved since.

No network here. Paywalled texts (ISO) are never stored; only edition, stage and dates.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_FILE = Path(__file__).resolve().parent.parent / "data" / "standards_watch.json"
_ENGINE = {"matches": "the engine implements the edition in force",
           "partly": "the engine implements an earlier text or part of it",
           "behind": "the engine is behind the current edition"}


@lru_cache(maxsize=1)
def data() -> dict:
    return json.loads(_FILE.read_text())


def items() -> list:
    return data()["items"]


def item(item_id: str) -> Optional[dict]:
    return next((i for i in items() if i["id"] == item_id), None)


def item_for_monitor(monitor_id: str) -> Optional[dict]:
    return next((i for i in items() if monitor_id in i["monitors"]), None)


def monitor_fingerprint(monitor_id: str) -> dict:
    it = item_for_monitor(monitor_id)
    if not it:
        raise KeyError(monitor_id)
    return it["monitors"][monitor_id]


def upcoming(it: dict) -> list:
    return [d for d in it["key_dates"] if not d["done"]]


def summary(item_id: str) -> str:
    """One paragraph: status, what is next, what the engine implements. Used by the agent and the knowledge index."""
    it = item(item_id)
    if not it:
        return ""
    eng = it["engine"]
    nxt = "; ".join(f"{d['date']}: {d['event']}" for d in upcoming(it))
    parts = [f"{it['name']} ({it['body']}). Status as read on {data()['as_of']}: {it['headline']}"]
    if nxt:
        parts.append(f"Upcoming: {nxt}.")
    parts.append(f"In the engine ({eng['where']}): {eng['holds']}. {_ENGINE[eng['state']].capitalize()}. {eng['note']}")
    parts.append("Not assessed: " + "; ".join(it["not_assessed"]) + ".")
    return " ".join(parts)


def overview() -> dict:
    """Machine-readable view for the API: every item with its engine state, plus the monitors' live freshness."""
    from . import source_monitor as SM
    st = {r["id"]: r for r in SM.status()["sources"]}
    out = []
    for it in items():
        mons = [{"id": m, "freshness": st[m]["freshness"], "message": st[m]["message"]} for m in it["monitors"] if m in st]
        out.append({"id": it["id"], "name": it["name"], "body": it["body"], "status": it["status"], "status_meaning": data()["status_vocabulary"][it["status"]],
                    "headline": it["headline"], "key_dates": it["key_dates"], "engine": it["engine"], "not_assessed": it["not_assessed"],
                    "sources": it["sources"], "monitors": mons})
    return {"as_of": data()["as_of"], "items": out, "note": data()["note"]}


def knowledge_text() -> str:
    return "== Standards and regulations the engine follows ==\n" + "\n\n".join(summary(i["id"]) for i in items())


def ids() -> list:
    return [i["id"] for i in items()]
