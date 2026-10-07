"""
cri/resident/linkedin.py — draft storage and approval-gated publishing to LinkedIn.

Publishing uses LinkedIn's own REST API directly (no third-party service). It needs two values in the engine's environment
(.env): LINKEDIN_ACCESS_TOKEN (w_member_social, or w_organization_social for a page) and LINKEDIN_AUTHOR_URN
(urn:li:person:… or urn:li:organization:…). Nothing is posted unless publish() is called with approved=True, and the dock only
does that from an explicit "Approve & publish" click on the previewed text and image. Without credentials the draft is kept and
the text/image can be copied or downloaded — drafting never depends on LinkedIn.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests

from . import memory as M

API = "https://api.linkedin.com"
VERSION = os.environ.get("LINKEDIN_API_VERSION", "202405")


def configured() -> bool:
    return bool(os.environ.get("LINKEDIN_ACCESS_TOKEN") and os.environ.get("LINKEDIN_AUTHOR_URN"))


def status() -> dict:
    return {"configured": configured(), "author": (os.environ.get("LINKEDIN_AUTHOR_URN") or "")[:40] or None,
            "how_to": None if configured() else "Add LINKEDIN_ACCESS_TOKEN and LINKEDIN_AUTHOR_URN to the engine .env to enable publishing. Drafts work without them."}


def _h(extra: Optional[dict] = None) -> dict:
    h = {"Authorization": f"Bearer {os.environ['LINKEDIN_ACCESS_TOKEN']}", "LinkedIn-Version": VERSION, "X-Restli-Protocol-Version": "2.0.0"}
    h.update(extra or {})
    return h


def _upload_image(png: bytes, author: str) -> str:
    r = requests.post(f"{API}/rest/images?action=initializeUpload", headers=_h({"Content-Type": "application/json"}),
                      json={"initializeUploadRequest": {"owner": author}}, timeout=30)
    r.raise_for_status()
    v = r.json()["value"]
    u = requests.put(v["uploadUrl"], data=png, headers={"Authorization": f"Bearer {os.environ['LINKEDIN_ACCESS_TOKEN']}"}, timeout=60)
    u.raise_for_status()
    return v["image"]


def publish(post_id: int, approved: bool = False) -> dict:
    """Publish a stored draft. Refuses unless approved=True, the draft exists, and it has not been published already."""
    p = M.get_post(post_id)
    if not p:
        return {"ok": False, "error": "no such draft"}
    if not approved:
        return {"ok": False, "error": "approval required — nothing was posted", "needs_approval": True}
    if p["status"] == "published":
        return {"ok": False, "error": "already published", "urn": p.get("urn")}
    if not configured():
        return {"ok": False, "error": "LinkedIn credentials are not set in the engine .env", "status": status()}
    author = os.environ["LINKEDIN_AUTHOR_URN"]
    try:
        body: dict = {"author": author, "commentary": p["text"], "visibility": "PUBLIC",
                      "distribution": {"feedDistribution": "MAIN_FEED", "targetEntities": [], "thirdPartyDistributionChannels": []},
                      "lifecycleState": "PUBLISHED", "isReshareDisabledByAuthor": False}
        if p.get("image"):
            body["content"] = {"media": {"id": _upload_image(p["image"], author), "altText": (p["title"] or "ClimRisk")[:120]}}
        r = requests.post(f"{API}/rest/posts", headers=_h({"Content-Type": "application/json"}), json=body, timeout=30)
        if r.status_code not in (200, 201):
            M.update_post(post_id, status="failed", meta={**p["meta"], "error": r.text[:300]})
            return {"ok": False, "error": f"LinkedIn returned {r.status_code}: {r.text[:200]}"}
        urn = r.headers.get("x-restli-id") or r.headers.get("X-RestLi-Id") or ""
        M.update_post(post_id, status="published", published_ts=time.time(), urn=urn)
        return {"ok": True, "urn": urn, "url": f"https://www.linkedin.com/feed/update/{urn}" if urn else None}
    except Exception as e:                                       # noqa: BLE001
        M.update_post(post_id, status="failed", meta={**p["meta"], "error": str(e)[:300]})
        return {"ok": False, "error": str(e)[:300]}
