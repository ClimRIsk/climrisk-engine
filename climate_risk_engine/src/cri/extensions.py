"""Optional private extensions, loaded only on a machine that lists them (env CRI_EXTENSIONS, or lines in ~/.climrisk/extensions).
Each extension is a module that may define make_router(auth_dep) -> iterable of FastAPI routers and/or handle(session, text) -> dict | None (a chat command).
A commercial build ships without the extension modules and without that file, so nothing is loaded."""
from __future__ import annotations

import importlib
import os
from pathlib import Path


def names() -> list:
    env = os.environ.get("CRI_EXTENSIONS", "")
    if env.strip():
        return [x.strip() for x in env.split(",") if x.strip()]
    p = Path.home() / ".climrisk" / "extensions"
    try:
        return [x.strip() for x in p.read_text().splitlines() if x.strip() and not x.startswith("#")]
    except Exception:                                              # noqa: BLE001
        return []


def modules() -> list:
    out = []
    for n in names():
        try:
            out.append(importlib.import_module(n))
        except Exception:                                          # noqa: BLE001
            continue
    return out
