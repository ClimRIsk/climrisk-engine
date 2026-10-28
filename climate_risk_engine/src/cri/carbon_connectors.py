"""
Registry connectors for the carbon-credit screen: automatic pickup of registry data without per-lot manual upload.

Why connectors and not a built-in scraper. None of the three registries that matter for legacy and voluntary credits publishes an open,
documented per-unit API that we could confirm: the CDM Registry publishes aggregate statistics as periodic PDFs; the Gold Standard Impact
Registry offers a CSV export in its web interface; Verra's registry terms of use govern access and we have not seen a bulk-data or API
provision. The open aggregate (CarbonPlan OffsetsDB) has paused updates and does not cover the CDM. Reading a registry's web pages
automatically could breach its terms, so this module does not do it.

What it does instead: connect to a source the operator is entitled to use, and keep the data fresh on a schedule.
  FileDrop   a folder or HTTPS URL that receives registry exports (for example a platform's own nightly export, a data licence delivery,
             or a file an analyst saves once and the schedule re-reads).
  HttpJson   a JSON API the operator is authorised to call (a registry data licence, a commercial carbon-data provider, or the operator's
             own platform), with the token read from an environment variable and a field map from the API's names to ours.
Every sync result carries its source, the time it was fetched and a content hash, so a screen is always labelled with how old its data is.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

from . import carbon_registry as CR
from .carbon_markets import screen_credits

_UA = {"User-Agent": "ClimRisk-connector/1.0"}


def _get(url: str, token_env: Optional[str], timeout: float = 30.0) -> bytes:
    if not url.lower().startswith("https://"):
        raise ValueError("connectors only fetch https:// addresses")
    h = dict(_UA)
    if token_env:
        tok = os.environ.get(token_env)
        if not tok:
            raise ValueError(f"environment variable {token_env} is not set")
        h["Authorization"] = "Bearer " + tok
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read(50_000_000)


class FileDrop:
    kind = "file_drop"

    def __init__(self, source: str, programme: Optional[str] = None, token_env: Optional[str] = None):
        self.source, self.programme, self.token_env = source, programme, token_env

    def fetch(self) -> tuple:
        if self.source.lower().startswith("http"):
            raw = _get(self.source, self.token_env)
            name = self.source
        else:
            p = Path(self.source)
            if p.is_dir():
                files = sorted(p.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
                if not files:
                    raise FileNotFoundError(f"no .csv files in {p}")
                p = files[0]
            raw, name = p.read_bytes(), str(p)
        return CR.import_csv(raw.decode("utf-8-sig", errors="replace"), self.programme), raw, name


class HttpJson:
    kind = "http_json"

    def __init__(self, url: str, programme: Optional[str] = None, token_env: Optional[str] = None,
                 records_path: str = "", field_map: Optional[dict] = None):
        self.url, self.programme, self.token_env, self.records_path = url, programme, token_env, records_path
        self.field_map = field_map or {}                      # API field name -> column name the importer knows (see carbon_registry.ALIASES)

    def fetch(self) -> tuple:
        raw = _get(self.url, self.token_env)
        data = json.loads(raw)
        for part in [p for p in self.records_path.split(".") if p]:
            data = data[part]
        if not isinstance(data, list):
            raise ValueError("records_path does not point at a list of records")
        rows = [{self.field_map.get(k, k): v for k, v in rec.items()} for rec in data]
        return CR.import_rows(rows, self.programme), raw, self.url


def sync(connector, current_year: int = 2026, screen: bool = True) -> dict:
    """Fetch, import and screen. The result says where the data came from and when, and never claims the registry itself was checked."""
    imp, raw, name = connector.fetch()
    out = {**imp, "provenance": {"connector": connector.kind, "source": name, "programme": connector.programme,
                                 "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sha256": hashlib.sha256(raw).hexdigest(),
                                 "n_records": len(imp["lots"]),
                                 "statement": "Data as delivered by this source at the time shown. Not verified against the registry by ClimRisk."}}
    if screen and imp["lots"]:
        out["screen"] = screen_credits(imp["lots"], current_year)
    return out
