"""
db.py — Supabase persistence layer for ClimRisk company registry.
=================================================================
Provides a thin async-optional wrapper around the Supabase Python client.

Connection is optional: if SUPABASE_URL / SUPABASE_SERVICE_KEY env vars
are not set, all operations silently no-op and the engine runs on the
in-memory seed data.  Set both vars to enable persistence.

Supabase table DDL (run once in the SQL editor):
─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS companies (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    sector      TEXT,
    hq_region   TEXT,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Optional: auto-update updated_at
CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER companies_updated_at
BEFORE UPDATE ON companies
FOR EACH ROW EXECUTE PROCEDURE _set_updated_at();

Row-level security (recommended for production):
    ALTER TABLE companies ENABLE ROW LEVEL SECURITY;
    CREATE POLICY "service_role_full_access" ON companies
        USING (true) WITH CHECK (true);
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.schemas import Company

logger = logging.getLogger(__name__)

_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
_SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
_TABLE = "companies"

_client = None          # supabase.Client — lazy init
_init_attempted = False


def _get_client():
    """Lazy-init the Supabase client. Returns None if not configured."""
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True

    if not _SUPABASE_URL or not _SUPABASE_KEY:
        logger.info(
            "Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_KEY not set). "
            "Running with in-memory company registry only."
        )
        return None

    try:
        from supabase import create_client  # type: ignore
        _client = create_client(_SUPABASE_URL, _SUPABASE_KEY)
        logger.info("Supabase client initialised: %s", _SUPABASE_URL)
        return _client
    except ImportError:
        logger.warning(
            "supabase-py not installed — run: pip install supabase. "
            "Falling back to in-memory registry."
        )
    except Exception as exc:
        logger.warning("Supabase init failed: %s. Falling back to in-memory.", exc)
    return None


def is_connected() -> bool:
    """Return True if a Supabase client is configured and initialised."""
    return _get_client() is not None


def load_companies() -> dict[str, "Company"]:
    """
    Fetch all company records from Supabase.

    Returns a dict {company_id: Company} on success, or {} if Supabase is
    not configured / unreachable.  Never raises — callers always get a dict.
    """
    client = _get_client()
    if client is None:
        return {}

    try:
        from ..data.schemas import Company  # local import to avoid circular
        response = client.table(_TABLE).select("*").execute()
        companies: dict[str, Company] = {}
        for row in (response.data or []):
            try:
                company = Company.model_validate(row["data"])
                companies[company.id] = company
            except Exception as exc:
                logger.warning(
                    "Skipping malformed company row id=%s: %s",
                    row.get("id"), exc,
                )
        logger.info("Loaded %d companies from Supabase.", len(companies))
        return companies
    except Exception as exc:
        logger.warning("Failed to load companies from Supabase: %s", exc)
        return {}


def save_company(company: "Company") -> bool:
    """
    Upsert a company to Supabase.

    Returns True on success, False if not connected or on error.
    """
    client = _get_client()
    if client is None:
        return False

    try:
        row = {
            "id":        company.id,
            "name":      company.name,
            "sector":    company.sector,
            "hq_region": company.hq_region,
            "data":      company.model_dump(),
        }
        client.table(_TABLE).upsert(row).execute()
        logger.info("Saved company %s to Supabase.", company.id)
        return True
    except Exception as exc:
        logger.warning("Failed to save company %s to Supabase: %s", company.id, exc)
        return False


def delete_company(company_id: str) -> bool:
    """
    Delete a company from Supabase by id.

    Returns True on success, False if not connected or on error.
    """
    client = _get_client()
    if client is None:
        return False

    try:
        client.table(_TABLE).delete().eq("id", company_id).execute()
        logger.info("Deleted company %s from Supabase.", company_id)
        return True
    except Exception as exc:
        logger.warning(
            "Failed to delete company %s from Supabase: %s", company_id, exc
        )
        return False
