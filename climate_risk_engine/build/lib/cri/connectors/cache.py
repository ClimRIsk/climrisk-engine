"""Simple file-based cache utility for connector data."""

from pathlib import Path
import json
import time
from typing import Callable, Optional


CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "cache"


def cached_fetch(
    key: str, fetch_fn: Callable[[], dict], ttl_hours: int = 24
) -> dict:
    """Check cache first, then call fetch_fn() and cache result.

    Args:
        key: Cache key (unique identifier for this data)
        fetch_fn: Callable that returns dict of data to cache
        ttl_hours: Time-to-live in hours (default 24)

    Returns:
        Cached or freshly fetched dict data
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / f"{key}.json"
    now = time.time()

    # Check if cache exists and is fresh
    if cache_path.exists():
        try:
            stat = cache_path.stat()
            age_seconds = now - stat.st_mtime
            if age_seconds < ttl_hours * 3600:
                # Cache is fresh
                with open(cache_path, "r") as f:
                    return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    # Cache miss or stale: fetch fresh data
    data = fetch_fn()

    # Save to cache
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # Silently fail on cache write errors

    return data
