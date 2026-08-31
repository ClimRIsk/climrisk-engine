"""Base class for climate data connectors."""

from abc import ABC, abstractmethod
from pathlib import Path
import json
from typing import Optional


class BaseConnector(ABC):
    """Abstract base for climate data fetchers with built-in file cache."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            # Default cache location
            cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self, **kwargs) -> dict:
        """Fetch data from the source, using cache if available.

        Subclasses must implement this to return a dict of results.
        """
        pass

    def _cache_path(self, key: str) -> Path:
        """Return the cache file path for a given key."""
        # Sanitize key to be safe as filename
        safe_key = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
        return self.cache_dir / f"{safe_key}.json"

    def _load_cache(self, key: str) -> Optional[dict]:
        """Load data from cache if it exists."""
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_cache(self, key: str, data: dict) -> None:
        """Save data to cache."""
        path = self._cache_path(key)
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass  # Silently fail on cache write errors
