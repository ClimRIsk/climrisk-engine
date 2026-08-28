"""Climate data connectors for CRI engine.

Provides abstract base class and implementations for fetching data from
open-source climate and energy datasets:
- WRI Aqueduct: water risk indicators
- NGFS: carbon price scenarios
- Our World in Data: energy and commodity demand
"""

from .base import BaseConnector
from .wri_aqueduct import WRIAqueductConnector
from .ngfs import NGFSConnector
from .owid import OWIDConnector
from .cache import cached_fetch

__all__ = [
    "BaseConnector",
    "WRIAqueductConnector",
    "NGFSConnector",
    "OWIDConnector",
    "cached_fetch",
]
