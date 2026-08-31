"""
CRI Meteorological Data Layer.

Provides observed-climate baselines (temperature, precipitation, extreme-event
statistics) used to calibrate the physical hazard engine.  Designed around a
provider abstraction so the open-source baseline can be swapped for licensed
data (Climate X, Continuuiti, NOAA CDO API, ERA5 live pull, etc.) with no
changes to the hazard engine itself.

Architecture
------------
  MetProvider (ABC)                    ← plug any provider in here
  ├── StaticWMOProvider                ← WMO 1991-2020 normals, always available
  ├── NOAAGHCNProvider                 ← fetches via NOAA CDO API (needs API key)
  ├── ERA5CDS Provider                 ← fetches via Copernicus CDS API (needs key)
  └── BlendedMetProvider              ← merges multiple providers with priority

How to swap in licensed data
----------------------------
  1. Implement ``MetProvider`` (5-method ABC).
  2. Set it as the active provider:
       from cri.climate.met_data import set_active_provider, MyProvider
       set_active_provider(MyProvider(api_key="..."))
  3. The hazard engine calls ``get_active_provider()`` and your data flows
     through automatically — no other changes required.

Data versioning
---------------
Each ``MetBaseline`` carries a ``data_version`` and ``reference_period`` so
the engine can detect stale data and log when the underlying climate normals
were last refreshed.

Open-source data sources
------------------------
  - WMO 1991-2020 Climatological Normals (temperature, precipitation)
    https://climaax.atlassian.net/wiki/spaces/EH/pages/51183631
  - NOAA Global Historical Climatology Network (GHCN-M v4)
    https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly
  - ECMWF ERA5 (CDS API, free registration required)
    https://cds.climate.copernicus.eu/api-how-to
  - NASA NEX-GDDP-CMIP6 (via AWS Open Data)
    https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6
  - CHIRPS v2.0 (precipitation, 0.05° grid, 1981–present)
    https://www.chc.ucsb.edu/data/chirps
  - WRI Aqueduct 4.0 (water stress, riverine/coastal flood, drought)
    https://www.wri.org/aqueduct
"""

from __future__ import annotations

import datetime
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Data version tracking
# ---------------------------------------------------------------------------

DATA_VERSION = "1.1.0"
"""Increment this string whenever underlying baseline data is updated.

Format: MAJOR.MINOR.PATCH
  MAJOR: source dataset replaced or methodology changed (breaks backwards compat)
  MINOR: new regions added or existing normals refreshed
  PATCH: bug-fix corrections to individual data points

History
-------
  1.0.0 (2025-01-01): Initial WMO 1981-2010 normals, 21 regions
  1.1.0 (2025-06-01): Updated to WMO 1991-2020 normals; added 5 new regions;
                      added wet/dry-day statistics and heat-event frequencies
"""

DATA_SOURCES_REGISTRY: dict[str, dict] = {
    "wmo_normals_1991_2020": {
        "description": "WMO 1991-2020 Climatological Normals",
        "url": "https://climexp.knmi.nl/start.cgi",
        "reference": "WMO (2020) Guidelines on the Calculation of Climate Normals, WMO-No. 1203",
        "last_refreshed": "2025-06-01",
        "variables": ["mean_temp_c", "precip_mm_yr", "hot_days_above_35c", "dry_days"],
    },
    "noaa_ghcn_v4": {
        "description": "NOAA Global Historical Climatology Network Monthly v4",
        "url": "https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-monthly",
        "api_docs": "https://www.ncei.noaa.gov/cdo-web/api/v2/",
        "last_refreshed": "2025-06-01",
        "variables": ["mean_temp_c", "precip_mm_yr"],
    },
    "era5_monthly": {
        "description": "ECMWF ERA5 monthly reanalysis (1991-2020 baseline)",
        "url": "https://cds.climate.copernicus.eu/",
        "api_docs": "https://cds.climate.copernicus.eu/api-how-to",
        "last_refreshed": "2025-06-01",
        "variables": ["mean_temp_c", "precip_mm_yr", "wind_speed_ms", "humidity_pct"],
    },
    "chirps_v2": {
        "description": "CHIRPS v2.0 precipitation (0.05° grid, 1981-present)",
        "url": "https://www.chc.ucsb.edu/data/chirps",
        "last_refreshed": "2025-06-01",
        "variables": ["precip_mm_yr", "dry_days", "wet_spell_days"],
    },
    "nasa_nex_gddp_cmip6": {
        "description": "NASA NEX-GDDP-CMIP6 (downscaled CMIP6 projections)",
        "url": "https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6",
        "api_docs": "https://registry.opendata.aws/nex-gddp-cmip6/",
        "last_refreshed": "2025-06-01",
        "variables": ["projected_temp_delta", "projected_precip_change_pct"],
    },
}


# ---------------------------------------------------------------------------
# Baseline dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetBaseline:
    """
    Observed-climate baseline statistics for a CRI region.

    All values represent 1991-2020 WMO climatological normals unless
    otherwise noted.  Use ``data_version`` + ``reference_period`` to
    detect stale data.

    Field notes
    -----------
    mean_temp_c          : Annual mean temperature (°C), 1991-2020
    precip_mm_yr         : Annual total precipitation (mm/yr)
    hot_days_above_35c   : Mean annual days with Tmax > 35°C
    extreme_precip_days  : Days with precip > 20 mm (heavy rain events/yr)
    dry_days_yr          : Days with < 1 mm precip
    wind_speed_ms        : Mean annual wind speed (m/s, 10m height)
    humidity_pct         : Mean relative humidity (%)
    wet_season_months    : List of calendar months in primary wet season
    drought_frequency_yr : Observed return period for meteorological drought
                           (SPI-3 < -1.5), in years
    heat_wave_freq_yr    : Observed annual frequency of heat-wave events
                           (3+ consecutive days with Tmax > mean + 5°C)
    cyclone_landfall_yr  : Historical annual probability of tropical cyclone
                           landfall (Cat 1+) near the region
    data_version         : CRI data version when this baseline was last updated
    reference_period     : Climate normal period (e.g., "1991-2020")
    primary_source       : Data source identifier from DATA_SOURCES_REGISTRY
    notes                : Any caveats or limitations
    """
    region: str
    mean_temp_c: float
    precip_mm_yr: float
    hot_days_above_35c: float           # days/yr
    extreme_precip_days: float          # days/yr with >20mm
    dry_days_yr: float                  # days/yr with <1mm
    wind_speed_ms: float                # m/s at 10m
    humidity_pct: float                 # %
    wet_season_months: list[int] = field(default_factory=list)
    drought_frequency_yr: float = 10.0  # return period (years)
    heat_wave_freq_yr: float = 1.0      # events/yr
    cyclone_landfall_yr: float = 0.0    # prob/yr
    data_version: str = DATA_VERSION
    reference_period: str = "1991-2020"
    primary_source: str = "wmo_normals_1991_2020"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def is_stale(self, current_version: str = DATA_VERSION) -> bool:
        """Return True if this baseline was built with an older data version."""
        return self.data_version != current_version


# ---------------------------------------------------------------------------
# WMO 1991-2020 regional baselines (static — always available)
# Sources: WMO Normals; GHCN-M v4; ERA5; CHIRPS v2; IBTRACS v4
# ---------------------------------------------------------------------------

_WMO_BASELINES: dict[str, MetBaseline] = {
    "AU-WA": MetBaseline(
        region="AU-WA",
        mean_temp_c=21.8,
        precip_mm_yr=310,
        hot_days_above_35c=62,
        extreme_precip_days=8,
        dry_days_yr=270,
        wind_speed_ms=5.8,
        humidity_pct=52,
        wet_season_months=[12, 1, 2, 3],
        drought_frequency_yr=4,
        heat_wave_freq_yr=3.2,
        cyclone_landfall_yr=0.08,
        notes="Pilbara/Kimberley baseline; inland semi-arid; TC season Nov-Apr",
    ),
    "AU-QLD": MetBaseline(
        region="AU-QLD",
        mean_temp_c=23.5,
        precip_mm_yr=620,
        hot_days_above_35c=45,
        extreme_precip_days=18,
        dry_days_yr=230,
        wind_speed_ms=4.2,
        humidity_pct=64,
        wet_season_months=[12, 1, 2, 3],
        drought_frequency_yr=5,
        heat_wave_freq_yr=2.0,
        cyclone_landfall_yr=0.12,
        notes="Tropical/subtropical; coral coast; TC risk Nov-Apr",
    ),
    "AU-SA": MetBaseline(
        region="AU-SA",
        mean_temp_c=18.5,
        precip_mm_yr=220,
        hot_days_above_35c=55,
        extreme_precip_days=5,
        dry_days_yr=290,
        wind_speed_ms=5.2,
        humidity_pct=44,
        wet_season_months=[6, 7, 8],
        drought_frequency_yr=3,
        heat_wave_freq_yr=4.0,
        notes="Mediterranean/arid; Mediterranean wet season; extreme heat events",
    ),
    "AU-NSW": MetBaseline(
        region="AU-NSW",
        mean_temp_c=17.2,
        precip_mm_yr=640,
        hot_days_above_35c=22,
        extreme_precip_days=14,
        dry_days_yr=215,
        wind_speed_ms=4.0,
        humidity_pct=60,
        wet_season_months=[3, 4, 5, 6],
        drought_frequency_yr=6,
        heat_wave_freq_yr=1.5,
    ),
    "AU-VIC": MetBaseline(
        region="AU-VIC",
        mean_temp_c=14.5,
        precip_mm_yr=580,
        hot_days_above_35c=14,
        extreme_precip_days=12,
        dry_days_yr=210,
        wind_speed_ms=4.5,
        humidity_pct=62,
        wet_season_months=[5, 6, 7, 8],
        drought_frequency_yr=7,
        heat_wave_freq_yr=1.2,
    ),
    "AU-NT": MetBaseline(
        region="AU-NT",
        mean_temp_c=27.5,
        precip_mm_yr=1600,
        hot_days_above_35c=120,
        extreme_precip_days=35,
        dry_days_yr=180,
        wind_speed_ms=3.8,
        humidity_pct=72,
        wet_season_months=[11, 12, 1, 2, 3, 4],
        drought_frequency_yr=6,
        heat_wave_freq_yr=4.5,
        cyclone_landfall_yr=0.10,
        notes="Tropical monsoon; Darwin proxy; wet/dry season contrast",
    ),
    "US-TX": MetBaseline(
        region="US-TX",
        mean_temp_c=19.0,
        precip_mm_yr=750,
        hot_days_above_35c=55,
        extreme_precip_days=16,
        dry_days_yr=185,
        wind_speed_ms=5.0,
        humidity_pct=62,
        wet_season_months=[4, 5, 9, 10],
        drought_frequency_yr=4,
        heat_wave_freq_yr=2.8,
        cyclone_landfall_yr=0.10,
        notes="Gulf Coast hurricanes; inland heat events; flash flooding",
    ),
    "US-WY": MetBaseline(
        region="US-WY",
        mean_temp_c=7.2,
        precip_mm_yr=360,
        hot_days_above_35c=8,
        extreme_precip_days=6,
        dry_days_yr=250,
        wind_speed_ms=7.5,
        humidity_pct=42,
        wet_season_months=[4, 5, 6],
        drought_frequency_yr=5,
        heat_wave_freq_yr=0.5,
        notes="High plains; wind energy potential; snowpack-driven hydrology",
    ),
    "US-OK": MetBaseline(
        region="US-OK",
        mean_temp_c=15.5,
        precip_mm_yr=860,
        hot_days_above_35c=42,
        extreme_precip_days=14,
        dry_days_yr=175,
        wind_speed_ms=5.8,
        humidity_pct=62,
        wet_season_months=[4, 5, 10, 11],
        drought_frequency_yr=4,
        heat_wave_freq_yr=2.2,
        notes="Tornado alley; flash drought risk; large precip variability",
    ),
    "CA-AB": MetBaseline(
        region="CA-AB",
        mean_temp_c=4.0,
        precip_mm_yr=460,
        hot_days_above_35c=6,
        extreme_precip_days=5,
        dry_days_yr=220,
        wind_speed_ms=4.2,
        humidity_pct=48,
        wet_season_months=[5, 6, 7],
        drought_frequency_yr=8,
        heat_wave_freq_yr=0.4,
        notes="Continental cold; wildfire season Jun-Sep; chinook wind effects",
    ),
    "CA-QC": MetBaseline(
        region="CA-QC",
        mean_temp_c=4.5,
        precip_mm_yr=980,
        hot_days_above_35c=4,
        extreme_precip_days=18,
        dry_days_yr=155,
        wind_speed_ms=3.5,
        humidity_pct=72,
        wet_season_months=[4, 5, 6, 10, 11],
        drought_frequency_yr=12,
        heat_wave_freq_yr=0.3,
        notes="Continental; snowmelt flooding; aluminium smelter water availability",
    ),
    "GB-ENG": MetBaseline(
        region="GB-ENG",
        mean_temp_c=10.8,
        precip_mm_yr=870,
        hot_days_above_35c=2,
        extreme_precip_days=15,
        dry_days_yr=155,
        wind_speed_ms=6.0,
        humidity_pct=78,
        wet_season_months=[10, 11, 12, 1],
        drought_frequency_yr=10,
        heat_wave_freq_yr=0.6,
        notes="Temperate maritime; summer drought risk in SE England increasing",
    ),
    "NL-NH": MetBaseline(
        region="NL-NH",
        mean_temp_c=10.2,
        precip_mm_yr=820,
        hot_days_above_35c=2,
        extreme_precip_days=14,
        dry_days_yr=140,
        wind_speed_ms=7.0,
        humidity_pct=82,
        wet_season_months=[9, 10, 11, 12],
        drought_frequency_yr=12,
        heat_wave_freq_yr=0.5,
        notes="Below sea level; Rhine/Meuse river flood risk; North Sea surge",
    ),
    "ZA": MetBaseline(
        region="ZA",
        mean_temp_c=17.8,
        precip_mm_yr=480,
        hot_days_above_35c=38,
        extreme_precip_days=10,
        dry_days_yr=235,
        wind_speed_ms=4.8,
        humidity_pct=55,
        wet_season_months=[10, 11, 12, 1, 2, 3],
        drought_frequency_yr=4,
        heat_wave_freq_yr=2.5,
        notes="Cape Fold mountains; Day Zero drought risk; semi-arid interior",
    ),
    "CL-02": MetBaseline(
        region="CL-02",
        mean_temp_c=14.5,
        precip_mm_yr=120,
        hot_days_above_35c=20,
        extreme_precip_days=3,
        dry_days_yr=310,
        wind_speed_ms=5.5,
        humidity_pct=35,
        wet_season_months=[6, 7, 8],
        drought_frequency_yr=3,
        heat_wave_freq_yr=1.5,
        notes="Atacama region; extreme aridity; glacier retreat driving water stress",
    ),
    "PE-01": MetBaseline(
        region="PE-01",
        mean_temp_c=10.0,
        precip_mm_yr=650,
        hot_days_above_35c=5,
        extreme_precip_days=20,
        dry_days_yr=180,
        wind_speed_ms=4.0,
        humidity_pct=68,
        wet_season_months=[11, 12, 1, 2, 3, 4],
        drought_frequency_yr=5,
        heat_wave_freq_yr=0.3,
        notes="Andean highlands; ENSO-driven interannual variability; glacial meltwater",
    ),
    "BR-PA": MetBaseline(
        region="BR-PA",
        mean_temp_c=26.8,
        precip_mm_yr=2200,
        hot_days_above_35c=85,
        extreme_precip_days=70,
        dry_days_yr=120,
        wind_speed_ms=2.5,
        humidity_pct=85,
        wet_season_months=[12, 1, 2, 3, 4, 5],
        drought_frequency_yr=6,
        heat_wave_freq_yr=1.0,
        notes="Equatorial Amazon; high rainfall variability; deforestation-driven drought",
    ),
    "ID-KI": MetBaseline(
        region="ID-KI",
        mean_temp_c=27.0,
        precip_mm_yr=2500,
        hot_days_above_35c=30,
        extreme_precip_days=80,
        dry_days_yr=100,
        wind_speed_ms=2.0,
        humidity_pct=88,
        wet_season_months=[10, 11, 12, 1, 2, 3],
        drought_frequency_yr=7,
        heat_wave_freq_yr=0.5,
        cyclone_landfall_yr=0.04,
        notes="Kalimantan; El Niño drought risk; peat fire season Jun-Oct",
    ),
    "CN-NM": MetBaseline(
        region="CN-NM",
        mean_temp_c=6.5,
        precip_mm_yr=290,
        hot_days_above_35c=18,
        extreme_precip_days=6,
        dry_days_yr=255,
        wind_speed_ms=6.0,
        humidity_pct=40,
        wet_season_months=[7, 8],
        drought_frequency_yr=4,
        heat_wave_freq_yr=1.5,
        notes="Inner Mongolia steppe; high wind erosion; persistent water stress",
    ),
    "IN-MH": MetBaseline(
        region="IN-MH",
        mean_temp_c=26.2,
        precip_mm_yr=1400,
        hot_days_above_35c=90,
        extreme_precip_days=35,
        dry_days_yr=190,
        wind_speed_ms=3.5,
        humidity_pct=72,
        wet_season_months=[6, 7, 8, 9],
        drought_frequency_yr=4,
        heat_wave_freq_yr=3.5,
        cyclone_landfall_yr=0.06,
        notes="Maharashtra; Arabian Sea cyclone exposure; monsoon-driven flood/drought cycle",
    ),
    "MN-01": MetBaseline(
        region="MN-01",
        mean_temp_c=1.0,
        precip_mm_yr=250,
        hot_days_above_35c=12,
        extreme_precip_days=4,
        dry_days_yr=260,
        wind_speed_ms=5.5,
        humidity_pct=45,
        wet_season_months=[6, 7, 8],
        drought_frequency_yr=4,
        heat_wave_freq_yr=0.8,
        notes="Mongolian steppe; dzud (severe winter) events; rapid warming rate",
    ),
    "global": MetBaseline(
        region="global",
        mean_temp_c=14.5,
        precip_mm_yr=990,
        hot_days_above_35c=20,
        extreme_precip_days=15,
        dry_days_yr=180,
        wind_speed_ms=3.5,
        humidity_pct=60,
        wet_season_months=[1, 2, 3, 10, 11, 12],
        drought_frequency_yr=8,
        heat_wave_freq_yr=1.0,
        notes="Global average fallback — use region-specific data wherever possible",
    ),
}


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------

class MetProvider(ABC):
    """
    Abstract interface for meteorological baseline data.

    Implement this to plug in any data source — open or licensed.

    Methods
    -------
    get_baseline(region)   → MetBaseline for a CRI region code
    get_baseline_all()     → dict[region, MetBaseline]
    source_name            → human-readable provider name
    data_version           → version string of the underlying data
    is_available()         → True if provider can serve requests right now
    refresh(regions)       → refresh baselines from upstream source (optional)
    """

    @property
    @abstractmethod
    def source_name(self) -> str: ...

    @property
    @abstractmethod
    def data_version(self) -> str: ...

    @abstractmethod
    def get_baseline(self, region: str) -> MetBaseline: ...

    @abstractmethod
    def get_baseline_all(self) -> dict[str, MetBaseline]: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    def refresh(self, regions: list[str] | None = None) -> int:
        """
        Fetch fresh data from upstream.  Returns number of regions refreshed.
        Default implementation is a no-op (static providers always up-to-date).
        """
        return 0

    def check_staleness(self) -> list[str]:
        """Return list of regions whose data_version does not match current."""
        stale = []
        for region, baseline in self.get_baseline_all().items():
            if baseline.is_stale(DATA_VERSION):
                stale.append(region)
        return stale


# ---------------------------------------------------------------------------
# Static WMO provider (always available — no API key needed)
# ---------------------------------------------------------------------------

class StaticWMOProvider(MetProvider):
    """
    WMO 1991-2020 Climatological Normals (static lookup table).

    This is the default provider.  It is always available and requires no
    network access or API keys.  Data is embedded in the CRI package and
    versioned with DATA_VERSION.

    To update:
        1. Revise ``_WMO_BASELINES`` in this module with new values
        2. Bump ``DATA_VERSION`` with a MINOR increment
        3. Document the change in the DATA_VERSION docstring
    """

    @property
    def source_name(self) -> str:
        return "WMO 1991-2020 Climatological Normals (static)"

    @property
    def data_version(self) -> str:
        return DATA_VERSION

    def is_available(self) -> bool:
        return True

    def get_baseline(self, region: str) -> MetBaseline:
        return _WMO_BASELINES.get(region, _WMO_BASELINES["global"])

    def get_baseline_all(self) -> dict[str, MetBaseline]:
        return dict(_WMO_BASELINES)


# ---------------------------------------------------------------------------
# NOAA GHCN provider (requires free NOAA CDO API key)
# ---------------------------------------------------------------------------

class NOAAGHCNProvider(MetProvider):
    """
    Fetches temperature and precipitation normals from the NOAA Climate Data
    Online (CDO) Web Services API v2.

    Free API key: https://www.ncdc.noaa.gov/cdo-web/token

    Note: NOAA CDO provides station-level data; we aggregate to CRI region
    codes using the representative stations defined in ``_NOAA_STATION_MAP``.

    This provider falls back to ``StaticWMOProvider`` when:
      - No API key is configured
      - The NOAA API is unavailable
      - A region has no mapped station
    """

    # Representative NOAA CDO station IDs for each CRI region
    # (GHCND:xxxxx or GHCNM:xxxxx format)
    _NOAA_STATION_MAP: dict[str, str] = {
        "AU-WA":  "GHCNM:ASN00009021",   # Perth Airport
        "AU-QLD": "GHCNM:ASN00040223",   # Brisbane Airport
        "AU-SA":  "GHCNM:ASN00023034",   # Adelaide Airport
        "AU-NSW": "GHCNM:ASN00066062",   # Sydney Observatory Hill
        "AU-VIC": "GHCNM:ASN00086282",   # Melbourne Airport
        "AU-NT":  "GHCNM:ASN00014015",   # Darwin Airport
        "US-TX":  "GHCNM:USW00012960",   # Houston Intercontinental
        "US-WY":  "GHCNM:USW00024089",   # Cheyenne Airport
        "US-OK":  "GHCNM:USW00013967",   # Oklahoma City Airport
        "CA-AB":  "GHCNM:CA003031093",   # Calgary Airport
        "CA-QC":  "GHCNM:CA007025440",   # Montreal Pierre Elliott Trudeau
        "GB-ENG": "GHCNM:UKE00105901",   # London Heathrow
        "NL-NH":  "GHCNM:NLE00152814",   # De Bilt
        "ZA":     "GHCNM:SFM00068816",   # Johannesburg OR Tambo
        "CL-02":  "GHCNM:CH000085406",   # Antofagasta
        "PE-01":  "GHCNM:PE000084469",   # Lima Callao
        "BR-PA":  "GHCNM:BR000082191",   # Belem
        "IN-MH":  "GHCNM:IN022021600",   # Mumbai Santacruz
        "CN-NM":  "GHCNM:CHM00054218",   # Hohhot
        "MN-01":  "GHCNM:MOM00044212",   # Ulaanbaatar
    }

    _API_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

    def __init__(self, api_key: str | None = None) -> None:
        """
        Args:
            api_key: NOAA CDO API key.  If None, attempts to read from
                     environment variable ``NOAA_CDO_API_KEY``.
        """
        self._api_key = api_key or os.environ.get("NOAA_CDO_API_KEY")
        self._fallback = StaticWMOProvider()
        self._cache: dict[str, MetBaseline] = {}

    @property
    def source_name(self) -> str:
        return "NOAA GHCN-M v4 + CDO API"

    @property
    def data_version(self) -> str:
        return DATA_VERSION

    def is_available(self) -> bool:
        return bool(self._api_key)

    def get_baseline(self, region: str) -> MetBaseline:
        if not self.is_available():
            return self._fallback.get_baseline(region)
        if region in self._cache:
            return self._cache[region]
        try:
            baseline = self._fetch_region(region)
            self._cache[region] = baseline
            return baseline
        except Exception:
            return self._fallback.get_baseline(region)

    def get_baseline_all(self) -> dict[str, MetBaseline]:
        if not self.is_available():
            return self._fallback.get_baseline_all()
        return {r: self.get_baseline(r) for r in _WMO_BASELINES}

    def refresh(self, regions: list[str] | None = None) -> int:
        """Fetch fresh normals from NOAA CDO and update cache."""
        if not self.is_available():
            return 0
        targets = regions or list(_WMO_BASELINES.keys())
        refreshed = 0
        for region in targets:
            try:
                self._cache[region] = self._fetch_region(region)
                refreshed += 1
            except Exception:
                pass
        return refreshed

    def _fetch_region(self, region: str) -> MetBaseline:
        """Internal: call NOAA CDO API for a single region."""
        try:
            import urllib.request
        except ImportError:
            raise RuntimeError("urllib not available")

        station_id = self._NOAA_STATION_MAP.get(region)
        if not station_id:
            return self._fallback.get_baseline(region)

        headers = {"token": self._api_key}
        # Fetch annual temperature normal (TAVG)
        url = (
            f"{self._API_BASE}/data?datasetid=NORMAL_ANN&stationid={station_id}"
            f"&datatypeid=ANN-TAVG-NORMAL&limit=1&units=metric"
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        mean_temp = None
        if data.get("results"):
            mean_temp = data["results"][0]["value"] / 10.0  # tenths of °C → °C

        # Fall back to static for any missing fields
        static = self._fallback.get_baseline(region)
        return MetBaseline(
            region=region,
            mean_temp_c=mean_temp if mean_temp is not None else static.mean_temp_c,
            precip_mm_yr=static.precip_mm_yr,          # use static for now
            hot_days_above_35c=static.hot_days_above_35c,
            extreme_precip_days=static.extreme_precip_days,
            dry_days_yr=static.dry_days_yr,
            wind_speed_ms=static.wind_speed_ms,
            humidity_pct=static.humidity_pct,
            wet_season_months=static.wet_season_months,
            drought_frequency_yr=static.drought_frequency_yr,
            heat_wave_freq_yr=static.heat_wave_freq_yr,
            cyclone_landfall_yr=static.cyclone_landfall_yr,
            data_version=DATA_VERSION,
            reference_period="1991-2020",
            primary_source="noaa_ghcn_v4",
            notes=f"Temperature from NOAA CDO station {station_id}; other fields from WMO static",
        )


# ---------------------------------------------------------------------------
# ERA5 provider (requires free CDS API key)
# ---------------------------------------------------------------------------

class ERA5CDSProvider(MetProvider):
    """
    Fetches monthly ERA5 reanalysis data from the Copernicus Climate Data Store
    (CDS) and derives 1991-2020 normals.

    Free registration: https://cds.climate.copernicus.eu/
    API key setup:     https://cds.climate.copernicus.eu/api-how-to

    ERA5 provides T2M (2m temperature), TP (total precipitation), U10/V10
    (10m wind), and RH (relative humidity) at 0.25° resolution, hourly.
    We aggregate to CRI region codes using bounding boxes.

    Requires: cdsapi Python package  (``pip install cdsapi``)
    """

    # Approximate bounding boxes for each CRI region [N, W, S, E]
    _REGION_BBOX: dict[str, tuple[float, float, float, float]] = {
        "AU-WA":  (-20, 114, -30, 128),
        "AU-QLD": (-16, 138, -28, 154),
        "AU-SA":  (-26, 128, -38, 141),
        "AU-NSW": (-28, 145, -38, 154),
        "AU-VIC": (-34, 140, -39, 150),
        "AU-NT":  (-10, 128, -20, 136),
        "US-TX":  (36, -106, 26, -94),
        "US-WY":  (45, -111, 41, -104),
        "US-OK":  (37, -103, 34, -95),
        "CA-AB":  (60, -120, 49, -110),
        "CA-QC":  (63, -80, 45, -57),
        "GB-ENG": (55, -6, 50, 2),
        "NL-NH":  (53, 4, 52, 5),
        "ZA":     (-22, 16, -35, 33),
        "CL-02":  (-21, -70, -26, -68),
        "PE-01":  (-8, -78, -18, -68),
        "BR-PA":  (-1, -55, -8, -48),
        "ID-KI":  (2, 108, -4, 118),
        "CN-NM":  (49, 100, 38, 122),
        "IN-MH":  (22, 72, 16, 80),
        "MN-01":  (50, 95, 42, 106),
    }

    def __init__(self, api_key: str | None = None,
                 api_url: str = "https://cds.climate.copernicus.eu/api/v2") -> None:
        self._api_key = api_key or os.environ.get("CDS_API_KEY")
        self._api_url = api_url
        self._fallback = StaticWMOProvider()
        self._cache: dict[str, MetBaseline] = {}

    @property
    def source_name(self) -> str:
        return "ECMWF ERA5 (Copernicus CDS)"

    @property
    def data_version(self) -> str:
        return DATA_VERSION

    def is_available(self) -> bool:
        try:
            import cdsapi  # noqa: F401
            return bool(self._api_key)
        except ImportError:
            return False

    def get_baseline(self, region: str) -> MetBaseline:
        if not self.is_available():
            return self._fallback.get_baseline(region)
        if region in self._cache:
            return self._cache[region]
        # ERA5 pulls are expensive — return static and schedule async refresh
        return self._fallback.get_baseline(region)

    def get_baseline_all(self) -> dict[str, MetBaseline]:
        return self._fallback.get_baseline_all()

    def refresh(self, regions: list[str] | None = None) -> int:
        """
        Download ERA5 monthly means for 1991-2020 and compute regional normals.

        This can take several minutes per region (ERA5 download + aggregation).
        Call from a background job, not inline.

        Returns:
            Number of regions successfully refreshed.
        """
        if not self.is_available():
            return 0
        try:
            import cdsapi
        except ImportError:
            return 0

        c = cdsapi.Client(url=self._api_url, key=self._api_key, quiet=True)
        targets = regions or list(self._REGION_BBOX.keys())
        refreshed = 0

        for region in targets:
            bbox = self._REGION_BBOX.get(region)
            if not bbox:
                continue
            try:
                import tempfile, os as _os
                with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                    c.retrieve(
                        "reanalysis-era5-land-monthly-means",
                        {
                            "product_type": "monthly_averaged_reanalysis",
                            "variable": ["2m_temperature", "total_precipitation",
                                         "10m_u_component_of_wind", "10m_v_component_of_wind"],
                            "year": [str(y) for y in range(1991, 2021)],
                            "month": [f"{m:02d}" for m in range(1, 13)],
                            "time": "00:00",
                            "area": list(bbox),
                            "format": "netcdf",
                        },
                        tmp.name,
                    )
                    baseline = self._parse_nc(tmp.name, region)
                    self._cache[region] = baseline
                    _os.unlink(tmp.name)
                    refreshed += 1
            except Exception:
                pass

        return refreshed

    def _parse_nc(self, nc_path: str, region: str) -> MetBaseline:
        """Aggregate ERA5 NetCDF to a MetBaseline (requires xarray + numpy)."""
        try:
            import xarray as xr
            import numpy as np
        except ImportError:
            return self._fallback.get_baseline(region)

        ds = xr.open_dataset(nc_path)
        static = self._fallback.get_baseline(region)

        # 2m temperature: K → °C
        t2m = ds["t2m"] - 273.15
        mean_temp = float(t2m.mean().values)

        # Precipitation: m/s monthly mean → mm/yr (ERA5 land uses m/s)
        tp = ds["tp"]
        # ERA5 monthly accumulated precip in m; sum over year and average across years
        precip_mm_yr = float(tp.sum(dim="time").mean().values) * 1000

        # Wind speed
        u10 = ds["u10"]; v10 = ds["v10"]
        wind = float(((u10**2 + v10**2)**0.5).mean().values)

        # Hot days proxy: fraction of time T > 35°C * 365
        hot_days = float((t2m > 35).sum(dim="time").mean().values) / 30 * 365 / (2021-1991)

        ds.close()
        return MetBaseline(
            region=region,
            mean_temp_c=round(mean_temp, 1),
            precip_mm_yr=round(precip_mm_yr, 0),
            hot_days_above_35c=round(hot_days, 1),
            extreme_precip_days=static.extreme_precip_days,
            dry_days_yr=static.dry_days_yr,
            wind_speed_ms=round(wind, 1),
            humidity_pct=static.humidity_pct,
            wet_season_months=static.wet_season_months,
            drought_frequency_yr=static.drought_frequency_yr,
            heat_wave_freq_yr=static.heat_wave_freq_yr,
            cyclone_landfall_yr=static.cyclone_landfall_yr,
            data_version=DATA_VERSION,
            reference_period="1991-2020",
            primary_source="era5_monthly",
            notes=f"ERA5 CDS download aggregated to region bbox {self._REGION_BBOX.get(region)}",
        )


# ---------------------------------------------------------------------------
# Blended provider: merges multiple sources with priority ordering
# ---------------------------------------------------------------------------

class BlendedMetProvider(MetProvider):
    """
    Merges results from multiple providers, using the first available source
    for each region.

    Priority order: providers[0] > providers[1] > ... > StaticWMO (final fallback)

    Usage:
        provider = BlendedMetProvider([
            ERA5CDSProvider(api_key="your-cds-key"),
            NOAAGHCNProvider(api_key="your-noaa-key"),
        ])
        baseline = provider.get_baseline("AU-WA")
    """

    def __init__(self, providers: list[MetProvider]) -> None:
        self._providers = providers
        self._static = StaticWMOProvider()

    @property
    def source_name(self) -> str:
        names = [p.source_name for p in self._providers if p.is_available()]
        return "BlendedMetProvider: " + " | ".join(names) if names else "BlendedMetProvider (static fallback)"

    @property
    def data_version(self) -> str:
        return DATA_VERSION

    def is_available(self) -> bool:
        return True   # always falls back to static

    def get_baseline(self, region: str) -> MetBaseline:
        for p in self._providers:
            if p.is_available():
                try:
                    return p.get_baseline(region)
                except Exception:
                    continue
        return self._static.get_baseline(region)

    def get_baseline_all(self) -> dict[str, MetBaseline]:
        return {r: self.get_baseline(r) for r in _WMO_BASELINES}

    def refresh(self, regions: list[str] | None = None) -> int:
        total = 0
        for p in self._providers:
            total += p.refresh(regions)
        return total


# ---------------------------------------------------------------------------
# Module-level active provider (singleton pattern)
# ---------------------------------------------------------------------------

_active_provider: MetProvider = StaticWMOProvider()


def get_active_provider() -> MetProvider:
    """Return the currently active meteorological data provider."""
    return _active_provider


def set_active_provider(provider: MetProvider) -> None:
    """
    Set the active meteorological data provider.

    Example — swap in ERA5 when CDS key is available:
        from cri.climate.met_data import set_active_provider, ERA5CDSProvider
        set_active_provider(ERA5CDSProvider(api_key=os.environ["CDS_API_KEY"]))

    Example — swap in a licensed provider:
        from my_licensed_provider import ClimateXProvider
        set_active_provider(ClimateXProvider(api_key=os.environ["CLIMATE_X_KEY"]))
    """
    global _active_provider
    _active_provider = provider


def get_met_baseline(region: str) -> MetBaseline:
    """
    Convenience function: get met baseline from the active provider.

    The hazard engine calls this instead of accessing ``_WMO_BASELINES``
    directly, so a provider swap affects all hazard calculations.
    """
    return _active_provider.get_baseline(region)


# ---------------------------------------------------------------------------
# Hazard calibration helpers (used by hazard_layers.py)
# ---------------------------------------------------------------------------

def observed_heat_days(region: str) -> float:
    """Historical annual hot-day count (Tmax > 35°C) from met baseline."""
    return get_met_baseline(region).hot_days_above_35c


def observed_precip_mm(region: str) -> float:
    """Historical annual precipitation (mm/yr) from met baseline."""
    return get_met_baseline(region).precip_mm_yr


def observed_drought_return(region: str) -> float:
    """Historical drought return period (years) from met baseline."""
    return get_met_baseline(region).drought_frequency_yr


def observed_cyclone_prob(region: str) -> float:
    """Historical annual cyclone landfall probability from met baseline."""
    return get_met_baseline(region).cyclone_landfall_yr


def heat_stress_baseline_multiplier(region: str) -> float:
    """
    Calibrate heat-stress baseline probability using observed hot-day counts.

    Uses the ratio of observed hot-day frequency to global average to scale
    the WRI heat baseline, ensuring region-specific calibration.

    Returns:
        Multiplier (0.5–3.0) to apply to the WRI heat baseline.
    """
    global_avg = _WMO_BASELINES["global"].hot_days_above_35c    # 20 days
    regional   = observed_heat_days(region)
    return min(3.0, max(0.5, regional / max(global_avg, 1.0)))


def precip_variability_index(region: str) -> float:
    """
    Dimensionless index of precipitation variability (0.5–2.5).

    High index → flood and drought risk amplified.
    Ratio of observed dry days to global average, normalised.
    """
    global_dry = _WMO_BASELINES["global"].dry_days_yr           # 180 days
    regional_dry = get_met_baseline(region).dry_days_yr
    return min(2.5, max(0.5, regional_dry / max(global_dry, 1.0)))


# ---------------------------------------------------------------------------
# Version check utility
# ---------------------------------------------------------------------------

def data_version_report() -> dict[str, Any]:
    """
    Return a report of the current data version state.

    Use this to check if baselines need refreshing before a model run.
    """
    provider = get_active_provider()
    stale = provider.check_staleness()
    return {
        "current_data_version": DATA_VERSION,
        "active_provider": provider.source_name,
        "provider_data_version": provider.data_version,
        "provider_available": provider.is_available(),
        "total_regions": len(_WMO_BASELINES),
        "stale_regions": stale,
        "stale_count": len(stale),
        "data_sources": list(DATA_SOURCES_REGISTRY.keys()),
        "last_checked": datetime.datetime.utcnow().isoformat() + "Z",
        "upgrade_instructions": {
            "noaa_ghcn": "Set env var NOAA_CDO_API_KEY and use NOAAGHCNProvider",
            "era5": "Set env var CDS_API_KEY, pip install cdsapi, and use ERA5CDSProvider",
            "licensed": "Implement MetProvider ABC and call set_active_provider(MyProvider(...))",
        },
    }
