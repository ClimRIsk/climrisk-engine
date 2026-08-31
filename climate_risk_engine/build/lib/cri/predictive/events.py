"""
Event detection logic — identifies climate/weather events from raw forecast data.

For each asset location the detector queries:
  1. Open-Meteo 16-day NWP forecast   → storm, heat wave, heavy rain, cold snap
  2. Open-Meteo 3-month seasonal      → drought development, persistent heat
  3. NOAA NHC advisory               → tropical cyclone track intersection
  4. GDACS active alerts             → live flood / cyclone / tsunami

Each detected event becomes an EventDetection with:
  - EventType (what kind of event)
  - Severity (0–5 scale)
  - time_horizon_days (when it starts / peaks)
  - duration_days (expected length of impact)
  - confidence (0–1; lower for seasonal outlooks)
  - metadata dict (raw metrics that triggered detection)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    TROPICAL_CYCLONE    = "tropical_cyclone"
    EXTRATROPICAL_STORM = "extratropical_storm"
    HEAT_WAVE           = "heat_wave"
    COLD_SNAP           = "cold_snap"
    HEAVY_RAIN_FLOOD    = "heavy_rain_flood"
    DROUGHT             = "drought"
    WILDFIRE_RISK       = "wildfire_risk"
    ACTIVE_DISASTER     = "active_disaster"    # GDACS live alert


class Severity(int, Enum):
    """0 = negligible  …  5 = catastrophic."""
    NEGLIGIBLE   = 0
    MINOR        = 1
    MODERATE     = 2
    SIGNIFICANT  = 3
    SEVERE       = 4
    EXTREME      = 5


# Saffir-Simpson → Severity mapping
_SS_SEVERITY = {
    "Tropical Depression": Severity.MINOR,
    "Tropical Storm":      Severity.MODERATE,
    "Category 1":          Severity.SIGNIFICANT,
    "Category 2":          Severity.SEVERE,
    "Category 3":          Severity.SEVERE,
    "Category 4":          Severity.EXTREME,
    "Category 5":          Severity.EXTREME,
}


# ---------------------------------------------------------------------------
# EventDetection dataclass
# ---------------------------------------------------------------------------

@dataclass
class EventDetection:
    event_type: EventType
    severity: Severity
    time_horizon_days: float          # days until event onset (0 = now/active)
    duration_days: float              # expected impact duration
    confidence: float                 # 0–1 (1 = deterministic NWP, 0.4 = seasonal)
    asset_id: str
    asset_lat: float
    asset_lon: float
    peak_date: Optional[str] = None   # ISO-8601 date of peak impact
    metadata: dict = field(default_factory=dict)  # raw metrics

    @property
    def urgency_score(self) -> float:
        """
        Composite urgency = severity × confidence / (1 + time_horizon_days/7).
        Higher = more urgent.  Used for alert ranking.
        """
        return (self.severity.value * self.confidence) / (1 + self.time_horizon_days / 7)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_from_forecast(
    asset_id: str,
    asset_lat: float,
    asset_lon: float,
    forecast_days: list,       # list[ForecastDay] from open_meteo_forecast
    seasonal_days: list,       # list[SeasonalDay] from open_meteo_forecast
) -> list[EventDetection]:
    """
    Run all rule-based detectors over NWP and seasonal forecast data.
    Returns a (possibly empty) list of EventDetection objects.
    """
    events: list[EventDetection] = []

    events += _detect_extratropical_storm(asset_id, asset_lat, asset_lon, forecast_days)
    events += _detect_heat_wave(asset_id, asset_lat, asset_lon, forecast_days, seasonal_days)
    events += _detect_cold_snap(asset_id, asset_lat, asset_lon, forecast_days)
    events += _detect_heavy_rain(asset_id, asset_lat, asset_lon, forecast_days)
    events += _detect_drought(asset_id, asset_lat, asset_lon, seasonal_days)
    events += _detect_wildfire_conditions(asset_id, asset_lat, asset_lon, forecast_days, seasonal_days)

    return events


def detect_from_nhc_storms(
    asset_id: str,
    asset_lat: float,
    asset_lon: float,
    storms: list,              # list[StormAdvisory] from noaa_nhc
    buffer_km: float = 150.0,
) -> list[EventDetection]:
    """
    Check NHC active storms against asset location.
    buffer_km adds indirect impact margin (storm surge, outer bands).
    """
    events: list[EventDetection] = []
    for storm in storms:
        if not storm.threatens(asset_lat, asset_lon, buffer_km=buffer_km):
            continue

        hours = storm.hours_to_closest_approach(asset_lat, asset_lon) or 0.0
        sev   = _SS_SEVERITY.get(storm.category, Severity.SIGNIFICANT)

        events.append(EventDetection(
            event_type=EventType.TROPICAL_CYCLONE,
            severity=sev,
            time_horizon_days=max(0.0, hours / 24),
            duration_days=_tc_duration_days(sev),
            confidence=0.85 if hours <= 72 else 0.60,
            asset_id=asset_id,
            asset_lat=asset_lat,
            asset_lon=asset_lon,
            peak_date=None,
            metadata={
                "storm_id":    storm.storm_id,
                "storm_name":  storm.name,
                "category":    storm.category,
                "wind_kmh":    round(storm.wind_kmh, 1),
                "hours_to_ca": round(hours, 1),
            },
        ))
    return events


def detect_from_gdacs(
    asset_id: str,
    asset_lat: float,
    asset_lon: float,
    gdacs_obs,            # GDACSObservation | None  from connectors.gdacs
) -> list[EventDetection]:
    """Translate a GDACS live observation into EventDetection(s)."""
    events: list[EventDetection] = []
    if gdacs_obs is None:
        return events

    if gdacs_obs.active_flood:
        events.append(EventDetection(
            event_type=EventType.ACTIVE_DISASTER,
            severity=Severity.SEVERE,
            time_horizon_days=0.0,
            duration_days=5.0,
            confidence=0.90,
            asset_id=asset_id,
            asset_lat=asset_lat,
            asset_lon=asset_lon,
            metadata={"source": "GDACS", "type": "flood"},
        ))
    if gdacs_obs.active_cyclone:
        events.append(EventDetection(
            event_type=EventType.TROPICAL_CYCLONE,
            severity=Severity.SEVERE,
            time_horizon_days=0.0,
            duration_days=3.0,
            confidence=0.90,
            asset_id=asset_id,
            asset_lat=asset_lat,
            asset_lon=asset_lon,
            metadata={"source": "GDACS", "type": "cyclone"},
        ))
    if gdacs_obs.active_tsunami:
        events.append(EventDetection(
            event_type=EventType.ACTIVE_DISASTER,
            severity=Severity.EXTREME,
            time_horizon_days=0.0,
            duration_days=7.0,
            confidence=0.95,
            asset_id=asset_id,
            asset_lat=asset_lat,
            asset_lon=asset_lon,
            metadata={"source": "GDACS", "type": "tsunami"},
        ))
    return events


# ---------------------------------------------------------------------------
# Internal detectors
# ---------------------------------------------------------------------------

def _detect_extratropical_storm(
    asset_id: str, lat: float, lon: float, days: list,
) -> list[EventDetection]:
    """
    Detect extratropical cyclone / severe windstorm from NWP forecast.
    Triggers when max sustained wind ≥ 60 km/h OR gust ≥ 90 km/h.
    """
    events = []
    RUN = 0       # consecutive trigger days needed for a "storm event"
    onset_day = None
    storm_days = []

    for fd in days:
        wind  = fd.wind_max_kmh or 0.0
        gust  = fd.wind_gust_kmh or 0.0
        wcode = fd.wmo_code or 0
        severe_wmo = wcode in (82, 85, 86, 95, 96, 99)

        if wind >= 60 or gust >= 90 or severe_wmo:
            storm_days.append(fd)
            if onset_day is None:
                onset_day = fd
        else:
            if len(storm_days) >= 1:
                # Commit detected storm
                peak = max(storm_days, key=lambda d: d.wind_max_kmh or 0)
                peak_wind  = peak.wind_max_kmh or 0.0
                peak_gust  = peak.wind_gust_kmh or 0.0
                peak_wcode = peak.wmo_code or 0

                sev = _wind_severity(peak_wind, peak_gust, peak_wcode)
                if sev >= Severity.MODERATE:
                    events.append(EventDetection(
                        event_type=EventType.EXTRATROPICAL_STORM,
                        severity=sev,
                        time_horizon_days=onset_day.days_from_today,
                        duration_days=len(storm_days),
                        confidence=0.80 if onset_day.days_from_today <= 5 else 0.55,
                        asset_id=asset_id,
                        asset_lat=lat,
                        asset_lon=lon,
                        peak_date=peak.date,
                        metadata={
                            "peak_wind_kmh":  round(peak_wind, 1),
                            "peak_gust_kmh":  round(peak_gust, 1),
                            "wmo_code":       peak_wcode,
                            "weather_label":  peak.weather_label,
                        },
                    ))
            onset_day  = None
            storm_days = []

    return events


def _detect_heat_wave(
    asset_id: str, lat: float, lon: float,
    days: list, seasonal: list,
) -> list[EventDetection]:
    """
    NWP: heat wave = Tmax ≥ 35°C for ≥ 3 consecutive days
    Seasonal: persistent heat = ensemble mean Tmax ≥ 33°C for ≥ 14 days
    """
    events = []
    # NWP
    streak, onset = [], None
    for fd in days:
        if (fd.tmax_c or 0) >= 35:
            streak.append(fd)
            if not onset:
                onset = fd
        else:
            if len(streak) >= 3:
                peak = max(streak, key=lambda d: d.tmax_c or 0)
                sev  = Severity.SIGNIFICANT if peak.tmax_c < 40 else Severity.SEVERE
                events.append(EventDetection(
                    event_type=EventType.HEAT_WAVE,
                    severity=sev,
                    time_horizon_days=onset.days_from_today,
                    duration_days=len(streak),
                    confidence=0.80 if onset.days_from_today <= 7 else 0.60,
                    asset_id=asset_id, asset_lat=lat, asset_lon=lon,
                    peak_date=peak.date,
                    metadata={"peak_tmax_c": peak.tmax_c, "streak_days": len(streak)},
                ))
            streak, onset = [], None

    # Seasonal outlook (persistent)
    if seasonal:
        hot_days = [d for d in seasonal if (d.tmax_c_mean or 0) >= 33]
        if len(hot_days) >= 14:
            sev = Severity.MODERATE
            p90_vals = [d.tmax_c_p90 for d in hot_days if d.tmax_c_p90 is not None]
            if p90_vals and max(p90_vals) >= 38:
                sev = Severity.SIGNIFICANT
            events.append(EventDetection(
                event_type=EventType.HEAT_WAVE,
                severity=sev,
                time_horizon_days=17,      # seasonal starts after 16-day NWP
                duration_days=len(hot_days),
                confidence=0.45,
                asset_id=asset_id, asset_lat=lat, asset_lon=lon,
                metadata={
                    "hot_days_count":  len(hot_days),
                    "mean_tmax_c":     round(sum(d.tmax_c_mean or 0 for d in hot_days) / len(hot_days), 1),
                    "horizon":         "seasonal",
                },
            ))

    return events


def _detect_cold_snap(
    asset_id: str, lat: float, lon: float, days: list,
) -> list[EventDetection]:
    """Cold snap = Tmin ≤ -5°C for ≥ 2 consecutive days (temperate assets)."""
    events = []
    streak, onset = [], None
    for fd in days:
        if (fd.tmin_c or 0) <= -5:
            streak.append(fd)
            if not onset:
                onset = fd
        else:
            if len(streak) >= 2:
                peak = min(streak, key=lambda d: d.tmin_c or 0)
                sev  = Severity.MODERATE if peak.tmin_c > -15 else Severity.SIGNIFICANT
                events.append(EventDetection(
                    event_type=EventType.COLD_SNAP,
                    severity=sev,
                    time_horizon_days=onset.days_from_today,
                    duration_days=len(streak),
                    confidence=0.75,
                    asset_id=asset_id, asset_lat=lat, asset_lon=lon,
                    peak_date=peak.date,
                    metadata={"min_tmin_c": peak.tmin_c, "streak_days": len(streak)},
                ))
            streak, onset = [], None
    return events


def _detect_heavy_rain(
    asset_id: str, lat: float, lon: float, days: list,
) -> list[EventDetection]:
    """
    Heavy rain / flood risk:
      - Single day ≥ 80 mm  → SIGNIFICANT
      - Single day ≥ 150 mm → SEVERE
      - 3-day accumulation ≥ 200 mm → SIGNIFICANT
    """
    events = []
    for i, fd in enumerate(days):
        rain = fd.precip_mm or 0.0
        if rain >= 80:
            sev = Severity.SEVERE if rain >= 150 else Severity.SIGNIFICANT
            events.append(EventDetection(
                event_type=EventType.HEAVY_RAIN_FLOOD,
                severity=sev,
                time_horizon_days=fd.days_from_today,
                duration_days=1.0,
                confidence=0.75 if fd.days_from_today <= 5 else 0.50,
                asset_id=asset_id, asset_lat=lat, asset_lon=lon,
                peak_date=fd.date,
                metadata={"precip_mm": rain, "wmo_code": fd.wmo_code},
            ))
        # 3-day accumulation
        if i + 2 < len(days):
            acc = (days[i].precip_mm or 0) + (days[i+1].precip_mm or 0) + (days[i+2].precip_mm or 0)
            if acc >= 200 and not any(e.peak_date in [days[i].date, days[i+1].date, days[i+2].date] for e in events):
                events.append(EventDetection(
                    event_type=EventType.HEAVY_RAIN_FLOOD,
                    severity=Severity.SIGNIFICANT,
                    time_horizon_days=days[i].days_from_today,
                    duration_days=3.0,
                    confidence=0.65,
                    asset_id=asset_id, asset_lat=lat, asset_lon=lon,
                    peak_date=days[i].date,
                    metadata={"3day_precip_mm": round(acc, 1)},
                ))
    return events


def _detect_drought(
    asset_id: str, lat: float, lon: float, seasonal: list,
) -> list[EventDetection]:
    """
    Drought signal from seasonal outlook.

    Criteria (all must be met to avoid false positives in humid climates):
      1. Ensemble mean daily precip < 2.0 mm/day for ≥ 21 days
         (rules out UK/NL/Germany/NW-Europe which average 2–4 mm/day)
      2. p10 (dry-scenario) daily precip < 0.5 mm/day for those same days
      3. Longest consecutive run of criterion-1 days ≥ 21

    Severity:
      MODERATE   : run 21–29 days
      SIGNIFICANT: run 30–59 days
      SEVERE     : run ≥ 60 days
    """
    events = []
    if not seasonal:
        return events

    # Pair mean + p10 per day
    pairs = [
        (d.precip_mm_mean, d.precip_mm_p10)
        for d in seasonal
        if d.precip_mm_mean is not None and d.precip_mm_p10 is not None
    ]
    if not pairs:
        return events

    overall_mean = sum(m for m, _ in pairs) / len(pairs)

    # Purely arid baseline — drought signal unreliable without climatological ref
    if overall_mean < 0.05:
        return events

    # Humid climate guard: if overall seasonal mean > 3 mm/day, this is a
    # wet region (UK, Netherlands, Germany, Ireland, NW France, Pacific NW US).
    # Drought can still occur in dry spells but requires stricter thresholds.
    strict = overall_mean > 3.0   # True for wet climates

    DRY_MEAN_THRESHOLD = 1.0 if strict else 2.0    # mm/day
    DRY_P10_THRESHOLD  = 0.2 if strict else 0.5    # mm/day

    # Find longest consecutive run of "drought condition" days
    max_run = 0
    current_run = 0
    onset_idx = None
    best_onset_idx = None

    for i, (mean, p10) in enumerate(pairs):
        if mean < DRY_MEAN_THRESHOLD and p10 < DRY_P10_THRESHOLD:
            current_run += 1
            if onset_idx is None:
                onset_idx = i
            if current_run > max_run:
                max_run = current_run
                best_onset_idx = onset_idx
        else:
            current_run = 0
            onset_idx   = None

    MIN_RUN = 30 if strict else 21
    if max_run < MIN_RUN:
        return events

    sev = (
        Severity.SEVERE      if max_run >= 60 else
        Severity.SIGNIFICANT if max_run >= 30 else
        Severity.MODERATE
    )

    events.append(EventDetection(
        event_type=EventType.DROUGHT,
        severity=sev,
        time_horizon_days=17 + (best_onset_idx or 0),
        duration_days=max_run,
        confidence=0.40,
        asset_id=asset_id, asset_lat=lat, asset_lon=lon,
        metadata={
            "drought_run_days":     max_run,
            "seasonal_mean_mm_d":   round(overall_mean, 2),
            "dry_mean_threshold":   DRY_MEAN_THRESHOLD,
            "horizon":              "seasonal",
            "strict_humid_mode":    strict,
        },
    ))
    return events


def _detect_wildfire_conditions(
    asset_id: str, lat: float, lon: float,
    days: list, seasonal: list,
) -> list[EventDetection]:
    """
    Wildfire weather conditions (not active fire — that's NASA FIRMS).
    Trigger: Tmax ≥ 35°C + precip < 2 mm + wind ≥ 30 km/h for ≥ 3 days.
    """
    events = []
    streak, onset = [], None
    for fd in days:
        hot  = (fd.tmax_c or 0) >= 35
        dry  = (fd.precip_mm or 0) < 2.0
        windy = (fd.wind_max_kmh or 0) >= 30
        if hot and dry and windy:
            streak.append(fd)
            if not onset:
                onset = fd
        else:
            if len(streak) >= 3:
                events.append(EventDetection(
                    event_type=EventType.WILDFIRE_RISK,
                    severity=Severity.SIGNIFICANT,
                    time_horizon_days=onset.days_from_today,
                    duration_days=len(streak),
                    confidence=0.65,
                    asset_id=asset_id, asset_lat=lat, asset_lon=lon,
                    peak_date=streak[0].date,
                    metadata={
                        "peak_tmax_c": max(d.tmax_c or 0 for d in streak),
                        "peak_wind_kmh": max(d.wind_max_kmh or 0 for d in streak),
                    },
                ))
            streak, onset = [], None
    return events


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

def _wind_severity(wind_kmh: float, gust_kmh: float, wmo_code: int) -> Severity:
    g = max(wind_kmh, gust_kmh * 0.8)
    if g < 60:
        return Severity.MINOR
    if g < 80:
        return Severity.MODERATE
    if g < 100:
        return Severity.SIGNIFICANT
    if g < 130:
        return Severity.SEVERE
    return Severity.EXTREME


def _tc_duration_days(sev: Severity) -> float:
    return {
        Severity.MINOR:       1.0,
        Severity.MODERATE:    2.0,
        Severity.SIGNIFICANT: 3.0,
        Severity.SEVERE:      5.0,
        Severity.EXTREME:     7.0,
    }.get(sev, 3.0)
