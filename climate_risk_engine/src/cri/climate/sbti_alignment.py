"""
SBTi target alignment and deviation check (absolute contraction approach).

Checks a company's stated absolute emissions target, and its reported emissions since the base year, against
the SBTi cross-sector absolute contraction pathways:

  * near-term Scope 1+2: at least 4.2 % of base-year emissions per year, linear (1.5 °C)
  * near-term Scope 3:   at least 2.5 % per year, linear (well-below 2 °C)
  * long-term / net-zero: at least 90 % absolute reduction by 2050 at the latest
  * near-term timeframe 5–10 years from submission; base year no earlier than 2015

Sources: SBTi Corporate Near-Term Criteria v5.x and SBTi Corporate Net-Zero Standard v1.x (absolute
contraction approach). Sector-specific pathways (SDA — power, cement, steel, aluminium …) are not applied here;
companies in those sectors may use them instead, and the result states this.

A deviation alert is raised when the latest reported emissions sit above the company's own straight-line path
to its target (and separately above the SBTi minimum path) by more than the tolerance.
"""
from __future__ import annotations

from typing import Optional

ACA_RATE = {"scope12": 0.042, "scope3": 0.025}
NET_ZERO_MIN_REDUCTION = 0.90
NET_ZERO_LATEST_YEAR = 2050
EARLIEST_BASE_YEAR = 2015
SOURCES = ["SBTi Corporate Near-Term Criteria v5 (absolute contraction: 4.2 %/yr Scope 1+2, 2.5 %/yr Scope 3)",
           "SBTi Corporate Net-Zero Standard (≥ 90 % absolute reduction by 2050)"]


def _linear(base: float, base_year: int, rate_per_year: float, year: int) -> float:
    return base * max(0.0, 1.0 - rate_per_year * (year - base_year))


def check(base_year: int, base_emissions_t: float, target_year: int, target_reduction_pct: float,
          scope: str = "scope12", reported: Optional[dict[int, float]] = None,
          net_zero_year: Optional[int] = None, net_zero_reduction_pct: Optional[float] = None,
          submission_year: Optional[int] = None, tolerance_pct: float = 5.0) -> dict:
    if scope not in ACA_RATE:
        raise ValueError("scope must be 'scope12' or 'scope3'")
    if base_emissions_t <= 0:
        raise ValueError("base_emissions_t must be > 0")
    if target_year <= base_year:
        raise ValueError("target_year must be after base_year")
    if not 0 < target_reduction_pct <= 100:
        raise ValueError("target_reduction_pct must be in (0, 100]")

    min_rate = ACA_RATE[scope]
    implied_rate = target_reduction_pct / 100.0 / (target_year - base_year)
    required_pct = min(100.0, 100.0 * min_rate * (target_year - base_year))
    issues, alerts = [], []

    if base_year < EARLIEST_BASE_YEAR:
        issues.append(f"Base year {base_year} is earlier than {EARLIEST_BASE_YEAR}; SBTi requires {EARLIEST_BASE_YEAR} or later.")
    if submission_year is not None:
        horizon = target_year - submission_year
        if not 5 <= horizon <= 10:
            issues.append(f"Near-term target year is {horizon} years after submission; SBTi requires 5–10 years.")
    ambition_ok = implied_rate + 1e-9 >= min_rate
    if not ambition_ok:
        issues.append(f"Target implies {100 * implied_rate:.2f} %/yr; the SBTi minimum for "
                      f"{'Scope 1+2 (1.5 °C)' if scope == 'scope12' else 'Scope 3 (well-below 2 °C)'} is "
                      f"{100 * min_rate:.1f} %/yr — a {required_pct:.0f} % cut by {target_year} would be needed.")

    nz = None
    if net_zero_year is not None or net_zero_reduction_pct is not None:
        nz_ok = (net_zero_year is not None and net_zero_year <= NET_ZERO_LATEST_YEAR and
                 net_zero_reduction_pct is not None and net_zero_reduction_pct >= 100 * NET_ZERO_MIN_REDUCTION)
        nz = {"year": net_zero_year, "reduction_pct": net_zero_reduction_pct, "meets_standard": nz_ok}
        if not nz_ok:
            issues.append(f"Net-zero target must cut at least {100 * NET_ZERO_MIN_REDUCTION:.0f} % by "
                          f"{NET_ZERO_LATEST_YEAR} at the latest.")

    own_rate = target_reduction_pct / 100.0 / (target_year - base_year)
    track = []
    for y, e in sorted((reported or {}).items()):
        y = int(y)
        if y < base_year:
            continue
        own = _linear(base_emissions_t, base_year, own_rate, y)
        sbti = _linear(base_emissions_t, base_year, min_rate, y)
        dev_own = 100.0 * (e - own) / own if own > 0 else None
        dev_sbti = 100.0 * (e - sbti) / sbti if sbti > 0 else None
        track.append({"year": y, "reported_t": e, "own_path_t": round(own, 1), "sbti_min_path_t": round(sbti, 1),
                      "change_vs_base_pct": round(100.0 * (e - base_emissions_t) / base_emissions_t, 2),
                      "deviation_vs_own_path_pct": None if dev_own is None else round(dev_own, 2),
                      "deviation_vs_sbti_path_pct": None if dev_sbti is None else round(dev_sbti, 2)})
    if track:
        last = track[-1]
        if last["deviation_vs_own_path_pct"] is not None and last["deviation_vs_own_path_pct"] > tolerance_pct:
            alerts.append({"type": "off_track_own_target", "year": last["year"],
                           "message": f"{last['year']} emissions are {last['deviation_vs_own_path_pct']:.1f} % above the "
                                      f"straight-line path to the company's own {target_year} target."})
        if last["deviation_vs_sbti_path_pct"] is not None and last["deviation_vs_sbti_path_pct"] > tolerance_pct:
            alerts.append({"type": "off_track_sbti_minimum", "year": last["year"],
                           "message": f"{last['year']} emissions are {last['deviation_vs_sbti_path_pct']:.1f} % above the "
                                      f"SBTi minimum ({100 * min_rate:.1f} %/yr) path."})
        yrs = last["year"] - base_year
        if yrs > 0:
            achieved_rate = (base_emissions_t - last["reported_t"]) / base_emissions_t / yrs
            remaining_years = target_year - last["year"]
            target_t = base_emissions_t * (1 - target_reduction_pct / 100.0)
            needed_rate = ((last["reported_t"] - target_t) / base_emissions_t / remaining_years
                           if remaining_years > 0 else None)
        else:
            achieved_rate, needed_rate = None, None
    else:
        achieved_rate, needed_rate = None, None

    return {
        "scope": "Scope 1+2" if scope == "scope12" else "Scope 3",
        "method": "Absolute contraction approach (cross-sector)",
        "base_year": base_year, "base_emissions_t": base_emissions_t,
        "target_year": target_year, "target_reduction_pct": target_reduction_pct,
        "implied_linear_rate_pct_per_yr": round(100 * implied_rate, 3),
        "sbti_minimum_rate_pct_per_yr": 100 * min_rate,
        "sbti_required_reduction_by_target_year_pct": round(required_pct, 1),
        "ambition_aligned": ambition_ok,
        "criteria_issues": issues,
        "net_zero": nz,
        "track_record": track,
        "achieved_rate_pct_per_yr": None if achieved_rate is None else round(100 * achieved_rate, 3),
        "rate_needed_from_now_pct_of_base_per_yr": None if needed_rate is None else round(100 * needed_rate, 3),
        "alerts": alerts,
        "notes": ["Sector-specific SBTi pathways (SDA) are not applied; power, cement, steel, aluminium, transport and "
                  "buildings companies may set intensity targets on those pathways instead.",
                  "This is a screening check, not an SBTi validation."],
        "sources": SOURCES,
    }
