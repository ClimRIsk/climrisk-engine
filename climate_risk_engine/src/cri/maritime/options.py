"""
Techno-economics of one abatement option, from figures YOU supply. The engine holds no cost database: a capex, fuel price or discount rate it made up would be exactly the invented number this project avoids.

  annuity factor      A = (1 - (1+r)^-n) / r        (r = 0: A = n)
  NPV                 = -capex + A x (annual savings - annual extra opex)
  abatement cost      = (capex / A + annual extra opex - annual savings) / annual CO2 saved       [currency per tCO2, levelised]
  payback (simple)    = capex / (annual savings - annual extra opex)
Annual savings = fuel saved x fuel price + CO2 saved (ETS-scope share) x allowance price + any other savings given.
"""
from __future__ import annotations

from typing import Optional


class OptionError(ValueError):
    pass


def annuity(r: float, n: float) -> float:
    if n <= 0:
        raise OptionError("life must be > 0 years")
    if r < 0:
        raise OptionError("discount rate must be >= 0")
    return float(n) if r == 0 else (1 - (1 + r) ** -n) / r


def evaluate(name: str, capex: float, life_years: float, discount_rate: float, co2_saved_t_per_year: float, fuel_saved_t_per_year: float = 0.0, fuel_price_per_t: Optional[float] = None,
             extra_opex_per_year: float = 0.0, other_savings_per_year: float = 0.0, allowance_price_per_t: Optional[float] = None, ets_scope_share: float = 0.0,
             fuel_extra_t_per_year: float = 0.0) -> dict:
    """fuel_saved_t_per_year may be negative-free: use fuel_extra_t_per_year for options that burn more fuel (for example a costlier fuel switch priced via extra opex)."""
    if capex < 0 or co2_saved_t_per_year == 0:
        raise OptionError("capex must be >= 0 and the CO2 saved per year must be non-zero")
    if not 0 <= ets_scope_share <= 1:
        raise OptionError("ets_scope_share must be between 0 and 1")
    A = annuity(discount_rate, life_years)
    missing = []
    fuel_val = None
    if fuel_saved_t_per_year or fuel_extra_t_per_year:
        if fuel_price_per_t is None:
            missing.append("fuel price per tonne (fuel saving not valued)")
        else:
            fuel_val = (fuel_saved_t_per_year - fuel_extra_t_per_year) * fuel_price_per_t
    ets_val = None
    if ets_scope_share > 0:
        if allowance_price_per_t is None:
            missing.append("allowance price per tonne (ETS saving not valued)")
        else:
            ets_val = co2_saved_t_per_year * ets_scope_share * allowance_price_per_t
    savings = (fuel_val or 0.0) + (ets_val or 0.0) + other_savings_per_year
    net = savings - extra_opex_per_year
    cost_per_t = (capex / A + extra_opex_per_year - savings) / co2_saved_t_per_year
    return {"option": name, "annuity_factor": A, "annual_savings": savings, "annual_net_cash": net, "npv": -capex + A * net,
            "abatement_cost_per_tco2": cost_per_t, "simple_payback_years": (capex / net) if net > 0 else None,
            "components": {"fuel_value_per_year": fuel_val, "ets_value_per_year": ets_val, "other_savings_per_year": other_savings_per_year, "extra_opex_per_year": extra_opex_per_year},
            "incomplete": bool(missing), "missing_inputs": missing,
            "note": "All inputs are yours; savings that lack a price are left out, so a flagged result understates the benefit. Negative abatement cost = the option pays for itself on these inputs. Currency is whatever you used throughout."}


def rank(options: list) -> list:
    """Cheapest abatement first (the marginal-abatement-cost ordering). Incomplete options are listed last."""
    done = [o for o in options if not o["incomplete"]]
    return sorted(done, key=lambda o: o["abatement_cost_per_tco2"]) + [o for o in options if o["incomplete"]]
