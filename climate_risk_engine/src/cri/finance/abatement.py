"""
finance/abatement.py — the operator's response to a carbon price.

Why this exists. Holding emissions flat for 25 years reports the cost of doing nothing. At the carbon prices in
NGFS Net Zero (several hundred US$/t by 2040) no operator keeps paying: it cuts emissions where the cut costs
less than the carbon price, or it closes. This module adds that response to the Monte Carlo.

Rules (stated so they can be challenged)
  • Each sector has a short list of levers: share of today's emissions it can remove, levelised cost per tonne
    abated (capital recovery + operating cost, US$/tCO2e), lead time, earliest year it is available, and the
    share of its levelised cost that is operating (the rest is capital).
  • Decision price = the carbon price the asset actually faces after free allocation AND after cost pass-through
    (the same quantity the loss model charges). A lever is triggered the first year that price reaches its
    levelised cost. The operator is myopic: it reacts to the price it sees, it does not know the scenario.
  • Abatement starts `lead` years after the trigger, so a late, sharp price (Delayed transition) is paid in full
    during the build time. Once started it persists (capital is sunk); it scales with sector output.
  • Residual emissions pay the carbon price; abated tonnes cost the levelised cost. The loss is the sum, so it
    is never larger than the unabated cost for the abated share — the operator only abates when it saves money.
  • Costs below zero (no-regret savings) are not credited: every lever costs at least 0.
  • Abatement shares are capped at 95 % of today's emissions in total.

Conservative choice: the decision price is net of pass-through, so abatement is only triggered when the plant itself
saves more than it spends. Where carbon cost is passed on, a real operator that abates can keep part of that price
(the EU ETS windfall effect) — not credited here, so abatement starts later than it might.

Sources are given per lever. `basis` says whether the numbers come from a published range or are analyst
judgement; judgement ranges are deliberately wide. Override the library with your own levers (site-specific
quotes beat any library) via `levers` in the request.

Not modelled: financing constraints, permitting, CO2 transport and storage availability beyond what the cost
range already carries, technology learning (costs held flat in real terms), and abatement by closing capacity
(that is the closure cap in distribution.py). Costs are indicative and mix source years; they are not CPI-aligned.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

MAX_ABATED = 0.95
FIRST_YEAR = 2026


def _L(name, share, mac, lead, avail, life, opex_share, basis, source):
    return {"name": name, "share": share, "mac": mac, "lead": lead, "avail": avail, "life": life,
            "opex_share": opex_share, "basis": basis, "source": source}


_CCS_DILUTE = ("IEA: capture from dilute streams (cement, power) US$40–120/t; cost ranges below add transport and "
               "storage where the source does not.")

LIBRARY = {
    "cement": [
        _L("Clinker substitution, alternative fuels, efficiency", (0.10, 0.20, 0.30), (0, 15, 50), (1, 2, 3), 2026, 20, 0.8,
           "analyst judgement", "Low-cost options; no single published curve — range kept wide."),
        _L("Carbon capture on the kiln", (0.50, 0.60, 0.70), (34, 90, 161), (4, 6, 8), 2030, 25, 0.4,
           "published range", "IEAGHG 2008/3: oxy-combustion US$34–60/t, post-combustion US$88–161/t CO2 avoided; "
                              "amine capture about US$90/t in recent estimates."),
    ],
    "steel:BF/BOF": [
        _L("Efficiency, scrap and process improvements", (0.05, 0.12, 0.20), (0, 25, 66), (1, 2, 4), 2026, 20, 0.8,
           "published range (upper end)", "Nature 2025 (s41586-025-09658-9): BF-BOF plant-level average abatement cost "
                                          "−US$80 to +US$66/t; negative values not credited here."),
        _L("Switch to DRI-EAF", (0.30, 0.45, 0.60), (15, 40, 65), (4, 6, 9), 2028, 30, 0.4,
           "published range", "Nature 2025: DRI-EAF abatement cost US$15–65/t CO2."),
        _L("Capture or hydrogen for the remainder", (0.10, 0.20, 0.30), (70, 130, 200), (5, 7, 10), 2032, 25, 0.5,
           "published range, upper end judgement", "Nature 2025: least-cost net-zero average about US$70/t; IEA capture "
                                                    "cost US$55–130/t; upper bound analyst judgement."),
    ],
    "steel:EAF": [
        _L("Efficiency and low-carbon fuels (few direct emissions on this route)", (0.10, 0.20, 0.30), (0, 20, 60), (1, 2, 4),
           2026, 20, 0.8, "analyst judgement", "Electric and DRI routes carry mostly indirect emissions (Scope 2, not in this "
                                                "asset's direct emissions)."),
    ],
    "power:coal": [
        _L("Gas co-firing or fuel switch", (0.15, 0.30, 0.45), (20, 45, 80), (2, 3, 5), 2026, 20, 0.85,
           "published range, share judgement", "Thunder Said Energy: about US$20/t avoided (US, 2010–19); competitive at a "
                                               "US$30–60/t carbon price; very sensitive to gas price (2022 spike)."),
        _L("CCS retrofit", (0.50, 0.60, 0.75), (55, 85, 130), (5, 6, 8), 2030, 25, 0.4,
           "published range", "IEA: US$55–80/t avoided (post-combustion, coal); recent studies US$86–130/t."),
    ],
    "power:gas": [
        _L("CCS retrofit", (0.80, 0.85, 0.90), (60, 70, 90), (4, 5, 7), 2030, 25, 0.4,
           "published range", "NETL 2023: US$65–72/t CO2 avoided including transport and storage (NGCC retrofit)."),
    ],
    "coal-mining": [
        _L("Methane drainage and ventilation-air-methane capture", (0.15, 0.30, 0.50), (5, 25, 60), (1, 2, 3), 2026, 15, 0.7,
           "analyst judgement", "IEA Global Methane Tracker: the no-cost share is lower for coal than for oil and gas; "
                                "ranges are judgement."),
    ],
    "oil-and-gas": [
        _L("Methane abatement (leak detection and repair, end venting and flaring)", (0.10, 0.20, 0.30), (0, 5, 21), (1, 2, 3),
           2026, 15, 0.7, "published range, share judgement",
           "IEA Global Methane Tracker 2026: about 30% of fossil methane avoidable at no net cost; 'low-cost' threshold about "
           "US$21/tCO2e (US$600/t CH4)."),
        _L("Electrification, efficiency and capture at gas processing", (0.20, 0.30, 0.40), (15, 45, 120), (3, 5, 7), 2028, 25, 0.5,
           "published range", "IEA: high-purity capture US$15–25/t (+ transport and storage); dilute streams US$40–120/t."),
    ],
    "refining": [
        _L("Efficiency, electrification, fuel-gas switching", (0.05, 0.10, 0.15), (0, 20, 60), (1, 2, 4), 2026, 20, 0.7,
           "analyst judgement", "No single published curve — range kept wide."),
        _L("Capture on hydrogen units and catalytic cracking", (0.25, 0.40, 0.55), (40, 80, 120), (4, 6, 8), 2030, 25, 0.4,
           "published range", "IEA: high-purity streams US$15–25/t, dilute US$40–120/t; blended, plus transport and storage."),
    ],
    "chemicals:ammonia": [
        _L("Capture of high-purity process CO2", (0.45, 0.55, 0.65), (25, 40, 55), (3, 4, 6), 2028, 25, 0.4,
           "published range + judgement", "IEA: capture from ammonia synthesis US$15–25/t (avg US$17.5); transport and "
                                          "storage US$10–30/t added (judgement)."),
    ],
}


def lever_key(act_key: Optional[str], cost_key: Optional[str]) -> Optional[str]:
    ck = cost_key or ""
    if act_key in ("power:coal", "coal-mining", "oil-and-gas", "refining", "cement"):
        return act_key
    if act_key in ("power:gas", "power:oil"):
        return "power:gas"
    if act_key == "steel" or ck.startswith("steel"):
        return "steel:EAF" if ck in ("steel:EAF", "steel:DRI") else "steel:BF/BOF"
    if ck == "chemicals:ammonia":
        return "chemicals:ammonia"
    return None


def library_for(act_key: Optional[str], cost_key: Optional[str]) -> dict:
    key = lever_key(act_key, cost_key)
    if key is None:
        return {"key": None, "levers": [], "note": "No abatement library for this sector: results show the unabated "
                "cost. Give your own levers (cost per tonne, share of emissions, lead time) to include a response."}
    return {"key": key, "levers": LIBRARY[key], "note": None}


def _rng3(v, name):
    if isinstance(v, (int, float)):
        return (float(v), float(v), float(v))
    t = tuple(float(x) for x in v)
    if len(t) != 3 or not (t[0] <= t[1] <= t[2]):
        raise ValueError(f"lever {name}: give a number or (low, mode, high) with low ≤ mode ≤ high")
    return t


def user_levers(items: list) -> list:
    """Validate request levers: [{name, share, mac, lead?, avail?, life?, opex_share?}]."""
    out = []
    for i, it in enumerate(items or []):
        name = str(it.get("name") or f"Lever {i + 1}")
        share, mac = _rng3(it["share"], name), _rng3(it["mac"], name)
        if share[2] > 1 or share[0] < 0:
            raise ValueError(f"lever {name}: share of emissions must be between 0 and 1")
        if mac[0] < 0:
            raise ValueError(f"lever {name}: cost per tonne must be ≥ 0")
        lead = _rng3(it.get("lead", 2), name)
        out.append(_L(name, share, mac, lead, int(it.get("avail", FIRST_YEAR)), int(it.get("life", 20)),
                      float(it.get("opex_share", 0.6)), "user-supplied", str(it.get("source") or "user-supplied")))
    return out


def _tri(rng, lo, mo, hi, n):
    if hi <= lo:
        return np.full(n, float(mo))
    return rng.triangular(lo, min(max(mo, lo), hi), hi, n)


def sample(levers: list, rng, n: int) -> dict:
    K = len(levers)
    share, mac, lead = np.zeros((n, K)), np.zeros((n, K)), np.zeros((n, K), int)
    for k, l in enumerate(levers):
        share[:, k] = _tri(rng, *l["share"], n)
        mac[:, k] = _tri(rng, *l["mac"], n)
        lead[:, k] = np.round(_tri(rng, *l["lead"], n)).astype(int)
    tot = share.sum(axis=1)
    share *= np.where(tot > MAX_ABATED, MAX_ABATED / np.where(tot > 0, tot, 1.0), 1.0)[:, None]
    return {"share": share, "mac": mac, "lead": lead,
            "avail": np.array([max(0, l["avail"] - FIRST_YEAR) for l in levers]),
            "opex": np.array([l["opex_share"] for l in levers]), "life": np.array([l["life"] for l in levers])}


def respond(q: np.ndarray, Ae: np.ndarray, S: dict, discount: float) -> dict:
    """q: net carbon price faced (US$/t, n×Y); Ae: baseline tonnes of the asset each year after output (n×Y).
    Returns abated fraction, levelised abatement cost, its operating part, and capital outlays (n×Y)."""
    n, Y = q.shape
    K = S["share"].shape[1]
    t = np.arange(Y)[None, :]
    frac, cost, opex, capex = (np.zeros((n, Y)) for _ in range(4))
    starts = np.full((n, K), Y)
    rows = np.arange(n)
    for k in range(K):
        hit = (q >= S["mac"][:, k][:, None]) & (t >= S["avail"][k])
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), Y)
        start = np.minimum(first + S["lead"][:, k], Y)
        starts[:, k] = start
        active = t >= start[:, None]
        sh = S["share"][:, k][:, None]
        c = Ae * sh * active * S["mac"][:, k][:, None]
        frac += sh * active
        cost += c
        opex += c * S["opex"][k]
        live = start < Y
        if live.any():
            r = max(discount, 1e-6)
            af = (1 - (1 + r) ** -float(S["life"][k])) / r
            at = Ae[rows, np.minimum(start, Y - 1)] * S["share"][:, k]
            amt = (1 - S["opex"][k]) * S["mac"][:, k] * at * af          # capital part, paid when construction ends
            idx = np.maximum(start - 1, 0)
            capex[rows[live], idx[live]] += amt[live]
    return {"frac": frac, "cost": cost, "opex": opex, "capex": capex, "starts": starts}


def summarise(levers: list, S: dict, R: dict, years: np.ndarray) -> list:
    Y = len(years)
    out = []
    for k, l in enumerate(levers):
        st = R["starts"][:, k]
        done = st < Y
        yr = years[np.minimum(st, Y - 1)]
        out.append({"name": l["name"], "share_of_emissions": l["share"], "cost_usd_per_t": l["mac"],
                    "lead_years": l["lead"], "available_from": l["avail"], "basis": l["basis"], "source": l["source"],
                    "prob_started_by_2030": float(np.mean(done & (yr <= 2030))),
                    "prob_started_by_2040": float(np.mean(done & (yr <= 2040))),
                    "prob_started_by_2050": float(np.mean(done)),
                    "median_start_year": int(np.median(yr[done])) if done.any() else None})
    return out
