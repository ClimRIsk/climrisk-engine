"""
Experience rating from an insurer's own claims: expected annual loss per cause for offshore wind (or any asset class), with honest uncertainty.

Offshore wind losses are dominated by causes no hazard model can see (subsea cables, installation, blades, drivetrain, electrical, vessel impact). Public claims
data do not exist, so the only defensible way to price them is from the insurer's (or the broker's) own claims, credibility-weighted against how little
exposure they cover. This module does that and nothing more.

Per cause c:  frequency  lambda_c = (a0 + n_c) / (b0 + E)       Gamma-Poisson posterior mean, n_c claims in E exposure units (turbine-years or km-years)
              severity   lognormal fitted to the claim amounts when n_c >= 8, otherwise the empirical mean with a wide factor
              AAL_c      = lambda_c x mean severity; its spread by posterior sampling of lambda and bootstrap of severities
              credibility Z_c = n_c / (n_c + k)   (k = 10 claims by default): how much of the answer comes from the data rather than from the prior
The prior (a0, b0) defaults to the insurer's all-cause rate shrunk to a weight of one pseudo-claim, and is returned in the output so a reviewer can change it.
If no claims are supplied the result is NOT ASSESSED (never zero).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

CAUSES = ("subsea cable", "installation and construction", "blade", "drivetrain", "electrical", "vessel impact", "weather", "other")


def _norm_cause(c: str) -> str:
    t = str(c or "").lower()
    for key, name in (("cable", "subsea cable"), ("install", "installation and construction"), ("construct", "installation and construction"), ("blade", "blade"), ("gearbox", "drivetrain"),
                      ("generator", "drivetrain"), ("bearing", "drivetrain"), ("drive", "drivetrain"), ("transformer", "electrical"), ("electric", "electrical"), ("converter", "electrical"),
                      ("vessel", "vessel impact"), ("collision", "vessel impact"), ("ship", "vessel impact"), ("storm", "weather"), ("weather", "weather"), ("lightning", "weather")):
        if key in t:
            return name
    return "other"


def assess(claims: Optional[list], exposure: float, exposure_unit: str = "turbine-years", asset_exposure: float = 1.0, k_full: float = 10.0, prior_weight: float = 1.0, seed: int = 5) -> dict:
    """`claims`: list of {amount, cause, (year)}; `exposure`: total exposure the claims arose from (turbine-years); `asset_exposure`: exposure to be priced per year
    (turbines in the farm). Returns expected annual loss for the priced asset, by cause, with a 5-95 % range and the credibility of each cause."""
    if not claims or not exposure or exposure <= 0:
        return {"status": "null", "reason": "no claims history supplied: loss from cables, installation, blades, drivetrain, electrical and vessel impact is NOT ASSESSED (not zero)"}
    rows = []
    for c in claims:
        try:
            a = float(c.get("amount", c.get("amount_usd")))
        except (TypeError, ValueError):
            continue
        if a > 0:
            rows.append((_norm_cause(c.get("cause")), a))
    if not rows:
        return {"status": "null", "reason": "claims supplied but none carried a positive amount; nothing could be assessed"}
    rng = np.random.default_rng(seed)
    n_all = len(rows)
    lam_all = n_all / exposure
    a0 = prior_weight; b0 = prior_weight / max(lam_all, 1e-12)
    out = {}
    total = np.zeros(4000); total_mean = 0.0
    for cause in CAUSES:
        amt = np.array([a for c, a in rows if c == cause])
        n = len(amt)
        a_post, b_post = (a0 / len(CAUSES) + n), (b0 + exposure)
        lam = a_post / b_post
        lams = rng.gamma(a_post, 1.0 / b_post, 4000)
        if n >= 8:
            lg = np.log(amt); mu, sd = float(lg.mean()), float(max(lg.std(ddof=1), 0.05))
            sev_mean = float(np.exp(mu + sd * sd / 2))
            sev_s = np.exp(mu + sd * sd / 2) * np.exp(rng.normal(0, sd / np.sqrt(n), 4000))
            kind = "lognormal"
        elif n >= 1:
            sev_mean = float(amt.mean()); sev_s = sev_mean * np.exp(rng.normal(0, 0.9 / np.sqrt(n), 4000)); kind = "empirical mean, wide spread (fewer than 8 claims)"
        else:
            sev_mean = float(np.mean([a for _, a in rows])); sev_s = sev_mean * np.exp(rng.normal(0, 1.0, 4000)); kind = "no claims of this cause: portfolio mean severity, wide spread"
        aal_unit = lams * sev_s                                              # per exposure unit per year
        z = n / (n + k_full)
        out[cause] = {"claims": int(n), "frequency_per_exposure_year": round(float(lam), 5), "mean_severity": round(sev_mean, 0), "severity_basis": kind,
                      "credibility": round(float(z), 2), "aal_per_exposure_unit": round(float(lam * sev_mean), 2),
                      "aal_asset": round(float(lam * sev_mean * asset_exposure), 0),
                      "aal_asset_p05_p95": [round(float(np.percentile(aal_unit, 5) * asset_exposure), 0), round(float(np.percentile(aal_unit, 95) * asset_exposure), 0)]}
        total += aal_unit * asset_exposure
        total_mean += lam * sev_mean * asset_exposure
    return {"status": "modelled", "exposure": exposure, "exposure_unit": exposure_unit, "n_claims": n_all, "asset_exposure": asset_exposure, "by_cause": out,
            "aal_asset": round(float(total_mean), 0), "aal_asset_p05_p95": [round(float(np.percentile(total, 5)), 0), round(float(np.percentile(total, 95)), 0)],
            "mean_credibility": round(float(np.mean([v["credibility"] for v in out.values() if v["claims"]])) if any(v["claims"] for v in out.values()) else 0.0, 2),
            "prior": {"all_cause_rate": round(lam_all, 5), "weight_claims": prior_weight, "k_full_credibility": k_full},
            "note": "Expected annual loss for the priced asset from the supplied claims only. Large or serial losses (a blade defect across a fleet, a cable failure with months of outage) are rare and may not be in the data: the 5 to 95 % range is a lower bound on the true uncertainty. Tail capital is not computed for this component."}
