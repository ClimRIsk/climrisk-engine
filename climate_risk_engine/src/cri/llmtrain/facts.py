"""Turn an insurer-view result into a compact facts string, and synthesise consistent results for training.
The same facts string is what the local model is shown at run time, so training matches use."""
from __future__ import annotations

import random


def fm(x: float) -> str:
    """$5.00m / $143.96m / $282.5k / $1.10bn (the formats the dashboard prints)."""
    a = abs(x)
    if a >= 1e9:
        return f"${x / 1e9:.2f}bn"
    if a >= 1e6:
        return f"${x / 1e6:.2f}m"
    if a >= 1e3:
        return f"${x / 1e3:.1f}k"
    return f"${x:.0f}"


def _row(rows, rp):
    for r in rows or []:
        if r.get("rp") == rp:
            return r
    return {}


def facts(res: dict) -> str:
    g = res["stats"]["gross"]
    tiv = res["tiv"]
    aal = g["aal"]
    oep = lambda rp: _row(g["rows"], rp).get("oep", 0.0)
    aep = lambda rp: _row(g["rows"], rp).get("aep", 0.0)
    bp = res["stats"]["by_peril"]
    out = [f"Schedule: {res['n_sites']} sites, total insured value {fm(tiv)}, {res['n_years']} simulated years.",
           f"Average annual loss (insured) {fm(aal)} ({100 * aal / tiv:.2f}% of TIV); ground-up before policy terms {fm(res['stats']['ground_up']['aal'])}.",
           f"1-in-100 occurrence loss {fm(oep(100))}; 1-in-100 aggregate {fm(aep(100))}; 1-in-250 occurrence {fm(oep(250))}.",
           f"Tail value at risk (99%) {fm(g['tvar99'])}.",
           f"Indicative technical premium {fm(res['premium']['technical_total'])} ({res['premium']['rate_permille']:.1f} per mille of TIV), not a quote.",
           "By peril, average annual loss: " + ", ".join(f"{k} {fm(v)}" for k, v in [("wind", bp.get("wind", 0)), ("river flood", bp.get("river", 0)), ("coastal flood", bp.get("coast", 0))]) + "."]
    s = sorted(res["sites"], key=lambda r: -r["aal"])[0]
    out.append(f"Largest site by average annual loss: {s['name']} {fm(s['aal'])} ({100 * s['aal'] / max(aal, 1):.0f}% of the portfolio).")
    ra = res.get("reinsurance") or {}
    if ra.get("ceded_aal"):
        out.append(f"Reinsurance ceded average annual loss {fm(ra['ceded_aal'])}; net average annual loss {fm(res['stats']['net']['aal'])}.")
    dl = res.get("deductible_ladder") or []
    if len(dl) >= 2:
        out.append(f"Deductible effect on average annual loss: {dl[0]['deductible_pct']:g}% of TIV {fm(dl[0]['aal'])}, {dl[-1]['deductible_pct']:g}% of TIV {fm(dl[-1]['aal'])}; on the 1-in-100 occurrence loss {fm(dl[0]['oep100'])} to {fm(dl[-1]['oep100'])}.")
    c = res.get("climate")
    if c:
        out.append(f"Climate {c['ssp'].upper()} {c['year']} {c['band']}: average annual loss {c['aal']['change_pct']:+.1f}% to {fm(c['aal']['scenario'])}; "
                   f"1-in-100 occurrence {c['tail']['oep100']['change_pct']:+.1f}% to {fm(c['tail']['oep100']['scenario'])}.")
    out.append("Limits: screening-grade model, wind and flood only, damage curves not calibrated to claims, correlation lengths assumed.")
    return " ".join(out)


def synth(rng: random.Random, climate: bool = True, reins: bool = True) -> dict:
    n = rng.randint(3, 14)
    tiv = rng.choice([2e8, 3.5e8, 6e8, 9e8, 1.1e9, 1.8e9, 2.6e9, 4e9]) * rng.uniform(0.8, 1.3)
    rate = 10 ** rng.uniform(-3.3, -2.0)                             # 0.05% .. 1% of TIV
    aal = tiv * rate
    oep20 = aal * rng.uniform(2.5, 6)
    oep100 = oep20 * rng.uniform(2.5, 7)
    oep250 = oep100 * rng.uniform(1.15, 1.7)
    tvar = oep100 * rng.uniform(1.15, 1.6)
    aep100 = oep100 * rng.uniform(1.0, 1.08)
    shares = [rng.random() ** 2 + 0.02 for _ in range(3)]
    s = sum(shares)
    wind, river, coast = (aal * x / s for x in shares)
    names = ["Plant", "Refinery", "Warehouse", "Terminal", "Mill", "Depot", "Smelter", "Works", "Hub"]
    towns = ["Chattogram", "Houston", "Manila", "Rotterdam", "Kolkata", "Osaka", "Miami", "Durban", "Santos", "Mumbai", "Shanghai", "Lagos", "Jakarta"]
    top = f"{rng.choice(names)} {rng.choice('ABCDEFG')}, {rng.choice(towns)}"
    top_aal = aal * rng.uniform(0.2, 0.85)
    ground = aal * rng.uniform(1.05, 1.6)
    capital = tvar * 0.3
    prem = (aal * 1.1 + 0.08 * capital) / 0.75
    ladder = []
    a0, o0 = ground, oep100 * rng.uniform(1.0, 1.06)
    for i, d in enumerate((0, 1, 2, 5, 10)):
        ladder.append({"deductible_pct": float(d), "aal": a0 * (1 - 0.16 * i) ** 1.2, "oep100": o0 * (1 - 0.03 * i)})
    res = {"n_sites": n, "tiv": tiv, "n_years": rng.choice([5000, 8000, 10000, 20000]),
           "stats": {"gross": {"rows": [{"rp": 20, "oep": oep20, "aep": oep20 * 1.02}, {"rp": 100, "oep": oep100, "aep": aep100}, {"rp": 250, "oep": oep250, "aep": oep250 * 1.04}],
                               "aal": aal, "tvar99": tvar},
                     "by_peril": {"wind": wind, "river": river, "coast": coast}, "ground_up": {"aal": ground}, "net": {"aal": aal}},
           "premium": {"technical_total": prem, "rate_permille": 1000 * prem / tiv},
           "sites": [{"name": top, "aal": top_aal}, {"name": "Other sites", "aal": aal - top_aal}],
           "deductible_ladder": ladder}
    if reins and rng.random() < 0.6:
        ceded = aal * rng.uniform(0.1, 0.6)
        res["reinsurance"] = {"ceded_aal": ceded}
        res["stats"]["net"]["aal"] = aal - ceded
    if climate and rng.random() < 0.7:
        ssp = rng.choice(["ssp126", "ssp245", "ssp370", "ssp585"])
        year = rng.choice([2030, 2050, 2070, 2100])
        sev = {"ssp126": 0.5, "ssp245": 1.0, "ssp370": 1.4, "ssp585": 1.8}[ssp] * (year - 2020) / 30
        da = rng.uniform(2, 9) * sev
        dt = da * rng.uniform(0.6, 2.2)
        res["climate"] = {"ssp": ssp, "year": year, "band": rng.choice(["low", "median", "high"]),
                          "aal": {"change_pct": da, "scenario": aal * (1 + da / 100)},
                          "tail": {"oep100": {"change_pct": dt, "scenario": oep100 * (1 + dt / 100)}}}
    return res
