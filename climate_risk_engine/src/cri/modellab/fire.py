"""Canadian Forest Fire Weather Index system (Van Wagner 1987): FFMC, DMC, DC, ISI, BUI, FWI from noon weather."""
from __future__ import annotations

import math

_LE = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.2, 8.0, 7.0, 6.0]        # DMC day-length factor, 46°N
_LF = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]        # DC day-length adjustment, 46°N


def ffmc(f0, t, h, w, rain):
    mo = 147.2 * (101 - f0) / (59.5 + f0)
    if rain > 0.5:
        rf = rain - 0.5
        mr = mo + 42.5 * rf * math.exp(-100.0 / (251 - mo)) * (1 - math.exp(-6.93 / rf))
        if mo > 150:
            mr += 0.0015 * (mo - 150) ** 2 * rf ** 0.5
        mo = min(mr, 250.0)
    ed = 0.942 * h ** 0.679 + 11 * math.exp((h - 100) / 10) + 0.18 * (21.1 - t) * (1 - math.exp(-0.115 * h))
    if mo > ed:
        ko = 0.424 * (1 - (h / 100) ** 1.7) + 0.0694 * w ** 0.5 * (1 - (h / 100) ** 8)
        m = ed + (mo - ed) * 10 ** (-(ko * 0.581 * math.exp(0.0365 * t)))
    else:
        ew = 0.618 * h ** 0.753 + 10 * math.exp((h - 100) / 10) + 0.18 * (21.1 - t) * (1 - math.exp(-0.115 * h))
        if mo < ew:
            k1 = 0.424 * (1 - ((100 - h) / 100) ** 1.7) + 0.0694 * w ** 0.5 * (1 - ((100 - h) / 100) ** 8)
            m = ew - (ew - mo) * 10 ** (-(k1 * 0.581 * math.exp(0.0365 * t)))
        else:
            m = mo
    m = min(max(m, 0.0), 250.0)                                   # Van Wagner & Pickett (1985): fine-fuel moisture stays within 0-250
    return min(59.5 * (250 - m) / (147.2 + m), 101.0)


def dmc(p0, t, h, rain, month, tropical=False):
    t = max(t, -1.1)
    le = 9.0 if tropical else _LE[month - 1]
    if rain > 1.5:
        re = 0.92 * rain - 1.27
        mo = 20 + math.exp(5.6348 - p0 / 43.43)
        b = 100 / (0.5 + 0.3 * p0) if p0 <= 33 else (14 - 1.3 * math.log(p0) if p0 <= 65 else 6.2 * math.log(p0) - 17.2)
        mr = mo + 1000 * re / (48.77 + b * re)
        p0 = max(244.72 - 43.43 * math.log(mr - 20), 0.0)
    return p0 + 100 * 1.894 * (t + 1.1) * (100 - h) * le * 1e-6


def dc(d0, t, rain, month, tropical=False):
    t = max(t, -2.8)
    lf = 1.4 if tropical else _LF[month - 1]
    if rain > 2.8:
        rd = 0.83 * rain - 1.27
        qo = 800 * math.exp(-d0 / 400)
        dr = 400 * math.log(800 / (qo + 3.937 * rd))
        d0 = max(dr, 0.0)
    v = max(0.36 * (t + 2.8) + lf, 0.0)
    return d0 + 0.5 * v


def isi(f, w):
    f = min(max(f, 0.0), 101.0)
    m = 147.2 * (101 - f) / (59.5 + f)
    return 0.208 * math.exp(0.05039 * w) * 91.9 * math.exp(-0.1386 * m) * (1 + m ** 5.31 / 4.93e7)


def bui(p, d):
    if p <= 0.4 * d:
        u = 0.8 * p * d / (p + 0.4 * d) if (p + 0.4 * d) else 0.0
    else:
        u = p - (1 - 0.8 * d / (p + 0.4 * d)) * (0.92 + (0.0114 * p) ** 1.7)
    return max(u, 0.0)


def fwi(i, u):
    fd = 0.626 * u ** 0.809 + 2 if u <= 80 else 1000 / (25 + 108.64 * math.exp(-0.023 * u))
    b = 0.1 * i * fd
    return math.exp(2.72 * (0.434 * math.log(b)) ** 0.647) if b > 1 else b


def run_series(temp, rh, wind_kmh, rain_mm, months, tropical=False, f0=85.0, p0=6.0, d0=15.0) -> dict:
    """Day-by-day indices from noon weather; returns lists FFMC, DMC, DC, ISI, BUI, FWI."""
    F, P, D = f0, p0, d0
    out = {k: [] for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}
    for t, h, w, r, mo in zip(temp, rh, wind_kmh, rain_mm, months):
        h = min(max(h, 1.0), 100.0)
        F = ffmc(F, t, h, w, r)
        P = dmc(P, t, h, r, mo, tropical)
        D = dc(D, t, r, mo, tropical)
        i = isi(F, w)
        u = bui(P, D)
        out["ffmc"].append(F); out["dmc"].append(P); out["dc"].append(D); out["isi"].append(i); out["bui"].append(u); out["fwi"].append(fwi(i, u))
    return out


def bench_properties() -> dict:
    """Properties every correct implementation must have: hand value of ISI, drying with heat/wind, wetting by rain, bounds."""
    bad = []
    if abs(isi(90.0, 10.0) - 7.09) > 0.15:
        bad.append(f"ISI(90,10) = {isi(90.0, 10.0):.2f}, hand value 7.09")
    if not (ffmc(85, 30, 30, 10, 0) > ffmc(85, 10, 30, 10, 0)):
        bad.append("hotter air must dry the fine fuels faster")
    if not (ffmc(85, 25, 40, 10, 0) > ffmc(85, 25, 90, 10, 0)):
        bad.append("humid air must keep fuels wetter")
    if not (ffmc(90, 25, 40, 10, 20) < ffmc(90, 25, 40, 10, 0)):
        bad.append("20 mm of rain must wet the fuels")
    if not (0 <= ffmc(60, 20, 50, 10, 0) <= 101):
        bad.append("FFMC outside 0–101")
    if not (dc(100, 30, 0, 7) > 100 and dc(300, 20, 40, 7) < 300):
        bad.append("DC must rise in dry heat and fall after heavy rain")
    if not (fwi(isi(92, 20), bui(60, 300)) > fwi(isi(80, 5), bui(20, 100))):
        bad.append("FWI must be larger for drier fuels and stronger wind")
    return {"name": "Fire-weather index: hand value and physical monotonicity", "metric": "violations", "value": float(len(bad)), "limit": 0.0, "ok": not bad,
            "note": "ISI(90, 10 km/h) ≈ 7.09 by hand; heat and wind dry, humidity and rain wet, indices stay in range." + (" FAILED: " + "; ".join(bad) if bad else "")}


def bench_rain_reset_and_spinup() -> dict:
    """Run a synthetic fortnight: a long dry hot spell must raise FWI steadily, and 40 mm of rain must collapse it."""
    n = 30
    s = run_series([30] * n, [30] * n, [20] * n, [0] * n, [7] * n)
    w = run_series([30] * n + [15], [30] * n + [90], [20] * n + [8], [0] * n + [40], [7] * (n + 1))
    ok = s["fwi"][-1] > s["fwi"][5] and w["fwi"][-1] < 0.5 * w["fwi"][-2] and s["dc"][-1] > s["dc"][0]
    return {"name": "Dry spell builds the danger; heavy rain resets it", "metric": f"FWI after 30 dry days {s['fwi'][-1]:.0f}, next day after 40 mm rain {w['fwi'][-1]:.0f}", "value": 0.0 if ok else 1.0, "limit": 0.5, "ok": bool(ok),
            "note": "Moisture codes behave like moisture."}


def run_benchmarks() -> list:
    return [bench_properties(), bench_rain_reset_and_spinup()]
