"""
Build the emission-factor library for the carbon-accounting (GHG inventory) module
→ src/cri/data/carbon_factors.json

Sources (all public; licences noted):
  DESNZ   UK Government GHG Conversion Factors for Company Reporting 2026, flat file (revised) — OGL v3.
          Activity factors for fuels, vehicles, refrigerants, travel, hotels, freight, waste, water, materials,
          homeworking, well-to-tank, UK electricity and T&D. AR5 GWP basis; per-gas splits kept so the engine
          can restate CH4/N2O on AR6.
  eGRID   US EPA eGRID2023 (rev2): subregion CO2/CH4/N2O output rates, state→subregion (by plant generation),
          grid gross loss by interconnect — public domain.
  Green-e 2025 Green-e Residual Mix Emission Rates (2023 data), by eGRID subregion — CO2 only.
  AIB     Association of Issuing Bodies, European Residual Mix 2025 (published May 2026): production-mix and
          residual-mix direct CO2 g/kWh.
  Ember   Yearly electricity data (CC BY 4.0): generation and lifecycle emissions by fuel; used only where
          no official national factor is included, restated to direct combustion (see _ember_direct).
  WB      World Bank WDI EG.ELC.LOSS.ZS, transmission & distribution losses (% of output) — CC BY 4.0.
  USEEIO  US EPA Supply Chain GHG Emission Factors v1.3 (2022 USD, purchaser price, with margins) — public domain.
  FRED    US CPI-U annual averages (CPIAUCSL) for deflating spend to 2022 USD.
  IPCC    AR6 WG1 Table 7.15 / 7.SM.7 GWP100 for refrigerant restatement.

Usage: python scripts/build_carbon_factors.py  [raw dir, default data/carbon/raw]
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "carbon" / "raw"
ENG = Path(__file__).resolve().parents[1]
OUT = ENG / "src" / "cri" / "data" / "carbon_factors.json"
EMBER = ENG / "data" / "lca" / "raw" / "ember_yearly.csv"
LB_PER_MWH_TO_G_PER_KWH = 0.45359237

# ── GWP sets ─────────────────────────────────────────────────────────────────
GWP = {
    "AR5": {"CH4": 28.0, "N2O": 265.0},   # DESNZ basis (AR5 without climate-carbon feedback)
    "AR6": {"CH4_fossil": 29.8, "CH4_nonfossil": 27.0, "N2O": 273.0},
}
# IPCC AR6 WG1 GWP100 for pure gases, and ASHRAE 34 blend compositions (mass fractions)
AR6_GASES = {"CO2": 1, "HFC-23": 14600, "HFC-32": 771, "HFC-125": 3740, "HFC-134a": 1530, "HFC-143a": 5810,
             "HFC-152a": 164, "HFC-227ea": 3600, "HFC-236fa": 8690, "HFC-245fa": 962, "HFC-365mfc": 914,
             "HFC-43-10mee": 1600, "HFO-1234yf": 0.501, "HFO-1234ze(E)": 1.37, "SF6": 25200, "NF3": 17400,
             "CF4": 7380, "C2F6": 12400, "R-290": 0.02, "R-600a": 0.0, "R-717": 0.0, "R-744": 1}
BLENDS = {"R-404A": {"HFC-125": .44, "HFC-143a": .52, "HFC-134a": .04},
          "R-407A": {"HFC-32": .20, "HFC-125": .40, "HFC-134a": .40},
          "R-407C": {"HFC-32": .23, "HFC-125": .25, "HFC-134a": .52},
          "R-407F": {"HFC-32": .30, "HFC-125": .30, "HFC-134a": .40},
          "R-410A": {"HFC-32": .50, "HFC-125": .50},
          "R-507A": {"HFC-125": .50, "HFC-143a": .50},
          "R-422D": {"HFC-125": .651, "HFC-134a": .315, "R-600a": .034},
          "R-448A": {"HFC-32": .26, "HFC-125": .26, "HFO-1234yf": .20, "HFC-134a": .21, "HFO-1234ze(E)": .07},
          "R-449A": {"HFC-32": .243, "HFC-125": .247, "HFO-1234yf": .253, "HFC-134a": .257},
          "R-452A": {"HFC-32": .11, "HFC-125": .59, "HFO-1234yf": .30},
          "R-454B": {"HFC-32": .689, "HFO-1234yf": .311},
          "R-513A": {"HFO-1234yf": .56, "HFC-134a": .44}}
REFRIGERANTS_AR6 = dict(AR6_GASES)
for b, comp in BLENDS.items():
    REFRIGERANTS_AR6[b] = round(sum(AR6_GASES[g] * f for g, f in comp.items()), 1)


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ── DESNZ 2026 ───────────────────────────────────────────────────────────────
def desnz() -> dict:
    df = pd.read_excel(RAW / "desnz2026_flat.xlsx", "Factors by Category", header=None)
    hdr = next(i for i in range(20) if "Scope" in [str(v) for v in df.iloc[i].values])
    df.columns = [str(c) for c in df.iloc[hdr]]
    df = df.iloc[hdr + 1:]
    val_col = [c for c in df.columns if c.startswith("GHG Conversion Factor")][0]
    recs: dict[tuple, dict] = {}
    for r in df.itertuples(index=False):
        d = dict(zip(df.columns, r))
        levels = tuple(str(d[k]).strip() if isinstance(d[k], str) else "" for k in ("Level 1", "Level 2", "Level 3", "Level 4"))
        text = str(d["Column Text"]).strip() if isinstance(d["Column Text"], str) else ""
        uom = str(d["UOM"]).strip()
        key = (str(d["Scope"]).strip(), levels, text, uom)
        rec = recs.setdefault(key, {"scope": key[0], "path": [x for x in levels if x], "variant": text, "unit": uom})
        g = str(d["GHG/Unit"]).strip()
        v = _num(d[val_col])
        if v is None:
            continue
        if g == "kg CO2e":
            rec["kgco2e"] = v
        elif g == "kg CO2e of CO2 per unit":
            rec["co2"] = v
        elif g == "kg CO2e of CH4 per unit":
            rec["ch4"] = v
        elif g == "kg CO2e of N2O per unit":
            rec["n2o"] = v
        else:
            rec.setdefault("other", {})[g] = v
    out = []
    for k, rec in recs.items():
        if "kgco2e" not in rec:
            continue
        rec["id"] = "desnz26:" + re.sub(r"[^a-z0-9]+", "-", " ".join(rec["path"] + [rec["variant"], rec["unit"]]).lower()).strip("-")
        out.append(rec)
    return {"source": "UK DESNZ GHG Conversion Factors for Company Reporting 2026 (flat file, revised), OGL v3",
            "gwp_basis": "AR5", "factors": out}


# ── US eGRID2023 + Green-e ───────────────────────────────────────────────────
def egrid() -> dict:
    x = pd.ExcelFile(RAW / "egrid2023.xlsx")
    sr = pd.read_excel(x, "SRL23", header=1)
    subs = {}
    for r in sr.itertuples(index=False):
        d = r._asdict()
        co2, ch4, n2o = d["SRCO2RTA"], d["SRCH4RTA"], d["SRN2ORTA"]
        subs[d["SUBRGN"]] = {"name": d["SRNAME"],
                             "g_per_kwh_ar6": round((co2 + ch4 * GWP["AR6"]["CH4_fossil"] + n2o * GWP["AR6"]["N2O"]) * LB_PER_MWH_TO_G_PER_KWH, 2),
                             "g_co2_per_kwh": round(co2 * LB_PER_MWH_TO_G_PER_KWH, 2)}
    us = pd.read_excel(x, "US23", header=1).iloc[0]
    us_g = round((us["USCO2RTA"] + us["USCH4RTA"] * GWP["AR6"]["CH4_fossil"] + us["USN2ORTA"] * GWP["AR6"]["N2O"]) * LB_PER_MWH_TO_G_PER_KWH, 2)
    pl = pd.read_excel(x, "PLNT23", header=1, usecols=["PSTATABB", "SUBRGN", "PLNGENAN"])
    pl = pl[pl["PLNGENAN"] > 0]
    state_sub = {}
    for st, g in pl.groupby("PSTATABB"):
        s = g.groupby("SUBRGN")["PLNGENAN"].sum().sort_values(ascending=False)
        state_sub[st] = {"dominant": s.index[0], "share": round(float(s.iloc[0] / s.sum()), 3),
                         "all": [str(v) for v in s.index[:4]]}
    gg = pd.read_excel(x, "GGL23", header=1)
    loss = {r.REGION: float(r.GGRSLOSS) for r in gg.itertuples(index=False)}
    interconnect = {s: ("Alaska" if s.startswith("AK") else "Hawaii" if s.startswith("HI") else
                        "ERCOT" if s == "ERCT" else "Western" if s in ("AZNM", "CAMX", "NWPP", "RMPA") else
                        "Puerto Rico" if s == "PRMS" else "Eastern") for s in subs}
    # Green-e 2025 residual mix (2023 data), adjusted system mix CO2 lb/MWh
    html = (RAW / "rs2025.html").read_text(encoding="utf-8", errors="ignore")
    txt = re.sub(r"<[^>]+>", "|", html)
    resid = {}
    cells = [c.strip() for c in re.sub(r"\|\s*(\|\s*)+", "|", txt).split("|")]
    for i, c in enumerate(cells):
        if re.fullmatch(r"[A-Z]{4}", c) and c in subs and i + 6 < len(cells):
            try:
                adj = float(cells[i + 6].replace(",", ""))
                resid[c] = round(adj * LB_PER_MWH_TO_G_PER_KWH, 2)
            except ValueError:
                continue
    return {"source": "US EPA eGRID2023 rev2 (public domain); GWP restated to AR6", "subregions": subs,
            "us_average_g_per_kwh_ar6": us_g, "state_subregion": state_sub, "grid_gross_loss": loss,
            "subregion_interconnect": interconnect,
            "residual_mix": {"source": "2025 Green-e Residual Mix Emission Rates (2023 data), adjusted system mix, CO2 only",
                             "g_co2_per_kwh": resid}}


# ── AIB Europe ───────────────────────────────────────────────────────────────
def aib(iso2to3: dict) -> dict:
    df = pd.read_excel(RAW / "aib_rm_2025.xlsx", "CO2", header=0)
    df.columns = ["cc", "production", "residual", "supplier"]
    out = {}
    for r in df.itertuples(index=False):
        cc = str(r.cc).strip()
        if len(cc) != 2:
            continue
        iso3 = iso2to3.get({"EL": "GR", "UK": "GB", "XK": "XK"}.get(cc, cc))
        if not iso3:
            continue
        out[iso3] = {"production_g_co2_per_kwh": _num(r.production), "residual_g_co2_per_kwh": _num(r.residual),
                     "supplier_mix_g_co2_per_kwh": _num(r.supplier)}
    return {"source": "AIB European Residual Mix 2025 (published 26 May 2026) — direct CO2", "year": 2025, "countries": out}


# ── Ember-derived direct factors (fallback) ──────────────────────────────────
# Ember reports lifecycle emissions by fuel. Direct (combustion) share from IPCC AR5 WG3 Annex III Table A.III.2
# medians: coal (PC) 760 of 820 gCO2e/kWh; gas (CC) 370 of 490; oil/other fossil taken as 0.85 (not in the
# table). Non-fossil sources: zero direct fossil CO2 (biogenic CO2 is reported outside the scopes).
DIRECT_SHARE = {"Coal": 760 / 820, "Gas": 370 / 490, "Other Fossil": 0.85}


def ember() -> dict:
    d = pd.read_csv(EMBER, low_memory=False)
    d = d[d["Area type"] == "Country or economy"]
    out = {}
    for iso3, g in d.groupby("ISO 3 code"):
        yr = int(g[(g["Category"] == "Power sector emissions") & (g["Unit"] == "mtCO2")]["Year"].max())
        x = g[g["Year"] == yr]
        gen = x[(x["Category"] == "Electricity generation") & (x["Unit"] == "TWh")].set_index("Variable")["Value"]
        em = x[(x["Category"] == "Power sector emissions") & (x["Unit"] == "mtCO2")].set_index("Variable")["Value"]
        lc = x[(x["Unit"] == "gCO2/kWh")]["Value"]
        tot = gen.get("Total Generation")
        if not tot or tot <= 0:
            continue
        direct_mt = sum(float(em.get(f, 0) or 0) * s for f, s in DIRECT_SHARE.items())
        out[str(iso3)] = {"year": yr, "direct_g_per_kwh": round(direct_mt / tot * 1000, 1),
                          "lifecycle_g_per_kwh": round(float(lc.iloc[0]), 1) if len(lc) else None}
    return {"source": "Ember yearly electricity data (CC BY 4.0); direct = lifecycle fossil emissions × IPCC AR5 direct share",
            "countries": out}


def wb_losses(iso_ok: set) -> dict:
    d = json.load(open(RAW / "wb_loss.json"))[1]
    best = {}
    for r in d:
        iso = r["countryiso3code"]
        if r["value"] is None or iso not in iso_ok:
            continue
        y = int(r["date"])
        if iso not in best or y > best[iso]["year"]:
            best[iso] = {"year": y, "loss_pct": round(float(r["value"]), 2)}
    return {"source": "World Bank WDI EG.ELC.LOSS.ZS (CC BY 4.0)", "countries": best}


def useeio() -> dict:
    out = {}
    with open(RAW / "useeio13.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["2017 NAICS Code"]] = {"title": r["2017 NAICS Title"],
                                         "kgco2e_per_usd": float(r["Supply Chain Emission Factors with Margins"])}
    cpi = {}
    with open(RAW / "cpi.csv") as fh:
        for r in csv.DictReader(fh):
            v = _num(r.get("CPIAUCSL"))
            if v:
                cpi[int(r["observation_date"][:4])] = v
    return {"source": "US EPA Supply Chain GHG Emission Factors v1.3 (USEEIO, 2022 USD, purchaser price incl. margins), "
                      "public domain; GWP AR5", "base_year": 2022, "naics": out,
            "cpi_u": {"source": "FRED CPIAUCSL annual average", "values": cpi}}


def main():
    iso2to3 = json.loads((ENG / "src" / "cri" / "data" / "iso2_iso3.json").read_text())
    D = desnz()
    uk = [f for f in D["factors"] if f["path"][:1] == ["UK electricity"]]
    uk_td = [f for f in D["factors"] if f["path"][:1] == ["Transmission and distribution"]]
    uk_wtt = [f for f in D["factors"] if f["path"][:1] == ["WTT- UK electricity"]]
    em = ember()
    data = {
        "version": "2026-09-30",
        "gwp": {"sets": GWP, "refrigerants_ar6": REFRIGERANTS_AR6,
                "refrigerant_source": "IPCC AR6 WG1 Ch.7 SM Table 7.SM.7 GWP100; blend compositions ASHRAE Standard 34"},
        "desnz": D,
        "electricity": {
            "GBR": {"desnz_generation": uk, "desnz_td": uk_td, "desnz_wtt": uk_wtt},
            "us_egrid": egrid(),
            "europe_aib": aib(iso2to3),
            "ember_derived": em,
            "td_losses": wb_losses(set(em["countries"]) | {"GBR", "USA"}),
        },
        "spend": useeio(),
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", OUT, round(OUT.stat().st_size / 1e6, 2), "MB;", len(D["factors"]), "DESNZ factors;",
          len(data["electricity"]["us_egrid"]["subregions"]), "eGRID subregions;",
          len(data["electricity"]["us_egrid"]["residual_mix"]["g_co2_per_kwh"]), "Green-e residual;",
          len(data["electricity"]["europe_aib"]["countries"]), "AIB countries;",
          len(em["countries"]), "Ember countries;", len(data["spend"]["naics"]), "NAICS")


if __name__ == "__main__":
    main()
