"""
Build the emission-factor database behind the ClimRisk LCA module (cri/lca).

Every factor carries its source. Downloads are cached under data/lca/raw/ (not shipped).

  Ember Yearly Electricity Data (CC BY 4.0)   country grid carbon intensity, lifecycle
                                              gCO2e/kWh (all gases, 100-yr), latest year
  UK DESNZ GHG Conversion Factors 2025 (OGL)  well-to-tank (upstream) factors for fuels,
                                              freight tonne-km factors (+ WTT)
  NGFS Phase 5 via IIASA (pyam)               electricity CO2 and generation by region →
                                              direct grid-intensity index 2025–2050
  IPCC 2006 Guidelines (stationary combustion,
  mineral / metal / chemical industry)        combustion factors and process defaults
                                              (typed in below with table references)

Usage:  python scripts/build_lca_data.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "lca" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "src" / "cri" / "data" / "lca_factors.json"

EMBER = "https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/yearly_full_release_long_format.csv"
DESNZ = ("https://assets.publishing.service.gov.uk/media/6846b6ea57f3515d9611f0dd/"
         "ghg-conversion-factors-2025-flat-format.xlsx")


def _get(url: str, name: str) -> Path:
    p = RAW / name
    if not p.exists():
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ClimRisk-LCA/1.0"}), timeout=300) as r:
            p.write_bytes(r.read())
    return p


# ── IPCC 2006 Guidelines, Vol. 2 (Energy) ─────────────────────────────────────
# CO2: Table 1.4 default emission factors (kg CO2 / TJ, NCV basis)
# NCV: Table 1.2 default net calorific values (TJ / Gg = GJ / t)
# CH4, N2O: Table 2.3 manufacturing industries & construction (kg / TJ)
FUELS = {
    #  key                 label                           CO2    NCV   CH4  N2O  biogenic  DESNZ WTT key
    "natural_gas":      ("Natural gas",                    56100, 48.0,  1, 0.1, False, ("Gaseous fuels", "Natural gas")),
    "fuel_oil":         ("Residual fuel oil",              77400, 40.4,  3, 0.6, False, ("Liquid fuels", "Fuel oil")),
    "diesel":           ("Gas/diesel oil",                 74100, 43.0,  3, 0.6, False, ("Liquid fuels", "Gas oil")),
    "lpg":              ("LPG",                            63100, 47.3,  1, 0.1, False, ("Gaseous fuels", "LPG")),
    "petcoke":          ("Petroleum coke",                 97500, 32.5,  3, 0.6, False, ("Solid fuels", "Petroleum coke")),
    "coal_bituminous":  ("Other bituminous coal",          94600, 25.8, 10, 1.5, False, ("Solid fuels", "Coal (industrial)")),
    "coal_subbituminous": ("Sub-bituminous coal",          96100, 18.9, 10, 1.5, False, ("Solid fuels", "Coal (industrial)")),
    "lignite":          ("Lignite",                       101000, 11.9, 10, 1.5, False, ("Solid fuels", "Coal (industrial)")),
    "coking_coal":      ("Coking coal",                    94600, 28.2, 10, 1.5, False, ("Solid fuels", "Coking coal")),
    "waste_oil":        ("Waste oils",                     73300, 40.2, 30, 4.0, False, None),
    "industrial_waste": ("Industrial wastes (fossil)",    143000, None, 30, 4.0, False, None),
    "tyres":            ("Waste tyres (fossil fraction)",  85000, None, 30, 4.0, False, None),
    "wood":             ("Wood / wood waste (biogenic)",  112000, 15.6, 30, 4.0, True,  None),
    "biomass_other":    ("Other primary solid biomass (biogenic)", 100000, 11.6, 30, 4.0, True, None),
}

GWP_AR6 = {"CO2": 1.0, "CH4_fossil": 29.8, "CH4_nonfossil": 27.0, "N2O": 273.0, "CF4": 7380.0, "C2F6": 12400.0}

# ── IPCC 2006 Vol. 3 process defaults (Tier 1) ────────────────────────────────
PROCESS = {
    "clinker_default_kgco2_per_t": {"value": 525.0,
        "source": "GCCA/WBCSD-CSI Cement CO2 & Energy Protocol v3.1 default calcination factor (525 kg CO2/t clinker); "
                  "IPCC 2006 Vol.3 Ch.2 Tier 1 gives 0.52 t (0.51 × 1.02 CKD correction)"},
    "cao_to_co2": {"value": 44.01 / 56.08, "source": "stoichiometry CaCO3 → CaO + CO2 (IPCC 2006 Vol.3 Eq. 2.4)"},
    "mgo_to_co2": {"value": 44.01 / 40.30, "source": "stoichiometry MgCO3 → MgO + CO2"},
    "raw_meal_toc_default": {"value": 0.002, "source": "GCCA/CSI Protocol default organic carbon content of raw meal (0.2 %)"},
    "raw_meal_to_clinker_default": {"value": 1.55, "source": "GCCA/CSI Protocol default raw meal : clinker ratio"},
    "steel_bof_tco2_per_t": {"value": 1.46, "source": "IPCC 2006 Vol.3 Table 4.1 Tier 1, basic oxygen furnace"},
    "steel_eaf_tco2_per_t": {"value": 0.08, "source": "IPCC 2006 Vol.3 Table 4.1 Tier 1, electric arc furnace"},
    "steel_dri_tco2_per_t": {"value": 0.70, "source": "IPCC 2006 Vol.3 Table 4.1 Tier 1, direct reduced iron"},
    "aluminium_prebake_tco2_per_t": {"value": 1.6, "source": "IPCC 2006 Vol.3 Table 4.10 Tier 1, prebake anode consumption"},
    "ammonia_gas_gj_per_t": {"value": 30.2, "source": "IPCC 2006 Vol.3 Table 3.1 Tier 1, modern natural-gas plant fuel requirement"},
}


def build():
    # Ember
    em = pd.read_csv(_get(EMBER, "ember_yearly.csv"), low_memory=False)
    ci = em[(em["Variable"] == "CO2 intensity") & (em["Area type"] == "Country or economy")].dropna(subset=["ISO 3 code"])
    latest = ci.sort_values("Year").groupby("ISO 3 code").last()
    grid = {iso: {"g_per_kwh": round(float(r["Value"]), 1), "year": int(r["Year"]), "name": r["Area"]}
            for iso, r in latest.iterrows()}
    world = ci[ci["Area"] == "World"] if (ci["Area"] == "World").any() else em[(em["Variable"] == "CO2 intensity") & (em["Area"] == "World")]
    world_row = world.sort_values("Year").iloc[-1] if len(world) else None

    # DESNZ
    x = pd.read_excel(_get(DESNZ, "desnz_2025_flat.xlsx"), "Factors by Category", header=None)
    hdr = x.index[x.iloc[:, 0].astype(str).str.contains("ID", na=False)][0]
    d = pd.read_excel(RAW / "desnz_2025_flat.xlsx", "Factors by Category", header=hdr)
    V = "GHG Conversion Factor 2025"
    kg = d[d["GHG/Unit"] == "kg CO2e"]

    def one(q):
        v = kg.query(q, engine="python")[V].dropna()
        return float(v.iloc[0]) if len(v) else None

    fuels = {}
    for k, (label, co2, ncv, ch4, n2o, bio, wtt) in FUELS.items():
        w = None
        if wtt:
            w = one(f"`Level 1`=='WTT- fuels' and `Level 2`=='{wtt[0]}' and `Level 3`=='{wtt[1]}' and UOM=='kWh (Net CV)'")
        fuels[k] = {"label": label, "co2_kg_per_gj": co2 / 1000, "ch4_kg_per_gj": ch4 / 1000, "n2o_kg_per_gj": n2o / 1000,
                    "ncv_gj_per_t": ncv, "biogenic": bio,
                    "wtt_kgco2e_per_gj": None if w is None else round(w * 277.778, 3),
                    "source": "IPCC 2006 Vol.2 Tables 1.2, 1.4, 2.3" + ("; WTT: UK DESNZ 2025" if w else "")}
    freight = {
        "truck": {"label": "HGV, all diesel, average laden",
                  "kgco2e_per_tkm": one("`Level 1`=='Freighting goods' and `Level 3`=='All HGVs' and `Column Text`=='Average laden' and UOM=='tonne.km'"),
                  "wtt_kgco2e_per_tkm": one("`Level 1`=='WTT- delivery vehs & freight' and `Level 3`=='All HGVs' and `Column Text`=='Average laden' and UOM=='tonne.km'")},
        "rail": {"label": "Freight train",
                 "kgco2e_per_tkm": one("`Level 1`=='Freighting goods' and `Level 3`=='Freight train' and UOM=='tonne.km'"),
                 "wtt_kgco2e_per_tkm": one("`Level 1`=='WTT- delivery vehs & freight' and `Level 3`=='Freight train' and UOM=='tonne.km'")},
        "ship": {"label": "Bulk carrier, average",
                 "kgco2e_per_tkm": one("`Level 1`=='Freighting goods' and `Level 3`=='Bulk carrier' and `Level 4`=='Average' and UOM=='tonne.km'"),
                 "wtt_kgco2e_per_tkm": one("`Level 1`=='WTT- delivery vehs & freight' and `Level 3`=='Bulk carrier' and `Level 4`=='Average' and UOM=='tonne.km'")},
    }
    for f in freight.values():
        f["source"] = "UK DESNZ GHG Conversion Factors 2025 (OGL v3.0)"

    # NGFS: direct grid CO2 intensity index (2025 = 1) by model region & scenario
    import pyam
    ng = RAW / "ngfs_elec.csv"
    if not ng.exists():
        pyam.read_iiasa("ngfs_phase_5", variable=["Emissions|CO2|Energy|Supply|Electricity", "Secondary Energy|Electricity"],
                        model=["GCAM 6.0 NGFS", "MESSAGEix-GLOBIOM 2.0-M-R12-NGFS", "REMIND-MAgPIE 3.3-4.8"]).to_csv(ng)
    n = pd.read_csv(ng); n.columns = [c.lower() for c in n.columns]
    years = [str(y) for y in range(2025, 2051, 5)]
    idx: dict = {}
    for (m, r, s), g in n.groupby(["model", "region", "scenario"]):
        e = g[g["variable"] == "Emissions|CO2|Energy|Supply|Electricity"]
        q = g[g["variable"] == "Secondary Energy|Electricity"]
        if e.empty or q.empty:
            continue
        inten = (e[years].values[0].astype(float) / q[years].values[0].astype(float))
        if not np.isfinite(inten[0]) or inten[0] <= 0:
            continue
        idx.setdefault(m, {}).setdefault(r, {})[s] = [round(float(max(v / inten[0], 0.0)), 4) if np.isfinite(v) else None
                                                       for v in inten]
    out = {
        "version": "2026-09-30",
        "gwp": {"set": "IPCC AR6 WG1 Table 7.15 (GWP100)", **GWP_AR6},
        "fuels": fuels,
        "process": PROCESS,
        "freight": freight,
        "grid": {"source": "Ember Yearly Electricity Data (CC BY 4.0) — lifecycle gCO2e/kWh of generation, all gases, "
                           "100-yr; excludes transmission & distribution losses",
                 "world_g_per_kwh": None if world_row is None else round(float(world_row["Value"]), 1),
                 "countries": grid},
        "grid_index": {"source": "NGFS Phase 5 (IIASA): Emissions|CO2|Energy|Supply|Electricity ÷ Secondary Energy|Electricity, "
                                 "indexed to 2025 = 1", "years": [int(y) for y in years], "data": idx},
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"lca factors: {len(fuels)} fuels, {len(grid)} grids, freight {[(k, v['kgco2e_per_tkm']) for k, v in freight.items()]} "
          f"→ {OUT.stat().st_size/1e3:.0f} kB")


if __name__ == "__main__":
    build()
