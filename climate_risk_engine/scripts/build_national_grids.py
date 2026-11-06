"""Write src/cri/data/national_grids.json: official national grid emission factors for Australia, Canada, Germany, China and Japan.

Every number below was read on 6 Nov 2026 from the publisher's own document (not from an aggregator):
  AUS  DCCEEW, Australian National Greenhouse Accounts Factors 2026, Table 1 (location-based) and Table 2 (residual mix) - PDF text read page by page.
  CAN  ECCC, National Inventory Report 1990-2024 (April 2026), Annex 7 workbook EN_Annex7_Electricity_Intensity.xlsx, tables A7-1 to A7-14.
  DEU  Umweltbundesamt, CLIMATE CHANGE 16/2026 (March 2026), Tables 1-3.
  CHN  MEE + NBS, Announcement 2025 No. 47 (31 Dec 2025), Annex tables 1-5; year-on-year figures from the MEE Q&A of 5 Jan 2026.
  JPN  MOE/METI, supplier emission factors for the SHK system, R6 results (published 9 Jan 2026, revised to rev4 on 3 Aug 2026): the substitute value.
Checks run before anything is written; a transcription error fails the build.   Run:  python scripts/build_national_grids.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "cri" / "data" / "national_grids.json"


# ── Australia ────────────────────────────────────────────────────────────────
# code, name, scope 2 kg CO2-e/kWh, scope 3 kg CO2-e/kWh  (Table 1, 2026 edition)
AU = [("NSW", "New South Wales and Australian Capital Territory", 0.60, 0.07), ("VIC", "Victoria", 0.74, 0.11), ("QLD", "Queensland", 0.65, 0.11),
      ("SA", "South Australia", 0.21, 0.03), ("WA-SWIS", "Western Australia - South West Interconnected System (SWIS)", 0.45, 0.10),
      ("WA-NWIS", "Western Australia - North Western Interconnected System (NWIS)", 0.56, 0.09), ("TAS", "Tasmania", 0.21, 0.03),
      ("NT", "Northern Territory - Darwin Katherine Interconnected System (DKIS)", 0.55, 0.13), ("NATIONAL", "National", 0.59, 0.10)]
AU_RESIDUAL = (0.79, 0.11)            # Table 2: residual mix factor for the market-based method (national)

# ── Canada ───────────────────────────────────────────────────────────────────
# code, name, generation intensity 2024, consumption intensity 2024, generation 2023, consumption 2023  (g CO2 eq / kWh; Annex 7 tables)
CA = [("CA", "Canada", 102.17, 110.55, 105.27, 112.86), ("NL", "Newfoundland and Labrador", 16.04, 17.04, 15.66, 16.70),
      ("PE", "Prince Edward Island", 3.19, 264.55, 7.19, 176.02), ("NS", "Nova Scotia", 585.93, 628.35, 574.59, 613.85),
      ("NB", "New Brunswick", 361.70, 375.48, 225.58, 233.99), ("QC", "Quebec", 2.10, 2.56, 1.48, 1.87),
      ("ON", "Ontario", 64.91, 73.31, 53.94, 58.86), ("MB", "Manitoba", 2.70, 3.04, 2.23, 2.50),
      ("SK", "Saskatchewan", 541.83, 580.72, 595.26, 631.74), ("AB", "Alberta", 334.76, 346.10, 420.30, 433.87),
      ("BC", "British Columbia", 16.71, 18.06, 16.58, 17.96), ("YT", "Yukon", 120.66, 133.57, 66.55, 74.00),
      ("NT", "Northwest Territories", 460.0, 490.0, 390.0, 420.0), ("NU", "Nunavut", 740.0, 760.0, 780.0, 800.0)]

# ── Germany ──────────────────────────────────────────────────────────────────
# year: (CO2 Mt, consumption TWh, CO2 g/kWh, inland-consumption CO2 g/kWh, GHG g CO2e/kWh without upstream, GHG with upstream, upstream, status)
DE = {2020: (187, 513, 365, 379, 373, 435, 62, "inventory"), 2021: (215, 530, 406, 421, 414, 477, 63, "inventory"),
      2022: (223, 516, 433, 457, 441, 503, 62, "inventory"), 2023: (172, 455, 379, 371, 387, 442, 56, "inventory"),
      2024: (158, 447, 353, 334, 361, 414, 53, "provisional"), 2025: (154, 449, 344, 330, 352, 406, 54, "estimate")}

# ── China ────────────────────────────────────────────────────────────────────
CN_NATIONAL = 0.5306
CN_NATIONAL_EXCL_MARKET_NONFOSSIL = 0.6096
CN_FOSSIL = 0.8273
CN_2022 = {"national": 0.5366, "national_excl_market_nonfossil": 0.5856, "fossil": 0.8325}        # MEE Q&A, 5 Jan 2026
CN_REGIONS = [("NORTH", "North China", "华北", 0.6361), ("NORTHEAST", "Northeast China", "东北", 0.5122), ("EAST", "East China", "华东", 0.5500),
              ("CENTRAL", "Central China", "华中", 0.5271), ("NORTHWEST", "Northwest China", "西北", 0.5543), ("SOUTH", "South China (China Southern Power Grid area)", "南方", 0.4042),
              ("SOUTHWEST", "Southwest China", "西南", 0.2472)]
CN_PROVINCES = [("BJ", "Beijing", "北京", 0.5554), ("TJ", "Tianjin", "天津", 0.6796), ("HE", "Hebei", "河北", 0.6516), ("SX", "Shanxi", "山西", 0.6634),
                ("NM", "Inner Mongolia", "内蒙古", 0.6479), ("LN", "Liaoning", "辽宁", 0.4878), ("JL", "Jilin", "吉林", 0.4671), ("HL", "Heilongjiang", "黑龙江", 0.5229),
                ("SH", "Shanghai", "上海", 0.5737), ("JS", "Jiangsu", "江苏", 0.5827), ("ZJ", "Zhejiang", "浙江", 0.4974), ("AH", "Anhui", "安徽", 0.6553),
                ("FJ", "Fujian", "福建", 0.4211), ("JX", "Jiangxi", "江西", 0.5836), ("SD", "Shandong", "山东", 0.6191), ("HA", "Henan", "河南", 0.5897),
                ("HB", "Hubei", "湖北", 0.4044), ("HN", "Hunan", "湖南", 0.4976), ("GD", "Guangdong", "广东", 0.4419), ("GX", "Guangxi", "广西", 0.4476),
                ("HI", "Hainan", "海南", 0.3648), ("CQ", "Chongqing", "重庆", 0.5581), ("SC", "Sichuan", "四川", 0.1564), ("GZ", "Guizhou", "贵州", 0.5683),
                ("YN", "Yunnan", "云南", 0.1333), ("SN", "Shaanxi", "陕西", 0.6335), ("GS", "Gansu", "甘肃", 0.4471), ("QH", "Qinghai", "青海", 0.1796),
                ("NX", "Ningxia", "宁夏", 0.6187), ("XJ", "Xinjiang", "新疆", 0.6021)]

# ── Japan ────────────────────────────────────────────────────────────────────
JP_SUBSTITUTE_T_PER_KWH = 0.000422


def _check() -> None:
    # Australia
    s2 = {c: a for c, _, a, _ in AU}
    assert len(AU) == 9 and all(0 < a < 1 and 0 < b < 0.3 for _, _, a, b in AU)
    assert min(v for k, v in s2.items() if k != "NATIONAL") <= s2["NATIONAL"] <= max(v for k, v in s2.items() if k != "NATIONAL")
    assert AU_RESIDUAL[0] > s2["NATIONAL"], "the residual mix should be dirtier than the grid average"
    # Canada
    assert len(CA) == 14
    prov = [r for r in CA if r[0] != "CA"]
    nat = next(r for r in CA if r[0] == "CA")
    assert min(r[2] for r in prov) <= nat[2] <= max(r[2] for r in prov)
    for c, _, g24, c24, g23, c23 in CA:
        assert c24 >= g24 - 0.005 and c23 >= g23 - 0.005, f"consumption intensity should not be below generation intensity ({c})"
    # Germany: CO2 = emissions / consumption (rounded inputs), GHG with upstream = without + upstream
    for y, (mt, twh, co2, inl, ghg, ghg_up, up, _) in DE.items():
        assert abs(mt * 1000 / twh - co2) <= 2.0, (y, mt * 1000 / twh, co2)
        assert abs(ghg + up - ghg_up) <= 1, (y, ghg + up, ghg_up)
        assert co2 < ghg < ghg_up
    # China
    assert len(CN_PROVINCES) == 30 and len(CN_REGIONS) == 7
    assert len({c for c, *_ in CN_PROVINCES}) == 30
    for _, _, _, v in CN_PROVINCES + CN_REGIONS:
        assert 0.1 < v < 0.8
    # the MEE Q&A states the year-on-year change; check our figures reproduce it
    assert round((CN_NATIONAL / CN_2022["national"] - 1) * 100, 1) == -1.1
    assert round((CN_NATIONAL_EXCL_MARKET_NONFOSSIL / CN_2022["national_excl_market_nonfossil"] - 1) * 100, 1) == 4.1
    assert round((CN_FOSSIL / CN_2022["fossil"] - 1) * 100, 1) == -0.6
    assert CN_NATIONAL < CN_NATIONAL_EXCL_MARKET_NONFOSSIL < CN_FOSSIL
    assert 0.0003 < JP_SUBSTITUTE_T_PER_KWH < 0.0006


def build() -> dict:
    _check()
    aus = {
        "name": "Australia", "publisher": "Department of Climate Change, Energy, the Environment and Water (DCCEEW), Australian Government",
        "title": "Australian National Greenhouse Accounts Factors 2026", "edition": [2026], "label": "NGA Factors 2026",
        "published": "2026 edition (the document refers to a decision of 24 July 2026, so it was issued after that date)",
        "url": "https://www.dcceew.gov.au/climate-change/publications/national-greenhouse-accounts-factors",
        "document_url": "https://www.dcceew.gov.au/sites/default/files/documents/national-greenhouse-accounts-factors-2026.pdf",
        "period": "financial years ending June (July to June)", "unit": "kg CO2-e/kWh in the source; stored as g CO2e/kWh",
        "gwp_basis": "AR5", "gases": "CO2+CH4+N2O (AR5, including methane from hydro dams)", "kind": "state or grid",
        "scope_statement": "Scope 2 factors are location-based, for electricity purchased from the grid of a state or interconnected system. Scope 3 factors cover upstream fuel supply and transmission/distribution losses. "
                           "Self-consumption of small-scale rooftop solar is deducted from the estimate (Table 1 notes).",
        "default_region": "NATIONAL",
        "regions": {c: {"name": n, "g_per_kwh": round(s * 1000), "scope3_g_per_kwh": round(t * 1000)} for c, n, s, t in AU},
        "aliases": {"ACT": "NSW", "NSW/ACT": "NSW", "NEW SOUTH WALES": "NSW", "VICTORIA": "VIC", "QUEENSLAND": "QLD", "SOUTH AUSTRALIA": "SA", "TASMANIA": "TAS",
                    "NORTHERN TERRITORY": "NT", "SWIS": "WA-SWIS", "NWIS": "WA-NWIS", "WA": "WA-SWIS", "WESTERN AUSTRALIA": "WA-SWIS", "AUSTRALIA": "NATIONAL"},
        "region_notes": {"WA-SWIS": "'WA' without a grid is taken as the South West Interconnected System (Perth area); the Pilbara/North West grid is 'WA-NWIS'."},
        "residual_mix": {"g_per_kwh": round(AU_RESIDUAL[0] * 1000), "scope3_g_per_kwh": round(AU_RESIDUAL[1] * 1000), "source": "NGA Factors 2026 Table 2, national residual mix factor for the market-based method"},
        "notes": ["The same table is republished every year, usually mid-year; check the edition (GET /data/sources).",
                  "The NGA workbook also holds fuel factors; only the electricity tables are held here."]}
    can = {
        "name": "Canada", "publisher": "Environment and Climate Change Canada (ECCC)", "title": "National Inventory Report 1990-2024, Annex 7: Electricity in Canada - summary and intensity tables",
        "edition": [2024, "2026-04-14"], "label": "NIR 2026 (data to 2024)", "published": "2026-04-14",
        "url": "https://open.canada.ca/data/en/dataset/779c7bcf-4982-47eb-af1b-a33618a05e5b",
        "document_url": "https://data-donnees.ec.gc.ca/data/substances/monitor/canada-s-official-greenhouse-gas-inventory/C-Tables-Electricity-Canada-Provinces-Territories/EN_Annex7_Electricity_Intensity.xlsx",
        "period": "calendar year; 2024 is preliminary (table footnote a)", "unit": "g CO2 eq / kWh", "gwp_basis": "AR5",
        "gwp_note": "The Annex 7 workbook says 'CO2 eq' but does not name the GWP set; Canada's inventory uses IPCC AR5 100-year values (stated in the NIR's methodology chapters, not re-read for this file). Treat as AR5 until confirmed.",
        "gases": "CO2+CH4+N2O (CO2 eq)", "kind": "province or territory", "default_region": "CA", "default_year": 2024,
        "scope_statement": "Generation intensity is GHG emitted per kWh generated in the jurisdiction (NAICS 22111). Consumption intensity adds unallocated energy (transmission losses, metering differences) and SF6 from electrical equipment, "
                           "and is on electricity consumed, so it reflects imports. Emissions from flooding of land for hydro dams are not included (footnote e).",
        "regions": {c: {"name": n, "g_per_kwh": {"2024": g24, "2023": g23}, "consumption_g_per_kwh": {"2024": c24, "2023": c23}} for c, n, g24, c24, g23, c23 in CA},
        "aliases": {"CANADA": "CA", "NEWFOUNDLAND": "NL", "NEWFOUNDLAND AND LABRADOR": "NL", "PEI": "PE", "PRINCE EDWARD ISLAND": "PE", "NOVA SCOTIA": "NS", "NEW BRUNSWICK": "NB",
                    "QUEBEC": "QC", "QUÉBEC": "QC", "ONTARIO": "ON", "MANITOBA": "MB", "SASKATCHEWAN": "SK", "ALBERTA": "AB", "BRITISH COLUMBIA": "BC", "YUKON": "YT",
                    "NORTHWEST TERRITORIES": "NT", "NUNAVUT": "NU"},
        "notes": ["Intensity values are rounded to reflect their accuracy (footnote n); the Northwest Territories and Nunavut are rounded to the nearest 10 g.",
                  "Prince Edward Island generates little but imports most of its electricity: generation intensity 3 g vs consumption intensity 265 g in 2024. For consumers there, use the consumption figure (returned alongside).",
                  "The provincial figures can move a lot year to year (Alberta 420 g in 2023, 335 g in 2024); say which year you report."]}
    de_series = {str(y): {"co2_mt": mt, "consumption_twh": twh, "co2_g_per_kwh": co2, "inland_consumption_co2_g_per_kwh": inl, "ghg_g_per_kwh": ghg,
                          "ghg_with_upstream_g_per_kwh": ghg_up, "upstream_g_per_kwh": up, "status": st}
                 for y, (mt, twh, co2, inl, ghg, ghg_up, up, st) in DE.items()}
    deu = {
        "name": "Germany", "publisher": "Umweltbundesamt (UBA), German Environment Agency",
        "title": "Entwicklung der spezifischen Treibhausgas-Emissionen des deutschen Strommix in den Jahren 1990 - 2025 (CLIMATE CHANGE 16/2026)", "edition": [2025], "label": "UBA CLIMATE CHANGE 16/2026 (years 1990-2025)",
        "published": "2026-03", "url": "https://www.umweltbundesamt.de/themen/luft/emissionen-von-luftschadstoffen/spezifische-emissionsfaktoren-fuer-den-deutschen",
        "document_url": "https://www.umweltbundesamt.de/system/files/medien/11850/publikationen/2026-03/16_2026_CC.pdf",
        "period": "calendar year; 2024 provisional, 2025 estimated", "unit": "g CO2 / kWh and g CO2 eq / kWh", "gwp_basis": "AR5",
        "gwp_note": "UBA converts CH4 and N2O with IPCC Fifth Assessment Report potentials (section 2.8).", "gases": "CO2+CH4+N2O (AR5), direct emissions of electricity consumed",
        "kind": "national", "default_region": "DE", "default_year": 2025,
        "scope_statement": "Direct emissions per kWh of the German electricity mix (power-station combustion, no upstream chain unless stated). Emissions of exports are counted in Germany, so the factor follows IPCC rules; "
                           "an inland-consumption variant that nets the trade balance is also given.",
        "regions": {"DE": {"name": "Germany", "series": de_series}}, "aliases": {"GERMANY": "DE", "DEUTSCHLAND": "DE"},
        "notes": ["The headline UBA 'Emissionsfaktor Strommix' is the CO2-only value (344 g in 2025). The engine returns the GHG (CO2e) value without upstream as g_per_kwh and keeps CO2 only and with-upstream alongside.",
                  "UBA's own text calls the 2023 to 2025 decline not necessarily lasting (net electricity imports, weak demand).",
                  "The 2025 value is an estimate; for a finished reporting year prefer 2023 (inventory-based) or 2024 (provisional) and say which."]}
    reg = {c: {"name": n, "name_zh": zh, "g_per_kwh": round(v * 1000, 1), "level": "region"} for c, n, zh, v in CN_REGIONS}
    reg.update({c: {"name": n, "name_zh": zh, "g_per_kwh": round(v * 1000, 1), "level": "province"} for c, n, zh, v in CN_PROVINCES})
    reg["CN"] = {"name": "China (national)", "name_zh": "全国", "g_per_kwh": round(CN_NATIONAL * 1000, 1), "level": "national"}
    chn = {
        "name": "China", "publisher": "Ministry of Ecology and Environment (MEE) with the National Bureau of Statistics (NBS)",
        "title": "Announcement 2025 No. 47: 2023 electricity CO2 emission factors (关于发布2023年电力二氧化碳排放因子的公告)", "edition": [2023], "label": "MEE/NBS electricity CO2 emission factors, data year 2023",
        "published": "2025-12-31", "url": "https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/t20251231_1139517.html",
        "document_url": "https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/W020251231726284332528.pdf",
        "qa_url": "https://www.mee.gov.cn/ywdt/zbft/202601/t20260105_1139911.shtml",
        "period": "calendar year 2023 (published about two years later)", "unit": "kg CO2/kWh in the source; stored as g CO2/kWh", "gwp_basis": "CO2 only", "gases": "CO2 only",
        "kind": "national, region or province", "default_region": "CN",
        "scope_statement": "Average CO2 per kWh of all electricity generated (thermal, hydro, wind, nuclear, solar), at national, regional and provincial level, for the location-based method. A second national value excludes non-fossil "
                           "electricity traded on the market (including green-power trading) and is meant for the market-based method; a third covers fossil-fuel electricity only. Same method and data sources as the 2021 and 2022 releases.",
        "regions": reg, "aliases": {"CHINA": "CN", **{zh: c for c, _, zh, _ in CN_REGIONS}, **{zh: c for c, _, zh, _ in CN_PROVINCES}, **{n.upper(): c for c, n, _, _ in CN_PROVINCES}},
        "residual_mix": {"g_per_kwh": round(CN_NATIONAL_EXCL_MARKET_NONFOSSIL * 1000, 1), "source": "MEE/NBS 2023, national electricity average excluding market-traded non-fossil electricity (Table 4), for the market-based method"},
        "extra": {"fossil_electricity_g_per_kwh": round(CN_FOSSIL * 1000, 1), "prior_year_2022": {k: round(v * 1000, 1) for k, v in CN_2022.items()}},
        "notes": ["Tibet, Hong Kong, Macau and Taiwan are not in the provincial table.",
                  "A buyer of traceable green electricity may count that quantity as zero; the rest uses the excluding-market-traded factor (MEE Q&A).",
                  "MEE also publishes a separate carbon-footprint factor (kgCO2e/kWh, 2024 data); it is not held here.",
                  "Province codes are ISO 3166-2:CN (HB is Hubei, HE is Hebei, HA is Henan, HN is Hunan)."]}
    jpn = {
        "name": "Japan", "publisher": "Ministry of the Environment and Ministry of Economy, Trade and Industry (SHK system)",
        "title": "Emission factors by electricity supplier for the GHG Emissions Calculation, Reporting and Publication System (R6 results, for use in FY2025 calculations)",
        "edition": [8, 4], "label": "SHK electricity supplier list, R8 submission (rev4)", "published": "2026-01-09 (revised to rev4, last on 2026-08-03)",
        "url": "https://policies.env.go.jp/earth/ghg-santeikohyo/calc.html", "document_url": "https://policies.env.go.jp/earth/ghg-santeikohyo/files/calc/r08_denki_coefficient_rev4.pdf",
        "period": "fiscal year R6 (April 2024 to March 2025) results", "unit": "t-CO2/kWh in the source; stored as g CO2/kWh", "gwp_basis": "CO2 only", "gases": "CO2 only (energy-origin CO2)",
        "kind": "supplier-specific", "default_region": "JP",
        "scope_statement": "Japan has no single official national grid factor. Factors are published per electricity retailer (basic and adjusted emission factors, some per menu). "
                           "The only country-wide number in the list is the substitute value (the table's asterisk), applied where a retailer's factor is abnormal or unavailable.",
        "regions": {"JP": {"name": "Japan (SHK substitute value)", "g_per_kwh": round(JP_SUBSTITUTE_T_PER_KWH * 1e6)}}, "aliases": {"JAPAN": "JP"},
        "notes": ["Use your retailer's published factor (basic or adjusted, by menu) when you know the supplier: enter it as supplier_g_per_kwh or location_g_per_kwh.",
                  "From the FY2025 reporting year the basic factor is published by menu, not only per company (list header).",
                  "Retailers that started supplying in R6 carry their R5 factor until the July 2026 update.",
                  "The ten general transmission and distribution operators carry a separate loss-compensation factor (0.000423 t-CO2/kWh in the list); that is not a consumption factor and is not used here."]}
    return {
        "_comment": "Official national grid emission factors, read from the publishers' own documents on 6 Nov 2026 and cross-checked by scripts/build_national_grids.py.",
        "retrieved": "2026-11-06", "countries": {"AUS": aus, "CAN": can, "DEU": deu, "CHN": chn, "JPN": jpn}}


if __name__ == "__main__":
    d = build()
    OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(d['countries'])} countries)")
