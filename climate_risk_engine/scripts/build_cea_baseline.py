"""Write src/cri/data/cea_baseline.json: the seed for the CEA CO2 Baseline Database (Version 22.0, August 2026).

The numbers below were transcribed by eye from the published User Guide PDF (Annexure I, Table S, Figures 4-6, revision history) and
are checked here before anything is written:
  * weighted average = column 2 (CO2) / column 5 (total net generation), within 0.0006 tCO2/MWh
  * before FY2022-23, total net generation = conventional + RE, within 0.02 BU. From FY2022-23 CEA adds cross-border and captive
    injection to the total (Annexure I note), so the gap is stored as an explicit adjustment instead of being forced to zero.
  * combined margin = (OM + BM) / 2, within 0.0011 (CEA rounds each to three decimals)
A transcription error fails the build. Run:  python scripts/build_cea_baseline.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "cri" / "data" / "cea_baseline.json"

WA = """2013-14 727.4 886.77 53.06 939.83 0.774
2014-15 805.4 972.04 61.72 1033.76 0.779
2015-16 846.3 1027.03 65.78 1092.81 0.774
2016-17 888.34 1072.84 81.55 1154.39 0.770
2017-18 922.18 1121.57 101.84 1223.41 0.754
2018-19 960.9 1165.16 126.76 1291.92 0.744
2019-20 928.14 1162.97 138.34 1301.31 0.713
2020-21 910.02 1147.52 147.25 1294.77 0.703
2021-22 1002.01 1230.09 170.91 1401.01 0.715
2022-23 1108.11 1320.18 203.55 1547.80 0.716
2023-24 1203.36 1408.45 225.83 1654.54 0.727
2024-25 1235.96 1461.85 255.01 1740.54 0.710
2025-26 1189.29 1424.19 310.58 1761.28 0.675"""
OM = {"2021-22": 0.960, "2022-23": 0.971, "2023-24": 0.961, "2024-25": 0.961, "2025-26": 0.963}
BM = {"2021-22": 0.869, "2022-23": 0.867, "2023-24": 0.552, "2024-25": 0.512, "2025-26": 0.446}
CM = {"2021-22": 0.915, "2022-23": 0.919, "2023-24": 0.757, "2024-25": 0.736, "2025-26": 0.705}
HIST = [
    ("1.0 Draft", "October 2006", "Draft for stakeholder consultation"),
    ("1.0", "November 2006", "Added data on 10 stations from the exclusion worksheet; adjusted to IPCC 2006 Guidelines values"),
    ("1.1", "December 2006", "Adjusted fuel emission factor of lignite"),
    ("2.0", "June 2007", "Added FY2005-06 data"),
    ("3.0", "December 2007", "Added FY2006-07; aligned with ACM0002 v07 and the Tool to calculate the emission factor for an electricity system (v01.1)"),
    ("4.0", "October 2008", "Added FY2007-08; adjusted regional grid delineation and fuel factors for uncertainty"),
    ("5.0", "November 2009", "Added FY2008-09"),
    ("6.0", "March 2011", "Added FY2009-10"),
    ("7.0", "January 2012", "Added FY2010-11"),
    ("8.0", "January 2013", "Added FY2011-12; database restricted to stations above 25 MW from FY2011-12"),
    ("9.0", "December 2013", "Added FY2012-13; retroactive build margin changes; updated GCVs of five stations"),
    ("10.0", "December 2014", "Added FY2013-14; separated Indian and imported coal; retroactive changes"),
    ("11.0", "April 2016", "Added FY2014-15; introduced the integrated single Indian grid (NEWNE and Southern synchronised); Bangladesh exports considered"),
    ("12.0", "May 2017", "Added FY2015-16"),
    ("13.0", "June 2018", "Added FY2016-17; Myanmar exports considered"),
    ("14.0", "December 2018", "Added FY2017-18"),
    ("15.0", "December 2019", "Added FY2018-19"),
    ("16.0", "March 2021", "Added FY2019-20; some retroactive changes to FY2018-19"),
    ("17.0", "October 2021", "Added FY2020-21"),
    ("18.0", "September 2022", "Added FY2021-22"),
    ("19.0", "December 2023", "Added FY2022-23; unit-wise emission factor for FY2022-23; captive power injection into the grid incorporated; renewable open-access transactions adjusted to avoid double counting"),
    ("20.0", "December 2024", "Added FY2023-24; build margin and combined margin now account for renewable generation"),
    ("21.0", "November 2025", "Added FY2024-25; actual captive data replaces provisional data for FY2022-23 and FY2023-24"),
    ("22.0", "August 2026", "Added FY2025-26; actual captive data replaces provisional data for FY2024-25"),
]


def build() -> dict:
    series = {}
    for line in WA.splitlines():
        fy, co2, conv, re_, tot, f = line.split()
        co2, conv, re_, tot, f = map(float, (co2, conv, re_, tot, f))
        assert abs(co2 / tot - f) < 0.0006, (fy, co2 / tot, f)
        start = int(fy[:4])
        adj = round(tot - conv - re_, 2)
        if start < 2022:
            assert abs(adj) < 0.02, (fy, adj)
            adj = 0.0
        series[fy] = {"co2_mt": co2, "net_generation_conventional_bu": conv, "re_generation_bu": re_,
                      "total_net_generation_bu": tot, "adjustment_bu_captive_and_cross_border": adj,
                      "weighted_average_tco2_per_mwh": f}
    margins = {}
    for k in OM:
        assert abs((OM[k] + BM[k]) / 2 - CM[k]) < 0.0011, k
        margins[k] = {"om": OM[k], "bm": BM[k], "cm": CM[k]}
    return {
        "_comment": "Seed values transcribed from the CEA CO2 Baseline Database User Guide, Version 22.0 (August 2026): Annexure I, Table S, Figures 4-6, revision history. "
                    "Built and cross-checked by scripts/build_cea_baseline.py. Fuel-level emission factors and station data are in the CEA XLSX and are NOT in this file.",
        "publisher": "Central Electricity Authority (CEA), Clean Energy & Energy Transition (CE&ET) Division, Ministry of Power, Government of India",
        "title": "CO2 Baseline Database for the Indian Power Sector",
        "version": "22.0", "published": "August 2026", "latest_fy": "2025-26", "retrieved": "2026-11-06",
        "listing_url": "https://cea.nic.in/cdm-co2-baseline-view?mappingId=234&lang=en",
        "user_guide_url": "https://cea.nic.in/api/cdm-report-version/view-pdf/en/82",
        "database_url": "https://cea.nic.in/api/cdm-report-version/view-pdf/en/83",
        "licence": "Official publication of the Government of India for the purpose of CDM baselines; reproduced as published data with attribution. Check CEA terms before redistributing.",
        "scope_statement": "The database is an official publication of the Government of India for the purpose of CDM baselines (User Guide s.1). It is CO2 only, covers grid-connected "
                           "stations, and is published once for the single Indian grid (all regional grids integrated in December 2013). CEA publishes no state-wise or utility-wise grid factors.",
        "definitions": {
            "weighted_average": "Average emission of all stations in the grid, weighted by net generation, including renewable generation (tCO2/MWh).",
            "om": "Operating margin: average emission from all stations excluding low-cost/must-run sources.",
            "bm": "Build margin: average emission of the 20% (by net generation) most recent capacity addition in the grid, including renewables.",
            "cm": "Combined margin: weighted average of OM and BM, here 50:50."},
        "method_notes": [
            "From FY2022-23 total CO2 and net generation are adjusted for cross-border electricity transfers and include electricity injected into the grid by grid-connected captive plants (Annexure I note). Earlier years are not on that basis, so the series has a break at FY2022-23.",
            "From FY2023-24 renewable generation is accounted for in the build margin and combined margin; FY2021-22 and FY2022-23 margins are on the earlier basis.",
            "Consistent with version 7.0 of the CDM Tool to calculate the emission factor for an electricity system.",
            "Factors are for fiscal years (April to March), in tCO2/MWh = kg CO2/kWh. They are direct CO2 from power stations on net generation: no CH4 or N2O, no upstream fuel supply chain, no transmission and distribution losses.",
            "CEA updates the database annually. The latest year's captive data is provisional and is replaced by actuals in the next version (see revision history)."],
        "weighted_average": series,
        "margins": margins,
        "revision_history": [{"version": v, "published": p, "changes": c} for v, p, c in HIST],
        "installed_capacity_mw_31_03_2026": {"coal": 221940.01, "lignite": 6620.00, "gas": 20122.41, "diesel": 589.20, "thermal_total": 249271.63,
                                              "nuclear": 8780, "hydro": 51414.66, "res": 223273.43, "total": 532739.72},
    }


if __name__ == "__main__":
    d = build()
    OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(d['weighted_average'])} fiscal years, {len(d['revision_history'])} versions)")
