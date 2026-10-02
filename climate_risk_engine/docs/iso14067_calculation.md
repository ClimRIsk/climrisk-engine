# ISO 14067:2018 in the ClimRisk engine — what it calculates

`POST /lca/study` (and the "ISO 14067 study" panel in the LCA workbench) runs the full product carbon footprint
calculation for cement, steel, aluminium, ammonia or any custom product, and `POST /lca/study/report[.xlsx]`
writes the study report. Every figure below is computed by the engine from the inputs you give it.

| ISO 14067 element | What the engine computes |
|---|---|
| System boundary | Cradle-to-gate A1–A3, cradle-to-customer A1–A4 (both *partial* CFPs), cradle-to-grave A1–C (full CFP); the total under each boundary is shown |
| A1–A3 product stage | Materials, inbound freight, process CO2, fuels (IPCC 2006) + well-to-tank, lifecycle grid electricity (Ember), CO2 capture |
| A4 distribution | Truck / rail / ship (DESNZ 2025) and air (DESNZ 2026 freight flights) |
| B use stage | Electricity and fuel over the service life; grid factor static or following an NGFS Phase 5 pathway year by year |
| C end of life | 139 DESNZ 2026 waste-disposal factors by material and treatment (landfill, combustion, recycling, composting, digestion); cut-off approach, no recycling credit |
| Allocation | Mass, economic and energy shares for multi-output processes; the selected method is applied and all three are shown |
| GHG types | Fossil, aircraft, biogenic emissions, biogenic removals, direct land-use change reported separately; aircraft radiative-forcing uplift shown separately |
| Cut-off | Excluded mass share and emission share against your stated criterion |
| Uncertainty | Monte Carlo over every flow (5,000 draws) → P5 / P50 / P95 of the boundary total |
| Interpretation | Hotspots to 80 % of the total, ±10 % sensitivity per flow, effect of each methodological choice (allocation, capture, biogenic uptake, aircraft RF, 2035 grid) |
| Data quality §6.3.5 | All ten criteria scored 1–5 from the inputs actually used |
| Study report | Clause 9 content as JSON, Markdown and Excel, including a communication statement and a reproducibility fingerprint |

## Choices the engine makes visible
- Biogenic CO2 uptake is assumed equal to biogenic emissions unless you set `biogenic_uptake_share`.
- Indirect land-use change is excluded; direct land-use change is included only if supplied.
- Recycling takes no credit for avoided virgin production.
- Service life and end-of-life shares are stated scenario inputs, not sampled.

## What the engine cannot calculate
Whether an independent reviewer has examined the study. Every report therefore shows "critical review: not reviewed"
until one is recorded, and a public comparative assertion needs that review under the standard.

## Factor caveats shown in results
- BF-BOF steel uses the IPCC Tier 1 process factor plus electricity and reads below the worldsteel 2023 average of 2.32 t CO2/t; a benchmark note is attached to every steel result.
- Aluminium excludes alumina refining and PFCs unless supplied; EAF and DRI routes exclude scrap / feedstock upstream unless supplied. All are listed under exclusions.

Source: worldsteel Sustainability Indicators 2023 (https://worldsteel.org/wp-content/uploads/Sustainability-indicators-report-2023.pdf).
