# BRSR Principle 6 (Environment): what ClimRisk covers and what it does not

Source for the indicator list and the assurance timeline: SEBI circular SEBI/HO/CFD/CFD-SEC-2/P/CIR/2023/122 of 12 July 2023 (BRSR Core, Annexure I). I read paras 3 to 5 and the Annexure I table for GHG, water, energy and waste (the fetched text, about lines 20 to 440). I did not read Annexure II (the full BRSR format) or the December 2024 industry-standards circular, so questions outside the BRSR Core table are listed as "not checked".

## Timeline (para 3.4)

BRSR disclosure in the updated format applies to the top 1000 listed entities from FY2023-24. Reasonable assurance of BRSR Core is phased: top 150 in FY2023-24, top 250 in FY2024-25, top 500 in FY2025-26, top 1000 in FY2026-27. Value-chain disclosures: top 250 on comply-or-explain from FY2024-25, limited assurance from FY2025-26 (para 4). An earlier version of the engine's text said assurance applied to the top 1000 from FY2024-25; that was wrong and is fixed in `outcomes/disclosure.py`.

## Principle 6 items in BRSR Core and where ClimRisk stands

| BRSR Core item (Principle 6 Essential question) | What the circular asks for | ClimRisk today | Honest status |
|---|---|---|---|
| GHG Scope 1 (Q7) | Fuel burned x emission factor (IPCC or accredited lab), minus carbon captured, plus process and fugitive emissions | Modelled Scope 1 from the engine's current-policies projection for 2026 | **Estimate only.** Not the bottom-up method; cannot be assured as it stands. Field `data_basis` now says so. |
| GHG Scope 2 (Q7) | Purchased energy x emission factor (IPCC or supplier-audited) | Modelled Scope 2; national grid factors are held (`docs/NATIONAL_GRIDS.md`, CEA baseline) | **Factors yes, consumption no.** The engine has the factor side; the filer's kWh must be supplied. |
| GHG intensity per rupee of revenue adjusted for PPP, and per unit of output (Q7) | Scope 1+2 / revenue from the audited P&L adjusted by PPP (USD/INR) | Computed per INR crore; PPP variant when `ppp_inr_per_intl_usd` is passed | **Partly.** Arithmetic is right; inputs are estimates unless the filer supplies audited revenue and PPP. Per-unit-of-output intensity is not computed. |
| Energy: total, % renewable, intensity (Q1) | Sum of fuels, purchased and self-generated energy | Not modelled (fields are null by design) | **Not covered.** Needs metering or ERP data. |
| Water: consumption, intensity (Q3), discharge by treatment level (Q4) | Input minus output meter logs | Not modelled. WRI Aqueduct basin stress is held as context | **Not covered.** Context only: it says where water stress is, not what a plant consumed. |
| Waste: 8 categories, recovery and disposal (Q9) | Weights by plastic, e-waste, bio-medical, C&D, battery, radioactive, hazardous, other | Not modelled (fields are null by design) | **Not covered.** |
| Environmental compliance (this module's "P6-E5") | n/a | Null fields | Not part of the BRSR Core table in this circular; not checked against the full format. |
| Leadership: physical and transition climate risk (this module's "P6-L1, L2") | Voluntary narrative and numbers | Physical hazard context, adaptation capex, carbon cost, EV range | **Covered as screening**, which is where ClimRisk is strongest. |

Other Principle 6 essential questions that are not in the BRSR Core table (energy-saving schemes for designated consumers, air emissions, zero liquid discharge, ecologically sensitive areas, environmental impact assessments and so on): **not checked** against Annexure II, and the engine does not model them.

## What ClimRisk can truthfully say to a BRSR filer

1. It supplies the **physical and transition risk** layer (Principle 6 leadership indicators) with sources, ranges and limits. That is the part nobody can read off a meter.
2. It supplies **emission factors and a modelled screening estimate** for Scope 1 and 2, clearly labelled as modelled, to be replaced by bottom-up data before assurance.
3. It does **not** produce assurance-grade energy, water or waste numbers, and says so in the output.

## Gaps worth closing, in order

1. Accept bottom-up Scope 1/2 inputs (fuel by type and quantity, purchased electricity by state or grid) and compute them with the circular's formulas and the held factors, with the factor source and year printed beside every number.
2. Per-unit-of-output intensity (needs the filer's sector-specific output).
3. Water: use Aqueduct basin context to flag high-stress sites for the filer, without inventing consumption.
4. Read Annexure II and the December 2024 industry standards and extend this map to every Principle 6 question.
