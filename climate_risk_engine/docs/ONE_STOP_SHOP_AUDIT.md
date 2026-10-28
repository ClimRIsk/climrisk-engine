# One-stop shop audit (28 Oct 2026)

Question asked: if we say "one stop shop" for climate financial risk, what is missing, and what must be scrutinised before we say it? Written from the repo as it stands (GAP_REGISTER, VALIDATION_LEDGER, module list, API). Grades are ours, not an expert's. Nothing here was re-run except where stated.

## 1. Verdict
Accurate today: **a screening and triage engine with unusually honest limits, covering physical hazards, transition/carbon, disclosure gap checks and a local assistant, in one API.**
Not accurate today: "one stop shop" if it implies a validated loss number, a bank-grade credit number, a regulator-ready submission, or live market data. Use "one place to screen", not "one stop shop", until sections 2 and 3 below are closed.

## 2. What is missing (ranked by what blocks a paying customer)

| # | Gap | Why it blocks | Effort |
|---|---|---|---|
| 1 | **Sector-specific vulnerability in the loss module (C1).** Offices, mines, solar and data centres are priced on the industrial flood curve. | Every loss number is on the wrong curve for most assets. Largest accuracy error we have named. | Medium: curves exist in `climate/damage_curves.py`; wiring plus tests |
| 2 | **No loss validation against a named event or claims.** Insurer simulation is 1.2-3.6x FEMA NRI for wind, 0-0.6x for inland flood, 0 coastal (grade C/D). | We can rank places; we cannot defend a premium. | Hard: needs claims or a market-loss source such as PERILS |
| 3 | **No first-party business-interruption model (C2); no supply-chain layer.** | BI dominated Uri, Thailand 2011, offshore wind. | Medium for BI; supply chain needs client data |
| 4 | **Credit risk is unvalidated.** `climate/credit_risk.py` PD/LGD/ECL/RWA has no test against defaults or losses. Competitors sell this. | Banks are the buyer for "financial risk"; we would be showing an unvalidated number. | Hard; honest option is to relabel as illustrative and keep it out of the headline |
| 5 | **Hazard data licence.** Hourly heat and rain depend on the free Open-Meteo tier (non-commercial, rate-limited); fallback fails two event checks (C3). | Cannot sell commercially on it. | Copernicus CDS account or paid plan; decision, not code |
| 6 | **No multi-tenant model.** I found a single API-key dependency (`_AUTH`) and a desktop licence file; no organisation, user, role or per-client data separation in `api/main.py`. | Needed before two clients share one deployment; also for audit trails. | Medium |
| 7 | **No regulator-format outputs.** No ECB/EBA stress-test template, no NGFS-scenario portfolio run packaged, no ISSB S2 / TCFD table output (S2 and TCFD exist only as knowledge sources). PCAF, SFDR PAI, EU Taxonomy and ESRS E1 gap modules exist. | Disclosure buyers want the filled template, not a gap list. | Medium per template |
| 8 | **Carbon market: no registry feed.** Connector exists; no real registry delivery tested; no credit price. | See GAP_REGISTER carbon section. | Needs data licence |
| 9 | **Perils with no site model:** convective-storm loss, windstorm onshore loss, wildfire loss, lightning, grid outage, heat derating, dust, snow load, subsidence (B table). Pluvial is modest skill; small/mid river depth not shipped. | Coverage claims must name these. | Per peril: weeks |
| 10 | **Geography:** hail, tornado US only; LiDAR England and Netherlands only; earthquake hazard US only as published. | Global claims overstate. | Per country |
| 11 | **Portfolio layer:** no optimisation, economic capital, climate-adjusted capital ratio, no projected company financials. | Competitor feature; matters for bank/asset-manager buyers. | Hard |
| 12 | **Operations:** one Dockerfile and `deploy/`; I did not verify CI, monitoring, backups, or load behaviour. Tests run in a sandbox; the full suite was **not** run in this pass. | "Production" is not shown. | Medium |

## 3. What must be scrutinised before we repeat it publicly

1. **Every grade in the validation ledger is ours.** Get one external reviewer (hydrologist or cat modeller) to sign off the B-grade rows; until then say "self-assessed".
2. **Carbon rules were written from the standards' text and not re-verified** (credit quality screen, ICVCM label, EU 2024/825 date, CBAM free-allocation factors, ETS arithmetic). The CER rules cite UNFCCC PA/A6.4/TP/4 and ICAO Oct 2025; neither has been read by a lawyer or the UNFCCC. Re-check each constant against the primary text before showing UNMAI.
3. **CBAM/ETS numbers in marketing**: confirm against Regulation (EU) 2023/956 and current Commission pages on the day we quote them.
4. **credit_risk.py coefficients "calibrated to ECB 2022"**: unsourced in the ledger. Either cite the paper and reproduce, or stop describing it as calibrated.
5. **Remaining website/README claims review** (open since earlier): "eight perils", "one report", any accuracy words. Check each against the ledger rows above.
6. **Wikipedia is still referenced in `analysis/site_sources.py`**; we purged it from the assistant, not from site analysis. Decide and remove.
7. **Open-Meteo terms** for anything shown to a customer or in a paid pilot.
8. **Event checks use one event per peril** (C7): never quote as tail validation.
9. **Resident assistant (C6)**: model not retrained on newer gold answers; do not let it quote premium numbers.
10. **Outreach claims**: the Leeds example was already overstated once; every named-site claim in `marketing/` needs a re-check against its cited run.
11. **Data provenance in reports**: confirm every report states source, date and "not assessed" lines (SILENT_DEGRADATION_AUDIT.md is the template; re-run it after the recent changes).
12. **Security basics**: single shared API key, local licence file marked "not tamper-proof" in its own docstring. Fine for a pilot, not for a bank's security review.

## 4. Suggested order (YC deadline 2 Nov; Upwork showcase)
This week, cheap and credibility-raising: (a) relabel credit risk as illustrative everywhere, (b) re-verify carbon constants against primary text, (c) finish the website claims review, (d) remove Wikipedia from site analysis, (e) run the full test suite and record the result.
Next 4-6 weeks: wire sector curves (gap 1), BI first-party layer (3), licensed hazard source (5), then one regulator template (7) for the first design partner.
Needs a partner, not code: claims data for loss validation (2), registry licence (8), external reviewer (3.1).

## 5. Status after the 28 Oct 2026 fixes

| Item | Status |
|---|---|
| Gap 1, sector vulnerability (flood) | **Closed for flood.** JRC tables for six asset classes by continent, stored as data (`data/jrc_flood_curves.json`), used per site; class and curve are stated in each result; missing continent tables use the published mean and say so. Transcribed from the CLIMADA-petals implementation, **not yet compared with the JRC report itself**. Wind is still one curve per cyclone basin. The untraceable `damage_curves.py` parameters are gone. |
| Gap 4 / scrutiny 4, credit risk | **Rebuilt.** The untraceable model is withdrawn (`POST /portfolio/credit` returns 410). `climate/structural_credit.py` is a Merton structural model: no fitted coefficients, asset volatility implied from the bank's own PD and leverage, climate as a loss of asset value from our hazard outputs and a carbon cost over EBITDA, LGD from collateral after damage, IRB capital from the Basel formula. Anything not supplied is "not assessed". **Not tested against observed climate-related defaults.** The dashboards still contain their own hand-coded IFRS 9 logic (open). |
| Scrutiny 2-3, carbon constants | **Checked against secondary sources, not primary text.** CBAM factors match several guides; the EUR-Lex text could not be retrieved. Found: a 17 Jul 2026 Commission proposal to extend the CBAM-sector phase-out to 2038 (not law, noted in output); SBTi Corporate Net-Zero Standard v2.0 (final 11 Jun 2026) is not encoded, so the SBTi rules are labelled v1.x in every result. |
| Everything else in section 2 | **Not done.** Loss validation (needs claims), business interruption, hazard-data licence, multi-tenancy, regulator templates, registry feed, remaining perils, geography, portfolio layer, operations. Each needs data we do not have, a decision, or a larger build. |
