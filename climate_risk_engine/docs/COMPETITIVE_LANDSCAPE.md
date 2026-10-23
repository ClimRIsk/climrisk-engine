# Who we compete with (23 October 2026)

Physical climate risk analytics is crowded. There are three kinds of competitor, and the small ones are numerous. Sources are web pages found on this date. Anything marked **(general knowledge)** was not re-checked and must be verified before it appears in a YC or investor document. Funding and client numbers are the vendors' own claims.

## 1. Banking and insurance risk platforms (sell to the same buyers, sell the whole stack)

| Vendor | What it offers | Source |
|---|---|---|
| **Prometeia** (Italy; 200+ clients and 500+ references in EMEA, own claim) | Asset geo-location from tax or company IDs; long-term hazard scenarios at a granular geographic level; damage to real estate and operations carried into PDs and LGDs; transition-risk engine with projected balance sheets; support for regulatory climate stress tests; insurance premium estimates for physical risk; portfolio alignment and a climate-risk-adjusted capital ratio. Scenarios start from NGFS. | [Prometeia ESG risk sheet](https://www.prometeia.com/cdn/upload/2022/7/36807b77-b7e6-4c7f-a129-cb04ebc340f9.pdf), [2023 sustainability report](https://www.prometeia.com/cdn/upload/2024/5/c9b3460d-dda4-426e-9733-1506a122b8df.pdf). The article the founder linked could not be read; method depth is unverified. |
| Moody's (RMS, 427), S&P (Trucost), MSCI, Aon, Willis Towers Watson, McKinsey | Axios reports that RMS, The Climate Service and 427 were bought by larger firms (which buyer took which target was not mapped there). MSCI is named as a competitor of CLIMORISK. | [Axios, Jupiter and BCG](https://www.axios.com/2022/06/14/jupiter-intel-climate-partnership-boston-consulting), [CB Insights, CLIMORISK](https://www.cbinsights.com/company/climorisk/alternatives-competitors). Product details **(general knowledge)** |
| Verisk, Munich Re, Swiss Re (CatNet), Moody's RMS | Catastrophe models and hazard viewers for insurers | **(general knowledge)** |

## 2. Climate-risk specialists (venture-backed, same pitch as ours)

| Vendor | Notes | Source |
|---|---|---|
| **Jupiter Intelligence** (US) | Physical risk analytics for global portfolios and assets; partner of Boston Consulting Group and Aon | [Axios](https://www.axios.com/2022/06/14/jupiter-intel-climate-partnership-boston-consulting), [CB Insights](https://www.cbinsights.com/compare/jupiter-intelligence-vs-xdi) |
| **XDI** (Australia) | Asset-level physical climate risk | [CB Insights](https://www.cbinsights.com/company/xdi/alternatives-competitors) |
| **Climate X** (UK) | Climate risk data and analytics for financial services, real estate and insurance; $18M Series A in June 2024, total raised about $25-27M depending on source | [AI Business](https://aibusiness.com/data/google-backed-climate-x-raises-18m-to-expand-climate-risk-platform), [PitchBook](https://pitchbook.com/profiles/company/469894-42) |
| **Cervest** (UK) | $30M Series A in May 2021 (Draper Esprit); no later round found | [Silicon Republic](https://www.siliconrepublic.com/start-ups/cervest-climate-risk-funding) |
| RiskThinking.AI, Risilience, Sust Global (EarthScan), CLIMORISK, Arcturus | Smaller specialists that came up in vendor comparisons | [CB Insights](https://www.cbinsights.com/company/riskthinkingai/alternatives-competitors), [Sust Global](https://www.cbinsights.com/company/sust-global/alternatives-competitors) |
| First Street, Fathom, ClimateAI, Ecoascent, Kita and others | Hazard-specific or sector-specific vendors **(general knowledge)** | not checked |

## 3. Free and open alternatives (set the price floor)

| Option | Notes | Source |
|---|---|---|
| OS-Climate physrisk | Open-source physical risk calculation library | [PyPI](https://pypi.org/project/physrisk-lib/0.20.0/) |
| CLIMADA (ETH Zurich) | Open-source catastrophe and impact modelling **(general knowledge)** | not checked |
| FEMA National Risk Index, Copernicus, national hazard maps | Free hazard layers; the engine already benchmarks against the FEMA NRI | in-repo benchmark |

## How we are different, and how we are not

Different, and checkable in the repo: every model carries a grade and a test in `VALIDATION_LEDGER.md`; reports show only tested hazards; missing data is null, never zero; site-level models (LiDAR flood, wildfire, cold-region, hail, tornado, marine heatwave) are built on open data.

Not different: breadth of coverage, client base, regulator relationships, and bank-grade packaging all favour the incumbents. Prometeia in particular sells to banks that already run regulatory stress tests. We should expect to be compared to them on that, and should win only where a buyer values a transparent, benchmarked site-level model over a full platform.

## Before this goes in a YC document

1. Verify every **(general knowledge)** row and the Axios acquisition mapping.
2. Read Prometeia's article and product pages directly; the public material found so far is marketing-level.
3. Say plainly that the field is crowded and name two or three direct competitors. Claiming no competition would hurt more than it helps.
