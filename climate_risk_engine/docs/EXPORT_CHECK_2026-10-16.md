# Export check, 16 October 2026

Question: do the downloaded files carry the same numbers as the screen / API?

- Excel run export (`POST /runs/export`) against `POST /runs` for BHP, scenario NZE 2050: 7 summary figures, 2 compression figures, dominant hazard, and 450 projection cells (25 years x 18 columns). 0 differences (tolerance = the rounding the export applies). Automated: `tests/test_exports_match.py`.
- Physical-risk PDF (`POST /reports/physical-risk.pdf`) against the JSON of the same request (Houston chemical site, no satellite image): 72 distinct numbers in the PDF text, all 72 found in the JSON. The overall rating and the three main exposures in the PDF headline (river flood HIGH, cyclone wind MODERATE, heat MODERATE) are those of the JSON.
- Limits: the PDF check proves a number in the file exists in the data, not that it sits next to the right label; the labels were read by eye on this one site. The portfolio CSV and the carbon workbook were not compared here.
- Note for readers: the Excel "Summary" sheet shows the company's base-year revenue (62,500) while the Projections sheet starts at the 2026 modelled revenue (45,491.5). Both are correct and different quantities; the first is a labelling risk.
