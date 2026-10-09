"""Collect real company disclosures about named climate and weather events from SEC EDGAR (public-domain filings).

    python -m cri.llmtrain.edgar_cases collect OUT_DIR [--max-docs-per-event 90] [--only KEY,KEY]
    python -m cri.llmtrain.edgar_cases rows    OUT_DIR TRAIN_DIR

A "case" is one (company, event) pair with at least one verbatim sentence from the company's own filing that names the event and states
an amount or an action. Nothing is paraphrased: the quote, the filing form, the date and the document address are stored with it, so every
case can be checked by opening the link. Response sentences (reinsurance, borrowing, insurance recoveries, generators, impairments ...)
are tagged by keyword so the model can learn how companies tackled the loss.

EDGAR asks automated clients to be polite: a few requests per second and a descriptive User-Agent. This module does 4 per second.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "ClimRisk research (open-source climate-risk project; polite crawler)", "Accept-Encoding": "identity"}
FTS = "https://efts.sec.gov/LATEST/search-index"

# key, display name, search phrase, start, end, hazard, region
EVENTS = [
    ("katrina-2005", "Hurricane Katrina", "Hurricane Katrina", "2005-08-29", "2006-12-31", "tropical cyclone", "United States"),
    ("sandy-2012", "Superstorm Sandy", "Superstorm Sandy", "2012-10-29", "2013-12-31", "tropical cyclone", "United States"),
    ("harvey-2017", "Hurricane Harvey", "Hurricane Harvey", "2017-08-25", "2018-06-30", "tropical cyclone", "United States"),
    ("irma-2017", "Hurricane Irma", "Hurricane Irma", "2017-09-06", "2018-06-30", "tropical cyclone", "United States and Caribbean"),
    ("maria-2017", "Hurricane Maria", "Hurricane Maria", "2017-09-20", "2018-06-30", "tropical cyclone", "Puerto Rico"),
    ("florence-2018", "Hurricane Florence", "Hurricane Florence", "2018-09-14", "2019-06-30", "tropical cyclone", "United States"),
    ("michael-2018", "Hurricane Michael", "Hurricane Michael", "2018-10-10", "2019-06-30", "tropical cyclone", "United States"),
    ("dorian-2019", "Hurricane Dorian", "Hurricane Dorian", "2019-09-01", "2020-06-30", "tropical cyclone", "Bahamas and United States"),
    ("laura-2020", "Hurricane Laura", "Hurricane Laura", "2020-08-27", "2021-06-30", "tropical cyclone", "United States"),
    ("ida-2021", "Hurricane Ida", "Hurricane Ida", "2021-08-29", "2022-06-30", "tropical cyclone", "United States"),
    ("ian-2022", "Hurricane Ian", "Hurricane Ian", "2022-09-28", "2023-06-30", "tropical cyclone", "United States"),
    ("idalia-2023", "Hurricane Idalia", "Hurricane Idalia", "2023-08-30", "2024-06-30", "tropical cyclone", "United States"),
    ("beryl-2024", "Hurricane Beryl", "Hurricane Beryl", "2024-07-08", "2025-03-31", "tropical cyclone", "United States and Caribbean"),
    ("helene-2024", "Hurricane Helene", "Hurricane Helene", "2024-09-26", "2025-06-30", "tropical cyclone", "United States"),
    ("milton-2024", "Hurricane Milton", "Hurricane Milton", "2024-10-09", "2025-06-30", "tropical cyclone", "United States"),
    ("uri-2021", "Winter Storm Uri", "Winter Storm Uri", "2021-02-15", "2022-06-30", "extreme cold", "United States"),
    ("elliott-2022", "Winter Storm Elliott", "Winter Storm Elliott", "2022-12-23", "2023-06-30", "extreme cold", "United States"),
    ("derecho-2020", "the August 2020 Midwest derecho", "derecho", "2020-08-10", "2021-03-31", "severe convective storm", "United States"),
    ("camp-2018", "the Camp Fire", "Camp Fire", "2018-11-08", "2019-12-31", "wildfire", "California"),
    ("tubbs-2017", "the 2017 California wildfires", "2017 Northern California wildfires", "2017-10-09", "2018-12-31", "wildfire", "California"),
    ("marshall-2021", "the Marshall Fire", "Marshall Fire", "2021-12-30", "2022-12-31", "wildfire", "Colorado"),
    ("maui-2023", "the Maui wildfires", "Maui wildfires", "2023-08-08", "2024-12-31", "wildfire", "Hawaii"),
    ("la-fires-2025", "the January 2025 Los Angeles wildfires", "Los Angeles wildfires", "2025-01-07", "2025-12-31", "wildfire", "California"),
    ("mcmurray-2016", "the Fort McMurray wildfire", "Fort McMurray wildfire", "2016-05-03", "2017-03-31", "wildfire", "Canada"),
    ("thailand-2011", "the 2011 Thailand floods", "Thailand floods", "2011-10-01", "2012-12-31", "flood", "Thailand"),
    ("jebi-2018", "Typhoon Jebi", "Typhoon Jebi", "2018-09-04", "2019-06-30", "tropical cyclone", "Japan"),
    ("europe-floods-2021", "the July 2021 European floods", "European floods", "2021-07-14", "2022-03-31", "flood", "Germany and Europe"),
    ("alberta-2013", "the 2013 Alberta floods", "Alberta floods", "2013-06-20", "2014-03-31", "flood", "Canada"),
    ("calgary-hail-2020", "the 2020 Calgary hailstorm", "Calgary hailstorm", "2020-06-13", "2021-03-31", "hail", "Canada"),
    ("australia-floods-2022", "the 2022 eastern Australia floods", "Australia floods", "2022-02-22", "2022-12-31", "flood", "Australia"),
    ("gabrielle-2023", "Cyclone Gabrielle", "Cyclone Gabrielle", "2023-02-12", "2023-12-31", "tropical cyclone", "New Zealand"),
    ("idai-2019", "Cyclone Idai", "Cyclone Idai", "2019-03-14", "2019-12-31", "tropical cyclone", "Mozambique and Zimbabwe"),
    ("mississippi-2022", "the 2022 Mississippi River low water", "Mississippi River low water", "2022-08-01", "2023-03-31", "drought", "United States"),
    ("rhine-2018", "the 2018 low Rhine water", "low water levels on the Rhine", "2018-08-01", "2019-06-30", "drought", "Germany and Europe"),
    ("panama-2023", "the Panama Canal drought", "Panama Canal drought", "2023-06-01", "2024-12-31", "drought", "Panama"),
    ("heat-dome-2021", "the 2021 Pacific Northwest heat dome", "heat dome", "2021-06-25", "2022-03-31", "extreme heat", "United States and Canada"),
    ("tx-hail-2023", "the 2023 severe convective storm season", "severe convective storm", "2023-03-01", "2024-03-31", "severe convective storm", "United States"),
    ("matthew-2016", "Hurricane Matthew", "Hurricane Matthew", "2016-10-04", "2017-06-30", "tropical cyclone", "United States and Haiti"),
    ("harvey-flood-ins", "flooding from Hurricane Harvey", "flooding associated with Hurricane Harvey", "2017-08-25", "2018-06-30", "flood", "United States"),
    ("ike-2008", "Hurricane Ike", "Hurricane Ike", "2008-09-13", "2009-12-31", "tropical cyclone", "United States"),
    ("gustav-2008", "Hurricane Gustav", "Hurricane Gustav", "2008-08-31", "2009-06-30", "tropical cyclone", "United States"),
    ("rita-2005", "Hurricane Rita", "Hurricane Rita", "2005-09-24", "2006-12-31", "tropical cyclone", "United States"),
    ("wilma-2005", "Hurricane Wilma", "Hurricane Wilma", "2005-10-24", "2006-12-31", "tropical cyclone", "United States"),
    ("irene-2011", "Hurricane Irene", "Hurricane Irene", "2011-08-27", "2012-06-30", "tropical cyclone", "United States"),
    ("isaac-2012", "Hurricane Isaac", "Hurricane Isaac", "2012-08-28", "2013-06-30", "tropical cyclone", "United States"),
    ("isaias-2020", "Tropical Storm Isaias", "Isaias", "2020-08-04", "2021-06-30", "tropical cyclone", "United States"),
    ("sally-2020", "Hurricane Sally", "Hurricane Sally", "2020-09-16", "2021-06-30", "tropical cyclone", "United States"),
    ("delta-2020", "Hurricane Delta", "Hurricane Delta", "2020-10-09", "2021-06-30", "tropical cyclone", "United States"),
    ("zeta-2020", "Hurricane Zeta", "Hurricane Zeta", "2020-10-28", "2021-06-30", "tropical cyclone", "United States"),
    ("fiona-2022", "Hurricane Fiona", "Hurricane Fiona", "2022-09-18", "2023-06-30", "tropical cyclone", "Puerto Rico and Canada"),
    ("nicole-2022", "Hurricane Nicole", "Hurricane Nicole", "2022-11-09", "2023-06-30", "tropical cyclone", "United States"),
    ("imelda-2019", "Tropical Storm Imelda", "Tropical Storm Imelda", "2019-09-17", "2020-06-30", "flood", "United States"),
    ("allison-2001", "Tropical Storm Allison", "Tropical Storm Allison", "2001-06-05", "2002-06-30", "flood", "United States"),
    ("colorado-2013", "the 2013 Colorado floods", "Colorado floods", "2013-09-11", "2014-06-30", "flood", "United States"),
    ("midwest-2019", "the 2019 Midwest flooding", "Midwest flooding", "2019-03-14", "2020-03-31", "flood", "United States"),
    ("tornado-2011", "the 2011 tornado outbreaks", "tornado outbreak", "2011-04-25", "2012-06-30", "severe convective storm", "United States"),
    ("joplin-2011", "the Joplin tornado", "Joplin tornado", "2011-05-22", "2012-06-30", "severe convective storm", "United States"),
    ("tx-freeze-2011", "the February 2011 Texas freeze", "February 2011 winter storm", "2011-02-02", "2011-12-31", "extreme cold", "United States"),
    ("polar-vortex-2014", "the 2014 polar vortex", "polar vortex", "2014-01-06", "2014-12-31", "extreme cold", "United States"),
    ("drought-2012", "the 2012 US drought", "2012 drought", "2012-06-01", "2013-06-30", "drought", "United States"),
    ("drought-california", "the California drought", "California drought", "2014-01-01", "2017-12-31", "drought", "California"),
    ("woolsey-2018", "the Woolsey Fire", "Woolsey Fire", "2018-11-08", "2019-12-31", "wildfire", "California"),
    ("dixie-2021", "the Dixie Fire", "Dixie Fire", "2021-07-13", "2022-06-30", "wildfire", "California"),
    ("psps-2019", "California public safety power shutoffs", "public safety power shutoff", "2019-10-01", "2020-12-31", "wildfire", "California"),
    ("canada-fires-2023", "the 2023 Canadian wildfires", "Canadian wildfires", "2023-05-01", "2024-03-31", "wildfire", "Canada"),
    ("australia-bushfire", "the 2019 to 2020 Australian bushfires", "Australian bushfires", "2019-12-01", "2020-12-31", "wildfire", "Australia"),
    ("kenya-drought", "the East Africa drought", "drought in East Africa", "2021-06-01", "2023-12-31", "drought", "East Africa"),
    ("india-heatwave-2022", "the 2022 South Asia heatwave", "heatwave", "2022-03-15", "2022-12-31", "extreme heat", "India and Pakistan"),
    ("europe-heat-2022", "the 2022 European heatwave and drought", "European drought", "2022-06-01", "2023-03-31", "drought", "Europe"),
    ("china-drought-2022", "the 2022 Sichuan drought and power rationing", "power rationing in Sichuan", "2022-08-01", "2023-03-31", "drought", "China"),
    ("pakistan-floods-2022", "the 2022 Pakistan floods", "Pakistan floods", "2022-08-20", "2023-06-30", "flood", "Pakistan"),
    ("germany-hail", "the German hailstorms", "hailstorm", "2013-07-27", "2014-06-30", "hail", "Germany"),
    ("cat-losses-generic", "catastrophe losses in the period", "catastrophe losses attributable to", "2017-01-01", "2025-12-31", "natural catastrophe", "United States"),
]

# Added for coverage (each is a quoted-phrase search in SEC filings; events that return nothing are simply skipped and reported)
EXTRA = [
    ("charley-2004", "Hurricane Charley", "Hurricane Charley", "2004-08-13", "2005-12-31", "tropical cyclone", "United States"),
    ("frances-2004", "Hurricane Frances", "Hurricane Frances", "2004-09-05", "2005-12-31", "tropical cyclone", "United States"),
    ("ivan-2004", "Hurricane Ivan", "Hurricane Ivan", "2004-09-16", "2005-12-31", "tropical cyclone", "United States and Caribbean"),
    ("jeanne-2004", "Hurricane Jeanne", "Hurricane Jeanne", "2004-09-26", "2005-12-31", "tropical cyclone", "United States"),
    ("dennis-2005", "Hurricane Dennis", "Hurricane Dennis", "2005-07-10", "2006-06-30", "tropical cyclone", "United States"),
    ("dean-2007", "Hurricane Dean", "Hurricane Dean", "2007-08-21", "2008-06-30", "tropical cyclone", "Caribbean and Mexico"),
    ("hanna-2020", "Hurricane Hanna", "Hurricane Hanna", "2020-07-25", "2021-03-31", "tropical cyclone", "United States and Mexico"),
    ("henri-2021", "Tropical Storm Henri", "Tropical Storm Henri", "2021-08-21", "2022-03-31", "tropical cyclone", "United States"),
    ("nicholas-2021", "Hurricane Nicholas", "Hurricane Nicholas", "2021-09-13", "2022-03-31", "tropical cyclone", "United States"),
    ("lee-2011", "Tropical Storm Lee", "Tropical Storm Lee", "2011-09-01", "2012-06-30", "flood", "United States"),
    ("fay-2008", "Tropical Storm Fay", "Tropical Storm Fay", "2008-08-18", "2009-06-30", "flood", "United States"),
    ("lane-2018", "Hurricane Lane", "Hurricane Lane", "2018-08-22", "2019-03-31", "tropical cyclone", "Hawaii"),
    ("debby-2024", "Hurricane Debby", "Hurricane Debby", "2024-08-05", "2025-06-30", "tropical cyclone", "United States"),
    ("francine-2024", "Hurricane Francine", "Hurricane Francine", "2024-09-11", "2025-06-30", "tropical cyclone", "United States"),
    ("otis-2023", "Hurricane Otis", "Hurricane Otis", "2023-10-25", "2024-06-30", "tropical cyclone", "Mexico"),
    ("haiyan-2013", "Typhoon Haiyan", "Typhoon Haiyan", "2013-11-08", "2014-06-30", "tropical cyclone", "Philippines"),
    ("hato-2017", "Typhoon Hato", "Typhoon Hato", "2017-08-23", "2018-03-31", "tropical cyclone", "Macau and Hong Kong"),
    ("mangkhut-2018", "Typhoon Mangkhut", "Typhoon Mangkhut", "2018-09-14", "2019-03-31", "tropical cyclone", "Philippines and Hong Kong"),
    ("debbie-2017", "Cyclone Debbie", "Cyclone Debbie", "2017-03-28", "2018-03-31", "tropical cyclone", "Australia"),
    ("jonas-2016", "Winter Storm Jonas", "Winter Storm Jonas", "2016-01-22", "2016-12-31", "extreme cold", "United States"),
    ("bomb-cyclone-2018", "the January 2018 bomb cyclone", "bomb cyclone", "2018-01-04", "2018-12-31", "extreme cold", "United States"),
    ("tx-drought-2011", "the 2011 Texas drought", "Texas drought", "2011-06-01", "2012-06-30", "drought", "United States"),
    ("moore-2013", "the Moore tornado", "Moore tornado", "2013-05-20", "2014-03-31", "severe convective storm", "United States"),
    ("tuscaloosa-2011", "the Tuscaloosa tornado", "Tuscaloosa", "2011-04-27", "2012-03-31", "severe convective storm", "United States"),
    ("denver-hail-2017", "the 2017 Denver hailstorm", "Denver hailstorm", "2017-05-08", "2018-03-31", "hail", "United States"),
    ("louisiana-2016", "the 2016 Louisiana floods", "Louisiana flood", "2016-08-12", "2017-06-30", "flood", "United States"),
    ("nashville-2010", "the 2010 Nashville flood", "Nashville flood", "2010-05-01", "2011-03-31", "flood", "United States"),
    ("memorial-day-2015", "the 2015 Memorial Day floods", "Memorial Day flood", "2015-05-25", "2016-03-31", "flood", "United States"),
    ("tax-day-2016", "the 2016 Tax Day flood", "Tax Day flood", "2016-04-17", "2017-03-31", "flood", "United States"),
    ("missouri-2011", "the 2011 Missouri River flooding", "Missouri River flooding", "2011-06-01", "2012-06-30", "flood", "United States"),
    ("midwest-2008", "the 2008 Midwest floods", "Midwest floods", "2008-06-01", "2009-06-30", "flood", "United States"),
    ("carr-2018", "the Carr Fire", "Carr Fire", "2018-07-23", "2019-06-30", "wildfire", "California"),
    ("thomas-2017", "the Thomas Fire", "Thomas Fire", "2017-12-04", "2018-12-31", "wildfire", "California"),
    ("kincade-2019", "the Kincade Fire", "Kincade Fire", "2019-10-23", "2020-12-31", "wildfire", "California"),
    ("gatlinburg-2016", "the Gatlinburg wildfire", "Gatlinburg", "2016-11-28", "2017-12-31", "wildfire", "Tennessee"),
    ("waldo-2012", "the Waldo Canyon fire", "Waldo Canyon", "2012-06-23", "2013-03-31", "wildfire", "Colorado"),
    ("ira-2022", "the US Inflation Reduction Act", "Inflation Reduction Act", "2022-08-16", "2025-06-30", "transition policy", "United States"),
    ("cpp-2015", "the US Clean Power Plan", "Clean Power Plan", "2015-08-03", "2017-12-31", "transition policy", "United States"),
    ("rggi", "the Regional Greenhouse Gas Initiative", "Regional Greenhouse Gas Initiative", "2009-01-01", "2024-12-31", "transition policy", "United States"),
    ("ca-cap-trade", "California cap-and-trade", "cap-and-trade", "2013-01-01", "2024-12-31", "transition policy", "California"),
    ("eu-ets", "the EU Emissions Trading System", "Emissions Trading System", "2012-01-01", "2025-06-30", "transition policy", "European Union"),
    ("coal-retire", "coal plant retirements and stranded costs", "coal plant retirement", "2012-01-01", "2025-06-30", "transition market", "United States"),
    ("ev-mandate", "the shift to electric vehicles", "transition to electric vehicles", "2019-01-01", "2025-06-30", "transition market", "Global"),
]

# Events outside the United States. Searched in the SEC's foreign-company forms (20-F annual report, 6-K press release, 40-F) as well as the US forms.
GLOBAL_EVENTS = [
    ("hagibis-2019", "Typhoon Hagibis", "Typhoon Hagibis", "2019-10-12", "2020-06-30", "tropical cyclone", "Japan"),
    ("faxai-2019", "Typhoon Faxai", "Typhoon Faxai", "2019-09-09", "2020-03-31", "tropical cyclone", "Japan"),
    ("japan-rain-2018", "the July 2018 western Japan floods", "heavy rain in western Japan", "2018-07-06", "2019-03-31", "flood", "Japan"),
    ("lekima-2019", "Typhoon Lekima", "Typhoon Lekima", "2019-08-09", "2020-03-31", "tropical cyclone", "China"),
    ("yagi-2024", "Typhoon Yagi", "Typhoon Yagi", "2024-09-06", "2025-06-30", "tropical cyclone", "Vietnam, China and the Philippines"),
    ("doksuri-2023", "Typhoon Doksuri", "Typhoon Doksuri", "2023-07-26", "2024-03-31", "tropical cyclone", "China and the Philippines"),
    ("henan-2021", "the 2021 Henan floods", "Henan floods", "2021-07-17", "2022-03-31", "flood", "China"),
    ("ketsana-2009", "Typhoon Ketsana", "Typhoon Ketsana", "2009-09-26", "2010-06-30", "tropical cyclone", "Philippines and Vietnam"),
    ("odette-2021", "Typhoon Odette", "Typhoon Odette", "2021-12-16", "2022-09-30", "tropical cyclone", "Philippines"),
    ("jakarta-2020", "the 2020 Jakarta floods", "Jakarta floods", "2020-01-01", "2020-12-31", "flood", "Indonesia"),
    ("chennai-2015", "the 2015 Chennai floods", "Chennai floods", "2015-11-30", "2016-09-30", "flood", "India"),
    ("kerala-2018", "the 2018 Kerala floods", "Kerala floods", "2018-08-08", "2019-06-30", "flood", "India"),
    ("amphan-2020", "Cyclone Amphan", "Cyclone Amphan", "2020-05-20", "2021-03-31", "tropical cyclone", "India and Bangladesh"),
    ("fani-2019", "Cyclone Fani", "Cyclone Fani", "2019-05-03", "2020-03-31", "tropical cyclone", "India"),
    ("biparjoy-2023", "Cyclone Biparjoy", "Cyclone Biparjoy", "2023-06-15", "2024-03-31", "tropical cyclone", "India and Pakistan"),
    ("yasi-2011", "Cyclone Yasi", "Cyclone Yasi", "2011-02-03", "2011-12-31", "tropical cyclone", "Australia"),
    ("qld-floods-2011", "the 2011 Queensland floods", "Queensland floods", "2010-12-25", "2011-12-31", "flood", "Australia"),
    ("seroja-2021", "Cyclone Seroja", "Cyclone Seroja", "2021-04-11", "2022-03-31", "tropical cyclone", "Australia"),
    ("winston-2016", "Cyclone Winston", "Cyclone Winston", "2016-02-20", "2016-12-31", "tropical cyclone", "Fiji"),
    ("xynthia-2010", "Storm Xynthia", "Xynthia", "2010-02-27", "2010-12-31", "windstorm", "France"),
    ("kyrill-2007", "Storm Kyrill", "Kyrill", "2007-01-18", "2007-12-31", "windstorm", "Europe"),
    ("ciara-2020", "Storm Ciara", "Storm Ciara", "2020-02-09", "2020-12-31", "windstorm", "United Kingdom and Europe"),
    ("eunice-2022", "Storm Eunice", "Storm Eunice", "2022-02-18", "2022-12-31", "windstorm", "United Kingdom and Europe"),
    ("desmond-2015", "Storm Desmond", "Storm Desmond", "2015-12-05", "2016-09-30", "flood", "United Kingdom"),
    ("boris-2024", "Storm Boris", "Storm Boris", "2024-09-12", "2025-06-30", "flood", "Central Europe"),
    ("emilia-2023", "the 2023 Emilia-Romagna floods", "Emilia-Romagna", "2023-05-01", "2024-03-31", "flood", "Italy"),
    ("daniel-2023", "Storm Daniel", "Storm Daniel", "2023-09-04", "2024-03-31", "flood", "Greece and Libya"),
    ("valencia-2024", "the 2024 Valencia floods", "Valencia floods", "2024-10-29", "2025-06-30", "flood", "Spain"),
    ("elbe-2013", "the 2013 central European floods", "floods in Central Europe", "2013-05-30", "2014-03-31", "flood", "Germany and Central Europe"),
    ("rs-brazil-2024", "the 2024 Rio Grande do Sul floods", "Rio Grande do Sul", "2024-04-28", "2025-03-31", "flood", "Brazil"),
    ("brazil-drought-2021", "the 2021 Brazilian drought", "Brazilian drought", "2021-06-01", "2022-06-30", "drought", "Brazil"),
    ("chile-megadrought", "the Chilean megadrought", "megadrought", "2014-01-01", "2024-12-31", "drought", "Chile"),
    ("argentina-drought-2023", "the 2023 Argentine drought", "drought in Argentina", "2022-11-01", "2023-12-31", "drought", "Argentina"),
    ("la-nina-colombia", "the 2010 to 2011 La Nina floods", "La Nina", "2010-09-01", "2011-12-31", "flood", "Colombia and Latin America"),
    ("day-zero-2018", "the Cape Town water crisis", "Day Zero", "2018-01-01", "2018-12-31", "drought", "South Africa"),
    ("kzn-2022", "the 2022 KwaZulu-Natal floods", "KwaZulu-Natal floods", "2022-04-11", "2023-03-31", "flood", "South Africa"),
    ("freddy-2023", "Cyclone Freddy", "Cyclone Freddy", "2023-02-21", "2023-12-31", "tropical cyclone", "Mozambique and Malawi"),
    ("kariba-2024", "the 2024 Kariba low water", "Kariba", "2024-01-01", "2025-03-31", "drought", "Zambia and Zimbabwe"),
    ("dubai-2024", "the April 2024 Dubai floods", "Dubai floods", "2024-04-16", "2025-03-31", "flood", "United Arab Emirates"),
    ("bc-floods-2021", "the 2021 British Columbia floods", "British Columbia floods", "2021-11-14", "2022-09-30", "flood", "Canada"),
    ("lytton-2021", "the Lytton wildfire", "Lytton", "2021-06-30", "2022-03-31", "wildfire", "Canada"),
    ("cbam", "the EU Carbon Border Adjustment Mechanism", "Carbon Border Adjustment Mechanism", "2021-07-14", "2025-06-30", "transition policy", "European Union"),
    ("china-ets", "China's national carbon market", "national carbon market", "2021-07-16", "2025-06-30", "transition policy", "China"),
    ("uk-ets", "the UK Emissions Trading Scheme", "UK Emissions Trading Scheme", "2020-12-01", "2025-06-30", "transition policy", "United Kingdom"),
]
ALL_EVENTS = EVENTS + EXTRA + [e for e in GLOBAL_EVENTS if e[0] not in {x[0] for x in EVENTS + EXTRA}]

RESP = [
    ("transfer", re.compile(r"reinsur|catastrophe bond|cat bond|retrocession|quota share|aggregate cover|insurance (recover|proceeds)|recoveries? from insur|business interruption (insurance|claim)", re.I)),
    ("financial", re.compile(r"revolv|credit facilit|borrow|securitiz|rights issue|equity (offering|raise)|issued (notes|debt|shares)|impairment|write-?off|reserves?\\b|dividend|liquidity|covenant|credit rating|\\brating (agenc|downgrade|action)", re.I)),
    ("operational", re.compile(r"generator|resum|restart|rebuild|repair|relocat|alternate (supplier|source)|shut ?down|curtail|reroute|inventory|backup power|restor", re.I)),
    ("strategic", re.compile(r"withdraw|non-?renew|exit|reduce (our )?exposure|raise(d)? rates|rate increase|underwriting (action|change)|divest|reprice", re.I)),
    ("policy", re.compile(r"regulator|legislat|commission approv|rate case|storm (cost )?recovery|securitization charge|fund", re.I)),
]
BOILER = re.compile(r"non-gaap|represents|defined as|forward-looking|safe harbor|undertakes? no obligation|risk factors|incorporated by reference|exhibit", re.I)
AMT = re.compile(r"\$\s?\d[\d,\.]*\s?(?:million|billion|thousand|m|bn)?|\d[\d,\.]*\s?(?:million|billion)\b", re.I)


def _get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read(4_000_000)
        enc = r.headers.get_content_charset() or "utf-8"
    return raw.decode(enc, errors="replace")


def search(ev, form_set, start_from=0):
    key, name, phrase, a, b, *_ = ev
    qs = urllib.parse.urlencode({"q": f'"{phrase}"', "forms": form_set, "dateRange": "custom", "startdt": a, "enddt": b, "from": start_from})
    return json.loads(_get(f"{FTS}?{qs}"))


def doc_url(hit):
    s = hit["_source"]
    cik = str(int(s["ciks"][0]))
    adsh, fname = hit["_id"].split(":", 1)
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh.replace('-', '')}/{fname}", s


def clean_text(html):
    import html as H
    t = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", html)
    t = re.sub(r"(?i)</(p|div|tr|li|br|h\d)>", "\n", t)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    t = H.unescape(t).replace("\xa0", " ")
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t)


def sentences(t):
    out = []
    for para in t.split("\n"):
        para = para.strip()
        if len(para) < 40:
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"(])", para):
            s = s.strip()
            if 8 <= len(s.split()) <= 70:
                out.append(s)
    return out


def classify(s):
    return [k for k, rx in RESP if rx.search(s)]


def extract(text, ev):
    """verbatim sentences that name the event and state an amount or an action"""
    kw = re.compile(re.escape(ev[2]), re.I)
    seen, out = set(), []
    for s in sentences(text):
        s = re.sub(r"^[•\u2022\-\u2013\*\s]+", "", s)
        if not kw.search(s) or not re.search(r"[.!?\u201d\"]$", s):          # must name the event and be a complete sentence
            continue
        if BOILER.search(s) or s.count("(") != s.count(")"):
            continue
        amts = AMT.findall(s)
        cls = classify(s)
        if not amts and not cls:
            continue
        key = re.sub(r"\W+", " ", s.lower())[:90]
        if key in seen:
            continue
        seen.add(key)
        out.append({"quote": s, "amounts": [a.strip() for a in amts][:6], "response": cls})
    return out[:6]


def _doc_job(args):
    url, src, ev = args
    time.sleep(0.25)
    try:
        t = clean_text(_get(url))
    except Exception as e:                                           # noqa: BLE001
        return {"url": url, "ok": False, "why": type(e).__name__}
    q = extract(t, ev)
    if not q:
        return {"url": url, "ok": True, "quotes": []}
    name = src["display_names"][0]
    company = re.sub(r"\s+\(.*", "", name).strip().title() if name.isupper() else re.sub(r"\s+\(.*", "", name).strip()
    return {"url": url, "ok": True, "company": company, "cik": src["ciks"][0], "form": src["form"], "filed": src["file_date"], "sic": (src.get("sics") or [""])[0], "quotes": q}


def collect(out: Path, max_docs=90, only=None, forms="8-K,10-Q,10-K", merge=False, pass_name="", fallback="8-K"):
    """merge=True: keep what is already in out/, and collect only events not yet done (or that failed before). Lets a second pass extend the first."""
    out.mkdir(parents=True, exist_ok=True)
    recs, report = [], {}
    if merge and (out / "edgar_cases.jsonl").exists():
        recs = [json.loads(x) for x in (out / "edgar_cases.jsonl").read_text().splitlines() if x.strip()]
        report = json.loads((out / "edgar_report.json").read_text()) if (out / "edgar_report.json").exists() else {}
    seen_urls = {x["url"] for x in recs}
    for ev in ALL_EVENTS:
        if only and ev[0] not in only:
            continue
        rk = ev[0] + (f"@{pass_name}" if pass_name else "")
        if merge and rk in report and "error" not in report[rk]:
            continue
        if merge and rk in report and report[rk].get("error") == "no hits":
            continue
        hits, err = [], None
        for form_set in (forms, fallback):                           # the combined-form query sometimes 500s; fall back to one form alone
            for frm in (0, 100, 200):
                for attempt in range(2):
                    try:
                        r = search(ev, form_set, frm)
                        break
                    except Exception as e:                           # noqa: BLE001
                        err, r = f"{type(e).__name__}: {str(e)[:60]}", None
                        time.sleep(1.5)
                if r is None:
                    break
                hits += r["hits"]["hits"]
                if len(r["hits"]["hits"]) < 100:
                    break
                time.sleep(0.3)
            if hits:
                break
        if not hits:
            report[rk] = {"error": err or "no hits"}
            print(rk, report[rk], flush=True)
            continue
        # prefer press-release exhibits (short, rich in figures), then main documents; one document per filing
        seen, jobs = set(), []
        for h in sorted(hits, key=lambda x: (0 if x["_source"].get("file_type", "").startswith("EX-99") else 1, -x["_score"])):
            u, s = doc_url(h)
            if u in seen or u in seen_urls:
                continue
            seen.add(u)
            jobs.append((u, s, ev))
        jobs = jobs[:max_docs]
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(_doc_job, jobs))
        n_ok = sum(1 for r in res if r.get("ok") and r.get("quotes"))
        for r in res:
            if r.get("ok") and r.get("quotes") and r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                recs.append({"event": ev[0], "event_name": ev[1], "hazard": ev[5], "region": ev[6], "company": r["company"], "cik": r["cik"], "form": r["form"], "filed": r["filed"], "url": r["url"], "quotes": r["quotes"]})
        report[rk] = {"hits": len(hits), "docs_tried": len(jobs), "docs_with_quotes": n_ok}
        (out / "edgar_cases.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in recs))
        (out / "edgar_report.json").write_text(json.dumps(report, indent=1))
        print(rk, report[rk], flush=True)
    pairs = {(r["event"], r["cik"]) for r in recs}
    summ = {"filings_with_quotes": len(recs), "company_event_cases": len(pairs), "companies": len({r["cik"] for r in recs}), "events": len({r["event"] for r in recs}),
            "quotes": sum(len(r["quotes"]) for r in recs)}
    (out / "edgar_summary.json").write_text(json.dumps(summ, indent=1))
    print(summ)
    return summ


# ── training rows ────────────────────────────────────────────────────────────

SYS = ("You are ClimRisk's climate-risk analyst. Answer using ONLY the numbered sources, which are sentences copied from company filings. Cite every factual sentence with its source number in square brackets, like [2]. "
       "If the sources do not contain the answer, say exactly what is missing instead of guessing. Never add a number that is not in the sources.")
from .heldout import EVENTS as HELD_EVENTS


def clean_quote(q: str):
    """A quote is only usable as a standalone cited sentence if it starts like one. Strips bullets and '(b)' markers; returns None for mid-sentence fragments."""
    q = re.sub(r"^[◦•▪·\-–—\*\s]+", "", (q or "").strip())
    q = re.sub(r"^\(\w{1,2}\)\s*", "", q)
    return q if re.match(r"[A-Z0-9“\"\$]", q) else None


def _cases(path: Path):
    by = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        for q in r["quotes"]:
            cq = clean_quote(q["quote"])
            if not cq:
                continue
            by.setdefault((r["event"], r["cik"]), {"event": r["event"], "event_name": r["event_name"], "company": r["company"], "items": []})["items"].append(
                {"form": r["form"], "filed": r["filed"], "url": r["url"], **{**q, "quote": cq}})
    return by


def _blk(items, rng):
    order = list(range(len(items)))
    rng.shuffle(order)
    pos, lines = {}, []
    for n, k in enumerate(order, 1):
        pos[k] = n
        text, _ = items[k]
        lines.append(f"[{n}] {text}")
    return "\n\n".join(lines), pos


def _item(c, i):
    it = c["items"][i]
    label = f"climate report {it['filed']}" if str(it["form"]).lower() == "climate report" else f"Form {it['form']} filed {it['filed']}"
    return (f"{c['company']}, {label}: {it['quote']}", (c["company"], i))


def _clean(t):
    return t.rstrip(".").rstrip()


def _row(kind, q, blk, ans, need, n, split, meta):
    return {"kind": kind, "split": split, "meta": {"need": need, "n": n, **meta},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Question: {q}\n\nSources:\n{blk}"}, {"role": "assistant", "content": ans}]}


def build_rows(path: Path, seed=5, per_case=2):
    rng = random.Random(seed)
    by = _cases(path)
    cases = list(by.values())
    by_event = {}
    for c in cases:
        by_event.setdefault(c["event"], []).append(c)
    rows = []
    for c in cases:
        held = c["event"] in HELD_EVENTS
        split = "test" if held else ("valid" if rng.random() < 0.08 else "train")
        pool = [x for x in cases if (x["event"] in HELD_EVENTS) == held and x is not c]
        for rep in range(per_case):
            n = min(len(c["items"]), rng.choice([1, 2, 3]))
            idx = sorted(rng.sample(range(len(c["items"])), n))
            others = rng.sample(pool, min(2, len(pool)))
            items = [_item(c, i) for i in idx] + [_item(o, rng.randrange(len(o["items"]))) for o in others]
            blk, pos = _blk(items, rng)
            ans = f"{c['company']} reported the following about {c['event_name']}. " + " ".join(f"{_clean(c['items'][i]['quote'])} [{pos[k]}]." for k, i in zip(range(n), idx))
            q = rng.choice([f"What did {c['company']} report about {c['event_name']}?", f"How was {c['company']} affected by {c['event_name']} according to its filings?",
                            f"What figures or actions did {c['company']} disclose for {c['event_name']}?"])
            rows.append(_row("case", q, blk, ans, [pos[k] for k in range(n)], len(items), split, {"type": "company", "case": f"{c['event']}:{c['company']}"}))
        resp = [i for i, it in enumerate(c["items"]) if it["response"]]
        if resp:
            i = rng.choice(resp)
            others = rng.sample(pool, min(2, len(pool)))
            items = [_item(c, i)] + [_item(o, rng.randrange(len(o["items"]))) for o in others]
            blk, pos = _blk(items, rng)
            kinds = "/".join(c["items"][i]["response"])
            ans = f"{c['company']} described this action ({kinds}) in response to {c['event_name']}: {_clean(c['items'][i]['quote'])} [{pos[0]}]."
            rows.append(_row("case", f"How did {c['company']} deal with the effects of {c['event_name']}?", blk, ans, [pos[0]], len(items), split, {"type": "company_response", "case": f"{c['event']}:{c['company']}"}))
    for ev, cs in by_event.items():
        held = ev in HELD_EVENTS
        split = "test" if held else "train"
        if len(cs) < 4:
            continue
        for _ in range(max(3, len(cs) // 4)):
            pick = rng.sample(cs, 4)
            items = [_item(c, rng.randrange(len(c["items"]))) for c in pick]
            blk, pos = _blk(items, rng)
            ans = f"The companies describe {pick[0]['event_name']} differently. " + " ".join(f"{pick[j]['company']}: {_clean(items[j][0].split(': ', 1)[1])} [{pos[j]}]." for j in range(4))
            rows.append(_row("case", f"How did different companies describe the impact of {pick[0]['event_name']}?", blk, ans, [pos[j] for j in range(4)], 4, split, {"type": "compare", "case": ev}))
        # unknown: a company that reported on the event is asked about a different event it did not report on
        for _ in range(max(1, len(cs) // 12)):
            c = rng.choice(cs)
            other_events = [e for e in by_event if e != ev]
            if not other_events:
                continue
            e2 = rng.choice(other_events)
            if any(x["company"] == c["company"] for x in by_event[e2]):
                continue
            items = [_item(c, rng.randrange(len(c["items"])))] + [_item(o, rng.randrange(len(o["items"]))) for o in rng.sample(by_event[e2], min(2, len(by_event[e2])))]
            blk, pos = _blk(items, rng)
            e2name = by_event[e2][0]["event_name"]
            ans = (f"The sources provided do not cover what {c['company']} reported about {e2name}. They only show that {c['company']} reported this about {c['event_name']}: "
                   f"{_clean(items[0][0].split(': ', 1)[1])} [{pos[0]}]. To answer I would need {c['company']}'s own filing for {e2name}.")
            rows.append(_row("case_gap", f"What did {c['company']} report about {e2name}?", blk, ans, [pos[0]], len(items), "test" if ev in HELD_EVENTS else "train", {"type": "company_unknown", "case": f"{ev}:{c['company']}"}))
    return rows


def write_rows(path: Path, out: Path):
    from collections import Counter
    rows = build_rows(path)
    out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "valid", "test"):
        with open(out / f"edgar_{s}.jsonl", "w") as f:
            for r in rows:
                if r["split"] == s:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({s: dict(Counter(r["meta"]["type"] for r in rows if r["split"] == s)) for s in ("train", "valid", "test")})


def main():
    a = sys.argv[1:]
    if a[0] == "collect":
        md = 90
        only = None
        merge = "--merge" in a
        forms = a[a.index("--forms") + 1] if "--forms" in a else "8-K,10-Q,10-K"
        pass_name = a[a.index("--pass") + 1] if "--pass" in a else ""
        fallback = a[a.index("--fallback") + 1] if "--fallback" in a else ("6-K" if "6-K" in forms else "8-K")
        if "--max-docs-per-event" in a:
            md = int(a[a.index("--max-docs-per-event") + 1])
        if "--only" in a:
            only = a[a.index("--only") + 1].split(",")
        collect(Path(a[1]), md, only, forms=forms, merge=merge, pass_name=pass_name, fallback=fallback)
    elif a[0] == "rows":
        write_rows(Path(a[1]) / "edgar_cases.jsonl", Path(a[2]))


if __name__ == "__main__":
    main()
