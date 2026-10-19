"""Golden sites: places where public knowledge gives an unambiguous coarse expectation.

Each expectation is a rating FLOOR (`min`) or CEILING (`max`) on one report zone, on the scale NEGLIGIBLE < LOW < MODERATE < HIGH < CRITICAL.
They are deliberately coarse ("exposed to hurricanes", "not exposed to hurricanes"), taken from well-known events/climatology, not from
this engine's output. Where the engine disagrees today the disagreement is recorded in KNOWN_GAPS with the reason; that list is the honest
statement of what the screening-grade engine gets wrong, and a fix is expected to REMOVE entries from it (the test then tells you to).

zones: river_flood, coastal_flood, surface_water, cyclone, wildfire, heat, slope, water
"""
SITES = {
 # name: (lat, lon, {zone: {"min": R} or {"max": R}}, why)
 "miami":        (25.76, -80.19, {"cyclone": {"min": "HIGH"}, "coastal_flood": {"min": "MODERATE"}}, "Hurricane Andrew 1992, Irma 2017; one of the most sea-level-exposed cities"),
 "new_orleans":  (29.95, -90.07, {"cyclone": {"min": "MODERATE"}, "coastal_flood": {"min": "MODERATE"}}, "Katrina 2005; city below sea level behind levees"),
 "houston":      (29.76, -95.37, {"cyclone": {"min": "MODERATE"}, "river_flood": {"min": "MODERATE"}}, "Harvey 2017 flooding; Gulf cyclone region"),
 "rotterdam":    (51.92, 4.48,   {"cyclone": {"max": "NEGLIGIBLE"}, "coastal_flood": {"max": "LOW"}, "river_flood": {"max": "LOW"}}, "no tropical cyclones; 1-in-10,000 year dike standard"),
 "dhaka":        (23.81, 90.41,  {"river_flood": {"min": "MODERATE"}, "cyclone": {"min": "LOW"}}, "Ganges-Brahmaputra-Meghna floodplain; Bay of Bengal cyclones"),
 "jakarta":      (-6.21, 106.85, {"coastal_flood": {"min": "MODERATE"}, "cyclone": {"max": "NEGLIGIBLE"}}, "land subsidence and tidal flooding; equatorial (no cyclones)"),
 "bangkok":      (13.75, 100.50, {"river_flood": {"min": "MODERATE"}}, "2011 Chao Phraya floods"),
 "brisbane":     (-27.47, 153.03, {"river_flood": {"min": "MODERATE"}}, "2011 Brisbane River flood"),
 "dresden":      (51.05, 13.74,  {"river_flood": {"min": "MODERATE"}}, "Elbe floods 2002 and 2013"),
 "cologne":      (50.94, 6.96,   {"river_flood": {"min": "MODERATE"}}, "Rhine flood history (1993, 1995)"),
 "ahrweiler":    (50.54, 7.09,   {"river_flood": {"min": "MODERATE"}}, "July 2021: 134 deaths in the Ahr valley (the engine's own insurance module cites this as a known miss)"),
 "chennai":      (13.08, 80.27,  {"river_flood": {"min": "MODERATE"}, "cyclone": {"min": "MODERATE"}}, "2015 Chennai floods; NE-monsoon cyclones"),
 "mumbai":       (19.08, 72.88,  {"coastal_flood": {"min": "LOW"}}, "2005 Mumbai floods; low-lying coast"),
 "manila":       (14.60, 120.98, {"cyclone": {"min": "HIGH"}}, "western-Pacific typhoon belt (Haiyan, Ketsana)"),
 "tokyo":        (35.68, 139.77, {"cyclone": {"min": "MODERATE"}}, "typhoon landfalls (Faxai, Hagibis 2019)"),
 "singapore":    (1.35, 103.82,  {"cyclone": {"max": "LOW"}}, "1.4 N: essentially no tropical-cyclone hazard (Typhoon Vamei 2001 was a freak exception)"),
 "reykjavik":    (64.15, -21.90, {"heat": {"max": "NEGLIGIBLE"}, "wildfire": {"max": "NEGLIGIBLE"}, "cyclone": {"max": "NEGLIGIBLE"}}, "sub-arctic oceanic climate"),
 "phoenix":      (33.45, -112.07, {"heat": {"min": "HIGH"}, "water": {"min": "HIGH"}, "cyclone": {"max": "NEGLIGIBLE"}, "coastal_flood": {"max": "NEGLIGIBLE"}}, "Sonoran desert: 40 C+ summers, Colorado River dependence"),
 "riyadh":       (24.71, 46.68,  {"heat": {"min": "HIGH"}, "water": {"min": "HIGH"}, "cyclone": {"max": "NEGLIGIBLE"}}, "hyper-arid; 45 C+ summers"),
 "delhi":        (28.61, 77.21,  {"heat": {"min": "HIGH"}, "water": {"min": "HIGH"}}, "pre-monsoon heatwaves above 45 C; severe water stress"),
 "kuwait_city":  (29.38, 47.99,  {"heat": {"min": "HIGH"}, "coastal_flood": {"max": "LOW"}}, "among the hottest cities on Earth (50 C+)"),
 "pasadena_fire":(34.18, -118.15, {"wildfire": {"min": "MODERATE"}}, "San Gabriel foothills: Station Fire 2009, Eaton Fire 2025"),
 "blue_mountains":(-33.72, 150.31, {"wildfire": {"min": "HIGH"}}, "Black Summer 2019-20"),
 "kathmandu":    (27.72, 85.32,  {"slope": {"min": "MODERATE"}}, "steep Himalayan terrain, monsoon landslides"),
 "death_valley": (36.46, -116.87, {"heat": {"min": "HIGH"}, "cyclone": {"max": "NEGLIGIBLE"}}, "world record air temperature 56.7 C"),
 "venice":       (45.44, 12.33,  {"coastal_flood": {"min": "MODERATE"}}, "acqua alta (2019 peak 1.87 m)"),
 "funafuti":     (-8.52, 179.20, {"coastal_flood": {"min": "MODERATE"}}, "atoll, highest ground ~4 m"),
 "male":         (4.18, 73.51,   {"coastal_flood": {"min": "MODERATE"}}, "Maldives, average elevation ~1.5 m"),
}


COASTAL = "Deltares 90 m undefended surge-and-tide map reads dry (or nearly) at the pin: its elevation model cannot resolve built-up ground, atolls or subsiding coasts. The engine reports 'outside the modelled storm-surge zone', which is what the data says, not what the place experiences."
RIVER = "The JRC/GloFAS river-flood map (30 arcsec) is dry or shallow at the 250 m pin: it misses small rivers and some large-river floodplains and ignores local flood history. For Ahrweiler the engine's own insurer module documents this miss (2021)."
HEAT = "The 'heat' zone is a land-cover and latitude indicator (built-up, tree canopy), not an air-temperature or WBGT model, so hot-desert and hot-city sites cannot reach HIGH. WBGT/UTCI heat exists in the model lab but is not in this panel."
KNOWN_GAPS = {
 ("miami", "coastal_flood"): COASTAL, ("new_orleans", "coastal_flood"): COASTAL, ("jakarta", "coastal_flood"): COASTAL,
 ("mumbai", "coastal_flood"): COASTAL, ("venice", "coastal_flood"): COASTAL, ("funafuti", "coastal_flood"): COASTAL, ("male", "coastal_flood"): COASTAL,
 ("dhaka", "river_flood"): RIVER, ("bangkok", "river_flood"): RIVER, ("dresden", "river_flood"): RIVER, ("cologne", "river_flood"): RIVER, ("ahrweiler", "river_flood"): RIVER,
 ("phoenix", "heat"): HEAT, ("riyadh", "heat"): HEAT, ("delhi", "heat"): HEAT, ("kuwait_city", "heat"): HEAT, ("death_valley", "heat"): HEAT,
 ("blue_mountains", "wildfire"): "Wildfire zone comes from vegetation cover and NDVI at the pin, not from fire-weather or fire history; Black Summer sites read MODERATE.",
}
