"""Question batteries for the two question families the agent is asked most: 'can I build X here, what would it cost' and 'ship from A to B, how much carbon/cost/disruption'.

Each case is (text, expected). For shipping, expected is the set of fields parse() must return (subset match; a key set to None means it must be absent).
For building, expected is the set of fields feasibility.parse() must return. `route` is the handler that must fire; 'none' means no building/voyage handler may claim it.
Phrasings were written to cover how people actually ask: names with or without MV, "from A to B", "A -> B", distances in nm/km/miles, speeds in knots,
ship sizes as dwt/TEU/GT, fuels by trade name, years, prices, and the same for buildings: storeys/floors/height/area, use types, place forms.
"""

SHIP = [
    ("Ship Ever Given is docked at Rotterdam and moves to Shanghai, 16000 nm away at 14 knots, how much carbon, what disruption and cost?",
     dict(name="Ever Given", origin="Rotterdam", dest="Shanghai", distance_nm=16000, speed_kn=14)),
    ("MV Atlantic Star Singapore to Los Angeles 85,000 dwt bulker 12 knots VLSFO $620/t",
     dict(name="Atlantic Star", origin="Singapore", dest="Los Angeles", dwt=85000, speed_kn=12, fuel="hfo", ship_type="bulk_carrier")),
    ("how much CO2 will a 8000 TEU container ship emit going from Hamburg to New York",
     dict(origin="Hamburg", dest="New York", teu=8000, ship_type="container_ship")),
    ("Vessel IMO 9811000 is moored at Port of Singapore and heading to Rotterdam, 8300 nm, 13.5 kn", dict(imo="9811000", origin="Singapore", dest="Rotterdam", distance_nm=8300, speed_kn=13.5)),
    ("The tanker Nordic Spirit sails from Ras Tanura to Rotterdam, what are the emissions and the ETS cost in 2026?", dict(name="Nordic Spirit", origin="Ras Tanura", dest="Rotterdam", year=2026, ship_type="oil_tanker")),
    ("bulk carrier Oceanic Dream from Port Hedland to Qingdao 3,400 nautical miles at 11.5 knots", dict(name="Oceanic Dream", origin="Port Hedland", dest="Qingdao", distance_nm=3400, speed_kn=11.5, ship_type="bulk_carrier")),
    ("container ship Maersk Elba, Rotterdam -> Singapore, 8,300 NM, 16 knots, burning VLSFO at 90 t/day", dict(name="Maersk Elba", origin="Rotterdam", dest="Singapore", distance_nm=8300, speed_kn=16, fuel="hfo", daily_t=90)),
    ("What if MV Pacific Pearl goes Busan to Long Beach 5,300 nm at 14 knots and delays 3 days at anchor?", dict(name="Pacific Pearl", origin="Busan", dest="Long Beach", distance_nm=5300, speed_kn=14, delay_days=3)),
    ("Ship Aurora, docked at Piraeus, going to Alexandria, uses 35 tonnes of MGO per day, 400 nm, 12 knots", dict(name="Aurora", origin="Piraeus", dest="Alexandria", distance_nm=400, speed_kn=12, fuel="mdo_mgo", daily_t=35)),
    ("A 150,000 dwt tanker is moving from Fujairah to Singapore, 3,300 nm. Emissions and cost if bunker is 650 USD/t?", dict(origin="Fujairah", dest="Singapore", dwt=150000, distance_nm=3300, ship_type="oil_tanker", bunker_usd_per_t=650)),
    ("vessel Ocean Titan Dubai to Mumbai 1,000 nm 14 knots LNG", dict(name="Ocean Titan", origin="Dubai", dest="Mumbai", distance_nm=1000, speed_kn=14, fuel="lng_otto_medium")),
    ("our container ship is at Shanghai and needs to reach Rotterdam, 10,500 nm, can we slow to 12 knots to cut carbon?", dict(origin="Shanghai", dest="Rotterdam", distance_nm=10500, speed_option_kn=12, ship_type="container_ship")),
    ("MT Sea Falcon moving from Houston to Antwerp, 4900 nm, 13 kn, hire rate 28000 USD/day", dict(name="Sea Falcon", origin="Houston", dest="Antwerp", distance_nm=4900, speed_kn=13, charter_usd_per_day=28000)),
    ("Ship called Baltic Wind will sail from Gdansk to Rotterdam", dict(name="Baltic Wind", origin="Gdansk", dest="Rotterdam")),
    ("how bad is a Red Sea diversion for the vessel Cape Pioneer going Jeddah to Rotterdam, 3,300 nm", dict(name="Cape Pioneer", origin="Jeddah", dest="Rotterdam", distance_nm=3300)),
    ("5000 TEU boxship from Colombo to Felixstowe at 18 knots in 2027", dict(origin="Colombo", dest="Felixstowe", teu=5000, speed_kn=18, year=2027, ship_type="container_ship")),
    ("MV Star Harmony, 60000 dwt bulk carrier, Santos to Rotterdam, 5,900 nm, 12 knots, VLSFO 600 per tonne", dict(name="Star Harmony", dwt=60000, origin="Santos", dest="Rotterdam", distance_nm=5900, speed_kn=12, bunker_usd_per_t=600)),
    ("ship docked at Tokyo moving to Vancouver 4,300 nautical miles", dict(origin="Tokyo", dest="Vancouver", distance_nm=4300)),
    ("vessel Mediterranean Queen sails Genoa - Barcelona, 300 nm at 20 knots on MDO", dict(name="Mediterranean Queen", origin="Genoa", dest="Barcelona", distance_nm=300, speed_kn=20, fuel="mdo_mgo")),
    ("tanker route Rotterdam to New York how many tonnes of CO2 for an 80,000 dwt ship", dict(origin="Rotterdam", dest="New York", dwt=80000, ship_type="oil_tanker")),
    ("Voyage of container ship MSC Gulsun from Tanjung Pelepas to Hamburg, 8,500 nm, 15 knots", dict(name="MSC Gulsun", origin="Tanjung Pelepas", dest="Hamburg", distance_nm=8500, speed_kn=15)),
    ("MV Horizon is going Singapore to Rotterdam via the Cape, 11,800 nm, 13 knots", dict(name="Horizon", origin="Singapore", dest="Rotterdam", distance_nm=11800, speed_kn=13)),
    ("A vessel at Chittagong going to Singapore, 1,200 nm, what is the carbon and the cost", dict(origin="Chittagong", dest="Singapore", distance_nm=1200)),
    ("ship Ocean Spirit from Valencia, Spain to Algeciras 380 nm 14 knots", dict(name="Ocean Spirit", origin="Valencia", dest="Algeciras", origin_country="Spain", distance_nm=380, speed_kn=14)),
    ("cruise ship Aurora Dream goes Southampton to New York 3,100 nm at 17 knots", dict(name="Aurora Dream", origin="Southampton", dest="New York", distance_nm=3100, speed_kn=17)),
    ("how much carbon does a ro-ro ferry emit on a 120 nm trip from Dover to Calais at 18 knots", dict(origin="Dover", dest="Calais", distance_nm=120, speed_kn=18)),
    ("Ship Harmony of the Seas 4,500 nm Miami to Barcelona 20 knots fuel 150 tonnes per day HFO", dict(name="Harmony of the Seas", origin="Miami", dest="Barcelona", distance_nm=4500, speed_kn=20, daily_t=150, fuel="hfo")),
    ("a bulker from Newcastle, Australia to Kashima 4,200 nm 12 knots", dict(origin="Newcastle", dest="Kashima", distance_nm=4200, speed_kn=12, ship_type="bulk_carrier")),
    ("Container vessel 14,000 TEU leaving Ningbo for Los Angeles: 5,600 nm, 17 knots, bunker 700 USD per tonne, what if the port is delayed 4 days", dict(origin="Ningbo", dest="Los Angeles", teu=14000, distance_nm=5600, speed_kn=17, bunker_usd_per_t=700)),
    ("What will the ETS bill be for a ship going Rotterdam to Houston, 4,800 nm, 15 knots, 2026, EUA 75 EUR", dict(origin="Rotterdam", dest="Houston", distance_nm=4800, speed_kn=15, year=2026)),
]

SHIP_NEGATIVE = [  # must not be claimed by the voyage handler
    "What is a tanker?",
    "Which ports are in the EU ETS?",
    "Show me flood risk for the port of Rotterdam",
    "How do ships get fuel?",
    "write a physical risk report for the port of Singapore",
    "How many storeys can I build at Rotterdam?",
]

BUILD = [
    ("Can I build a 10-storey building here, what would it cost?", dict(storeys=10)),
    ("Can I build a 10 story building at 25.77, -80.19 and what will it cost?", dict(storeys=10)),
    ("is it feasible to put up a twelve floor apartment block in Dhaka", dict(storeys=12, use="residential")),
    ("what would it cost to construct a 20-storey office tower in Singapore", dict(storeys=20, use="office")),
    ("I want to develop a 6 storey hotel in Phuket, is it allowed and what budget do I need?", dict(storeys=6, use="hotel")),
    ("could I build a 40 m high-rise near the river in Cologne", dict(height_m=40)),
    ("can i build a 3 level car park in Houston", dict(storeys=3, use="car park")),
    ("how much would a 5,000 square metre warehouse cost to build in Rotterdam", dict(area_m2=5000, use="warehouse")),
    ("is it safe to build a factory of 20,000 m2 in Chennai? what are the risks and costs", dict(area_m2=20000, use="factory")),
    ("Can I build a house in Miami Beach? what will it cost", dict(use="house")),
    ("can we build a data centre in Frankfurt, what would it cost", dict(use="data centre")),
    ("planning a 15 floor residential tower in Jakarta - feasibility and cost?", dict(storeys=15, use="residential")),
    ("budget for a hospital at 51.5, -0.12", dict(use="hospital")),
    ("we want to build a school with 3 floors in Manila, can we, and what would it cost?", dict(storeys=3, use="school")),
    ("Is it possible to build a 30-storey tower in Mumbai near the coast?", dict(storeys=30)),
    ("what does it cost to put up a 8 storey block of flats at Lagos", dict(storeys=8, use="residential")),
    ("could i erect a 100 m tall building in Doha", dict(height_m=100)),
    ("how feasible is a 25 floor hotel in Dubai Marina", dict(storeys=25, use="hotel")),
    ("can I build a 2-storey home in Houston and what will it cost me?", dict(storeys=2, use="house")),
    ("Cost to build a 12,000 sq ft warehouse in Savannah", dict(area_ft2=12000, use="warehouse")),
    ("Build a 4 storey student hostel in Kolkata - allowed? cost?", dict(storeys=4)),
    ("what would a 60-storey skyscraper in Shenzhen cost and can it be built there", dict(storeys=60)),
    ("is it viable to develop a 7-storey mixed-use building in Amsterdam", dict(storeys=7)),
    ("do you think I can construct a 9 floors building in Karachi?", dict(storeys=9)),
    ("feasibility of a 3 storey shopping mall at 13.08, 80.27", dict(storeys=3, use="mall")),
]

BUILD_NEGATIVE = [  # building words but a different tool or no question about building something
    "show me the 3D twin of a 10-storey building in Miami",
    "what flood protection do I need for my 10 storey building",
    "simulate a flood at my building",
    "what is the physical risk of the port of Singapore",
    "what is a storey",
    "write a physical risk report for 25.77, -80.19",
]

# Held-out set: written after the parser was tuned on SHIP above; never used to tune it. Reported separately.
SHIP_HELDOUT = [
    ("Our bulker Cape Aurora (IMO 9876543) is at Port Hedland and has to get to Qingdao, 3,350 nm. If she steams at 12.5 knots what are the CO2 and the cost?", dict(imo="9876543", origin="Port Hedland", dest="Qingdao", distance_nm=3350, speed_kn=12.5)),
    ("Estimate emissions for the VLCC Front Eagle going from Ras Tanura to Ningbo, around 6,500 nautical miles, 13 kt", dict(name="Front Eagle", origin="Ras Tanura", dest="Ningbo", distance_nm=6500, speed_kn=13)),
    ("what would a 12,000 TEU boxship burn between Yantian and Rotterdam at 18 knots", dict(origin="Yantian", dest="Rotterdam", teu=12000, speed_kn=18, ship_type="container_ship")),
    ("Hi, MV Lady Maria leaves Gdynia tomorrow for Rotterdam (about 600 nm). She makes 11 knots. What's the CO2 and ETS?", dict(name="Lady Maria", origin="Gdynia", dest="Rotterdam", distance_nm=600, speed_kn=11)),
    ("Tanker Elbe Star: Antwerp to Houston, 4,900 nm, 80,000 dwt, bunker $640 per tonne, hire 25,000/day", dict(name="Elbe Star", origin="Antwerp", dest="Houston", distance_nm=4900, dwt=80000, bunker_usd_per_t=640)),
    ("how many tonnes of fuel does a Suezmax need from Primorsk to Rotterdam, 1,300 nm", dict(origin="Primorsk", dest="Rotterdam", distance_nm=1300)),
    ("ship docked in Mombasa heading to Mumbai 2,400 nm 13 knots uses 28 t/day", dict(origin="Mombasa", dest="Mumbai", distance_nm=2400, speed_kn=13, daily_t=28)),
    ("MV Sunrise sails Kaohsiung to Manila 520 nm what is the fuel at 14 kn, MGO", dict(name="Sunrise", origin="Kaohsiung", dest="Manila", distance_nm=520, speed_kn=14, fuel="mdo_mgo")),
    ("vessel Polar Light, Murmansk to Rotterdam, 2,300 nautical miles, 70,000 dwt, ice class, speed 12 knots", dict(name="Polar Light", origin="Murmansk", dest="Rotterdam", distance_nm=2300, dwt=70000, speed_kn=12)),
    ("Compare slow steaming for a container ship going Shanghai to Los Angeles 5,700 nm at 20 knots versus 16 knots", dict(origin="Shanghai", dest="Los Angeles", distance_nm=5700, speed_kn=20, ship_type="container_ship")),
    ("A 40,000 dwt chemical tanker goes from Rotterdam to Fujairah, 6,300 nm, 14 knots, in 2026. EUA is 80 EUR per tonne", dict(origin="Rotterdam", dest="Fujairah", dwt=40000, distance_nm=6300, speed_kn=14, year=2026, eua_price=80, ship_type="chemical_tanker")),
    ("ro-ro ferry Stena Line from Harwich to Hook of Holland, 110 nm, 20 knots", dict(origin="Harwich", dest="Hook of Holland", distance_nm=110, speed_kn=20)),
    ("Container ship ZIM Antwerp is moving from Haifa to Piraeus (750 nm). 15 knots.", dict(name="ZIM Antwerp", origin="Haifa", dest="Piraeus", distance_nm=750, speed_kn=15)),
    ("we have a panamax bulker at Gladstone going to Kashima, 4,200 nm at 12 knots - what if there is a 5 day delay at anchor?", dict(origin="Gladstone", dest="Kashima", distance_nm=4200, speed_kn=12, delay_days=5)),
    ("how much carbon for a 6,000 km voyage by container ship from Rotterdam to Barcelona at 16 knots", dict(origin="Rotterdam", dest="Barcelona", speed_kn=16, ship_type="container_ship")),
]

BUILD_HELDOUT = [
    ("I own a plot at 19.0760, 72.8777 and I'd like to put up a 14 storey residential building. What are the site risks and roughly what would it cost?", dict(storeys=14, use="residential")),
    ("how risky is it to build a 4000 m2 distribution centre in Memphis", dict(area_m2=4000, use="warehouse")),
    ("we're considering a 22-floor office in Lagos, is it feasible and what's the budget", dict(storeys=22, use="office")),
    ("what will it cost to construct a hospital in Tacloban", dict(use="hospital")),
    ("Is it ok to build a 5 storey hotel right on the beach in Cancun?", dict(storeys=5, use="hotel")),
    ("thinking of erecting a 50 m tall tower in Rotterdam - allowed? cost?", dict(height_m=50)),
    ("can i build a 2 story home in Tampa, Florida, how much would that be", dict(storeys=2, use="house")),
    ("we need a new 30,000 sq ft factory near Houston. can we build there and what would it cost", dict(area_ft2=30000, use="factory")),
    ("is a 40-storey skyscraper feasible in Jakarta?", dict(storeys=40)),
    ("would it be possible to build a 3-floor school building in Dhaka and what budget should I plan", dict(storeys=3, use="school")),
    ("How much would it cost to develop a 10 storey apartment block in Manchester?", dict(storeys=10, use="residential")),
    ("what about a data center in Singapore, can we build it, cost?", dict(use="data centre")),
]
BUILD_HELDOUT_NEGATIVE = [
    "build a 3D twin of my 6 storey hotel",
    "how do I protect my warehouse from flooding",
    "what is the cost of carbon for a ship",
    "report on building flood risk in Miami",
    "my house is flooding what do I do",
]

SHIP_NEEDS_ROUTE = ["How much carbon does my container ship emit?", "what's the fuel and ETS cost for a bulk carrier at 12 knots", "vessel IMO 9811000 emissions"]
BUILD_NO_SPEC = [("Can I build in Miami Beach, what would it cost?", {}), ("is it feasible to build a building here and what's the budget", {})]
BUILD_NOT_BUILDING = ["how much does it cost to build a model for credit risk", "can I build a team for the app", "what does it cost to develop a website"]
