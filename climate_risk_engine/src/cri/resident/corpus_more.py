"""Extra training material for the resident agent's intent model (added 29 Oct 2026).

What it adds: the two question families that were rule-only (voyage, feasibility), an explicit off-topic class so that questions the agent has no tool for are
recognised as such instead of being forced onto the nearest tool, more phrasings for the intents that scored lowest on an independent test set, and
typing noise (lower case, missing punctuation, dropped letters, swapped letters) so that hurried messages are not out-of-distribution.

Rule for contributors: nothing here may be copied from tests/nlu_battery.py. That file measures; this one trains. tests/test_resident_training.py checks for overlap.
"""
from __future__ import annotations

import random
import re

SLOTS_MORE = {
    "port": ["Rotterdam", "Singapore", "Shanghai", "Hamburg", "Los Angeles", "Santos", "Busan", "Antwerp", "Jebel Ali", "Houston", "Mumbai", "Piraeus", "Durban", "Felixstowe", "Tanjung Pelepas", "Colombo"],
    "ship": ["container ship", "tanker", "bulk carrier", "bulker", "vessel", "ship", "LNG carrier", "ro-ro", "car carrier", "VLCC", "cruise ship", "MV Atlas", "MT Orion"],
    "use": ["hotel", "warehouse", "factory", "office block", "apartment building", "hospital", "school", "data centre", "shopping mall", "car park", "tower", "house", "logistics hub"],
    "n": ["4", "8", "12", "20", "35", "six", "fifteen", "forty"],
    "city": ["Dhaka", "Lagos", "Miami", "Jakarta", "Manila", "Mumbai", "Houston", "Rotterdam", "Chennai", "Karachi", "Amsterdam", "Shenzhen", "Doha", "Savannah"],
}

T_NEW = {
    "voyage": [
        "how much CO2 does a {ship} emit going from {port} to {port}", "{ship} {port} to {port}, what are the emissions and the ETS bill", "what does it cost in fuel to send a {ship} from {port} to {port}",
        "carbon footprint of a voyage {port} - {port}", "EU ETS exposure for a {ship} sailing to {port}", "if the {ship} slows down on the way to {port} what do we save", "emissions per trip for a {ship} from {port}",
        "{ship} bunkers and CO2 for {port} to {port} at 14 knots", "how long does {port} to {port} take and what does it burn", "FuelEU penalty for a {ship} calling at {port}",
        "what happens to the voyage cost if the {ship} has to go round the Cape", "would a delay at {port} cost us extra fuel", "voyage carbon calculator {port} {port}", "ship emissions estimate",
        "tonnes of fuel for a {ship} crossing from {port} to {port}", "how many days and how much fuel from {port} to {port} for our {ship}", "{ship} is leaving {port} for {port}, work out the emissions and cost",
        "Red Sea diversion cost for a {ship} bound for {port}", "what is the IMO class average burn for a {ship}", "price the ETS allowances for a {ship} voyage to {port}",
    ],
    "feasibility": [
        "can I build a {n} storey {use} in {city}", "what would a {use} cost to build at {city} and is the site safe", "is it viable to develop a {use} near {place}", "feasibility of a {use} in {city}",
        "we are thinking about a {n}-floor {use} at {place}, what are the risks and the budget", "can we put a {use} on this plot", "how much to construct a {use} in {city}", "is {place} a sensible place for a {n} storey {use}",
        "what site hazards matter for a new {use} in {city}", "budget and flood risk for a {use} at {place}", "is it allowed to build a {use} at {place}", "what will hazards add to the build cost of a {use} in {city}",
        "thinking of erecting a {use} on {place}", "construction cost for a {n} storey {use}", "do I have a problem building a {use} here", "{use} development at {city}: can it be done and what does it cost",
        "should I build a {use} at {place}, hazards and cost", "risk screening for a planned {use} in {city}", "viability of constructing a {use} near the coast at {city}", "what would stop me building a {n} storey {use} here",
    ],
    "offtopic": [
        "what is the capital of {offt}", "who invented the telephone", "tell me a recipe for lasagne", "how do I fix a leaking tap", "write a poem about the sea", "translate good night into Spanish", "translate hello into Italian", "how do you say thank you in French", "say good morning in Japanese", "translate thanks into Portuguese",
        "what's 45 divided by 9", "best phone under 500 dollars", "who is the president of Brazil", "how many calories in an avocado", "suggest a film for tonight", "what time is it in Tokyo",
        "give me a workout plan", "how do I change my password", "explain how a bicycle works", "what's a good name for a dog", "who won the last Champions League", "how do I learn guitar",
        "recommend a book about history", "what should I cook tonight", "help me write a cover letter", "convert 70 kg to pounds", "how tall is Mount Everest", "what is the meaning of life",
        "plan my holiday in Italy", "debug my python script", "which stocks should I buy", "what is bitcoin worth", "draft an email to my landlord", "how do I get a refund from an airline",
        "tell me about the Roman empire", "can you play music", "what's the best pizza topping", "latest football scores", "who painted the Mona Lisa", "how do vaccines work",
    ],
    "scenario": ["combine a {rp} flood with higher seas at {place}", "what does {year} look like for the {sector} site", "contrast a gentle and a harsh outcome for {place}",
                 "put {place} under pressure with a stronger storm and rising sea", "what is the range of outcomes for {company} by {year}", "how would things change if the planet warms two degrees", "what if the worst forecast comes true for {place}", "give me upside and downside cases for {place}", "will {place} still be viable in {year}",
                 "project {place} forward to {year} under high emissions", "vary the sea level and the storm strength and show the spread for {place}", "what are the plausible futures for {company}"],
    "report_physical": ["I need a document covering all physical hazards at {place}", "hazard exposure report for the {sector} site at {place}", "client-ready write up of the flood and wind risk at {place}",
                        "summarise physical risk at {place} in a report", "physical risk dossier for {place}"],
    "report_carbon": ["put together our greenhouse gas inventory as a report", "annual emissions report for {company}", "scope 1, 2 and 3 write up for the auditor", "carbon accounting pack for {company}",
                      "prepare the emissions disclosure report for {company}"],
    "risk_analysis": ["what is {company}'s climate exposure", "how much could climate cost {company}", "analyse the risk profile of {company}", "give me a risk read on {company}",
                      "what are the dollar losses from climate for {company}", "score {company} for climate risk"],
    "explain": ["why is that the answer", "how come the rating is so high", "what is a return period", "what does design wind speed mean", "what do you mean by that", "where do these numbers come from",
                "what is the difference between physical and transition risk", "how confident are you", "what is the source of this", "can you explain how the flood is calculated", "what does moderate mean here"],
    "publish_post": ["ok go for it, post", "yes publish", "fine, put it out", "that's good, post it to LinkedIn", "ship the post", "confirmed - publish"],
    "history": ["what have we got done so far", "what did you run earlier", "show everything you did today", "my past work", "recent outputs", "list what you produced last week"],
    "help": ["what is this", "how does it work", "what are you able to do", "give me examples of what to ask", "tell me what you can help with", "show me what is possible"],
    "research": ["what is the newest hazard hitting plants like ours", "is a La Nina year coming and what does it change", "tell me about the most recent storm that damaged factories",
                 "any new event I could write about", "any news about floods that matter to us", "what disasters are going on", "check recent hazard alerts", "look at what is happening in the world that touches our sites"],
}

T_NEW2 = {
    "report_transition": ["write up how a rising carbon price hurts {company}", "report on policy exposure of the {sector} client", "document the carbon price and regulation risk for {company}"],
    "assets": ["list the refineries {company} owns", "give me the {sector} plants owned by {company}", "which {sector} facilities does {company} run", "what are all the sites {company} has"],
    "report_financial": ["a report of the expected climate losses at {company}", "write down the expected annual loss for {company} in a report", "climate loss report for {company}'s lenders"],
    "explain": ["how did you get to that", "walk me through how this was worked out", "what exactly is a digital twin model", "what is a storm surge", "how is the water depth worked out", "what does a rating of high mean"],
    "voyage": ["what if the Suez route is closed for the {ship} going to {port}", "ship voyage and the Red Sea closing", "carbon cost of rerouting a {ship} around Africa to {port}"],
    "feasibility": ["tower of {n} floors in {city}, feasible?", "is a {n}-floor {use} workable at {place}", "would a {use} work on this ground, and what would it cost"],
    "history": ["what was the last thing you did", "which job did you finish most recently", "last run please"],
}

_OFFT = ["Peru", "Canada", "Kenya", "Norway", "Japan", "Chile", "Egypt", "Vietnam"]


def noisy(u: str, rng: random.Random) -> str:
    """A hurried version of an utterance: lower case, no end punctuation, sometimes a dropped or swapped letter."""
    s = u.lower().rstrip("?.! ")
    if rng.random() < 0.5 and len(s) > 8:
        i = rng.randrange(2, len(s) - 2)
        if s[i].isalpha():
            s = s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if rng.random() < 0.3:
        s = re.sub(r"\b(please|could you|can you|i would like to|i want to)\b\s*", "", s)
    return s.strip()
