"""
cri/resident/corpus.py — the training corpus for the resident agent's intent model.

Each intent has many phrasing templates with slots ({place}, {company}, {sector}, {hazard}, {rp}); utterances are the
templates filled with slot values. Splitting is by TEMPLATE, never by utterance, so the held-out score measures
generalisation to phrasings the model has not seen, not memorisation of slot values.
"""
from __future__ import annotations

import random

SLOTS = {
    "place": ["22.2645, 91.805", "Chattogram", "23.20 90.65", "Futtsu", "Tokyo Bay", "Houston ship channel", "Rotterdam", "29.7, -95.0",
              "Jebel Ali", "Mumbai", "the Meghna plant", "Kiln 2", "site A", "my refinery", "our cement works", "Gulf Coast", "Manila", "Miami"],
    "company": ["JERA", "Holcim", "NTPC", "ExxonMobil", "Heidelberg Materials", "Tata Steel", "Adani Power", "Shell", "Cemex", "Vale", "Duke Energy", "Reliance"],
    "sector": ["cement", "steel", "power", "refinery", "chemicals", "mining", "aluminium", "data centre", "port", "real estate"],
    "hazard": ["flood", "cyclone", "wind", "storm surge", "sea level rise", "typhoon", "hurricane", "heat", "river flooding", "coastal flooding"],
    "rp": ["1-in-100", "100-year", "1 in 50", "250 year", "10-year", "1-in-500"],
    "year": ["2030", "2040", "2050"],
}

T = {
    "twin": [
        "build a 3D twin of {place}", "show me {place} in 3D", "open the asset twin for {place}", "model {place} in 3D and replay the {hazard}",
        "3D model of {place}", "visualise {hazard} at {place}", "show how a {rp} {hazard} hits {place}", "replay {hazard} on {place}",
        "can you construct a 3D digital twin of {place}", "let me see {place} flooding in 3D", "simulate the {hazard} at {place} and show it",
        "render the buildings at {place} and run the {hazard}", "twin {place}", "drop the 3D viewer on {place}", "walk me through what {hazard} does to {place}",
        "I want to watch the water rise at {place}", "show the {sector} plant at {place} in three dimensions", "animate a {hazard} over {place}",
        "put the site {place} on the 3D model", "3d view {place} {hazard}",
    ],
    "protect": [
        "how can I protect {place} from {hazard}", "what protection does {place} need against a {rp} {hazard}", "how do I harden this {sector} site against {hazard}",
        "recommend measures for {place}", "what should we build to stop the {hazard} at {place}", "which mitigation works best for {hazard}",
        "how high should the flood wall be at {place}", "should we raise the equipment at {place}", "give me protection options for the {sector} plant",
        "how do I make {place} resilient to {hazard}", "what fixes the {hazard} problem here", "suggest adaptation measures for {place}",
        "what would it take to keep {place} dry in a {rp} flood", "best way to defend the {sector} works from {hazard}", "solutions for {hazard} at {place}",
        "rank the protection measures for {place}", "does a barrier work at {place}", "what retrofit stops wind damage at {place}",
    ],
    "scenario": [
        "build scenarios for {place}", "what if sea level rises at {place}", "run a scenario with {hazard} at {rp} for {place}", "compare scenarios for {company}",
        "scenario analysis for {sector} under net zero 2050", "what happens to {place} by {year}", "stress test {place} against {hazard}",
        "create a scenario set for {place} and {company}", "what if the {hazard} is 20 percent worse at {place}", "run delayed transition against {company}",
        "give me best case and worst case for {place}", "show me different futures for {place} in {year}", "scenario lab for {place}",
        "what if water level is higher by one metre at {place}", "model a stronger {hazard} for {place}", "try a {rp} {hazard} plus sea level rise by {year}",
        "build me the scenarios and solutions for {place}", "explore scenarios for the {sector} portfolio",
    ],
    "report_physical": [
        "write the physical risk report for {place}", "physical risk assessment for {place}", "give me a physical risk report on {company}",
        "produce a report on {hazard} exposure at {place}", "physical report for the {sector} site", "generate the physical hazard report for {place}",
        "I need a story report on physical risk at {place}", "report on how {place} is exposed to {hazard}", "customer report physical risk {place}",
        "make a physical risk report with the 3D findings for {place}", "physical risk write up {place}", "send me a physical report for {company} assets",
    ],
    "report_transition": [
        "write the transition risk report for {company}", "transition risk report for the {sector} sector", "give me a transition risk assessment for {company}",
        "report on carbon price exposure for {company}", "produce a transition report under net zero 2050 for {company}", "transition report {company}",
        "how exposed is {company} to the energy transition, as a report", "stranded asset report for {company}", "policy risk report for {company}",
        "generate a transition risk story for {company}", "I need the transition risk write up for the {sector} client",
    ],
    "report_financial": [
        "write the financial risk report for {company}", "climate financial risk report for {company}", "financial impact report for {company} under NGFS",
        "produce the loss distribution report for {company}", "EBITDA at risk report for {company}", "financial risk report for the {sector} portfolio",
        "give me a financial risk story for {company}", "report on climate value at risk for {company}", "credit risk report for {company}",
        "make the financial climate report for the lender", "financial translation report {company}",
    ],
    "report_lca": [
        "write the life cycle assessment report for {sector}", "LCA report for our {sector} product", "carbon footprint report of a tonne of {sector}",
        "generate the LCA report for {company}", "dynamic LCA report for {sector}", "product footprint report {sector}", "life cycle report for the {sector} product",
        "report on cradle to gate emissions for {sector}", "I need an LCA report for the customer",
    ],
    "report_iso14067": [
        "produce the ISO 14067 report for {sector}", "ISO 14067 carbon footprint study report for {sector}", "write the CFP study report for {company}",
        "ISO 14067 report", "product carbon footprint study with cradle to grave for {sector}", "generate the ISO 14067 study document",
        "clause 9 report for the {sector} footprint", "ISO 14067 calculation report for {company}", "carbon footprint of a product study report",
    ],
    "report_carbon": [
        "write the carbon audit report for {company}", "carbon accounting report for {company}", "greenhouse gas inventory report for {company}",
        "scope 1 2 3 audit report", "produce the carbon audit with assurance pack for {company}", "GHG audit report {company}", "emissions inventory report for the {sector} client",
        "carbon audit story for {company}", "give me the carbon accounting write up",
    ],
    "report_cbam": [
        "write the CBAM report for {company}", "CBAM exposure report for {sector} exports", "carbon border adjustment report for {company}",
        "produce a CBAM report", "how much CBAM will {company} pay, as a report", "CBAM liability report for {sector}", "EU border carbon report {company}",
        "CBAM story for the {sector} exporter", "generate the CBAM certificate cost report for {company}",
    ],
    "risk_analysis": [
        "run the risk analyst for {company}", "full climate risk analysis of {company}", "assess {company}", "how risky is {company} under {hazard} and carbon prices",
        "run the loss distribution on {company}", "analyse {company} physical and transition together", "climate value at risk of {company}",
        "evaluate {company} under net zero 2050", "give me the numbers for {company}", "assess climate risk for {company}",
    ],
    "assets": [
        "where are {company}'s plants", "which assets does {company} own", "list the sites of {company}", "show me {company} facilities on the map",
        "what does {company} operate", "find the plants of {company}", "most exposed assets of {company}", "company assets for {company}",
        "which {company} sites are in cyclone zones", "map {company}",
    ],
    "research": [
        "what is happening this week", "scan for recent disasters near my assets", "any storms threatening the plants", "research the latest climate event",
        "what is the news on typhoons", "find a current event worth writing about", "check GDACS for new cyclones", "look for floods that hit industrial sites this week",
        "what climate events matter for {company}", "research what happened at {place}", "is there a new {hazard} event", "run the weekly research scan",
        "find a story for this week", "which live events touch our asset register", "El Nino status and what it means for assets", "research desk scan",
    ],
    "draft_post": [
        "write a LinkedIn post about this", "draft a LinkedIn post on the {hazard} event", "turn this into a LinkedIn post", "make a post about {place} for LinkedIn",
        "LinkedIn post about {company} exposure", "write the post", "draft the article and the LinkedIn post", "create a short post for LinkedIn from the findings",
        "write something for LinkedIn about the {hazard}", "linkedin draft", "prepare a post for our company page", "write a LinkedIn update on {place}",
    ],
    "publish_post": [
        "post it on LinkedIn", "publish the post", "yes post it", "go ahead and publish to LinkedIn", "approve and post", "send that post live",
        "publish it now", "post this to our LinkedIn page", "looks good, post it", "approved, publish", "share the post",
    ],
    "explain": [
        "why is {place} beyond design wind", "explain this result", "what does that mean", "why did the flood reach the equipment", "how was this calculated",
        "explain the {rp} {hazard} at {place}", "why is the rating high", "what is the basis of the depth", "how does the solver work", "what are the assumptions",
        "can you explain the {hazard} finding", "why does {company} score so high", "what does design wind mean", "how reliable is this number",
        "how accurate is the flood model", "tell me the limits of this result",
    ],
    "remember": [
        "remember that tank 3 at {place} is 18 metres tall", "the design wind of the silo is 60 m/s", "note that the floor of the MCC room is 1.2 m above ground",
        "remember {place} was commissioned in 1994", "our plant {place} flooded in 2019 by 0.6 m", "keep in mind the substation equipment sits 2 m above floor",
        "save this: {company} is our client", "remember that the {sector} plant has a flood wall of 2 m", "store the fact that {place} is a {sector} site",
        "for future reference the kiln is 90 m high", "log an incident at {place} in 2021", "remember my client is {company}",
    ],
    "recall": [
        "what do you remember about {place}", "what did I tell you about the plant", "show what you know about {company}", "recall my notes on {place}",
        "what facts do you have for {place}", "list what you remember", "what have you learned about my sites", "what is in your memory",
    ],
    "history": [
        "what did you do so far", "show my history", "what runs have we done", "list previous reports", "what did we work on yesterday", "show the log of actions",
        "which posts have you published", "what is the status of the research runs", "show recent activity",
    ],
    "help": [
        "what can you do", "help", "how do I use you", "what are your capabilities", "show me the options", "what can I ask", "how does this agent work",
        "what tools do you have", "guide me", "what do you know how to do",
    ],
    "smalltalk": [
        "hello", "hi there", "thanks", "thank you very much", "good morning", "great work", "ok", "cool", "nice", "bye", "that is helpful", "perfect thanks",
    ],
}


T_MORE = {
    "twin": ["I want to see {place} as a 3D model", "bring up the three dimensional view for {place}", "load {place} into the viewer", "create the digital twin",
             "can I see the buildings and water at {place}", "show me how deep the water gets at {place}", "visual replay of the {hazard} at {place}", "open 3D for {place}"],
    "protect": ["what can we do about {hazard} at {place}", "how do we stop the water at {place}", "which measure is the most effective for {hazard}", "design a flood defence for {place}",
                "how should we defend the {sector} plant", "what would you recommend to reduce damage from {hazard}", "do we need a wall around {place}", "advise on protecting equipment at {place}"],
    "scenario": ["what if the {hazard} gets worse", "what could happen at {place} in the future", "compare a mild and a severe case for {place}", "run what if cases for {company}",
                 "show alternative futures for the {sector} sector", "test different return periods at {place}", "try a hotter and wetter world at {place}", "vary the scenario and rank the outcomes for {place}"],
    "assets": ["show me the plants owned by {company}", "what facilities does {company} have", "where does {company} operate", "which sites belong to {company}",
               "locate the {company} assets", "pull up the asset list for {company}", "how many plants does {company} have", "give me {company} locations"],
    "research": ["what is going on in the world right now", "any new events I should know about", "scan the news for climate hazards", "what has just hit industrial assets",
                 "check live hazards", "is anything happening at the moment", "do the research for this week", "look at recent cyclones and floods"],
    "explain": ["why", "what does this mean", "how come the equipment is flooded", "tell me why this is happening", "how do you know this",
                "what is behind this number", "what does that rating mean", "is this reliable"],
    "recall": ["what do you know about {place}", "do you remember {place}", "what notes do we have on {company}", "tell me what you stored", "what have I told you before"],
    "history": ["what have you done recently", "show previous work", "recent runs", "what did you publish before", "list my past posts and reports"],
    "help": ["what do you do", "what can I ask you", "how can you help", "list your features", "I do not know what to ask", "what is this agent for"],
    "smalltalk": ["hey", "hello agent", "thanks a lot", "awesome", "good evening", "that worked", "great thanks", "goodbye"],
    "remember": ["note this down: {place} has a 3 m flood wall", "please keep in mind that {company} is a client", "the tank at {place} is 20 metres high", "add to your memory that the plant was built in 1988"],
    "report_cbam": ["CBAM report please", "calculate the border tax liability as a report for {company}", "how much carbon border tax as a document for {sector}"],
    "report_iso14067": ["give me the ISO 14067 document", "carbon footprint study as a report", "make the CFP report"],
    "risk_analysis": ["how exposed is {company}", "what is the risk to {company}", "run the numbers on {company}", "quantify climate risk for {company}"],
    "publish_post": ["post it", "publish", "yes publish", "confirm and post to LinkedIn"],
    "draft_post": ["write a post", "turn the research into a post", "draft a post about the event", "LinkedIn copy for this finding"],
}
for _k, _v in T_MORE.items():
    T[_k].extend(_v)

from . import corpus_more as _CM                                   # noqa: E402  (added 29 Oct 2026: voyage, feasibility, offtopic, extra phrasings, typing noise)
SLOTS.update(_CM.SLOTS_MORE)
SLOTS["offt"] = _CM._OFFT
for _k, _v in list(_CM.T_NEW.items()) + list(_CM.T_NEW2.items()) + list(_CM.T_NEW3.items()):
    T.setdefault(_k, []).extend(_v)


def utterances(seed: int = 7, per_template: int = 3, noise: bool = True) -> list:
    """[(text, intent, template_index)] — deterministic."""
    rng = random.Random(seed)
    out = []
    for intent, temps in T.items():
        for ti, t in enumerate(temps):
            names = [n for n in SLOTS if "{" + n + "}" in t]
            n_variants = per_template if names else 1
            seen = set()
            for _ in range(n_variants * 3):
                vals = {n: rng.choice(SLOTS[n]) for n in names}
                u = t.format(**vals)
                if u not in seen:
                    seen.add(u); out.append((u, intent, ti))
                if len(seen) >= n_variants:
                    break
            if noise:                                                  # a hurried copy of the first variant, same template group (so the split stays leak-free)
                first = next(iter(seen), None)
                if first:
                    out.append((_CM.noisy(first, rng), intent, ti))
    return out


def split_by_template(rows: list, test_every: int = 5) -> tuple:
    """Hold out every `test_every`-th template of each intent (whole templates, so test phrasings are unseen)."""
    tr, te = [], []
    for u, i, ti in rows:
        (te if ti % test_every == test_every - 1 else tr).append((u, i))
    return tr, te


def real_rows() -> list:
    """[(text, "offtopic")] — REAL out-of-scope queries written by people (CLINC150 train + validation out-of-scope rows, CC BY 3.0,
    Larson et al., EMNLP 2019; retrieved 29 Oct 2026; see data/provenance.py entry real_oos_clinc150). Never includes the test rows used by
    tools/measure_real_oos.py. Returns [] if the file is absent so the engine still imports."""
    from pathlib import Path
    f = Path(__file__).resolve().parent.parent / "data" / "real_oos_clinc150.txt"
    if not f.exists():
        return []
    return [(l.strip(), "offtopic") for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
