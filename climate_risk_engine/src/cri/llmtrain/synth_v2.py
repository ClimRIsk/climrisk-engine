"""v2 behaviours, built from original text (no licence exposure):
  chain    risk-chain cited Q&A: numbered blocks (scenario / policy / hazard / asset) -> a cited causal chain, distractor blocks ignored
  refuse   boundary refusals: the context has hazard or policy text but not what the question asks for (valuation, PD/LGD, volumes, categories)
  narr     disclosure-style narration of an engine result string, copying every token exactly
Every answer is composed from the same fields that built the blocks, so it is correct by construction."""
from __future__ import annotations

import random

from .dataset import QA_SYS
from .facts import fm

CHAIN_SYS = QA_SYS
NARR_SYS = ("You write formal climate-risk disclosure language for a risk manager. Using ONLY the figures given, write 60 to 110 words of plain prose. "
            "Copy every number, name, scenario and year exactly as written; never add a number that is not given. State that scenarios are not forecasts. No lists, no headings.")

SECTORS = ["steel", "cement", "aluminium", "fertiliser", "refinery", "chemicals", "power"]
PLACES = ["Chattogram", "Rotterdam", "Houston", "Mumbai", "Gdansk", "Durban", "Kaohsiung", "Santos", "Jubail", "Port Klang"]
SCEN = {
    "Delayed Transition": ("climate policy stays weak until about 2030 and then tightens abruptly", "carbon prices jump sharply once policy tightens, with little time for firms to adapt", "a sudden, sharp regulatory shock"),
    "Current Policies": ("only policies already in force are assumed and no new ones are added", "carbon prices stay low while physical damage keeps rising", "slow transition pressure and growing physical damage"),
    "Net Zero 2050": ("policy tightens early and smoothly so that net zero is reached around mid-century", "carbon prices rise steadily from the start, giving firms time to invest", "an early, orderly cost signal"),
    "Divergent Net Zero": ("net zero is reached but policy is uneven across sectors and countries", "carbon prices are higher in some sectors than others, which raises the total cost of the transition", "uneven and costlier transition pressure"),
}


def _b(i, text):
    return f"[{i}] {text}"


def _blocks(items, rng):
    """items: [(text, role)] in any order -> shuffled numbered block, role -> position"""
    idx = list(range(len(items)))
    rng.shuffle(idx)
    pos, lines = {}, []
    for n, k in enumerate(idx, 1):
        text, role = items[k]
        pos[role] = n
        lines.append(_b(n, text))
    return "\n\n".join(lines), pos


def _asset(rng):
    sec, place = rng.choice(SECTORS), rng.choice(PLACES)
    capex = rng.random() < 0.5
    exp = rng.choice([20, 35, 50, 65, 80])
    water = rng.choice(["a river", "an estuary", "groundwater wells", "seawater once-through cooling"])
    cap = ("has no capital budget allocated to emissions reduction in its current plan" if capex else "has set aside capital for emissions reduction in its current plan")
    txt = f"The {sec} plant at {place} {cap}. About {exp}% of its output is exported to the European Union. Cooling water is drawn from {water}."
    return {"sec": sec, "place": place, "capex": capex, "exp": exp, "water": water, "text": txt}


def _distractors(rng, k):
    pool = ["The Atlantic hurricane season runs from 1 June to 30 November, and an average season has 14 named storms.",
            "Greenhouse gases trap heat in the atmosphere, and carbon dioxide is the largest contributor to human-caused warming.",
            "A tide gauge measures the height of the sea surface relative to a fixed point on land.",
            "Drought is a prolonged period of abnormally low rainfall that reduces water for people and industry.",
            "The Saffir-Simpson scale rates a hurricane from 1 to 5 using only its maximum sustained wind speed.",
            "Scope 3 emissions are all indirect emissions along a company's value chain other than purchased energy."]
    return [(t, "x") for t in rng.sample(pool, k)]


def chain(rng):
    kind = rng.choice(["scenario", "cbam", "compound", "vuln"])
    a = _asset(rng)
    ds = _distractors(rng, rng.randint(1, 2))
    if kind == "scenario":
        name = rng.choice(list(SCEN))
        timing, mech, shock = SCEN[name]
        items = [(f"Under the NGFS {name} scenario, {timing}. {mech[0].upper() + mech[1:]}.", "s"), (a["text"], "a")] + ds
        blk, pos = _blocks(items, rng)
        q = rng.choice([f"Under {name}, what policy factor creates the largest cash-flow pressure for the {a['sec']} plant?", f"How does the {name} scenario affect the {a['sec']} plant at {a['place']}?"])
        if a["capex"]:
            tail = f"Source [{pos['a']}] says the plant has no capital allocated to emissions reduction, so it has little ability to avoid the cost, and the pressure would fall mostly on operating margin. That last step is an inference from the sources, not a figure."
        else:
            tail = f"Source [{pos['a']}] says the plant has capital set aside for emissions reduction, which would soften the impact, although the sources give no amount. That last step is an inference, not a figure."
        ans = f"According to [{pos['s']}], under {name} {timing}, and {mech}. This is {shock}. {tail}"
        need = [pos["s"], pos["a"]]
    elif kind == "cbam":
        pol = "The EU Carbon Border Adjustment Mechanism phases in between 2026 and 2034 as free allocation under the EU Emissions Trading System is phased out, so the share of embedded emissions that importers must pay for rises each year until 2034."
        items = [(pol, "p"), (a["text"], "a")] + ds
        blk, pos = _blocks(items, rng)
        q = rng.choice([f"How does the CBAM phase-in change the cost exposure of the {a['sec']} plant at {a['place']}?", f"Why does the {a['sec']} plant's EU export cost rise over time?"])
        ans = (f"According to [{pos['p']}], the mechanism phases in between 2026 and 2034 as free allocation is phased out, so the share of embedded emissions that must be paid for rises each year. "
               f"Source [{pos['a']}] says about {a['exp']}% of the plant's output is exported to the European Union, so that share of output carries a growing cost over the period. The sources give no carbon price or tonnage, so the size of the cost cannot be stated.")
        need = [pos["p"], pos["a"]]
    elif kind == "compound":
        stress = rng.choice(["high", "extremely high"])
        hz = f"The site has {stress} baseline water stress, and the number of days above 35 degrees Celsius is projected to rise by mid-century."
        asset = f"The {a['sec']} plant at {a['place']} draws cooling water from {a['water']} and relies on it for continuous operation."
        items = [(hz, "h"), (asset, "a")] + ds
        blk, pos = _blocks(items, rng)
        q = f"What combined water and heat risk should be checked for the {a['sec']} plant?"
        ans = (f"Source [{pos['h']}] reports {stress} baseline water stress together with more very hot days. Source [{pos['a']}] says the plant relies on cooling water from {a['water']}. "
               f"Taken together, the sources point to a cooling-water constraint: hotter water and air raise cooling demand just as water is scarcer, which could limit output or raise water and utility costs. "
               f"The sources give no volumes or prices, so the size of the effect is not stated.")
        need = [pos["h"], pos["a"]]
    else:
        depth = rng.choice([0.8, 1.0, 1.5, 2.0])
        hz = f"The modelled 1-in-100 year flood depth at the site is {depth} m."
        typ = rng.choice(["terminal", "realestate"])
        asset = ("The site is a maritime container terminal whose revenue depends on days of continuous berth operation." if typ == "terminal" else "The site is a commercial property portfolio valued on the replacement cost of its buildings.")
        items = [(hz, "h"), (asset, "a")] + ds
        blk, pos = _blocks(items, rng)
        q = f"What does a 1-in-100 year flood mean for this site?"
        eff = ("operational downtime, because floodwater stops berth and yard operations and revenue is lost for the days the site is closed" if typ == "terminal" else "structural damage and a write-down of building value, because the loss is the cost of repairing or replacing flooded buildings")
        ans = (f"Source [{pos['h']}] gives a 1-in-100 year flood depth of {depth} m, a 1% chance of being exceeded in any year. Source [{pos['a']}] describes the site type, and for this type the main consequence is {eff}. "
               f"The same depth would cause a different kind of loss at a different type of asset. The sources give no damage ratio or downtime days, so no loss figure is stated.")
        need = [pos["h"], pos["a"]]
    return {"kind": "chain", "meta": {"need": need, "n": blk.count("\n\n") + 1},
            "messages": [{"role": "system", "content": CHAIN_SYS}, {"role": "user", "content": f"Question: {q}\n\nSources:\n{blk}"}, {"role": "assistant", "content": ans}]}


REF = [
    ("What is the exact projected change in the asset's enterprise value under this pathway?", "any enterprise value, balance sheet or valuation for the asset", "the asset's valuation and financial statements"),
    ("What is the probability of default for the borrower under this scenario?", "any probability of default, credit rating or borrower financials", "the borrower's financials and a credit model"),
    ("What is the loss given default on the loan?", "any loan terms, collateral value or recovery data", "the loan terms and collateral valuation"),
    ("How many days of downtime will the site have per year?", "any downtime estimate or operating schedule", "an operating schedule and a depth-damage or disruption model"),
    ("What is the annual CBAM cost for this plant?", "export volumes, embedded emissions or a carbon price", "export tonnage, emission intensity and the certificate price"),
    ("Does the company's scope 3 disclosure cover all fifteen categories?", "any company disclosure or scope 3 inventory", "the company's published scope 3 inventory"),
    ("Is the transition plan aligned with the EU taxonomy?", "the plan's content, targets or capital expenditure", "the plan itself and its capital expenditure breakdown"),
    ("Which GCM overestimates the tail at this site?", "any comparison of climate models at this location", "model-by-model output for the site"),
    ("How much will insurance premiums rise at this location?", "any insurance terms, claims or premium data", "the policy terms and a loss model"),
]
CTX = [
    "The site has elevated wildfire exposure and a modelled 1-in-100 year flood depth of {d} m.",
    "Under the NGFS Delayed Transition scenario, carbon prices rise sharply after 2030.",
    "The EU Carbon Border Adjustment Mechanism phases in between 2026 and 2034.",
    "Baseline water stress at the site is high and heat days are projected to increase.",
    "Climate model projections under SSP2-4.5 diverge more after 2050 than before it.",
]


def refuse(rng):
    q, missing, need = rng.choice(REF)
    ctx = rng.sample(CTX, rng.randint(2, 3))
    ctx = [c.format(d=rng.choice([0.8, 1.2, 1.5])) for c in ctx]
    blk = "\n\n".join(_b(i + 1, c) for i, c in enumerate(ctx))
    have = "; ".join(f"{c[0].lower() + c[1:].rstrip('.')} [{i + 1}]" for i, c in enumerate(ctx[:2]))
    ans = rng.choice([
        f"The sources provided do not cover {missing}. They only state that {have}. That is not enough to answer without guessing. To answer I would need {need}.",
        f"I cannot answer that from these sources: they do not contain {missing}. What they do provide is that {have}. I would need {need}."])
    return {"kind": "refuse", "meta": {"n": len(ctx)},
            "messages": [{"role": "system", "content": CHAIN_SYS}, {"role": "user", "content": f"Question: {q}\n\nSources:\n{blk}"}, {"role": "assistant", "content": ans}]}


def narr(rng):
    asset = rng.choice(["Refinery", "Cement plant", "Steel works", "Container terminal", "Power station", "Chemical works", "Aluminium smelter"])
    scen = rng.choice(["NGFS Current Policies", "NGFS Net Zero 2050", "NGFS Delayed Transition", "NGFS Divergent Net Zero", "SSP2-4.5", "SSP5-8.5"])
    year = rng.choice([2030, 2040, 2050, 2070])
    el = round(rng.uniform(1.5, 32), 1)
    car = rng.uniform(5e6, 8e8)
    extra = []
    if rng.random() < 0.5:
        extra.append(f"AAL: {fm(car * rng.uniform(0.005, 0.03))}")
    if rng.random() < 0.4:
        extra.append(f"EBITDA at risk: {rng.uniform(2, 24):.1f}%")
    if rng.random() < 0.3:
        extra.append(f"Covenant headroom: {rng.uniform(5, 60):.1f}%")
    if rng.random() < 0.15:
        extra.append(f"PD: {rng.uniform(0.5, 6):.1f}%, LGD: {rng.randint(25, 60)}%")
    fields = [f"Asset: {asset}", f"Scenario: {scen}", f"Year: {year}", f"Expected Loss: {el}%", f"Capital-at-Risk: {fm(car)}"] + extra
    facts = "[" + ", ".join(fields) + "]"
    s = [f"Under the {scen} scenario, the expected loss for the {asset.lower()} reaches {el}% by {year}.",
         f"This corresponds to a Capital-at-Risk of {fm(car)}."]
    for e in extra:
        k, v = e.split(": ", 1)
        s.append({"AAL": f"The average annual loss is {v}.", "EBITDA at risk": f"EBITDA at risk is {v}.", "Covenant headroom": f"Covenant headroom is {v}."}.get(k, f"The figures provided also state {e}, taken as given and not computed here."))
    s.append("These are scenario results, not forecasts, and they depend on the pathway assumptions and the data behind them.")
    return {"kind": "narr", "meta": {"facts": facts},
            "messages": [{"role": "system", "content": NARR_SYS}, {"role": "user", "content": f"Facts: {facts}\n\nWrite a risk disclosure paragraph."}, {"role": "assistant", "content": " ".join(s)}]}


def build(n_chain=220, n_ref=160, n_narr=220, seed=0):
    rng = random.Random(seed)
    return [chain(rng) for _ in range(n_chain)] + [refuse(rng) for _ in range(n_ref)] + [narr(rng) for _ in range(n_narr)]
