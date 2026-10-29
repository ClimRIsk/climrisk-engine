"""Set C: written on 29 Oct 2026 AFTER the real out-of-scope data was added and the final retrain was done; no template was written from it and none may be.
Same single author as sets A and B, so it is a fresh check, not an independent one. In-domain assistant-register phrasings (the register the real
out-of-scope questions use), so it tests whether real everyday data made the model refuse real work."""
BATTERY = {
    "twin": ["pull up our Futtsu plant in three dimensions", "can you show Kiln 2 as a 3D model with a storm hitting it", "open the viewer for Jebel Ali"],
    "protect": ["what is the cheapest way to stop the substation flooding", "advise me on defences for the Mumbai warehouse against surge", "what can we change at the plant to survive a typhoon"],
    "scenario": ["show me best and worst case for Rotterdam in 2040", "what happens to our Miami site if the sea is 1 m higher", "give me a range of outcomes for the Gulf Coast refinery"],
    "report_physical": ["draw up a hazard report for the Houston site", "write a document about flood exposure at Chattogram port"],
    "report_transition": ["put together a report on carbon price risk for Vale", "document how net zero rules would affect our cement client"],
    "report_financial": ["write a report on the losses climate could cause Duke Energy", "financial climate risk write up for the lender on Reliance"],
    "report_carbon": ["produce an emissions inventory report covering scopes 1 to 3 for Cemex", "annual GHG report for the auditors"],
    "report_cbam": ["write the border carbon tax report for our steel exports to the EU"],
    "risk_analysis": ["how much climate risk is Shell carrying", "give me the numbers on climate exposure for Tata Steel"],
    "assets": ["show me every plant NTPC runs", "where are Holcim's cement works"],
    "research": ["look up what extreme weather is going on this week", "check if any hurricane is heading for the Gulf"],
    "help": ["what sort of things can I ask here", "tell me what you are for"],
    "smalltalk": ["hi there", "thank you very much", "see you later"],
    "voyage": ["a bulk carrier sails Newcastle to Qingdao, what is the carbon cost", "how much CO2 for a container ship from Shanghai to Rotterdam via the Cape"],
    "feasibility": ["is it a good idea to put up a 12 storey hotel in Miami", "could I build apartments on this coastal plot near Manila"],
    "unknown": ["what is the best pasta shape for carbonara", "find me a cheap flight to Lisbon", "show me the football table", "look up how to tie a tie", "who sings this song"],
}
