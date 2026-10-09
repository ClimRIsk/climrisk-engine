"""One definition of 'events the model never trains on', shared by every training-row builder (curated cards, SEC filings, XBRL financials, world data).

Held-out means the WHOLE event: no curated fact, filing quote, financial sentence or macro figure about it appears in a training or validation row,
not even as a distractor. The app still knows these events: the retrieval index holds them, and the model's job is to read the sources it is given,
which is exactly what the held-out test measures.
"""
# curated-card ids and SEC-collector event keys are the same strings where an event is in both
EVENTS = [
    "ian-2022", "uri-2021", "maui-2023", "thailand-2011", "jebi-2018", "idalia-2023",           # US and Asia
    "basf-rhine-2018", "rhine-2018", "panama-2023", "tsmc-2021", "valencia-2024", "rs-brazil-2024", "kariba-2024",   # Europe, Americas, Asia, Africa
    "henan-2021", "kzn-2022", "emilia-2023",                                                    # world-data pass
]
