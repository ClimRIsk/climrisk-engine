# Question batteries: shipping and building (29 Oct 2026)

Why: the agent had been written against a few example sentences. Real questions vary: ports with no "from", arrows and dashes, "13 kn", "between A and B", distances in km, ship names with "of the", a year after "in", a price with no unit, building questions by floor count, height, floor area or use. The batteries in `tests/battery_cases.py` are the training set for that: every phrasing that failed is added there, and `tests/test_question_battery.py` fails if it regresses. `tools/run_battery.py` prints every miss.

## Result
| Set | Before | After |
|---|---|---|
| Shipping, 30 questions + 6 that must not route | 10 of 36 fully right (28%) | 36 of 36 |
| Shipping held-out, 15 questions written after tuning | 12 of 15 (80%) on first run | 15 of 15 after fixing the 3 misses |
| Building, 25 questions + 6 negatives | not parseable beyond storey count (a 12-storey question parsed, a warehouse or a factory did not route at all) | 31 of 31 |
| Building held-out, 12 + 5 negatives | 11 of 12 on first run | 12 of 12 after one fix |

The honest generalisation figure is the first-run held-out score (about 80 to 90%), not the post-fix 100%. All phrasings were written by the same author as the parser, so real users will find more. The way to improve it is to paste each real failure into `battery_cases.py`.

## What changed
- `maritime/nlparse.py` (new): numbers and prices are lifted out first and replaced by a delimiter so a ship name or port cannot run through them; then the ship name; then the route (`from A to B`, `between A and B`, `A to B`, `A -> B`, `A - B`, `at A ... to B`). A name/port split that had to be guessed (no "from") is flagged in the reply. Speeds in kn/kt/kts/knots; distances in nm, nautical miles, km (converted) and miles (read as statute, flagged); size in dwt/TEU/GT/cbm with "k"; t/day with the speed it applies to; bunker, EUA and hire prices; delay days; what-if speed ("slow to 12 knots") kept apart from the base speed.
- Fixes found on the way: "LNG" as a fuel made the ship an LNG carrier; "chemical tanker" was classed as an oil tanker; "Port Hedland" lost its first word; "ETS cost in 2026" read 2026 as an EUA price.
- Ship questions with no route now ask for it (with the example), instead of being left to the general classifier.
- Follow-ups: "what if she slows to 10 knots", "and a 5 day delay", "bunker at $700/t", "same ship to Singapore" re-run the last voyage with that change (kept for an hour).
- `resident/feasibility.py`: building questions now parse storeys, height, floor area (ft2 converted) and use (residential, hotel, office, warehouse, factory, data centre, hospital, school, mall, car park, house). Routing no longer needs a storey count. Use-specific notes are general design practice and are labelled as such; the 3.2 m per floor used to turn storeys into height is a stated rule of thumb. Still no price: the engine holds no construction-cost, land or soil data.
- `resident/nlu.py` bug: "5,000 m2", "20,000 sq ft" and "30,000 sq ft" were read as latitude/longitude (5.0, 0.0), so the site was placed in the Gulf of Guinea. Two whole numbers with no hemisphere letter are no longer a coordinate.

## Not covered
- English only; regex parsing, so unusual word order can still fail.
- Port names are resolved online (OpenStreetMap), not tested offline here.
- Building: no cost figures, no embodied-carbon or operating-carbon estimate, no code-specific design loads.
- The learned intent classifier (`resident_nlu.json`) has no voyage or feasibility class; these two route by rules first. Retraining it was not needed for these families.
