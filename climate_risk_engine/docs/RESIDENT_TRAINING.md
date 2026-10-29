# Resident agent: training and measurement (29 Oct 2026)

The agent routes a message in two steps: rules first (voyage, building feasibility, recorded events, documents, autopilot, knowledge, stress, scenario, source, profile, lab, case, extension), then the learned intent model (`data/resident_nlu.json`; TF-IDF word and character n-grams, logistic regression in numpy, plus keyword cues). Below a confidence floor the agent asks instead of guessing.

## How it is measured
Two independent utterance sets, written separately from the training templates:
- Set A (`tests/nlu_battery.py`, 208 utterances, 24 labels). Failures on A guided which templates were added, so A is a tuning set.
- Set B (`tests/nlu_battery_b.py`, 98 utterances). Written before the final retrain; no template was written from it. B is the figure to quote.
`tests/test_resident_training.py` fails if either score drops, if an off-topic message is sent to a tool, if a climate message is called off-topic, or if a battery sentence is copied into the training corpus (a short list of stock phrases such as "help" or "hello" is exempt).

## Result
| | Baseline (21 classes, 756 examples) | After (24 classes, ~1,600 examples) |
|---|---|---|
| Set A, all labels | 81.7% (voyage 0/8, feasibility 0/8, 3 of 8 off-topic sent to a tool or to smalltalk) | 95.2% |
| Set B | not measured | 91.8% (90 of 98: 6 asked instead, 2 wrong) |
| Off-topic questions ("recommend a good laptop" had landed on *protect*) | 5 of 8 handled | 8 of 8 on A and 6 of 6 on B go to "outside what I do" |
| Corpus template cross-validation | 0.763 (21 easier classes) | 0.70 (24 classes, typing noise included); this figure is not comparable and is not the quality measure |

A wrong tool is worse than a question, so the floor was set where the number of wrong picks stopped falling: 0.30 (from 0.35). On the 306 utterances above, 0.30 gave the same wrong count as 0.35 and 3 more correct answers.

## What changed
- New classes: `voyage`, `feasibility` (the rules handle these first; the classifier catches a phrasing the rules miss and the agent asks for the route or the site and building), and `offtopic`.
- Off-topic: trained on 36 non-climate message patterns (trivia, recipes, translation, maths, shopping, coding). A message is called off-topic only at 50% or more; below that it is an ordinary "not sure" and nothing is guessed.
- About 110 new phrasings for the weakest intents (scenario, explain, research, physical and carbon reports, risk analysis, history, help, publish), and a hurried copy (lower case, no punctuation, a dropped or swapped letter) of every template.
- `tools/retrain_resident.py` retrains from the corpus plus logged user corrections and prints before and after on both sets and every miss.

## Known weak spots
Remaining misses: "ok send it" (heard as small talk), "how does this work" (help vs explain), "what is a digital twin" (heard as the twin tool; the agent's info-question check usually turns this into an explanation), report requests that give only the standard number ("14067 write-up please"), and a few scenario and protection phrasings with no stock keyword.
The test sets are English and written by one author. Real traffic will find more; log them, add the pattern (not the sentence) to `corpus_more.py`, and retrain.
