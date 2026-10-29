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

## Real data added (29 Oct 2026)
Until now the intent model had never seen a sentence written by a person who was not me: every training example was a template filled with slot values. The only real-people data that exists for this task is general-purpose, so it is used for what it can teach: what is NOT a climate question.

**Source.** CLINC150 (Larson et al., "An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction", EMNLP 2019), Hugging Face `clinc/clinc_oos`, licence CC BY 3.0 (attribution required; kept in `src/cri/data/real_oos_clinc150.txt` and in the provenance registry). 350 out-of-scope queries from the train (250) and validation (100) splits, label `offtopic`. They cover banking, cars, trivia, sport, TV, health, shopping and similar, in everyday-assistant English from 2019. None is about climate, hazards, emissions or shipping, so there is still **no real climate-domain training data**, and no real message from a user of this product.

**Held-out check.** 185 further out-of-scope queries from the CLINC150 *test* split (`tests/data/clinc_oos_heldout.txt`, never used in training; `tools/measure_real_oos.py`). Rows that touch our own domain (earthquake maps, "what is global warming", local temperature trends, energy bills, power outages, a drought question) were left out by hand because the right answer is arguable. The agent does cover hazards and could reasonably answer some of them.

| On 185 real out-of-scope queries never seen in training | Before | After |
|---|---|---|
| Replied "outside what I do" | 12 (6.5%) | 174 (94.1%) |
| Asked what was meant | 149 | 8 |
| Sent to a climate tool (wrong) | 24 (13.0%) | 3 (1.6%) |

The 24 before were things like "check current prime mortgage rate" (research), "i have a rash, what can i use for it" (help), "give me my wallet location" (assets). My own off-topic set had missed all of them: it contained 8 tidy trivia questions, and the real ones are messier and more varied. The 3 remaining are "what do you do all day while i'm at work", "why did negan return to his jail cell" and "can you explain a good risk strategy to me".

**Cost to the climate side, and what was done about it.** With the real rows alone, in-domain accuracy fell (set A 95.2% to 91.4%, set B 91.8% to 87.8%) because everyday verbs ("show me", "check", "find", "look up") had become evidence for "not climate". Two fixes:
1. 62 new templates for the climate intents in the same everyday register (`T_NEW3` in `corpus_more.py`).
2. A domain-vocabulary veto in `nlu.py` (`DOMAIN_WORDS`): if a message contains flood, cyclone, emissions, carbon, asset, voyage, scope 1-3, CBAM, 14067 and similar words (plus a few "what can you do" phrases), it is never called off-topic; the best remaining intent has to clear the confidence floor or the agent asks. The veto only removes a refusal; it never picks an intent.

**Resulting figures (shipped model).**

| Set | Before real data | After | Caveat |
|---|---|---|---|
| A (208) | 95.2% | 96.2% (5 wrong) | tuning set |
| B (98) | 91.8% | 96.9% (2 wrong) | no longer clean: its misses shaped templates and the veto, so this gain is partly fitted |
| C (38, `tests/nlu_battery_c.py`) | 86.8% | 89.5% | written after the real data went in, but two of its misses later informed templates; 38 is small, so one sentence is 2.6 points |
| 185 real out-of-scope | 13.0% wrong | 1.6% wrong | the only set here that is not written by me |

The in-domain figures should be read as "unchanged within noise or slightly better", not as an improvement claim. The real-data gain is the off-topic result.

**What this does not show.** No real user has typed into this agent, so we still do not know how people will phrase shipping, building or reporting questions. The in-domain sets are all one author. CLINC150 is US English from 2019, so non-US English, other languages and code-switching are untested. The 350 rows are 17% of the 2,120 training rows and the model is still a small numpy classifier, not a language model; it cannot understand a question it has no pattern for, it can only decline or ask.

**To keep improving.** Log real messages (the agent already stores corrections as feedback rows; `tools/retrain_resident.py` reads them). Any real message the agent mishandles goes into a battery only after being recorded here, and its pattern into `corpus_more.py`.
