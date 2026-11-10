# Resident agent stress audit (10 Nov 2026)

Probes: `tools/stress_agent.py` (63 adversarial messages through `Agent.handle`, network blocked, tools stubbed) and `tools/stress_nlu.py` (calibration, typo and wrapper robustness on batteries A+B+C, 344 utterances; corpus balance; hashing; vocabulary). Run with `PYTHONPATH=src:tests`. All probe sentences were written by one author on the day; none is in the training corpus. Nothing below has been fixed yet.

## Defects (wrong behaviour, ranked)

| # | Finding | Evidence |
|---|---|---|
| 1 | **Wrong site, silently.** When coordinates in the message cannot be parsed or are invalid, the agent uses the last site and proceeds. | "twin at 91.5, 200.0", "twin at 12, 77", "twin at 12°58'N 77°35'E", "twin at lat 12.97 lon 77.59" all opened the previous site; "risk at 0, 0" started a physical report for Sydney |
| 2 | **Conversation state is global, not per session.** `memory.get_state("last_place")` has no session key. | A new session inherited 19.07, 72.87 from another session. On the hosted API this crosses users |
| 3 | **Swapped lon,lat accepted.** "77.59, 12.97" is taken as latitude 77.59 with no plausibility check or echo-back question. | twin opened at 77.59 N |
| 4 | **Negation is not understood.** | "do not publish the post" -> publish_post (0.99); "don't build a twin, just tell me what a twin is" -> twin |
| 5 | **One intent per message.** The second request is dropped without saying so. | "build a twin ... and then write the CBAM report" -> CBAM only; "list plants and run physical risk on each" -> asks for a site |
| 6 | **Engine capabilities the agent cannot reach or refuses.** | "sfdr pai numbers for my fund" -> "outside what I do" (the engine has `regulatory/sfdr_pai.py`); "csrd e1 gap check" -> hazard news scan; "pcaf financed emissions" and "1 in 200 flood cost at rotterdam" -> voyage; insurance premium -> knowledge search |
| 7 | **Non-English is refused as off-topic.** | German and Chinese flood questions -> "outside what I do"; Hindi and Tamil code-switch -> wrong tool or asks |
| 8 | **Domain-word veto lets off-topic through.** | "storm in a teacup", "hurricane cocktail", "401k asset allocation" -> knowledge search job; "factory reset on my phone" -> history |
| 9 | **Degenerate input starts work.** | "flood" x3000 and a NUL-prefixed "flood" started a protect job on the last site |
| 10 | Follow-ups without context words fail. | "same for mumbai" -> asks; "and for 2050?" -> knowledge search, not the last scenario |

Held: publish never happened without a draft and approval (injection strings included); no crash on 20 kB, emoji, SQL or script input; India grid factor answered from CEA with source; Milton cost asked for the wind instead of inventing a number.

## The intent model (measured)

| Measure | Result |
|---|---|
| Clean batteries, in-domain (n=325) | 96.0% correct, 1.8% wrong tool |
| 1 / 2 / 3 random typos per message | 87.1% / 77.8% / 68.0% correct (rest mostly asks; wrong tool stays about 1-2%) |
| "please can you " in front | 82.8% |
| "hey quick one, " in front | 89.8%, wrong tool 4.0% |
| Calibration | 0.8-1.0: 100% (n=255); 0.6-0.8: 98%; 0.4-0.6: 81%; 0.2-0.4: 40% |
| Held-out templates, model only (stored in the model file) | 60.1%; grouped CV 71.6% |
| Feature hashing | 21,841 distinct features in 16,384 buckets; 16,064 share a bucket |
| Thinnest classes | report_lca 34 rows from 9 templates; report_cbam 44 from 12 |
| Engine terms absent from the training vocabulary | sfdr, pcaf, esrs, csrd, tcfd, issb, taxonomy, insurance, premium, wildfire, drought, hail, tornado, landslide, earthquake, biodiversity, pd, lgd, merton, sbti, abatement, methane, cii, corsia, registry, cea, egrid, ssp, rcp, utci, wbgt and others |

Structural gaps: bag of n-grams (no word order, so no negation or scope); single label (no multi-intent); 24 classes against 163 API routes; slots are regexes outside the model; no language detection; conversation state not modelled. The 96% is on sets by the same author as the corpus; the 60% held-out-template figure and the typo and wrapper drops are closer to what a stranger will see.

## Stress tests not yet run

- Real users: zero real in-domain messages logged.
- Multi-turn dialogue battery (context carry, correction, "no, the other site").
- Answer correctness end to end: agent reply numbers against the engine tool output for the same inputs.
- Local language model path: red-team of `converse.py` and `answer.py` (number check exists; no adversarial test set).
- Concurrency on the agent (3-worker pool, global state), job cancellation, long-run memory.
- Engine repo has no CI workflow; 931 tests run only by hand, never on Windows or macOS in automation.
- Security: no penetration test, no auth/rate-limit test on `/resident`, no prompt-injection test through uploaded documents (`userdocs.py`) or learned web pages (`knowledge.py`).
- Hazard validation still open in `GAP_REGISTER.md` (one event per peril, no loss validation, JRC table transcription unchecked).
