# RAG recce, 19 Oct 2026 (no code changed; experiments in `tools/rag_recce_experiments.py`)

Baseline (case library, 4,672 passages, 132 events): paraphrase p1 hit@1 0.76 / p2 0.68, p2 hit@6 0.88, routing to the case library p1 0.96 / p2 0.80, false routes 0/25, entity questions 1.0 (template-made, optimistic).

## Findings
1. **Curated cards exist for only 25 of 132 events** (107 events have filing sentences and XBRL figures only). Routing uses similarity to curated cards, so questions about the other 107 events are routed by name matching alone.
2. **The final ranking throws away a good signal.** Ranking events purely by dense similarity (max passage per event) gets p1 hit@1 0.96 versus 0.76 for the current blend. It loses on p2 (0.44 versus 0.68). Plain rank fusion of the two does not beat the current blend on both sets (best p1 0.92 with p2 down to 0.52). One fixed weight cannot serve both question styles.
3. **p1 and p2 are different question types.** p1 asks for a specific case in different words; p2 asks abstract mechanism questions ("why can damages from fires started by electrical equipment wipe out a grid operator's equity"). Many p2 questions legitimately match several events (PG&E, Maui, Kincade for utility wildfire liability; Thailand, Chennai, Rio Grande do Sul for factory flooding), so one-gold-event scoring under-states retrieval. The gold lists are not adjudicated.
4. **Blend weight is flat:** SEM_W 0.35 to 0.9 moves p2 hit@1 only 0.68 to 0.72.
5. **The embedding model is small** (bge-small, 384 dimensions). Not yet tried: a larger embedder or a cross-encoder reranker on the top 20.
6. **The knowledge index (`knowledge.py`) is BM25 only**; the dense layer covers the case library and user documents.
7. **Generation is not benchmarked.** The honesty check (every number in a source, every [n] exists) is tested, but answer faithfulness and completeness are not measured.

## Proposed order
1. Adjudicate gold labels (multi-gold) and write a fresh held-out set p3 before changing anything; report all three splits.
2. Event-level descriptors for the 107 uncovered events built only from their own source text; use them for routing and event ranking.
3. Try a reranker and a larger embedder; keep only what wins on p3 without hurting p1/p2 or false routes.
4. Add dense retrieval to the knowledge index.
5. Add a small answer-quality set (faithfulness and coverage) for the generator.
