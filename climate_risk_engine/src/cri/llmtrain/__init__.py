"""cri.llmtrain — build a fine-tuning set for ClimRisk's local language model from open sources and the engine's own output.

  sources   the open-source list, each with its licence and whether it may be used for TRAINING or only for retrieval
  collect   fetch + clean + chunk -> corpus.jsonl (with url, licence, date)
  dataset   corpus + engine results + hand-written examples -> train/valid/test chat JSONL
  evaluate  score a model on the held-out set (citation validity, refusal, number faithfulness)

The model is taught HOW to answer (cite numbered sources, say what is missing, copy engine numbers exactly).
Facts and numbers still come from retrieved sources and engine results, never from the weights.
"""
