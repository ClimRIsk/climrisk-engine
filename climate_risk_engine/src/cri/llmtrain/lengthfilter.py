"""
cri/llmtrain/lengthfilter.py — keep only training rows that fit the training window without losing any of the answer.

Why: `mlx_lm lora --mask-prompt --max-seq-length N` truncates each row to N tokens and computes loss only on the answer tokens. A row whose prompt alone fills the window
has zero answer tokens left, which makes the loss 0/0 = NaN and poisons every weight after it. A row that is merely truncated teaches the model to stop mid-sentence.
This script counts real tokens with the model's own tokenizer and drops (never rewrites) rows that do not fit, so what remains is unchanged, source-composed text.

    python lengthfilter.py DS_IN DS_OUT [--model mlx-community/Llama-3.2-3B-Instruct-4bit] [--max-seq 1536] [--margin 36] [--min-target 8]

Files handled: train/valid/test .jsonl and their _meta.jsonl (kept line-for-line aligned). Writes DS_OUT/length_report.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def count(tok, messages: list) -> tuple:
    """-> (prompt_tokens, total_tokens) for a chat row whose last message is the assistant answer."""
    full = tok.apply_chat_template(messages, tokenize=True)
    prompt = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full = full["input_ids"] if isinstance(full, dict) else full
    prompt = prompt["input_ids"] if isinstance(prompt, dict) else prompt
    return len(prompt), len(full)


def keep(tok, messages: list, max_seq: int, margin: int, min_target: int) -> tuple:
    """-> (keep?, reason, prompt_tokens, total_tokens). Needs the whole row inside the window (minus a safety margin) and a real answer."""
    if not messages or messages[-1].get("role") != "assistant":
        return False, "no_assistant_answer", 0, 0
    p, t = count(tok, messages)
    if t - p < min_target:
        return False, "answer_too_short", p, t
    if t > max_seq - margin:
        return False, "too_long", p, t
    return True, "ok", p, t


def filter_split(tok, src: Path, dst: Path, name: str, max_seq: int, margin: int, min_target: int) -> dict:
    rows = [json.loads(l) for l in open(src / f"{name}.jsonl") if l.strip()]
    meta_path = src / f"{name}_meta.jsonl"
    metas = [json.loads(l) for l in open(meta_path) if l.strip()] if meta_path.exists() else None
    if metas is not None and len(metas) != len(rows):
        raise ValueError(f"{name}: {len(rows)} rows but {len(metas)} meta rows; refusing to guess the alignment")
    out_rows, out_meta, why, kinds_dropped, longest = [], [], Counter(), Counter(), 0
    for i, r in enumerate(rows):
        ok, reason, p, t = keep(tok, r["messages"], max_seq, margin, min_target)
        why[reason] += 1
        longest = max(longest, t if ok else 0)
        if ok:
            out_rows.append(r)
            if metas is not None:
                out_meta.append(metas[i])
        elif metas is not None:
            kinds_dropped[str(metas[i].get("kind"))] += 1
    with open(dst / f"{name}.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if metas is not None:
        with open(dst / f"{name}_meta.jsonl", "w") as f:
            for m in out_meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return {"in": len(rows), "kept": len(out_rows), "dropped": dict(why), "dropped_by_kind": dict(kinds_dropped), "longest_kept_tokens": longest}


def run(src: str, dst: str, tok, max_seq: int = 1536, margin: int = 36, min_target: int = 8) -> dict:
    s, d = Path(src), Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    rep = {"max_seq": max_seq, "margin": margin, "min_target": min_target}
    for name in ("train", "valid", "test"):
        if (s / f"{name}.jsonl").exists():
            rep[name] = filter_split(tok, s, d, name, max_seq, margin, min_target)
    for extra in ("dataset_report.json",):
        if (s / extra).exists():
            (d / extra).write_text((s / extra).read_text())
    (d / "length_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    ap.add_argument("--max-seq", type=int, default=1536)
    ap.add_argument("--margin", type=int, default=36)
    ap.add_argument("--min-target", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoTokenizer          # in the MLX environment already (mlx_lm depends on it)
    tok = AutoTokenizer.from_pretrained(a.model)
    rep = run(a.src, a.dst, tok, a.max_seq, a.margin, a.min_target)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
