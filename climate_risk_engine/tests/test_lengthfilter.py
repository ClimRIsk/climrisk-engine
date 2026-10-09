import json

from cri.llmtrain import lengthfilter as LF


class Enc(dict):                                     # like transformers' BatchEncoding: len() counts keys, not tokens
    pass


class Tok:                                           # 1 token per 3 characters, plus a fixed generation prompt
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert tokenize is False
        return "".join(m["content"] + "<|e|>" for m in messages) + ("<|a|>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=True):
        n = len(text) // 3
        return Enc(input_ids=list(range(n)), attention_mask=[1] * n)


def row(prompt_chars, answer_chars):
    return {"messages": [{"role": "system", "content": "s" * 30}, {"role": "user", "content": "u" * prompt_chars}, {"role": "assistant", "content": "a" * answer_chars}]}


def test_drops_rows_that_would_lose_their_answer_and_keeps_alignment(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    rows = [row(300, 300), row(4400, 100), row(300, 310), row(10000, 600), row(300, 2), row(1200, 1200), row(500, 200)]    # ok, too long, ok, far too long, no answer, ok, ok
    kinds = ["case", "chain", "case", "qa", "qa", "narr", "gold"]
    keep_idx = [0, 2, 5, 6]
    (src / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (src / "train_meta.jsonl").write_text("".join(json.dumps({"kind": k, "i": i}) + "\n" for i, k in enumerate(kinds)))
    rep = LF.run(str(src), str(dst), Tok(), max_seq=1536, margin=36, min_target=8)
    t = rep["train"]
    kept = [json.loads(l) for l in open(dst / "train.jsonl")]
    meta = [json.loads(l) for l in open(dst / "train_meta.jsonl")]
    assert t["in"] == 7 and t["kept"] == len(kept) == len(meta) == 4
    assert t["dropped"]["too_long"] == 2 and t["dropped"]["answer_too_short"] == 1
    assert [m["i"] for m in meta] == keep_idx                                                    # same rows, same order, meta aligned
    assert kept == [rows[i] for i in keep_idx]                                                   # rows are copied unchanged
    assert t["longest_kept_tokens"] <= 1536 - 36


def test_misaligned_meta_is_refused(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (src / "train.jsonl").write_text(json.dumps(row(10, 50)) + "\n")
    (src / "train_meta.jsonl").write_text("")
    try:
        LF.run(str(src), str(tmp_path / "o"), Tok())
        assert False
    except ValueError as e:
        assert "alignment" in str(e)


def test_wrong_token_counting_cannot_silently_empty_the_dataset(tmp_path):
    class Bad(Tok):
        def encode(self, text, add_special_tokens=True):
            return []                                  # every row would look like it has no answer
    src = tmp_path / "in"
    src.mkdir()
    (src / "train.jsonl").write_text("".join(json.dumps(row(300, 300)) + "\n" for _ in range(4)))
    try:
        LF.run(str(src), str(tmp_path / "o"), Bad())
        assert False
    except RuntimeError as e:
        assert "refusing" in str(e)
