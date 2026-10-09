import json

from cri.llmtrain import lengthfilter as LF


class Tok:                                           # 1 token per 3 characters, plus a fixed generation prompt
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        n = sum(len(m["content"]) for m in messages) // 3 + 5 * len(messages)
        return list(range(n + (3 if add_generation_prompt else 0)))


def row(prompt_chars, answer_chars):
    return {"messages": [{"role": "system", "content": "s" * 30}, {"role": "user", "content": "u" * prompt_chars}, {"role": "assistant", "content": "a" * answer_chars}]}


def test_drops_rows_that_would_lose_their_answer_and_keeps_alignment(tmp_path):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    rows = [row(300, 300), row(4000, 100), row(10000, 600), row(300, 2), row(1200, 1200)]    # ok, too long once answer added? 4000/3=1333+..., very long, no answer, ok
    kinds = ["case", "chain", "case", "qa", "narr"]
    (src / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (src / "train_meta.jsonl").write_text("".join(json.dumps({"kind": k, **r}) + "\n" for k, r in zip(kinds, rows)))
    rep = LF.run(str(src), str(dst), Tok(), max_seq=1536, margin=36, min_target=8)
    t = rep["train"]
    kept = [json.loads(l) for l in open(dst / "train.jsonl")]
    meta = [json.loads(l) for l in open(dst / "train_meta.jsonl")]
    assert t["in"] == 5 and t["kept"] == len(kept) == len(meta)
    assert t["dropped"]["too_long"] >= 1 and t["dropped"]["answer_too_short"] == 1
    assert [m["kind"] for m in meta] == [k for k, r in zip(kinds, rows) if r in kept]            # same rows, same order
    assert all(r["messages"][-1]["content"].startswith("a") for r in kept)                       # rows are copied unchanged
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
