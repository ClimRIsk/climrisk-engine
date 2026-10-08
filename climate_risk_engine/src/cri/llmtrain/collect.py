"""Fetch the open sources and write corpus.jsonl.  python -m cri.llmtrain.collect OUT_DIR"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..resident import knowledge as K
from . import sources as S

MAX_PASSAGES = 80


def _fetch_one(row):
    url, title, licence, topic, train = row
    try:
        raw = K._get(url, timeout=30)
        t, text = K._strip_html(raw) if "<" in raw[:2000] else ("", raw)
        ps = K._chunk(text)[:MAX_PASSAGES]
        if not ps:
            return {"url": url, "title": title, "ok": False, "why": "no readable text"}
        return {"url": url, "title": title, "licence": licence, "topic": topic, "train": train, "ok": True, "passages": ps, "fetched": time.strftime("%Y-%m-%d")}
    except Exception as e:                                           # noqa: BLE001
        return {"url": url, "title": title, "ok": False, "why": f"{type(e).__name__}: {str(e)[:100]}"}


def _wiki(title):
    try:
        for (t, url, text) in K._wiki_extracts([title]):
            text = K._CUT.sub("", "\n" + text)
            ps = K._chunk(text)[:40]
            if ps:
                return {"url": url, "title": t + " (Wikipedia)", "licence": "CC BY-SA 4.0 (retrieval only)", "topic": "wiki", "train": False, "ok": True, "passages": ps, "fetched": time.strftime("%Y-%m-%d")}
    except Exception as e:                                           # noqa: BLE001
        return {"url": title, "title": title, "ok": False, "why": f"{type(e).__name__}"}
    return {"url": title, "title": title, "ok": False, "why": "missing"}


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(6) as ex:
        a = list(ex.map(_fetch_one, S.SOURCES))
        w = list(ex.map(_wiki, list(dict.fromkeys(S.WIKI_EXTRA + K.WIKI_TITLES))))
    rows = a + w
    with open(out / "corpus.jsonl", "w") as f:
        for r in rows:
            if r["ok"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"sources_ok": sum(r["ok"] for r in rows), "sources_failed": [{"url": r["url"], "why": r["why"]} for r in rows if not r["ok"]],
           "passages_train": sum(len(r["passages"]) for r in rows if r["ok"] and r["train"]),
           "passages_retrieval_only": sum(len(r["passages"]) for r in rows if r["ok"] and not r["train"]),
           "by_topic": {}}
    for r in rows:
        if r["ok"] and r["train"]:
            rep["by_topic"][r["topic"]] = rep["by_topic"].get(r["topic"], 0) + len(r["passages"])
    (out / "collect_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:3000])


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "llm"))
