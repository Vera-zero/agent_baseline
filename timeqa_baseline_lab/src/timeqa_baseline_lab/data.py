from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pyarrow.ipc as ipc


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    source_idx: str


@dataclass
class QAItem:
    idx: str
    question: str
    targets: List[str]


def load_corpus(path: str | Path) -> List[Document]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    docs = []
    for d in data["documents"]:
        docs.append(
            Document(
                doc_id=d["doc_id"],
                title=d["title"],
                content=d["content"],
                source_idx=d["source_idx"],
            )
        )
    return docs


def load_questions_from_arrow(path: str | Path, limit: int = 100) -> List[QAItem]:
    with Path(path).open("rb") as f:
        table = ipc.open_stream(f).read_all()

    raw = table.to_pydict()
    size = len(raw["idx"])
    if limit > 0:
        size = min(size, limit)

    items = []
    for i in range(size):
        items.append(
            QAItem(
                idx=raw["idx"][i],
                question=raw["question"][i],
                targets=list(raw["targets"][i]),
            )
        )
    return items


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
