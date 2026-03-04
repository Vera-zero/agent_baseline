from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

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
    level: Optional[str] = None
    time_relation: Optional[str] = None
    doc_id: Optional[str] = None


def load_corpus(path: str | Path) -> List[Document]:
    """
    Legacy function: Load corpus from old format (documents array).

    Expected format:
    {
        "documents": [
            {"doc_id": str, "title": str, "content": str, "source_idx": str},
            ...
        ]
    }
    """
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
    """Legacy function: Load questions from Arrow format."""
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
                level=raw.get("level", [None] * size)[i] if "level" in raw else None,
                time_relation=raw.get("time_relation", [None] * size)[i] if "time_relation" in raw else None,
                doc_id=raw.get("doc_id", [None] * size)[i] if "doc_id" in raw else None,
            )
        )
    return items


def _detect_dataset_type(data: dict) -> str:
    """
    Detect dataset type from JSON structure.

    Returns:
        'tempreason' | 'timeqa' | 'unknown'
    """
    if "contents" in data and isinstance(data["contents"], list):
        # Check if it's tempreason format
        if data["contents"] and "fact_context" in data["contents"][0]:
            return "tempreason"

    if "datas" in data and isinstance(data["datas"], list):
        # Check if it's timeqa format
        if data["datas"] and "context" in data["datas"][0]:
            return "timeqa"

    return "unknown"


def load_unified_dataset(
    path: str | Path,
    limit: int = 0,
    dataset_type: Optional[str] = None
) -> Tuple[List[Document], List[QAItem]]:
    """
    Load dataset from unified test_ffinal.json format.

    Supports both tempreason and timeqa formats:
    - tempreason: {"contents": [...], "questions_num": N}
    - timeqa: {"datas": [...], "content_num": N}

    Args:
        path: Path to test_ffinal.json or similar unified format file
        limit: Maximum number of questions to load (0 = all)
        dataset_type: Force dataset type ('tempreason' or 'timeqa'), None = auto-detect

    Returns:
        (documents, qa_items): Tuple of document list and QA item list
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Auto-detect or use specified dataset type
    if dataset_type is None:
        dataset_type = _detect_dataset_type(data)
    else:
        dataset_type = dataset_type.lower()

    if dataset_type == "tempreason":
        return _load_tempreason_format(data, limit)
    elif dataset_type == "timeqa":
        return _load_timeqa_format(data, limit)
    else:
        raise ValueError(
            f"Unknown dataset format in {path}. "
            f"Expected 'contents' (tempreason) or 'datas' (timeqa) field."
        )


def _load_tempreason_format(data: dict, limit: int) -> Tuple[List[Document], List[QAItem]]:
    """
    Load TEMPREASON format data.

    Format:
    {
        "contents": [
            {
                "fact_context": "...",
                "question_list": [
                    {
                        "question": "...",
                        "text_answers": {"text": [...]},
                        "id": "...",
                        ...
                    }
                ]
            }
        ]
    }
    """
    documents = []
    qa_items = []
    global_qa_idx = 0

    for doc_id, item in enumerate(data.get("contents", [])):
        # Extract document
        doc = Document(
            doc_id=str(doc_id),
            title=f"TEMPREASON_DOC_{doc_id}",
            content=item.get("fact_context", ""),
            source_idx=str(doc_id),
        )
        documents.append(doc)

        # Extract questions
        for q_data in item.get("question_list", []):
            if limit > 0 and global_qa_idx >= limit:
                break

            # Extract answers from text_answers.text field
            text_answers = q_data.get("text_answers", {})
            targets = text_answers.get("text", []) if isinstance(text_answers, dict) else []

            qa = QAItem(
                idx=q_data.get("id", f"Q{global_qa_idx}"),
                question=q_data.get("question", ""),
                targets=targets if isinstance(targets, list) else [targets],
                level=q_data.get("level"),
                time_relation=q_data.get("time_relation"),
                doc_id=str(doc_id),
            )
            qa_items.append(qa)
            global_qa_idx += 1

        if limit > 0 and global_qa_idx >= limit:
            break

    return documents, qa_items


def _load_timeqa_format(data: dict, limit: int) -> Tuple[List[Document], List[QAItem]]:
    """
    Load TIMEQA format data.

    Format:
    {
        "datas": [
            {
                "idx": "...",
                "context": "...",
                "questions_list": [
                    {
                        "question": "...",
                        "targets": [...],
                        ...
                    }
                ]
            }
        ]
    }
    """
    documents = []
    qa_items = []
    global_qa_idx = 0

    for doc_id, item in enumerate(data.get("datas", [])):
        # Extract document
        doc = Document(
            doc_id=str(doc_id),
            title=item.get("idx", f"TIMEQA_DOC_{doc_id}"),
            content=item.get("context", ""),
            source_idx=str(doc_id),
        )
        documents.append(doc)

        # Extract questions
        for q_data in item.get("questions_list", []):
            if limit > 0 and global_qa_idx >= limit:
                break

            targets = q_data.get("targets", [])

            qa = QAItem(
                idx=f"Q{global_qa_idx}",
                question=q_data.get("question", ""),
                targets=targets if isinstance(targets, list) else [targets],
                level=q_data.get("level"),
                time_relation=q_data.get("time_relation"),
                doc_id=str(doc_id),
            )
            qa_items.append(qa)
            global_qa_idx += 1

        if limit > 0 and global_qa_idx >= limit:
            break

    return documents, qa_items


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
