from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence


def _to_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_answer(text: object) -> str:
    """Normalize answer text (aligned with QAaP/ReAct style)."""
    s = _to_text(text)

    # QAaP uses unidecode; we mirror it with stdlib Unicode normalization.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.lower().replace("-", " ")
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 used in QA projects like QAaP/ReAct."""
    pred = normalize_answer(prediction)
    gold = normalize_answer(ground_truth)

    if pred in {"yes", "no", "noanswer"} and pred != gold:
        return 0.0
    if gold in {"yes", "no", "noanswer"} and pred != gold:
        return 0.0

    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def substring_recall(prediction: str, targets: Sequence[str]) -> float:
    p = normalize_answer(prediction)
    if not p:
        return 0.0
    for t in targets:
        g = normalize_answer(t)
        if g and (g in p or p in g):
            return 1.0
    return 0.0


def em_f1(prediction: str, targets: Sequence[str]) -> Dict[str, float]:
    """Compute EM/F1 against multi-reference targets (take max over refs)."""
    if not targets:
        return {"em": 0.0, "f1": 0.0}

    pred_norm = normalize_answer(prediction)
    em = 0.0
    f1 = 0.0

    for t in targets:
        tgt_norm = normalize_answer(t)
        em = max(em, float(pred_norm == tgt_norm))
        f1 = max(f1, token_f1(pred_norm, tgt_norm))
        if em == 1.0:
            f1 = 1.0
            break

    return {"em": em, "f1": f1}


def mean(values: Iterable[float]) -> float:
    vals: List[float] = [float(v) for v in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
