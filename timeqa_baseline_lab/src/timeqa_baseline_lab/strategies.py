from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .chunking import Chunk
from .llm import BaseGenerator
from .migrated import run_react
from .retriever import ContrieverRetriever


SYSTEM_PROMPT = "You are a precise QA assistant. Always provide a final answer clearly."


@dataclass
class StrategyOutput:
    answer: str
    retrieved: List[Chunk]
    trace: List[str]


def _format_context(chunks: List[Chunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[Chunk {i}] Title: {c.title}\n{c.text}")
    return "\n\n".join(parts)


def build_zero_shot_cot_prompt(question: str) -> str:
    return (
        f"Question: {question}\n"
        "Think step by step silently, then output only:\nFinal Answer: <answer>"
    )


def build_rag_cot_prompt(question: str, chunks: List[Chunk]) -> str:
    return (
        "Use the retrieved context to answer the question.\n"
        "If uncertain, give the most likely answer based on evidence.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{_format_context(chunks)}\n\n"
        "Think step by step, then output only:\nFinal Answer: <answer>"
    )


def zero_shot_cot(llm: BaseGenerator, question: str) -> StrategyOutput:
    prompt = build_zero_shot_cot_prompt(question)
    ans = llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
    return StrategyOutput(answer=ans, retrieved=[], trace=[])


def rag_cot(llm: BaseGenerator, retriever: ContrieverRetriever, question: str, top_k: int) -> StrategyOutput:
    ctx = retriever.search(question, top_k=top_k)
    prompt = build_rag_cot_prompt(question, ctx)
    ans = llm.generate(prompt, system_prompt=SYSTEM_PROMPT)
    return StrategyOutput(answer=ans, retrieved=ctx, trace=[])


def react(llm: BaseGenerator, retriever: ContrieverRetriever, question: str, top_k: int, max_steps: int = 6) -> StrategyOutput:
    answer, retrieved, trace = run_react(
        llm=llm,
        retriever=retriever,
        question=question,
        top_k=top_k,
        max_steps=max_steps,
    )
    return StrategyOutput(answer=answer, retrieved=retrieved, trace=trace)