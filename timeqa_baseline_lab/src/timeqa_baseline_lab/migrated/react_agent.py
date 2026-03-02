from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

from ..chunking import Chunk
from ..llm import BaseGenerator
from ..retriever import ContrieverRetriever


SYSTEM_PROMPT = "You are a careful reasoning agent that follows Action format exactly."


def _asset_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / name


def _load_react_examples() -> str:
    path = _asset_path("react_prompts_naive.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "webthink_simple6" in data:
            return str(data["webthink_simple6"])
    return ""


def _first_sentences(text: str, n: int = 5) -> str:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sents: List[str] = []
    for p in paragraphs:
        parts = [s.strip() for s in p.split(". ") if s.strip()]
        sents.extend(parts)
    if not sents:
        return text[:400]
    head = sents[:n]
    return ". ".join(x.rstrip(".") for x in head) + "."


def _extract_action(raw: str, step_id: int) -> Tuple[str, str, str]:
    text = raw.strip()
    m = re.search(rf"Thought\s*{step_id}\s*:\s*(.*?)\nAction\s*{step_id}\s*:\s*(.*)", text, re.I | re.S)
    if m:
        thought = m.group(1).strip()
        action_line = m.group(2).strip()
    else:
        action_line = text
        thought = text.split("\n")[0].strip()

    m2 = re.search(r"(Search|Lookup|Finish)\[(.*?)\]", action_line, re.I | re.S)
    if not m2:
        return thought, "Search", ""
    return thought, m2.group(1).capitalize(), m2.group(2).strip()


def _format_evidence(chunks: List[Chunk], top_k: int) -> str:
    lines = []
    for i, c in enumerate(chunks[:top_k], start=1):
        lines.append(f"[Chunk {i}] {c.title}\n{c.text}")
    return "\n\n".join(lines)


def run_react(
    llm: BaseGenerator,
    retriever: ContrieverRetriever,
    question: str,
    top_k: int,
    max_steps: int = 6,
) -> Tuple[str, List[Chunk], List[str]]:
    """ReAct pipeline migrated from ReAct-master (Thought/Action/Observation loop)."""
    instruction = (
        "Solve a question answering task with interleaving Thought, Action, Observation steps. "
        "Action can be: Search[entity], Lookup[keyword], Finish[answer]."
    )
    fewshot = _load_react_examples()
    prompt = instruction + "\n\n" + fewshot + "\nQuestion: " + question + "\n"

    trace: List[str] = []
    collected: List[Chunk] = []
    current_page = ""
    lookup_keyword = ""
    lookup_list: List[str] = []
    lookup_idx = 0
    answer = ""

    for i in range(1, max_steps + 1):
        thought_action = llm.generate(prompt + f"Thought {i}:", system_prompt=SYSTEM_PROMPT)
        thought, action_type, action_arg = _extract_action(thought_action, i)

        if action_type == "Search":
            query = action_arg if action_arg else question
            found = retriever.search(query, top_k=top_k)
            collected.extend(found)
            if found:
                current_page = found[0].text
                obs = _first_sentences(current_page)
            else:
                obs = f"Could not find {query}."
            lookup_keyword, lookup_list, lookup_idx = "", [], 0

        elif action_type == "Lookup":
            kw = action_arg
            if not current_page:
                obs = "No page loaded. Use Search first."
            else:
                if kw != lookup_keyword:
                    lookup_keyword = kw
                    sentences = [s.strip() for s in current_page.split(". ") if s.strip()]
                    lookup_list = [s for s in sentences if kw.lower() in s.lower()]
                    lookup_idx = 0

                if lookup_idx >= len(lookup_list):
                    obs = "No more results."
                else:
                    obs = f"(Result {lookup_idx + 1} / {len(lookup_list)}) {lookup_list[lookup_idx]}"
                    lookup_idx += 1

        elif action_type == "Finish":
            answer = action_arg
            obs = "Episode finished."
        else:
            obs = "Invalid action."

        step_str = (
            f"Thought {i}: {thought}\n"
            f"Action {i}: {action_type}[{action_arg}]\n"
            f"Observation {i}: {obs}\n"
        )
        trace.append(step_str)
        prompt += step_str

        if action_type == "Finish" and answer:
            break

    dedup = {}
    for c in collected:
        dedup[c.chunk_id] = c
    collected = list(dedup.values())

    if not answer:
        final_prompt = (
            f"Question: {question}\n"
            f"Evidence:\n{_format_evidence(collected, top_k)}\n\n"
            "Output only: Final Answer: <answer>"
        )
        answer = llm.generate(final_prompt, system_prompt=SYSTEM_PROMPT)

    return answer, collected[:top_k], trace
