from __future__ import annotations

from dataclasses import dataclass
from typing import List

from transformers import AutoTokenizer

from .data import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source_idx: str
    text: str
    start_token: int
    end_token: int


class TokenChunker:
    def __init__(self, tokenizer_name: str, chunk_size: int, chunk_overlap: int, min_chunk_size: int):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: Document) -> List[Chunk]:
        token_ids = self.tokenizer.encode(doc.content, add_special_tokens=False)
        if len(token_ids) <= self.chunk_size:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            return [
                Chunk(
                    chunk_id=f"{doc.doc_id}-chunk-0000",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_idx=doc.source_idx,
                    text=text,
                    start_token=0,
                    end_token=len(token_ids),
                )
            ]

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        start = 0
        idx = 0
        while start < len(token_ids):
            end = min(start + self.chunk_size, len(token_ids))
            piece = token_ids[start:end]
            if not piece:
                break
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-chunk-{idx:04d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source_idx=doc.source_idx,
                    text=self.tokenizer.decode(piece, skip_special_tokens=True),
                    start_token=start,
                    end_token=end,
                )
            )
            idx += 1
            if end >= len(token_ids):
                break
            start += step

        if len(chunks) >= 2:
            last_len = chunks[-1].end_token - chunks[-1].start_token
            if last_len < self.min_chunk_size:
                prev = chunks[-2]
                merged_ids = token_ids[prev.start_token:chunks[-1].end_token]
                chunks[-2] = Chunk(
                    chunk_id=prev.chunk_id,
                    doc_id=prev.doc_id,
                    title=prev.title,
                    source_idx=prev.source_idx,
                    text=self.tokenizer.decode(merged_ids, skip_special_tokens=True),
                    start_token=prev.start_token,
                    end_token=chunks[-1].end_token,
                )
                chunks.pop()

        return chunks

    def chunk_corpus(self, docs: List[Document]) -> List[Chunk]:
        out: List[Chunk] = []
        for d in docs:
            out.extend(self.chunk_document(d))
        return out
