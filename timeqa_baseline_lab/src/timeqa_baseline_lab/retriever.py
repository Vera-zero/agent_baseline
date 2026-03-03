from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .chunking import Chunk

try:
    from FlagEmbedding import BGEM3FlagModel
    HAS_BGEM3 = True
except ImportError:
    HAS_BGEM3 = False


class ContrieverRetriever:
    def __init__(self, model_name: str = "facebook/contriever", device: str = "cpu"):
        self.device = self._resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.chunks: List[Chunk] = []
        self.embeddings: torch.Tensor | None = None

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @torch.no_grad()
    def _embed_texts(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=1)
            vecs.append(pooled.cpu())
        return torch.cat(vecs, dim=0)

    def build_or_load_index(self, chunks: List[Chunk], cache_dir: str) -> None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        chunks_path = cache / "chunks.jsonl"
        emb_path = cache / "chunk_embeddings.pt"

        if chunks_path.exists() and emb_path.exists():
            loaded_chunks = []
            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    loaded_chunks.append(Chunk(**json.loads(line)))
            self.chunks = loaded_chunks
            self.embeddings = torch.load(emb_path, map_location="cpu")
            return

        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self._embed_texts(texts)

        with chunks_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        torch.save(self.embeddings, emb_path)

    @torch.no_grad()
    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        return [chunk for chunk, _ in self.search_with_scores(query, top_k=top_k)]

    @torch.no_grad()
    def search_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        q = self._embed_texts([query])[0]
        sim = torch.mv(self.embeddings, q)
        k = min(top_k, sim.shape[0])
        values, indices = torch.topk(sim, k=k)
        idx_list = indices.cpu().numpy().tolist()
        val_list = values.cpu().numpy().tolist()
        return [(self.chunks[i], float(v)) for i, v in zip(idx_list, val_list)]


def simple_recall(pred: str, targets: List[str]) -> float:
    p = pred.strip().lower()
    if not p:
        return 0.0
    for t in targets:
        t_low = t.strip().lower()
        if t_low and (t_low in p or p in t_low):
            return 1.0
    return 0.0


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


class BGEM3Retriever:
    """Retriever using BGE-M3 model with FAISS indexing."""

    def __init__(self, model_path: str = "/workspace/models/bge-m3", device: str = "cpu", use_fp16: bool = False):
        if not HAS_BGEM3:
            raise ImportError("FlagEmbedding is required for BGEM3Retriever. Install with: pip install FlagEmbedding")

        self.device = self._resolve_device(device)
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16 and self.device != "cpu",
            device=self.device
        )

        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray | None = None

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _embed_texts(self, texts: List[str], batch_size: int = 12, max_length: int = 512) -> np.ndarray:
        """Embed texts using BGE-M3 model."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # BGE-M3 encode returns a dict with 'dense_vecs', 'lexical_weights', 'colbert_vecs'
            # We use dense_vecs for retrieval
            result = self.model.encode(batch, batch_size=len(batch), max_length=max_length)
            dense_vecs = result['dense_vecs']
            all_embeddings.append(dense_vecs)

        return np.vstack(all_embeddings)

    def build_or_load_index(self, chunks: List[Chunk], cache_dir: str) -> None:
        """Build or load embeddings index from cache."""
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        chunks_path = cache / "chunks_bgem3.jsonl"
        emb_path = cache / "chunk_embeddings_bgem3.npy"

        if chunks_path.exists() and emb_path.exists():
            loaded_chunks = []
            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    loaded_chunks.append(Chunk(**json.loads(line)))
            self.chunks = loaded_chunks
            self.embeddings = np.load(emb_path)
            return

        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self._embed_texts(texts)

        with chunks_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(emb_path, self.embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Search for top-k most similar chunks."""
        return [chunk for chunk, _ in self.search_with_scores(query, top_k=top_k)]

    def search_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for top-k most similar chunks with scores."""
        if self.embeddings is None or not self.chunks:
            return []

        # Encode query
        query_result = self.model.encode([query], batch_size=1, max_length=512)
        q_emb = query_result['dense_vecs'][0]

        # Normalize embeddings for cosine similarity
        q_norm = q_emb / np.linalg.norm(q_emb)
        emb_norm = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Compute similarities
        similarities = np.dot(emb_norm, q_norm)

        # Get top-k
        k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))

        return results
