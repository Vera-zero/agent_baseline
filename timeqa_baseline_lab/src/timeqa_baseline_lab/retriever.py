from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
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
        self.index: faiss.Index | None = None

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
        index_path = cache / "index.faiss"
        meta_path = cache / "index_meta.pkl"

        if chunks_path.exists() and index_path.exists() and meta_path.exists():
            loaded_chunks = []
            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    loaded_chunks.append(Chunk(**json.loads(line)))
            self.chunks = loaded_chunks

            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                _ = pickle.load(f)  # metadata if needed
            return

        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self._embed_texts(texts)

        # Convert to numpy and normalize for cosine similarity
        emb_np = embeddings.numpy().astype('float32')
        faiss.normalize_L2(emb_np)

        # Build FAISS index
        dimension = emb_np.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(emb_np)

        # Save chunks
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({}, f)  # Save empty metadata for compatibility

    @torch.no_grad()
    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        return [chunk for chunk, _ in self.search_with_scores(query, top_k=top_k)]

    @torch.no_grad()
    def search_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []

        # Embed and normalize query
        q = self._embed_texts([query])
        q_np = q.numpy().astype('float32')
        faiss.normalize_L2(q_np)

        # Search using FAISS
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_np, k)

        # Return results
        idx_list = indices[0].tolist()
        val_list = scores[0].tolist()
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
        self.index: faiss.Index | None = None

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
        index_path = cache / "index_bgem3.faiss"
        meta_path = cache / "index_bgem3_meta.pkl"

        if chunks_path.exists() and index_path.exists() and meta_path.exists():
            loaded_chunks = []
            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    loaded_chunks.append(Chunk(**json.loads(line)))
            self.chunks = loaded_chunks

            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                _ = pickle.load(f)  # metadata if needed
            return

        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self._embed_texts(texts)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Save chunks
        with chunks_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({}, f)  # Save empty metadata for compatibility

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        """Search for top-k most similar chunks."""
        return [chunk for chunk, _ in self.search_with_scores(query, top_k=top_k)]

    def search_with_scores(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for top-k most similar chunks with scores."""
        if self.index is None or not self.chunks:
            return []

        # Encode query
        query_result = self.model.encode([query], batch_size=1, max_length=512)
        q_emb = query_result['dense_vecs'].astype('float32')

        # Normalize
        faiss.normalize_L2(q_emb)

        # Search using FAISS
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, k)

        # Return results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append((self.chunks[idx], float(score)))

        return results
