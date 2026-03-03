"""
BGE-M3 based retrieval script for MRAG
Uses BGE-M3 model with FAISS indexing for efficient retrieval
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss

sys.path.append('./')
from utils import eval_recall, save_json_file, load_json_file
from contriever.src.evaluation import SimpleTokenizer, has_answer

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    raise ImportError("FlagEmbedding is required. Install with: pip install FlagEmbedding")


class BGEM3Indexer:
    """FAISS-based indexer for BGE-M3 embeddings."""

    def __init__(
        self,
        model_path: str = "/workspace/models/bge-m3",
        device: str = "cpu",
        use_fp16: bool = False
    ):
        """Initialize BGE-M3 model and FAISS index."""
        print(f"Loading BGE-M3 model from {model_path}...")
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16 and device != "cpu",
            device=device
        )
        self.index = None
        self.doc_ids = []
        self.doc_metadata = []

    def build_index(self, documents: List[Dict[str, Any]], batch_size: int = 12, max_length: int = 512):
        """Build FAISS index from documents."""
        print(f"Building index for {len(documents)} documents...")

        # Extract texts and metadata
        texts = []
        for doc in documents:
            # Combine title and text for embedding
            text = f"{doc['title']} {doc['text']}"
            texts.append(text)
            self.doc_ids.append(doc['id'])
            self.doc_metadata.append(doc)

        # Encode documents in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"Encoding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}")
            result = self.model.encode(batch, batch_size=len(batch), max_length=max_length)
            dense_vecs = result['dense_vecs']
            all_embeddings.append(dense_vecs)

        embeddings = np.vstack(all_embeddings).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Index built with {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 1000, max_length: int = 512) -> List[Dict[str, Any]]:
        """Search for top-k documents using BGE-M3."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        # Encode query
        result = self.model.encode([query], batch_size=1, max_length=max_length)
        query_emb = result['dense_vecs'].astype('float32')

        # Normalize
        faiss.normalize_L2(query_emb)

        # Search
        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))

        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.doc_metadata[idx].copy()
            doc['score'] = float(score)
            results.append(doc)

        return results

    def save_index(self, output_dir: str):
        """Save FAISS index and metadata."""
        os.makedirs(output_dir, exist_ok=True)

        # Save FAISS index
        index_path = os.path.join(output_dir, 'bgem3_index.faiss')
        faiss.write_index(self.index, index_path)

        # Save metadata
        meta_path = os.path.join(output_dir, 'bgem3_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'doc_ids': self.doc_ids,
                'doc_metadata': self.doc_metadata
            }, f, ensure_ascii=False)

        print(f"Index saved to {output_dir}")

    def load_index(self, index_dir: str):
        """Load FAISS index and metadata."""
        # Load FAISS index
        index_path = os.path.join(index_dir, 'bgem3_index.faiss')
        self.index = faiss.read_index(index_path)

        # Load metadata
        meta_path = os.path.join(index_dir, 'bgem3_metadata.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.doc_ids = data['doc_ids']
            self.doc_metadata = data['doc_metadata']

        print(f"Index loaded from {index_dir}")


def main():
    # Configuration
    data_path = './TempRAGEval/TempRAGEval.json'
    corpus_path = './enwiki-dec2021/psgs_w100.json'  # Path to Wikipedia corpus
    output_path = './TempRAGEval/BGEM3_output/TempRAGEval.json'
    index_dir = './bgem3_index'

    top_k = 1000
    num_examples = None
    rebuild_index = True
    device = "cuda" if os.path.exists('/dev/nvidia0') else "cpu"
    use_fp16 = device == "cuda"

    # Initialize
    tokenizer = SimpleTokenizer()
    indexer = BGEM3Indexer(device=device, use_fp16=use_fp16)

    # Load or build index
    if os.path.exists(index_dir) and not rebuild_index:
        print("Loading existing index...")
        indexer.load_index(index_dir)
    else:
        print("Building new index...")
        # Load corpus
        if not os.path.exists(corpus_path):
            print(f"Error: Corpus not found at {corpus_path}")
            print("Please specify the correct path to the Wikipedia corpus")
            return

        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)

        # Build index
        indexer.build_index(corpus)
        indexer.save_index(index_dir)

    # Load questions
    print(f"Loading questions from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    if num_examples:
        examples = examples[:num_examples]

    # Retrieve for each question
    print(f"\nRetrieving for {len(examples)} questions...")
    for k, ex in enumerate(examples):
        print(f'--{k + 1}/{len(examples)}--')

        question = ex['question']
        answers = ex['answers']

        # Search
        results = indexer.search(question, top_k=top_k)

        # Add hasanswer flag
        ctxs = []
        for doc in results:
            text = doc['title'] + ' ' + doc['text']
            h = has_answer(answers, text, tokenizer)
            ctx = {
                'id': doc['id'],
                'title': doc['title'],
                'text': doc['text'],
                'score': doc['score'],
                'hasanswer': h
            }
            ctxs.append(ctx)

        ex['ctxs'] = ctxs

    # Evaluate
    print('\nBGE-M3 Retrieval Performance')
    print('Answer Recall:')
    eval_recall(examples, ctxs_key='ctxs', ans_key='answers')
    print('\nEvidence Recall:')
    eval_recall(examples, ctxs_key='ctxs', ans_key='gold_evidences')

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_json_file(output_path, examples)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
