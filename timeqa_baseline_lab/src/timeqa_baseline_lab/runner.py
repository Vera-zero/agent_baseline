from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any

from .chunking import TokenChunker
from .config import ExperimentConfig
from .data import QAItem, append_jsonl, iter_jsonl, load_corpus, load_questions_from_arrow, load_unified_dataset
from .evaluation import em_f1, mean
from .llm import build_generator
from .retriever import BGEM3Retriever, ContrieverRetriever
from .strategies import rag_cot, react, zero_shot_cot


def build_retriever(cfg: ExperimentConfig, docs):
    """Build and initialize retriever."""
    # Create chunker and chunk documents
    # Use local contriever tokenizer
    chunker = TokenChunker(
        tokenizer_name="/workspace/models/contriever",  # Use local contriever tokenizer
        chunk_size=cfg.chunk.chunk_size,
        chunk_overlap=cfg.chunk.chunk_overlap,
        min_chunk_size=cfg.chunk.min_chunk_size,
    )
    chunks = chunker.chunk_corpus(docs)

    retriever_type = cfg.retriever.type.lower()
    cache_dir = Path(cfg.io.cache_dir)

    if retriever_type == "bgem3":
        retriever = BGEM3Retriever(
            model_path=cfg.retriever.model_path,
            use_fp16=cfg.retriever.use_fp16,
            device=cfg.retriever.device,
        )
    elif retriever_type == "contriever":
        retriever = ContrieverRetriever(
            model_name=cfg.retriever.model_name,
            device=cfg.retriever.device,
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    retriever.build_or_load_index(chunks, cache_dir=cache_dir)
    return retriever


def load_data(cfg: ExperimentConfig):
    """Load corpus and questions based on config."""
    # Try unified format first
    if cfg.data.unified_data_path:
        docs, questions = load_unified_dataset(
            cfg.data.unified_data_path,
            limit=cfg.run.max_questions,
            dataset_type=cfg.data.dataset_type or None,
        )
        return docs, questions

    # Fall back to legacy format
    if cfg.data.corpus_path and cfg.data.question_arrow_path:
        docs = load_corpus(cfg.data.corpus_path)
        questions = load_questions_from_arrow(
            cfg.data.question_arrow_path,
            limit=cfg.run.max_questions if cfg.run.max_questions > 0 else 100000,
        )
        return docs, questions

    raise ValueError(
        "No data configured. Set either 'unified_data_path' or "
        "'corpus_path' + 'question_arrow_path' in config."
    )


def run_single_question(cfg: ExperimentConfig, llm, retriever, qa_item: QAItem) -> Dict[str, Any]:
    """Run a single question through the selected strategy."""
    strategy = cfg.run.strategy
    start_time = time.time()

    # Get strategy parameters
    strategy_params = cfg.run.strategy_params or {}

    if strategy == "zero_shot_cot":
        result = zero_shot_cot(llm, qa_item.question)
    elif strategy == "rag_cot":
        top_k = strategy_params.get("top_k", cfg.retriever.top_k)
        result = rag_cot(llm, retriever, qa_item.question, top_k=top_k)
    elif strategy == "react":
        top_k = strategy_params.get("top_k", cfg.retriever.top_k)
        max_steps = strategy_params.get("max_steps", 6)
        result = react(llm, retriever, qa_item.question, top_k=top_k, max_steps=max_steps)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    end_time = time.time()
    query_time = end_time - start_time

    # Build result record - preserve all QAItem fields
    record = {
        "idx": qa_item.idx,
        "question": qa_item.question,
        "targets": qa_item.targets,
        "output": result.answer,
        "query_time": query_time,
        "retrieved_count": len(result.retrieved),
        "trace_count": len(result.trace),
    }

    # Add optional fields if present
    if qa_item.level is not None:
        record["level"] = qa_item.level
    if qa_item.time_relation is not None:
        record["time_relation"] = qa_item.time_relation
    if qa_item.doc_id is not None:
        record["doc_id"] = qa_item.doc_id

    return record


def load_existing_results(result_path: Path) -> tuple[List[Dict], set]:
    """Load existing results and return processed question indices."""
    existing_results = []
    processed_indices = set()

    if result_path.exists():
        try:
            for record in iter_jsonl(result_path):
                existing_results.append(record)
                processed_indices.add(record["idx"])
            print(f"📂 Found existing results: {len(processed_indices)} questions completed")
        except Exception as e:
            print(f"⚠️  Failed to load existing results: {e}")
            existing_results = []
            processed_indices = set()

    return existing_results, processed_indices


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Run complete experiment: load data, run strategy, save results, compute metrics.

    This function follows the pattern from GraphRAG/all.py:
    - Loads data and initializes models
    - Runs strategy on each question
    - Saves results incrementally (every save_every questions)
    - Supports resume from checkpoint
    - Computes final evaluation metrics
    """
    print(f"=== Running Experiment ===")
    print(f"Strategy: {cfg.run.strategy}")
    print(f"Max questions: {cfg.run.max_questions if cfg.run.max_questions > 0 else 'ALL'}")

    # Load data
    print("\n📥 Loading data...")
    docs, questions = load_data(cfg)
    print(f"   Loaded {len(docs)} documents, {len(questions)} questions")

    # Initialize LLM
    print("\n🤖 Initializing LLM...")
    llm = build_generator(cfg.model)
    print(f"   Model: {cfg.model.provider} - {cfg.model.model_name or cfg.model.model}")

    # Initialize retriever (if needed)
    retriever = None
    if cfg.run.strategy in ["rag_cot", "react"]:
        print("\n🔍 Building retriever index...")
        retriever = build_retriever(cfg, docs)
        print(f"   Retriever: {cfg.retriever.type} with {len(retriever.chunks)} chunks")

    # Setup output path
    # Extract dataset name from data path
    dataset_name = "unknown"
    if cfg.data.unified_data_path:
        # Extract dataset name from path like "/workspace/ETE-Graph/dataset/timeqa/test_ffinal.json"
        dataset_path = Path(cfg.data.unified_data_path)
        dataset_name = dataset_path.parent.name  # Gets "timeqa" from the parent directory
    elif cfg.data.corpus_path:
        # For legacy format
        corpus_path = Path(cfg.data.corpus_path)
        dataset_name = corpus_path.parent.name

    # Build output path: /workspace/ETE-Graph/QAresult/{数据集名}/{方法名}
    output_base = Path("/workspace/ETE-Graph/QAresult")
    output_dir = output_base / dataset_name / cfg.run.strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use strategy name and timestamp in output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_filename = f"{cfg.run.strategy}_{timestamp}.jsonl"
    result_path = output_dir / result_filename

    # Load existing results (resume support)
    all_results = []
    processed_indices = set()

    if cfg.run.resume:
        # Try to find the most recent result file for this strategy
        existing_files = sorted(output_dir.glob(f"{cfg.run.strategy}_*.jsonl"), reverse=True)
        if existing_files:
            result_path = existing_files[0]  # Use most recent
            all_results, processed_indices = load_existing_results(result_path)

    # Process questions
    print(f"\n📝 Processing questions...")
    print(f"   Output: {result_path}")
    save_interval = cfg.run.save_every
    questions_since_last_save = 0

    for idx, qa_item in enumerate(questions):
        # Skip if already processed
        if cfg.run.resume and qa_item.idx in processed_indices:
            continue

        print(f"\n[{idx+1}/{len(questions)}] Question: {qa_item.question[:60]}...")
        if qa_item.level:
            print(f"   Level: {qa_item.level}")

        try:
            record = run_single_question(cfg, llm, retriever, qa_item)
            all_results.append(record)
            processed_indices.add(qa_item.idx)
            questions_since_last_save += 1

            print(f"   Answer: {record['output'][:80]}...")
            print(f"   Time: {record['query_time']:.2f}s")

            # Save incrementally
            if questions_since_last_save >= save_interval:
                # Rewrite entire file to maintain consistency
                with result_path.open("w", encoding="utf-8") as f:
                    for result in all_results:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"\n💾 Progress saved: {len(processed_indices)}/{len(questions)} questions")
                questions_since_last_save = 0

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    # Final save
    with result_path.open("w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"\n✅ All results saved to: {result_path}")

    # Compute metrics
    print("\n📊 Computing metrics...")
    metrics = compute_metrics(all_results)

    # Save metrics
    metrics_path = output_dir / f"{cfg.run.strategy}_{timestamp}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"   Metrics saved to: {metrics_path}")

    return metrics


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute evaluation metrics from results."""
    if not results:
        return {"total": 0, "em": 0.0, "f1": 0.0}

    em_scores = []
    f1_scores = []
    level_metrics = {}  # Per-level metrics

    for record in results:
        output = record.get("output", "")
        targets = record.get("targets", [])
        level = record.get("level", "unknown")

        scores = em_f1(output, targets)
        em_scores.append(scores["em"])
        f1_scores.append(scores["f1"])

        # Track per-level metrics
        if level not in level_metrics:
            level_metrics[level] = {"em": [], "f1": [], "count": 0}
        level_metrics[level]["em"].append(scores["em"])
        level_metrics[level]["f1"].append(scores["f1"])
        level_metrics[level]["count"] += 1

    # Overall metrics
    metrics = {
        "total": len(results),
        "em": mean(em_scores),
        "f1": mean(f1_scores),
    }

    # Per-level metrics
    if level_metrics:
        metrics["by_level"] = {}
        for level, data in level_metrics.items():
            metrics["by_level"][level] = {
                "count": data["count"],
                "em": mean(data["em"]),
                "f1": mean(data["f1"]),
            }

    return metrics
