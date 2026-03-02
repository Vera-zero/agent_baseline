from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    corpus_path: str
    question_arrow_path: str


@dataclass
class ModelConfig:
    provider: str = "vllm"  # hf / api / vllm
    model_name: str = "qwen3-32b"
    model: str = "deepseek-chat"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    device: str = "auto"
    torch_dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 180
    batch_size: int = 1


@dataclass
class RetrieverConfig:
    type: str = "contriever"
    model_name: str = "facebook/contriever"
    top_k: int = 5


@dataclass
class ChunkConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 100
    min_chunk_size: int = 500


@dataclass
class RunConfig:
    strategy: str = "zero_shot"
    max_questions: int = 100
    resume: bool = True
    save_every: int = 1
    seed: int = 42
    load_strategy_config: bool = True
    strategy_config_dir: str = "./configs/methods"
    strategy_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IOConfig:
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    run_name: str = "timeqa_baseline"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    retriever: RetrieverConfig
    chunk: ChunkConfig
    run: RunConfig
    io: IOConfig


def _merge(defaults: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build_experiment(cfg: Dict[str, Any]) -> ExperimentConfig:
    exp = ExperimentConfig(
        data=DataConfig(**cfg["data"]),
        model=ModelConfig(**cfg["model"]),
        retriever=RetrieverConfig(**cfg["retriever"]),
        chunk=ChunkConfig(**cfg["chunk"]),
        run=RunConfig(**cfg["run"]),
        io=IOConfig(**cfg["io"]),
    )

    provider = (exp.model.provider or "local").lower()
    if provider in {"hf", "local"}:
        exp.model.provider = "local"
        if not exp.model.model_name:
            raise ValueError("model.model_name is required when model.provider=local")
    elif provider in {"api", "remote"}:
        exp.model.provider = "remote"
        if not exp.model.model:
            raise ValueError("model.model is required when model.provider=remote")
        if not exp.model.base_url:
            raise ValueError("model.base_url is required when model.provider=remote")
    elif provider == "vllm":
        exp.model.provider = "vllm"
        if not exp.model.model_name:
            raise ValueError("model.model_name is required when model.provider=vllm")
        if not exp.model.base_url:
            raise ValueError("model.base_url is required when model.provider=vllm")
    else:
        raise ValueError(f"Invalid model.provider={exp.model.provider}. Use local/hf, remote/api, or vllm")

    exp.model.batch_size = max(1, int(exp.model.batch_size))
    exp.model.max_retries = max(1, int(exp.model.max_retries))
    exp.model.timeout = max(1, int(exp.model.timeout))

    Path(exp.io.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(exp.io.output_dir).mkdir(parents=True, exist_ok=True)
    return exp


def load_yaml_dict(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml_dict(path)
    defaults = {
        "data": {},
        "model": {},
        "retriever": {},
        "chunk": {},
        "run": {},
        "io": {},
    }
    cfg = _merge(defaults, raw)
    return _build_experiment(cfg)


def merge_config(base: ExperimentConfig, override: Dict[str, Any]) -> ExperimentConfig:
    base_dict = {
        "data": asdict(base.data),
        "model": asdict(base.model),
        "retriever": asdict(base.retriever),
        "chunk": asdict(base.chunk),
        "run": asdict(base.run),
        "io": asdict(base.io),
    }
    merged = _merge(base_dict, override or {})
    return _build_experiment(merged)