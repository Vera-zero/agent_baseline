﻿from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from timeqa_baseline_lab.config import load_config, load_yaml_dict, merge_config
from timeqa_baseline_lab.runner import run_experiment


STRATEGIES = ["zero_shot_cot", "rag_cot", "react"]


def _apply_strategy_config(cfg, base_config_path: str):
    if not cfg.run.load_strategy_config:
        return cfg
    cfg_dir = Path(cfg.run.strategy_config_dir)
    if not cfg_dir.exists():
        fallback = Path(base_config_path).resolve().parent / "methods"
        if fallback.exists():
            cfg_dir = fallback
    path = cfg_dir / f"{cfg.run.strategy}.yaml"
    override = load_yaml_dict(path)
    if not override:
        return cfg
    return merge_config(cfg, override)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all baseline strategies")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--disable_strategy_config", action="store_true", help="Do not load configs/methods/<strategy>.yaml")
    args = parser.parse_args()

    base = load_config(args.config)
    if args.max_questions is not None:
        base.run.max_questions = args.max_questions
    if args.disable_strategy_config:
        base.run.load_strategy_config = False

    results = {}
    for s in STRATEGIES:
        cfg = copy.deepcopy(base)
        cfg.run.strategy = s
        cfg = _apply_strategy_config(cfg, args.config)
        metrics = run_experiment(cfg)
        results[s] = metrics
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\\nAll done")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()