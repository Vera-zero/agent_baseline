from __future__ import annotations

import argparse
import json
from pathlib import Path

from timeqa_baseline_lab.config import load_config, load_yaml_dict, merge_config
from timeqa_baseline_lab.runner import run_experiment


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
    parser = argparse.ArgumentParser(description="Run TimeQA baseline experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--strategy", type=str, default=None, help="Override strategy")
    parser.add_argument("--max_questions", type=int, default=None, help="Override max questions; <=0 means full")
    parser.add_argument("--disable_strategy_config", action="store_true", help="Do not load configs/methods/<strategy>.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.strategy:
        cfg.run.strategy = args.strategy
    if args.max_questions is not None:
        cfg.run.max_questions = args.max_questions
    if args.disable_strategy_config:
        cfg.run.load_strategy_config = False

    cfg = _apply_strategy_config(cfg, args.config)

    metrics = run_experiment(cfg)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
