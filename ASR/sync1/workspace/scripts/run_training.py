"""CLI script for running FunASR training.

Usage:
    python scripts/run_training.py --train-data data/train.jsonl --val-data data/val.jsonl
    python scripts/run_training.py --train-data data/train.jsonl --val-data data/val.jsonl --learning-rate 5e-5 --max-epoch 10 --gpu-ids 4
    python scripts/run_training.py --train-data data/train.jsonl --val-data data/val.jsonl --tag iter_001
    python scripts/run_training.py --dry-run --train-data data/train.jsonl --val-data data/val.jsonl

Wraps funasr_trainer.Trainer. Reads default params from the latest
versioned config in workspace/configs/training/ or from
workspace/configs/funasr_config.yaml. Outputs structured JSON to stdout.
"""

import argparse
import json
import sys

from scripts.core.config_utils import load_config
from scripts.execution.funasr_trainer import Trainer


def main(argv: list[str] | None = None) -> None:
    """Entry point for the run_training CLI."""
    parser = argparse.ArgumentParser(
        description="Run FunASR model training."
    )
    parser.add_argument(
        "--train-data", default=None,
        help="Path to training data (FunASR jsonl). Default: from config.",
    )
    parser.add_argument(
        "--val-data", default=None,
        help="Path to validation data (FunASR jsonl). Default: from config.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to FunASR config YAML file.",
    )
    parser.add_argument(
        "--tag", default="",
        help="Experiment tag for output directory naming.",
    )

    # Training parameter overrides
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--validate-interval", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=None)

    # Dry-run mode
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the training command without executing.",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)

    # ── 数据解析：默认使用增强数据 ──
    # 业务强制约束：train/val 必须经过增强才能训练
    # 解析优先级：CLI args > config.data.augmented_*_jsonl > config.data.*_jsonl
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    require_aug = aug_cfg.get("require_augmentation", True)

    # 如果用户没显式传 --train-data，默认用增强后的路径
    train_data = (
        args.train_data
        or data_cfg.get("augmented_train_jsonl")
        or data_cfg.get("train_jsonl")
    )
    val_data = (
        args.val_data
        or data_cfg.get("augmented_val_jsonl")
        or data_cfg.get("val_jsonl")
    )
    if not train_data or not val_data:
        parser.error("--train-data and --val-data are required (or set in config)")

    # ── 强制校验：训练数据必须是增强后的数据 ──
    if require_aug:
        augmented_dir = aug_cfg.get("augmented_dir", "data/augmented")
        augmented_train_hint = data_cfg.get(
            "augmented_train_jsonl", f"{augmented_dir}/train_aug.jsonl"
        )
        augmented_val_hint = data_cfg.get(
            "augmented_val_jsonl", f"{augmented_dir}/val_aug.jsonl"
        )

        def _is_augmented(path: str) -> bool:
            """增强数据的路径必须在 augmented_dir 下，或文件名带 _aug 后缀。"""
            import os as _os
            norm = _os.path.normpath(path).replace("\\", "/")
            return (
                augmented_dir.rstrip("/") in norm
                or norm.rsplit("/", 1)[-1].rstrip(".jsonl").endswith("_aug")
            )

        if not _is_augmented(train_data):
            parser.error(
                f"[Augmentation Required] train data '{train_data}' 不是增强数据。\n"
                f"  业务要求 train/val 必须经过加噪。请先运行：\n"
                f"    python scripts/run_augmentation.py --all\n"
                f"  然后使用 {augmented_train_hint} 作为训练输入。\n"
                f"  如需跳过此校验（仅调试用），在 funasr_config.yaml 中设置 "
                f"augmentation.require_augmentation: false"
            )
        if not _is_augmented(val_data):
            parser.error(
                f"[Augmentation Required] val data '{val_data}' 不是增强数据。\n"
                f"  请使用 {augmented_val_hint}，或先运行 run_augmentation.py --all 生成。"
            )

        # 进一步校验文件存在
        from pathlib import Path as _Path
        if not _Path(train_data).exists():
            parser.error(
                f"[Augmentation Required] 增强训练数据不存在：{train_data}\n"
                f"  请先运行：python scripts/run_augmentation.py --all"
            )
        if not _Path(val_data).exists():
            parser.error(
                f"[Augmentation Required] 增强验证数据不存在：{val_data}\n"
                f"  请先运行：python scripts/run_augmentation.py --all"
            )

    # Apply CLI overrides to config
    if args.learning_rate is not None:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.max_epoch is not None:
        config.setdefault("training", {})["max_epoch"] = args.max_epoch
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.gpu_ids is not None:
        config.setdefault("training", {})["gpu_ids"] = args.gpu_ids
    if args.num_workers is not None:
        config.setdefault("training", {})["num_workers"] = args.num_workers
    if args.validate_interval is not None:
        config.setdefault("training", {})["validate_interval"] = args.validate_interval
    if args.resume is not None:
        config.setdefault("training", {})["resume"] = args.resume

    # Build override params for FunASR command
    override_params = {}
    if args.learning_rate is not None:
        override_params["++optim_conf.lr"] = args.learning_rate
    if args.max_epoch is not None:
        override_params["++train_conf.max_epoch"] = args.max_epoch
    if args.batch_size is not None:
        override_params["++dataset_conf.batch_size"] = args.batch_size

    trainer = Trainer(config, base_dir=".")

    if args.dry_run:
        result = trainer.get_training_command(
            train_data, val_data, override_params, tag=args.tag
        )
        result["mode"] = "dry_run"
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    result = trainer.train(
        train_data,
        val_data,
        override_params,
        tag=args.tag,
    )

    result["mode"] = "full"
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
