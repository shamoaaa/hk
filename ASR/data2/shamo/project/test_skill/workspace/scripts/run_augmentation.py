"""CLI script for ASR data augmentation using MUSAN noise dataset.

业务强制约束：train / val / test 都必须经过增强后才能用于训练和评估。
建议在每次迭代开始前，运行 --all 同时生成 train / val / test 的增强版本。

Usage:
    # 一次性增强 train + val + test（推荐，迭代开始前执行）
    python scripts/run_augmentation.py --all

    # 单独增强某一份数据
    python scripts/run_augmentation.py --dataset-path data/train.jsonl --output-jsonl data/augmented/train_aug.jsonl
    python scripts/run_augmentation.py --dataset-path data/test.jsonl  --output-jsonl data/augmented/test_aug.jsonl

    # 单策略增强
    python scripts/run_augmentation.py --all --strategy babble
    python scripts/run_augmentation.py --dataset-path data/train.jsonl --strategy reverb

Uses real audio augmentation via funasr_noise_augment.NoiseAugmentor
(babble/music/noise/reverb with soundfile + scipy). Reads augmentation
config from funasr_config.yaml. Outputs structured JSON to stdout.

Requirements: 7.1, 7.3, 7.5, 7.6
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.core.config_utils import load_config, save_versioned_config
from scripts.execution.funasr_noise_augment import NoiseAugmentor


def _augment_one(
    augmentor: NoiseAugmentor,
    dataset_path: str,
    output_jsonl: str,
    strategy: str | None,
) -> dict:
    """Augment a single dataset (train OR val OR test)."""
    if strategy:
        result = augmentor.augment_single_strategy(
            jsonl_path=dataset_path,
            output_jsonl=output_jsonl,
            strategy_type=strategy,
        )
    else:
        result = augmentor.augment_dataset(
            jsonl_path=dataset_path,
            output_jsonl=output_jsonl,
        )
    result["input_path"] = dataset_path
    result["output_path"] = output_jsonl
    return result


def main(argv: list[str] | None = None) -> None:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run real audio data augmentation using MUSAN noise dataset. "
            "业务强制约束：train 和 test 必须经过增强才能用于训练和评估。"
        )
    )
    parser.add_argument(
        "--all", action="store_true",
        help=(
            "Augment train / val / test in one call (using paths from config). "
            "Output goes to {augmentation.augmented_dir}/{split}_aug.jsonl."
        ),
    )
    parser.add_argument(
        "--dataset-path", default=None,
        help=(
            "Path to a single jsonl. Ignored when --all is set. "
            "Default: data.train_jsonl from config."
        ),
    )
    parser.add_argument(
        "--output-jsonl", default=None,
        help="Explicit output path for --dataset-path mode.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Override augmentation.augmented_dir. Output files named "
            "{split}_aug.jsonl live under this directory."
        ),
    )
    parser.add_argument(
        "--noise-dir", default=None,
        help="Path to MUSAN noise dataset. Default: from config (augmentation.noise_dir).",
    )
    parser.add_argument(
        "--strategy",
        choices=["babble", "music", "noise", "reverb", "speed", "volume"],
        default=None,
        help="Apply a single augmentation strategy instead of all configured strategies.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML. Default: configs/funasr_config.yaml.",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    aug_cfg = config.setdefault("augmentation", {})
    aug_cfg["enabled"] = True

    if args.noise_dir:
        aug_cfg["noise_dir"] = args.noise_dir
    if args.output_dir:
        aug_cfg["augmented_dir"] = args.output_dir
    augmented_dir = aug_cfg.get("augmented_dir", "data/augmented")

    augmentor = NoiseAugmentor(config, base_dir=".")
    augmentor.prepare_noise_data()

    # ── Pre-augmentation quality check (Ch3/Ch5: 数据入口校验) ──
    pre_check_results = {}
    if args.all:
        splits_to_check = [
            ("train", data_cfg.get("train_jsonl")),
            ("val", data_cfg.get("val_jsonl")),
            ("test", data_cfg.get("test_jsonl")),
        ]
    else:
        ds = args.dataset_path or data_cfg.get("train_jsonl")
        splits_to_check = [("input", ds)]

    from scripts.analysis.dataset_analyzer import DatasetAnalyzer
    import logging as _log
    checker = DatasetAnalyzer(check_audio_exists=False)
    critical_count = 0
    for split_name, path in splits_to_check:
        if not path or not Path(path).exists():
            continue
        report = checker.generate_quality_report(dataset_path=path)
        anomalies = report.anomalies if hasattr(report, 'anomalies') else []
        critical = [a for a in anomalies if a.anomaly_type in ("empty_label", "corrupted_audio")]
        pre_check_results[split_name] = {
            "total_anomalies": len(anomalies),
            "critical": len(critical),
            "types": {}
        }
        for a in anomalies:
            pre_check_results[split_name]["types"][a.anomaly_type] = (
                pre_check_results[split_name]["types"].get(a.anomaly_type, 0) + 1
            )
        critical_count += len(critical)

        # Print check results in real-time
        if anomalies:
            type_summary = ", ".join(f"{k}={v}" for k, v in pre_check_results[split_name]["types"].items())
            _log.info("[Pre-check] %s: %d anomalies (%s)", split_name, len(anomalies), type_summary)
        else:
            _log.info("[Pre-check] %s: clean ✓", split_name)

    if critical_count > 0:
        _log.warning(
            "[Pre-check] ⚠ %d critical anomalies (empty_label/corrupted_audio) will be auto-filtered.",
            critical_count,
        )
    else:
        _log.info("[Pre-check] No critical anomalies. All data OK.")

    # ── Mode 1: --all (augment train + val + test) ──
    if args.all:
        splits = [
            ("train", data_cfg.get("train_jsonl"), data_cfg.get(
                "augmented_train_jsonl", f"{augmented_dir}/train_aug.jsonl"
            )),
            ("val", data_cfg.get("val_jsonl"), data_cfg.get(
                "augmented_val_jsonl", f"{augmented_dir}/val_aug.jsonl"
            )),
            ("test", data_cfg.get("test_jsonl"), data_cfg.get(
                "augmented_test_jsonl", f"{augmented_dir}/test_aug.jsonl"
            )),
        ]

        all_results = {}
        for split_name, input_path, output_path in splits:
            if not input_path:
                all_results[split_name] = {
                    "status": "skipped",
                    "reason": f"data.{split_name}_jsonl not set in config",
                }
                continue
            if not Path(input_path).exists():
                all_results[split_name] = {
                    "status": "skipped",
                    "reason": f"input file not found: {input_path}",
                }
                continue
            all_results[split_name] = _augment_one(
                augmentor, input_path, output_path, args.strategy
            )

        # Save augmentation config snapshot
        save_versioned_config("configs/augmentation", "augment", aug_cfg)

        # ── Post-augmentation validation (Ch5/Ch14: 增强产物校验) ──
        _log.info("[Post-check] Validating augmented outputs...")
        post_check_results = {}
        for split_name, input_path, output_path in splits:
            if not output_path or not Path(output_path).exists():
                continue
            post_report = checker.generate_quality_report(dataset_path=output_path)
            post_anomalies = post_report.anomalies if hasattr(post_report, 'anomalies') else []
            post_check_results[split_name] = {
                "total_anomalies": len(post_anomalies),
                "types": {}
            }
            for a in post_anomalies:
                post_check_results[split_name]["types"][a.anomaly_type] = (
                    post_check_results[split_name]["types"].get(a.anomaly_type, 0) + 1
                )
            if post_anomalies:
                type_summary = ", ".join(f"{k}={v}" for k, v in post_check_results[split_name]["types"].items())
                _log.warning("[Post-check] %s: %d anomalies (%s)", split_name, len(post_anomalies), type_summary)
            else:
                _log.info("[Post-check] %s: clean ✓", split_name)

        output = {
            "mode": "all",
            "augmented_dir": augmented_dir,
            "results": all_results,
            "pre_check": pre_check_results,
            "post_check": post_check_results,
        }
        json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")

        return

    # ── Mode 2: single --dataset-path ──
    dataset_path = args.dataset_path or data_cfg.get("train_jsonl")
    if not dataset_path:
        parser.error("--dataset-path required (or set data.train_jsonl in config, or use --all)")

    # Derive output path: <augmented_dir>/<input_stem>_aug.jsonl
    if args.output_jsonl:
        output_jsonl = args.output_jsonl
    else:
        input_stem = Path(dataset_path).stem
        output_jsonl = str(Path(augmented_dir) / f"{input_stem}_aug.jsonl")

    result = _augment_one(augmentor, dataset_path, output_jsonl, args.strategy)
    result["output_dir"] = augmented_dir
    result["mode"] = "single"
    result["pre_check"] = pre_check_results

    save_versioned_config("configs/augmentation", "augment", aug_cfg)

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
