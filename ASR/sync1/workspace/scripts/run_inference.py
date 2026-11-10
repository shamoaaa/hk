"""CLI script for running FunASR model inference.

Usage:
    python scripts/run_inference.py --model-dir models/iterations/iter_001 --test-data data/test/test.jsonl --output data/test/test_set.jsonl
    python scripts/run_inference.py --model-dir models/iterations/iter_001 --test-data data/test/test.jsonl --output data/test/test_set.jsonl --format funasr
    python scripts/run_inference.py --model-dir models/iterations/iter_001 --test-data data/test/test.jsonl --evaluate

Wraps funasr_evaluator.Evaluator. Loads model with FunASR AutoModel,
runs inference on test data, and outputs test_set.jsonl in the format
expected by analyze_results.py.

Outputs structured JSON summary to stdout.
"""

import argparse
import json
import sys

from scripts.core.config_utils import load_config
from scripts.execution.funasr_evaluator import Evaluator


def main(argv: list[str] | None = None) -> None:
    """Entry point for the run_inference CLI."""
    parser = argparse.ArgumentParser(
        description="Run FunASR model inference and evaluation."
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--test-data", default=None,
        help="Path to test data (FunASR jsonl). Default: from config.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write inference results. Default: data/test/test_set.jsonl",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--format", choices=["project", "funasr"], default="project",
        help="Output format: 'project' (audio_path/reference_text/hypothesis_text/scene_type) "
             "or 'funasr' (key/source/target/hypothesis). Default: project.",
    )
    parser.add_argument(
        "--tag", default="",
        help="Evaluation tag for report naming.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run full evaluation with CER/WER computation and report generation.",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip report generation.",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)

    # ── 数据解析：默认使用增强数据 ──
    # 业务强制约束：test 必须经过增强才能用于评估
    # 优先级：CLI args > config.data.augmented_test_jsonl > config.data.test_jsonl
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    require_aug = aug_cfg.get("require_augmentation", True)

    test_data = (
        args.test_data
        or data_cfg.get("augmented_test_jsonl")
        or data_cfg.get("test_jsonl")
    )
    if not test_data:
        parser.error("--test-data is required (or set data.test_jsonl in config)")

    # ── 强制校验：评估数据必须是增强后的数据 ──
    if require_aug:
        augmented_dir = aug_cfg.get("augmented_dir", "data/augmented")
        augmented_test_hint = data_cfg.get(
            "augmented_test_jsonl", f"{augmented_dir}/test_aug.jsonl"
        )

        def _is_augmented(path: str) -> bool:
            import os as _os
            norm = _os.path.normpath(path).replace("\\", "/")
            return (
                augmented_dir.rstrip("/") in norm
                or norm.rsplit("/", 1)[-1].rstrip(".jsonl").endswith("_aug")
            )

        if not _is_augmented(test_data):
            parser.error(
                f"[Augmentation Required] test data '{test_data}' 不是增强数据。\n"
                f"  业务要求 train/test 必须经过加噪后才能用于评估。\n"
                f"  请先运行：python scripts/run_augmentation.py --all\n"
                f"  然后使用 {augmented_test_hint} 作为评估输入。\n"
                f"  如需跳过此校验（仅调试用），在 funasr_config.yaml 中设置 "
                f"augmentation.require_augmentation: false"
            )

        from pathlib import Path as _Path
        if not _Path(test_data).exists():
            parser.error(
                f"[Augmentation Required] 增强测试数据不存在：{test_data}\n"
                f"  请先运行：python scripts/run_augmentation.py --all"
            )

    if args.no_report:
        config.setdefault("evaluation", {})["generate_report"] = False

    evaluator = Evaluator(config, base_dir=".")

    if args.evaluate:
        # Full evaluation mode
        result = evaluator.evaluate(
            model_dir=args.model_dir,
            test_jsonl=test_data,
            tag=args.tag,
        )

        # Also write output file if specified
        if args.output:
            evaluator.run_inference_only(
                model_dir=args.model_dir,
                test_jsonl=test_data,
                output_path=args.output,
                output_format=args.format,
            )
            result["output_path"] = args.output

        # Remove full samples list from stdout output (too verbose)
        result.pop("samples", None)

        output = {
            "mode": "evaluate",
            "status": "completed",
            **result,
        }
    else:
        # Inference-only mode
        output_path = args.output or "data/test/test_set.jsonl"
        result = evaluator.run_inference_only(
            model_dir=args.model_dir,
            test_jsonl=test_data,
            output_path=output_path,
            output_format=args.format,
        )

        output = {
            "mode": "inference",
            **result,
        }

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
