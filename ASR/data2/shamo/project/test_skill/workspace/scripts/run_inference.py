"""CLI script for running FunASR model inference (pure inference, no analysis).

Usage:
    # 推理 + 后处理（默认模式）
    python scripts/run_inference.py --model-dir models/iterations/iter_001 --tag iter_001

    # 推理，不加后处理（对比用）
    python scripts/run_inference.py --model-dir models/iterations/iter_001 --tag iter_001 --no-postprocess

    # 指定输出路径
    python scripts/run_inference.py --model-dir models/iterations/iter_001 --output reports/evaluation/inference_iter_001.jsonl

    # 指定测试数据
    python scripts/run_inference.py --model-dir models/iterations/iter_001 --test-data data/augmented/test_aug.jsonl --tag iter_001

输出格式（每行一个 JSON）：
    {"audio_path": "...", "reference_text": "...", "hypothesis_text": "...", "hypothesis_raw": "...", "scene_type": "..."}

    - hypothesis_text: 后处理后的识别文本（--no-postprocess 时与 hypothesis_raw 相同）
    - hypothesis_raw: 模型原始输出（未经后处理）

两种模式对比后处理效果：
    1. 默认模式（--postprocess）：输出同时包含 raw 和 processed，stdout 报告 postprocess_cer_delta
    2. 无后处理模式（--no-postprocess）：hypothesis_text == hypothesis_raw

分析评估请使用 analyze_results.py 读取输出文件。
"""

import argparse
import json
import sys

from scripts.core.config_utils import load_config
from scripts.execution.funasr_evaluator import Evaluator


def main(argv: list[str] | None = None) -> None:
    """Entry point for the run_inference CLI."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    parser = argparse.ArgumentParser(
        description="Run FunASR model inference. Outputs per-sample results as JSONL."
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
        help="Path to write inference results JSONL. Default: reports/evaluation/inference_{tag}.jsonl",
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
        help="Tag for output file naming (e.g. iteration ID).",
    )
    parser.add_argument(
        "--postprocess", action="store_true", default=True,
        dest="postprocess",
        help="Apply post-processing to hypothesis (default: enabled).",
    )
    parser.add_argument(
        "--no-postprocess", action="store_false",
        dest="postprocess",
        help="Skip post-processing, output raw model hypothesis only.",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)

    # ── 数据解析：默认使用增强数据 ──
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

    # ── 确定输出路径 ──
    if args.output:
        output_path = args.output
    elif args.tag:
        output_path = f"reports/evaluation/inference_{args.tag}.jsonl"
    else:
        output_path = "reports/evaluation/inference_results.jsonl"

    # ── 禁用报告生成（纯推理模式） ──
    config.setdefault("evaluation", {})["generate_report"] = False

    evaluator = Evaluator(config, base_dir=".")

    result = evaluator.run_inference_only(
        model_dir=args.model_dir,
        test_jsonl=test_data,
        output_path=output_path,
        output_format=args.format,
        apply_postprocess=args.postprocess,
    )

    output = {
        "mode": "inference",
        "status": "completed",
        "model_dir": args.model_dir,
        "test_data": test_data,
        "output_path": output_path,
        "num_samples": result.get("num_samples", 0),
        "postprocess_applied": args.postprocess,
        "tag": args.tag,
        "message": f"Inference results written to {output_path}. Run analyze_results.py to compute CER/WER.",
    }

    # Include CER/WER summary and postprocess delta if available
    if result.get("cer") is not None:
        output["cer"] = result["cer"]
        output["wer"] = result["wer"]
    if result.get("cer_raw") is not None:
        output["cer_raw"] = result["cer_raw"]
        output["wer_raw"] = result["wer_raw"]
        output["postprocess_cer_delta"] = result["postprocess_cer_delta"]
        output["postprocess_wer_delta"] = result["postprocess_wer_delta"]

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
