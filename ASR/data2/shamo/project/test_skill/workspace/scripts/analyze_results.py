"""CLI script for analyzing ASR inference results.

Usage:
    # 分析推理结果，生成评估报告
    python scripts/analyze_results.py --test-set reports/evaluation/inference_iter_001.jsonl --tag iter_001

    # 指定 badcase 阈值
    python scripts/analyze_results.py --test-set reports/evaluation/inference_iter_001.jsonl --tag iter_001 --badcase-threshold 0.1

    # 与基线报告对比
    python scripts/analyze_results.py --test-set reports/evaluation/inference_iter_002.jsonl --tag iter_002 --baseline-report reports/evaluation/eval_report_iter_001.json

读取 run_inference.py 输出的逐样本 JSONL 文件，计算：
- 整体 CER/WER
- 按场景分组的 CER/WER
- 错误类型分布（substitution/insertion/deletion）
- Badcase 列表（CER 超阈值的样本）
- 可选：与基线报告对比

输出：
- reports/evaluation/eval_report_{tag}.json（结构化报告，含 badcases）
- reports/evaluation/eval_report_{tag}.txt（人类可读报告）
- stdout: JSON 摘要

输入 JSONL 格式（每行一个 JSON）：
    {"audio_path": "...", "reference_text": "...", "hypothesis_text": "...", "scene_type": "..."}

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from scripts.core.config_utils import read_jsonl
from scripts.core.models import EvaluationReport, SceneMetrics
from scripts.analysis.result_analyzer import EvalSample, ResultAnalyzer


def _load_samples(test_set_path: str, use_raw: bool = False) -> list[EvalSample]:
    """Load evaluation samples from a JSONL file.

    Each line must be a JSON object with at least ``reference_text`` and
    ``hypothesis_text``.  ``audio_path`` and ``scene_type`` are optional.

    Parameters
    ----------
    use_raw : bool
        If True and ``hypothesis_raw`` field exists, use it instead of
        ``hypothesis_text`` for evaluation (to measure model-only performance
        without post-processing).
    """
    records = read_jsonl(test_set_path)
    samples: list[EvalSample] = []
    for record in records:
        if use_raw and "hypothesis_raw" in record:
            hyp = record["hypothesis_raw"]
        else:
            hyp = record.get("hypothesis_text", "")
        samples.append(
            EvalSample(
                audio_path=record.get("audio_path", ""),
                reference_text=record.get("reference_text", ""),
                hypothesis_text=hyp,
                scene_type=record.get("scene_type", ""),
            )
        )
    return samples


def _load_baseline_report(path: str) -> EvaluationReport:
    """Load a baseline EvaluationReport from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scene_metrics = [
        SceneMetrics(**sm) for sm in data.get("scene_metrics", [])
    ]
    return EvaluationReport(
        overall_cer=data.get("overall_cer", data.get("cer", 0.0)),
        overall_wer=data.get("overall_wer", data.get("wer", 0.0)),
        scene_metrics=scene_metrics,
        error_type_distribution=data.get("error_type_distribution", {}),
    )


def _write_json_report(output: dict, tag: str) -> str:
    """Write the JSON evaluation report to reports/evaluation/."""
    report_dir = Path("reports/evaluation")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"eval_report_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return str(path)


def _write_txt_report(output: dict, tag: str) -> str:
    """Write a human-readable text report."""
    report_dir = Path("reports/evaluation")
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"eval_report_{tag}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ASR Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Tag: {tag}\n")
        f.write(f"Samples: {output.get('total_samples', 0)}\n")
        f.write(f"CER: {output.get('overall_cer', 0):.4f} ({output.get('overall_cer', 0)*100:.2f}%)\n")
        f.write(f"WER: {output.get('overall_wer', 0):.4f} ({output.get('overall_wer', 0)*100:.2f}%)\n\n")

        scene_metrics = output.get("scene_metrics", [])
        if scene_metrics:
            f.write("-" * 60 + "\n")
            f.write("Scene-level Metrics:\n")
            f.write("-" * 60 + "\n\n")
            for sm in scene_metrics:
                f.write(
                    f"  {sm['scene_type']:20s}  "
                    f"CER={sm['cer']:.4f}  "
                    f"WER={sm['wer']:.4f}  "
                    f"(n={sm['sample_count']})\n"
                )
            f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("Error Type Distribution:\n")
        f.write("-" * 60 + "\n\n")
        for etype, count in output.get("error_type_distribution", {}).items():
            f.write(f"  {etype}: {count}\n")
        f.write("\n")

        badcases = output.get("badcases", [])
        if badcases:
            f.write("-" * 60 + "\n")
            f.write(f"Badcases (CER > threshold, total {len(badcases)}):\n")
            f.write("-" * 60 + "\n\n")
            for bc in badcases[:50]:  # 最多显示 50 个
                key = bc.get("audio_path", "").rsplit("/", 1)[-1] or "unknown"
                f.write(f"[{key}] REF: {bc.get('reference_text', '')}\n")
                f.write(f"           HYP: {bc.get('hypothesis_text', '')}\n\n")
            if len(badcases) > 50:
                f.write(f"  ... and {len(badcases) - 50} more\n")

    return str(path)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the analyze_results CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze ASR inference results: compute CER/WER, generate badcases and evaluation report."
    )
    parser.add_argument(
        "--test-set", required=True,
        help="Path to inference results JSONL (output of run_inference.py).",
    )
    parser.add_argument(
        "--tag", default="default",
        help="Tag for report naming (e.g. iteration ID).",
    )
    parser.add_argument(
        "--badcase-threshold", type=float, default=0.0,
        help="CER threshold for badcase (default: 0.0 = any error is a badcase).",
    )
    parser.add_argument(
        "--baseline-report", default=None,
        help="Path to a baseline eval_report JSON for comparison.",
    )
    parser.add_argument(
        "--use-raw", action="store_true", default=False,
        help="Use hypothesis_raw instead of hypothesis_text for CER/WER calculation (compare without post-processing).",
    )
    parser.add_argument(
        "--no-report-file", action="store_true",
        help="Only output to stdout, don't write report files.",
    )

    args = parser.parse_args(argv)

    # Load test samples
    samples = _load_samples(args.test_set, use_raw=args.use_raw)
    if not samples:
        json.dump({"status": "error", "message": f"No samples found in {args.test_set}"}, sys.stdout)
        sys.stdout.write("\n")
        return

    # Run evaluation
    analyzer = ResultAnalyzer(samples)
    report = analyzer.evaluate_model(
        model_path="",
        test_set_path=args.test_set,
    )

    # Generate badcases
    badcases = analyzer.generate_badcases(threshold=args.badcase_threshold)

    # Classify errors
    error_distribution = analyzer.classify_errors_from_badcases(badcases)

    # Build badcase list for output
    badcase_list = []
    for bc in badcases:
        badcase_list.append({
            "audio_path": bc.audio_path,
            "reference_text": bc.reference_text,
            "hypothesis_text": bc.hypothesis_text,
            "scene_type": getattr(bc, "scene_type", ""),
            "cer": bc.cer,
        })

    # Build output
    output: dict = {
        "test_set_path": args.test_set,
        "tag": args.tag,
        "total_samples": len(samples),
        "overall_cer": report.overall_cer,
        "overall_wer": report.overall_wer,
        "scene_metrics": [asdict(sm) for sm in report.scene_metrics],
        "error_type_distribution": report.error_type_distribution,
        "badcase_threshold": args.badcase_threshold,
        "badcase_count": len(badcases),
        "badcases": badcase_list,
    }

    # Optional baseline comparison
    if args.baseline_report:
        baseline = _load_baseline_report(args.baseline_report)
        comparison = analyzer.compare_with_baseline(baseline, report)
        output["baseline_comparison"] = comparison

    # Write report files
    if not args.no_report_file:
        json_path = _write_json_report(output, args.tag)
        txt_path = _write_txt_report(output, args.tag)
        output["report_json"] = json_path
        output["report_txt"] = txt_path

    # Stdout summary (without full badcase list to keep it concise)
    stdout_output = {k: v for k, v in output.items() if k != "badcases"}
    stdout_output["badcase_count"] = len(badcase_list)
    stdout_output["status"] = "success"

    json.dump(stdout_output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")

    from scripts.core.execution_logger import log_execution
    log_execution("analyze_results", stdout_output)


if __name__ == "__main__":
    main()
