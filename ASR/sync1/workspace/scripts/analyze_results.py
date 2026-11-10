"""CLI script for analyzing ASR model evaluation results.

Usage:
    python scripts/analyze_results.py --model-path models/iter_001 --test-set data/test.jsonl
    python scripts/analyze_results.py --test-set data/test.jsonl --badcase-threshold 0.3
    python scripts/analyze_results.py --test-set data/test.jsonl --baseline-report reports/baseline.json

Reads a test set (JSONL file with evaluation samples), computes overall and
per-scene CER/WER, generates badcase lists, classifies errors, and optionally
compares against a baseline report.  Outputs a structured JSON evaluation
report to stdout.

Test set JSONL format (one JSON object per line):
    {"audio_path": "...", "reference_text": "...", "hypothesis_text": "...", "scene_type": "..."}

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import argparse
import json
import sys
from dataclasses import asdict

from scripts.core.config_utils import read_json, read_jsonl
from scripts.core.models import EvaluationReport, SceneMetrics
from scripts.analysis.result_analyzer import EvalSample, ResultAnalyzer


def _load_samples(test_set_path: str) -> list[EvalSample]:
    """Load evaluation samples from a JSONL file.

    Each line must be a JSON object with at least ``reference_text`` and
    ``hypothesis_text``.  ``audio_path`` and ``scene_type`` are optional.
    """
    records = read_jsonl(test_set_path)
    samples: list[EvalSample] = []
    for record in records:
        samples.append(
            EvalSample(
                audio_path=record.get("audio_path", ""),
                reference_text=record.get("reference_text", ""),
                hypothesis_text=record.get("hypothesis_text", ""),
                scene_type=record.get("scene_type", ""),
            )
        )
    return samples


def _load_baseline_report(path: str) -> EvaluationReport:
    """Load a baseline EvaluationReport from a JSON file."""
    data = read_json(path)
    scene_metrics = [
        SceneMetrics(**sm) for sm in data.get("scene_metrics", [])
    ]
    return EvaluationReport(
        overall_cer=data.get("overall_cer", 0.0),
        overall_wer=data.get("overall_wer", 0.0),
        scene_metrics=scene_metrics,
        error_type_distribution=data.get("error_type_distribution", {}),
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the analyze_results CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze ASR model evaluation results from a test set."
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Path to the model (informational, recorded in output).",
    )
    parser.add_argument(
        "--test-set",
        required=True,
        help=(
            "Path to the test set JSONL file. Each line: "
            '{"audio_path": "...", "reference_text": "...", '
            '"hypothesis_text": "...", "scene_type": "..."}'
        ),
    )
    parser.add_argument(
        "--badcase-threshold",
        type=float,
        default=0.3,
        help="CER threshold for badcase generation (default: 0.3).",
    )
    parser.add_argument(
        "--baseline-report",
        default=None,
        help="Path to a baseline evaluation report JSON for comparison.",
    )

    args = parser.parse_args(argv)

    # Load test samples
    samples = _load_samples(args.test_set)

    # Run evaluation
    analyzer = ResultAnalyzer(samples)
    report = analyzer.evaluate_model(
        model_path=args.model_path,
        test_set_path=args.test_set,
    )

    # Generate badcases with the specified threshold
    badcases = analyzer.generate_badcases(threshold=args.badcase_threshold)

    # Classify errors across badcases
    error_distribution = analyzer.classify_errors_from_badcases(badcases)

    # Build output
    output: dict = {
        "model_path": args.model_path,
        "test_set_path": args.test_set,
        "overall_cer": report.overall_cer,
        "overall_wer": report.overall_wer,
        "scene_metrics": [asdict(sm) for sm in report.scene_metrics],
        "error_type_distribution": report.error_type_distribution,
        "badcase_threshold": args.badcase_threshold,
        "badcase_count": len(badcases),
        "badcases": [asdict(bc) for bc in badcases],
        "badcase_error_distribution": error_distribution,
        "total_samples": len(samples),
    }

    # Optional baseline comparison
    if args.baseline_report:
        baseline = _load_baseline_report(args.baseline_report)
        comparison = analyzer.compare_with_baseline(baseline, report)
        output["baseline_comparison"] = comparison

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
