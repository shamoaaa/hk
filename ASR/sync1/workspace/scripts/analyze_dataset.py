"""CLI script for analyzing ASR dataset quality.

Usage:
    python scripts/analyze_dataset.py --dataset-path data/train.jsonl
    python scripts/analyze_dataset.py --dataset-path data/train.jsonl --check-audio-exists

Reads a FunASR jsonl file (or legacy manifest) and outputs a structured
JSON dataset quality report to stdout.

Expected FunASR jsonl format (one JSON object per line):
    {"key": "...", "source": "...", "source_len": N, "target": "...", "target_len": M}

Analyses performed (metadata only — does NOT load audio):
- Basic stats: sample count, estimated total/avg duration, duration distribution
- Text stats: character distribution, word frequency, sentence length
- Anomaly detection: empty labels, missing source_len, char/frame ratio anomaly
- (Optional) missing audio file check with --check-audio-exists
- Scene distribution: only when ``scene_type`` field is present in samples
- SNR analysis: only when ``snr`` field is present in samples

Requirements: 3.1, 3.3, 3.4, 3.6
"""

import argparse
import json
import sys
from dataclasses import asdict

from scripts.analysis.dataset_analyzer import DatasetAnalyzer


def main(argv: list[str] | None = None) -> None:
    """Entry point for the analyze_dataset CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze ASR dataset quality from a FunASR jsonl file."
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help=(
            "Path to a .jsonl dataset file (FunASR format preferred), "
            "or a directory containing manifest.jsonl."
        ),
    )
    parser.add_argument(
        "--min-chars-per-frame-ratio",
        type=float,
        default=0.03,
        help=(
            "Minimum char/source_len ratio below which samples are flagged "
            "as potentially mis-annotated (default: 0.03, roughly 1 char per 2s)."
        ),
    )
    parser.add_argument(
        "--max-chars-per-frame-ratio",
        type=float,
        default=0.5,
        help=(
            "Maximum char/source_len ratio above which samples are flagged "
            "as potentially mis-annotated (default: 0.5, roughly 8 chars/s)."
        ),
    )
    parser.add_argument(
        "--check-audio-exists",
        action="store_true",
        help=(
            "If set, verify that audio files referenced by 'source' exist on disk. "
            "Slower but catches missing files in the anomaly report."
        ),
    )
    parser.add_argument(
        "--min-scene-samples",
        type=int,
        default=None,
        help=(
            "Minimum samples per scene for underrepresentation flagging "
            "(only applies if samples have 'scene_type' field)."
        ),
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=None,
        help=(
            "SNR threshold (dB). Samples with 'snr' field below this are "
            "flagged as low-quality. Only applies when samples carry 'snr'."
        ),
    )
    parser.add_argument(
        "--chars-per-second-min",
        type=float,
        default=1.0,
        help="Minimum chars/second for duration-mismatch detection.",
    )
    parser.add_argument(
        "--chars-per-second-max",
        type=float,
        default=25.0,
        help="Maximum chars/second for duration-mismatch detection.",
    )

    args = parser.parse_args(argv)

    analyzer = DatasetAnalyzer(
        min_chars_per_frame_ratio=args.min_chars_per_frame_ratio,
        max_chars_per_frame_ratio=args.max_chars_per_frame_ratio,
        check_audio_exists=args.check_audio_exists,
        chars_per_second_range=(args.chars_per_second_min, args.chars_per_second_max),
        snr_threshold=args.snr_threshold,
        min_scene_samples=args.min_scene_samples,
    )

    report = analyzer.generate_quality_report(dataset_path=args.dataset_path)

    output = asdict(report)
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
