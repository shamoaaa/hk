"""CLI script for collecting and summarizing training metrics.

Usage:
    python scripts/collect_metrics.py --log-dir logs/training/

Reads training metrics from JSONL files in the specified directory and
outputs a structured JSON summary to stdout.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

from scripts.core.config_utils import read_jsonl
from scripts.core.models import TrainingMetrics
from scripts.analysis.training_monitor import TrainingMonitor


def main(argv: list[str] | None = None) -> None:
    """Entry point for the collect_metrics CLI."""
    parser = argparse.ArgumentParser(
        description="Collect and summarize training metrics from JSONL logs."
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Path to the directory containing training_metrics.jsonl",
    )
    parser.add_argument(
        "--stagnation-threshold",
        type=float,
        default=0.01,
        help="Threshold for stagnation detection (default: 0.01)",
    )
    parser.add_argument(
        "--stagnation-window",
        type=int,
        default=5,
        help="Window size for stagnation detection (default: 5)",
    )

    args = parser.parse_args(argv)

    log_dir: str = args.log_dir
    stagnation_threshold: float = args.stagnation_threshold
    stagnation_window: int = args.stagnation_window

    # Read existing JSONL metrics
    jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")
    if os.path.exists(jsonl_path):
        records = read_jsonl(jsonl_path)
    else:
        records = []

    # Create a TrainingMonitor and populate it with loaded metrics
    monitor = TrainingMonitor(log_dir=log_dir, collect_interval=1)

    for record in records:
        raw_metrics = {
            "train_loss": record.get("train_loss", 0.0),
            "val_loss": record.get("val_loss"),
            "learning_rate": record.get("learning_rate", 0.0),
            "gradient_norm": record.get("gradient_norm", 0.0),
            "cer": record.get("cer"),
            "wer": record.get("wer"),
            "timestamp": record.get("timestamp", ""),
        }
        # Use collect_interval=1 so every step is collected.
        # We feed the original step directly.
        monitor.collect_metrics(
            step=record.get("step", 0),
            epoch=record.get("epoch", 0),
            raw_metrics=raw_metrics,
        )

    # Compute epoch metrics for each unique epoch
    epochs = sorted({m.epoch for m in monitor.metrics})
    metrics_summary = [monitor.compute_epoch_metrics(epoch) for epoch in epochs]

    # Get training curves
    training_curves = monitor.get_training_curves()

    # Check for stagnation and numerical anomalies
    alerts: list[dict] = []

    stagnation_alert = monitor.check_stagnation(
        threshold=stagnation_threshold, window=stagnation_window
    )
    if stagnation_alert is not None:
        alerts.append(asdict(stagnation_alert))

    for m in monitor.metrics:
        anomaly_alert = monitor.check_numerical_anomaly(m.train_loss)
        if anomaly_alert is not None:
            alert_dict = asdict(anomaly_alert)
            alert_dict["step"] = m.step
            alert_dict["epoch"] = m.epoch
            alerts.append(alert_dict)

    output = {
        "metrics_summary": metrics_summary,
        "training_curves": training_curves,
        "alerts": alerts,
        "total_steps": len(monitor.metrics),
        "total_epochs": len(epochs),
    }

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
