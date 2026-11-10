"""Tests for the collect_metrics CLI script.

Covers:
- CLI with valid log directory containing sample metrics
- CLI with empty log directory
- CLI missing required --log-dir argument
"""

import json
import os
import sys

import pytest

from scripts.core.config_utils import write_jsonl
from scripts.collect_metrics import main


class TestCollectMetricsCLI:
    """Tests for the collect_metrics.py CLI entry point."""

    def test_cli_with_valid_log_dir(self, tmp_workspace, capsys):
        """Run CLI with a valid JSONL file and verify JSON output structure."""
        log_dir = str(tmp_workspace / "logs" / "training")
        jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")

        records = [
            {
                "step": 0,
                "epoch": 0,
                "train_loss": 2.5,
                "val_loss": 2.8,
                "learning_rate": 1e-4,
                "gradient_norm": 1.2,
                "cer": 0.45,
                "wer": 0.55,
                "timestamp": "2024-01-01T00:00:00",
            },
            {
                "step": 1,
                "epoch": 0,
                "train_loss": 1.8,
                "val_loss": 2.1,
                "learning_rate": 1e-4,
                "gradient_norm": 0.9,
                "cer": 0.35,
                "wer": 0.45,
                "timestamp": "2024-01-01T00:10:00",
            },
            {
                "step": 2,
                "epoch": 1,
                "train_loss": 1.2,
                "val_loss": 1.5,
                "learning_rate": 5e-5,
                "gradient_norm": 0.7,
                "cer": 0.25,
                "wer": 0.35,
                "timestamp": "2024-01-01T00:20:00",
            },
        ]
        write_jsonl(jsonl_path, records)

        main(["--log-dir", log_dir])

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        # Verify top-level keys
        assert "metrics_summary" in output
        assert "training_curves" in output
        assert "alerts" in output
        assert "total_steps" in output
        assert "total_epochs" in output

        # Verify totals
        assert output["total_steps"] == 3
        assert output["total_epochs"] == 2

        # Verify metrics_summary has entries for each epoch
        assert len(output["metrics_summary"]) == 2
        epoch0 = output["metrics_summary"][0]
        assert epoch0["epoch"] == 0
        assert epoch0["sample_count"] == 2
        assert epoch0["avg_cer"] == pytest.approx(0.40)
        assert epoch0["avg_wer"] == pytest.approx(0.50)

        epoch1 = output["metrics_summary"][1]
        assert epoch1["epoch"] == 1
        assert epoch1["sample_count"] == 1
        assert epoch1["avg_cer"] == pytest.approx(0.25)
        assert epoch1["avg_wer"] == pytest.approx(0.35)

        # Verify training_curves structure
        curves = output["training_curves"]
        assert "loss" in curves
        assert "cer_wer" in curves
        assert "learning_rate" in curves
        assert len(curves["loss"]["steps"]) == 3
        assert len(curves["cer_wer"]["steps"]) == 3
        assert len(curves["learning_rate"]["steps"]) == 3

    def test_cli_with_empty_log_dir(self, tmp_workspace, capsys):
        """Run CLI with an empty log dir (no JSONL file) and verify valid JSON output."""
        log_dir = str(tmp_workspace / "logs" / "training")

        main(["--log-dir", log_dir])

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["total_steps"] == 0
        assert output["total_epochs"] == 0
        assert output["metrics_summary"] == []
        assert output["alerts"] == []
        assert output["training_curves"]["loss"]["steps"] == []

    def test_cli_missing_log_dir(self):
        """Running without --log-dir should exit with an error."""
        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code != 0
