"""Tests for the TrainingMonitor class.

Covers:
- collect_metrics stores to memory and JSONL file
- compute_epoch_metrics returns correct averages
- compute_epoch_metrics with no matching epoch returns appropriate result
- compute_epoch_metrics filters by epoch correctly
"""

import os

import pytest

from scripts.core.config_utils import read_jsonl
from scripts.analysis.training_monitor import TrainingMonitor


# ---------------------------------------------------------------------------
# collect_metrics
# ---------------------------------------------------------------------------


class TestCollectMetrics:
    """Tests for TrainingMonitor.collect_metrics."""

    def test_stores_to_memory(self, tmp_workspace):
        """Collected metrics are stored in the in-memory list."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        result = monitor.collect_metrics(
            step=0,
            epoch=0,
            raw_metrics={
                "train_loss": 2.5,
                "val_loss": 2.8,
                "learning_rate": 1e-4,
                "gradient_norm": 1.2,
                "cer": 0.45,
                "wer": 0.55,
            },
        )

        assert result is not None
        assert len(monitor.metrics) == 1
        assert monitor.metrics[0].train_loss == 2.5
        assert monitor.metrics[0].cer == 0.45

    def test_stores_to_jsonl(self, tmp_workspace):
        """Collected metrics are persisted to a JSONL file."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        monitor.collect_metrics(
            step=0,
            epoch=0,
            raw_metrics={
                "train_loss": 2.5,
                "val_loss": 2.8,
                "learning_rate": 1e-4,
                "gradient_norm": 1.2,
            },
        )

        jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")
        assert os.path.exists(jsonl_path)

        records = read_jsonl(jsonl_path)
        assert len(records) == 1
        assert records[0]["train_loss"] == 2.5

    def test_skips_non_interval_steps(self, tmp_workspace):
        """Steps that are not multiples of collect_interval are skipped."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        result = monitor.collect_metrics(
            step=50,
            epoch=0,
            raw_metrics={"train_loss": 1.0, "learning_rate": 1e-4, "gradient_norm": 0.5},
        )

        assert result is None
        assert len(monitor.metrics) == 0

    def test_collects_multiple_steps(self, tmp_workspace):
        """Multiple aligned steps are all collected."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        for step in [0, 100, 200]:
            monitor.collect_metrics(
                step=step,
                epoch=0,
                raw_metrics={
                    "train_loss": 2.5 - step * 0.005,
                    "learning_rate": 1e-4,
                    "gradient_norm": 1.0,
                },
            )

        assert len(monitor.metrics) == 3

        jsonl_path = os.path.join(log_dir, "training_metrics.jsonl")
        records = read_jsonl(jsonl_path)
        assert len(records) == 3

    def test_adds_timestamp_when_missing(self, tmp_workspace):
        """A timestamp is auto-generated when not provided in raw_metrics."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        result = monitor.collect_metrics(
            step=0,
            epoch=0,
            raw_metrics={"train_loss": 1.0, "learning_rate": 1e-4, "gradient_norm": 0.5},
        )

        assert result is not None
        assert result.timestamp != ""

    def test_uses_provided_timestamp(self, tmp_workspace):
        """When raw_metrics includes a timestamp, it is used as-is."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        result = monitor.collect_metrics(
            step=0,
            epoch=0,
            raw_metrics={
                "train_loss": 1.0,
                "learning_rate": 1e-4,
                "gradient_norm": 0.5,
                "timestamp": "2024-06-01T12:00:00",
            },
        )

        assert result.timestamp == "2024-06-01T12:00:00"


# ---------------------------------------------------------------------------
# compute_epoch_metrics
# ---------------------------------------------------------------------------


class TestComputeEpochMetrics:
    """Tests for TrainingMonitor.compute_epoch_metrics."""

    def _populate(self, monitor: TrainingMonitor):
        """Insert a known set of metrics across two epochs."""
        data = [
            (0, 0, 0.40, 0.50),
            (100, 0, 0.30, 0.40),
            (200, 1, 0.20, 0.30),
            (300, 1, 0.10, 0.20),
        ]
        for step, epoch, cer, wer in data:
            monitor.collect_metrics(
                step=step,
                epoch=epoch,
                raw_metrics={
                    "train_loss": 1.0,
                    "learning_rate": 1e-4,
                    "gradient_norm": 0.5,
                    "cer": cer,
                    "wer": wer,
                },
            )

    def test_correct_averages(self, tmp_workspace):
        """Average CER and WER are computed correctly for a given epoch."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        self._populate(monitor)

        result = monitor.compute_epoch_metrics(epoch=0)

        assert result["epoch"] == 0
        assert result["sample_count"] == 2
        assert result["avg_cer"] == pytest.approx(0.35)
        assert result["avg_wer"] == pytest.approx(0.45)

    def test_filters_by_epoch(self, tmp_workspace):
        """Only metrics belonging to the requested epoch are included."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        self._populate(monitor)

        result = monitor.compute_epoch_metrics(epoch=1)

        assert result["epoch"] == 1
        assert result["sample_count"] == 2
        assert result["avg_cer"] == pytest.approx(0.15)
        assert result["avg_wer"] == pytest.approx(0.25)

    def test_no_matching_epoch(self, tmp_workspace):
        """When no metrics match the epoch, averages are None and count is 0."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        self._populate(monitor)

        result = monitor.compute_epoch_metrics(epoch=99)

        assert result["epoch"] == 99
        assert result["sample_count"] == 0
        assert result["avg_cer"] is None
        assert result["avg_wer"] is None

    def test_partial_cer_wer(self, tmp_workspace):
        """Metrics with None CER/WER are excluded from the averages."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        # Step 0: has CER/WER
        monitor.collect_metrics(
            step=0,
            epoch=0,
            raw_metrics={
                "train_loss": 1.0,
                "learning_rate": 1e-4,
                "gradient_norm": 0.5,
                "cer": 0.40,
                "wer": 0.50,
            },
        )
        # Step 100: no CER/WER
        monitor.collect_metrics(
            step=100,
            epoch=0,
            raw_metrics={
                "train_loss": 0.8,
                "learning_rate": 1e-4,
                "gradient_norm": 0.4,
            },
        )
        # Step 200: has CER/WER
        monitor.collect_metrics(
            step=200,
            epoch=0,
            raw_metrics={
                "train_loss": 0.6,
                "learning_rate": 1e-4,
                "gradient_norm": 0.3,
                "cer": 0.20,
                "wer": 0.30,
            },
        )

        result = monitor.compute_epoch_metrics(epoch=0)

        # sample_count includes all 3 metrics for the epoch
        assert result["sample_count"] == 3
        # But averages only use the 2 metrics with CER/WER
        assert result["avg_cer"] == pytest.approx(0.30)
        assert result["avg_wer"] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# check_stagnation
# ---------------------------------------------------------------------------


class TestCheckStagnation:
    """Tests for TrainingMonitor.check_stagnation."""

    @staticmethod
    def _populate_epochs(monitor: TrainingMonitor, avg_losses: list[float]):
        """Populate the monitor with one metric per epoch having the given losses."""
        for epoch, loss in enumerate(avg_losses):
            monitor.collect_metrics(
                step=epoch * 100,
                epoch=epoch,
                raw_metrics={
                    "train_loss": loss,
                    "learning_rate": 1e-4,
                    "gradient_norm": 0.5,
                },
            )

    def test_stagnation_detected(self, tmp_workspace):
        """Alert is returned when loss barely decreases over 5 epochs."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        # Loss decreases by only 0.001 each epoch (below threshold of 0.01)
        self._populate_epochs(monitor, [2.0, 1.999, 1.998, 1.997, 1.996])

        alert = monitor.check_stagnation(threshold=0.01, window=5)

        assert alert is not None
        assert alert.alert_type.value == "stagnation"
        assert alert.epoch == 4

    def test_no_stagnation(self, tmp_workspace):
        """No alert when loss decreases significantly."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        # Loss decreases by 0.1 each epoch (above threshold of 0.01)
        self._populate_epochs(monitor, [2.0, 1.9, 1.8, 1.7, 1.6])

        alert = monitor.check_stagnation(threshold=0.01, window=5)

        assert alert is None

    def test_stagnation_insufficient_epochs(self, tmp_workspace):
        """Returns None when fewer than window epochs are available."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        self._populate_epochs(monitor, [2.0, 1.999, 1.998])

        alert = monitor.check_stagnation(threshold=0.01, window=5)

        assert alert is None

    def test_stagnation_custom_window(self, tmp_workspace):
        """Stagnation detected with a custom window of 3."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        # 6 epochs: first 3 decrease well, last 3 stagnate
        self._populate_epochs(monitor, [2.0, 1.5, 1.0, 0.999, 0.998, 0.997])

        alert = monitor.check_stagnation(threshold=0.01, window=3)

        assert alert is not None
        assert alert.alert_type.value == "stagnation"
        assert alert.epoch == 5


# ---------------------------------------------------------------------------
# check_numerical_anomaly
# ---------------------------------------------------------------------------


class TestCheckNumericalAnomaly:
    """Tests for TrainingMonitor.check_numerical_anomaly."""

    def test_nan_detected(self, tmp_workspace):
        """NaN loss returns a NAN_INF alert."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        alert = monitor.check_numerical_anomaly(float("nan"))

        assert alert is not None
        assert alert.alert_type.value == "nan_inf"

    def test_inf_detected(self, tmp_workspace):
        """Positive Inf loss returns a NAN_INF alert."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        alert = monitor.check_numerical_anomaly(float("inf"))

        assert alert is not None
        assert alert.alert_type.value == "nan_inf"

    def test_neg_inf_detected(self, tmp_workspace):
        """Negative Inf loss returns a NAN_INF alert."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        alert = monitor.check_numerical_anomaly(float("-inf"))

        assert alert is not None
        assert alert.alert_type.value == "nan_inf"

    def test_normal_value_no_alert(self, tmp_workspace):
        """A normal finite float returns None."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        alert = monitor.check_numerical_anomaly(1.5)

        assert alert is None


# ---------------------------------------------------------------------------
# get_training_curves
# ---------------------------------------------------------------------------


class TestGetTrainingCurves:
    """Tests for TrainingMonitor.get_training_curves."""

    @staticmethod
    def _populate(monitor: TrainingMonitor):
        """Insert metrics with a mix of full and partial CER/WER data."""
        entries = [
            # step, epoch, train_loss, val_loss, lr, grad_norm, cer, wer
            (0, 0, 2.5, 2.8, 1e-4, 1.2, 0.45, 0.55),
            (100, 0, 1.8, 2.1, 1e-4, 0.9, 0.35, 0.45),
            (200, 1, 1.2, None, 5e-5, 0.7, None, None),
            (300, 1, 0.9, 1.1, 5e-5, 0.5, 0.20, 0.30),
        ]
        for step, epoch, tl, vl, lr, gn, cer, wer in entries:
            raw: dict = {
                "train_loss": tl,
                "learning_rate": lr,
                "gradient_norm": gn,
            }
            if vl is not None:
                raw["val_loss"] = vl
            if cer is not None:
                raw["cer"] = cer
            if wer is not None:
                raw["wer"] = wer
            monitor.collect_metrics(step=step, epoch=epoch, raw_metrics=raw)

    def test_returns_loss_curves(self, tmp_workspace):
        """Loss curve data contains correct steps and values."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        self._populate(monitor)

        curves = monitor.get_training_curves()

        assert curves["loss"]["steps"] == [0, 100, 200, 300]
        assert curves["loss"]["train_loss"] == [2.5, 1.8, 1.2, 0.9]
        assert curves["loss"]["val_loss"] == [2.8, 2.1, None, 1.1]

    def test_returns_cer_wer_curves(self, tmp_workspace):
        """CER/WER curves only include entries where both are not None."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        self._populate(monitor)

        curves = monitor.get_training_curves()

        # Step 200 has cer=None, wer=None → excluded
        assert curves["cer_wer"]["steps"] == [0, 100, 300]
        assert curves["cer_wer"]["cer"] == [0.45, 0.35, 0.20]
        assert curves["cer_wer"]["wer"] == [0.55, 0.45, 0.30]

    def test_returns_learning_rate_curve(self, tmp_workspace):
        """Learning rate curve includes all collected steps."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)
        self._populate(monitor)

        curves = monitor.get_training_curves()

        assert curves["learning_rate"]["steps"] == [0, 100, 200, 300]
        assert curves["learning_rate"]["values"] == [1e-4, 1e-4, 5e-5, 5e-5]

    def test_empty_metrics(self, tmp_workspace):
        """Empty dict structure when no metrics have been collected."""
        log_dir = str(tmp_workspace / "logs" / "training")
        monitor = TrainingMonitor(log_dir=log_dir, collect_interval=100)

        curves = monitor.get_training_curves()

        assert curves["loss"] == {"steps": [], "train_loss": [], "val_loss": []}
        assert curves["cer_wer"] == {"steps": [], "cer": [], "wer": []}
        assert curves["learning_rate"] == {"steps": [], "values": []}
