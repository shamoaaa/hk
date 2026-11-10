"""Tests for the param_tuner module.

Covers:
- generate_config: generating TrainingConfig from optimization suggestions
- apply_config: applying config with versioned persistence
- check_early_anomaly: detecting NaN/Inf and abnormal loss growth
- rollback: restoring previous configurations
- get_param_history: retrieving history from database
- LR schedule strategy support (warmup-decay, cosine annealing, step decay)

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import io
import json
import math
import os
import sys

import pytest

from scripts.core.config_utils import list_versions, read_training_config
from scripts.core.database import Database
from scripts.core.models import (
    LRScheduleType,
    OptimizationSuggestion,
    TrainingConfig,
    TrainingMetrics,
)
from scripts.execution.param_tuner import ParamTuner


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tuner(tmp_workspace):
    """Return a ParamTuner with a valid initial config."""
    config_dir = str(tmp_workspace / "configs" / "training")
    initial = TrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        warmup_steps=2000,
        weight_decay=0.01,
        lr_schedule=LRScheduleType.WARMUP_DECAY,
        max_epochs=50,
        gradient_clip=5.0,
        extra_params={"label_smoothing": 0.1},
    )
    return ParamTuner(config_dir=config_dir, initial_config=initial)


@pytest.fixture
def tuner_with_db(tmp_workspace, db):
    """Return a ParamTuner with database integration."""
    config_dir = str(tmp_workspace / "configs" / "training")
    initial = TrainingConfig(
        learning_rate=1e-4,
        batch_size=32,
        warmup_steps=2000,
        weight_decay=0.01,
        lr_schedule=LRScheduleType.WARMUP_DECAY,
        max_epochs=50,
        gradient_clip=5.0,
    )
    return ParamTuner(config_dir=config_dir, initial_config=initial, db=db)


@pytest.fixture
def config_dir(tmp_workspace):
    """Return the training config directory path."""
    return str(tmp_workspace / "configs" / "training")


def _make_suggestion(
    changes: dict,
    priority: int = 3,
    dimension: str = "param",
) -> OptimizationSuggestion:
    """Helper to create an OptimizationSuggestion."""
    return OptimizationSuggestion(
        dimension=dimension,
        title="Test suggestion",
        description="Test description",
        priority=priority,
        confidence=0.8,
        config_changes=changes,
        rationale="Test rationale",
    )


def _make_metrics(
    step: int, train_loss: float, epoch: int = 0
) -> TrainingMetrics:
    """Helper to create a TrainingMetrics instance."""
    return TrainingMetrics(
        step=step,
        epoch=epoch,
        train_loss=train_loss,
        val_loss=None,
        learning_rate=1e-4,
        gradient_norm=1.0,
    )


# =============================================================================
# generate_config Tests
# =============================================================================


class TestGenerateConfig:
    def test_generate_from_single_suggestion(self, tuner):
        suggestions = [
            _make_suggestion({"learning_rate": 5e-5, "batch_size": 64})
        ]
        result = tuner.generate_config(suggestions)
        assert result.learning_rate == 5e-5
        assert result.batch_size == 64
        # Unchanged fields preserved
        assert result.warmup_steps == 2000
        assert result.weight_decay == 0.01

    def test_generate_preserves_unchanged_fields(self, tuner):
        suggestions = [_make_suggestion({"learning_rate": 2e-4})]
        result = tuner.generate_config(suggestions)
        assert result.learning_rate == 2e-4
        assert result.batch_size == 32
        assert result.warmup_steps == 2000
        assert result.weight_decay == 0.01
        assert result.lr_schedule == LRScheduleType.WARMUP_DECAY
        assert result.max_epochs == 50
        assert result.gradient_clip == 5.0

    def test_generate_with_lr_schedule_change(self, tuner):
        suggestions = [
            _make_suggestion({"lr_schedule": "cosine_annealing"})
        ]
        result = tuner.generate_config(suggestions)
        assert result.lr_schedule == LRScheduleType.COSINE_ANNEALING

    def test_generate_with_step_decay_schedule(self, tuner):
        suggestions = [_make_suggestion({"lr_schedule": "step_decay"})]
        result = tuner.generate_config(suggestions)
        assert result.lr_schedule == LRScheduleType.STEP_DECAY

    def test_generate_with_warmup_decay_schedule(self, tuner):
        suggestions = [_make_suggestion({"lr_schedule": "warmup_decay"})]
        result = tuner.generate_config(suggestions)
        assert result.lr_schedule == LRScheduleType.WARMUP_DECAY

    def test_generate_ignores_non_param_suggestions(self, tuner):
        suggestions = [
            _make_suggestion(
                {"learning_rate": 1e-3}, dimension="model"
            ),
            _make_suggestion(
                {"batch_size": 64}, dimension="data"
            ),
        ]
        result = tuner.generate_config(suggestions)
        # Nothing should change since dimension != "param"
        assert result.learning_rate == 1e-4
        assert result.batch_size == 32

    def test_generate_with_multiple_param_suggestions(self, tuner):
        suggestions = [
            _make_suggestion(
                {"learning_rate": 5e-5}, priority=5
            ),
            _make_suggestion(
                {"learning_rate": 1e-3, "batch_size": 64}, priority=3
            ),
        ]
        # Higher priority applied first, then lower priority overwrites
        result = tuner.generate_config(suggestions)
        assert result.learning_rate == 1e-3  # lower priority applied last
        assert result.batch_size == 64

    def test_generate_with_empty_suggestions(self, tuner):
        result = tuner.generate_config([])
        # Should return a copy of current config
        assert result.learning_rate == tuner.current_config.learning_rate
        assert result.batch_size == tuner.current_config.batch_size

    def test_generate_with_all_fields(self, tuner):
        suggestions = [
            _make_suggestion({
                "learning_rate": 2e-4,
                "batch_size": 16,
                "warmup_steps": 1000,
                "weight_decay": 0.05,
                "lr_schedule": "cosine_annealing",
                "max_epochs": 100,
                "gradient_clip": 3.0,
            })
        ]
        result = tuner.generate_config(suggestions)
        assert result.learning_rate == 2e-4
        assert result.batch_size == 16
        assert result.warmup_steps == 1000
        assert result.weight_decay == 0.05
        assert result.lr_schedule == LRScheduleType.COSINE_ANNEALING
        assert result.max_epochs == 100
        assert result.gradient_clip == 3.0

    def test_generate_does_not_modify_current_config(self, tuner):
        original = tuner.current_config
        suggestions = [
            _make_suggestion({"learning_rate": 9e-9, "batch_size": 999})
        ]
        tuner.generate_config(suggestions)
        # Current config should be unchanged
        assert tuner.current_config.learning_rate == original.learning_rate
        assert tuner.current_config.batch_size == original.batch_size

    def test_generate_with_extra_params(self, tuner):
        suggestions = [
            _make_suggestion({
                "learning_rate": 5e-5,
                "custom_param": "value",
            })
        ]
        result = tuner.generate_config(suggestions)
        assert result.learning_rate == 5e-5
        assert result.extra_params.get("custom_param") == "value"
        # Original extra_params preserved
        assert result.extra_params.get("label_smoothing") == 0.1


# =============================================================================
# apply_config Tests
# =============================================================================


class TestApplyConfig:
    def test_apply_updates_current_config(self, tuner):
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        tuner.apply_config(new_config)
        current = tuner.current_config
        assert current.learning_rate == 5e-5
        assert current.batch_size == 64
        assert current.lr_schedule == LRScheduleType.COSINE_ANNEALING

    def test_apply_saves_to_history(self, tuner):
        original = tuner.current_config
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        tuner.apply_config(new_config)
        history = tuner.history
        assert len(history) == 1
        assert history[0].learning_rate == original.learning_rate

    def test_apply_persists_versioned_yaml(self, tuner, config_dir):
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        filepath = tuner.apply_config(new_config)
        assert os.path.exists(filepath)

        versions = list_versions(config_dir, "train")
        assert len(versions) == 1

        loaded = read_training_config(versions[0])
        assert loaded.learning_rate == 5e-5
        assert loaded.batch_size == 64
        assert loaded.lr_schedule == LRScheduleType.COSINE_ANNEALING

    def test_apply_multiple_configs_creates_versions(self, tuner, config_dir):
        for lr in [1e-4, 5e-5, 2e-5]:
            config = TrainingConfig(
                learning_rate=lr,
                batch_size=32,
                warmup_steps=2000,
                weight_decay=0.01,
                lr_schedule=LRScheduleType.WARMUP_DECAY,
                max_epochs=50,
                gradient_clip=5.0,
            )
            tuner.apply_config(config)

        versions = list_versions(config_dir, "train")
        assert len(versions) == 3

    def test_apply_records_change_history(self, tuner):
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        tuner.apply_config(new_config)
        history = tuner.get_change_history()
        assert len(history) == 1
        assert history[0]["operation"] == "apply_config"
        assert "timestamp" in history[0]

    def test_apply_with_db_saves_param_history(self, tuner_with_db, db):
        # Insert the iteration record first to satisfy FK constraint
        db.insert("iterations", {
            "iteration_id": "iter_001",
            "status": "running",
            "created_at": "2024-01-01T00:00:00",
        })
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        tuner_with_db.apply_config(new_config, iteration_id="iter_001")
        history = tuner_with_db.get_param_history()
        assert len(history) == 1
        assert history[0].config.learning_rate == 5e-5
        assert history[0].iteration_id == "iter_001"

    def test_apply_returns_filepath(self, tuner):
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.WARMUP_DECAY,
            max_epochs=50,
            gradient_clip=5.0,
        )
        filepath = tuner.apply_config(new_config)
        assert filepath.endswith(".yaml")
        assert "train" in filepath


# =============================================================================
# check_early_anomaly Tests
# =============================================================================


class TestCheckEarlyAnomaly:
    def test_nan_loss_detected(self, tuner):
        metrics = [
            _make_metrics(0, 2.5),
            _make_metrics(10, 2.3),
            _make_metrics(50, float("nan")),
        ]
        assert tuner.check_early_anomaly(metrics) is True

    def test_inf_loss_detected(self, tuner):
        metrics = [
            _make_metrics(0, 2.5),
            _make_metrics(50, float("inf")),
        ]
        assert tuner.check_early_anomaly(metrics) is True

    def test_negative_inf_loss_detected(self, tuner):
        metrics = [
            _make_metrics(0, 2.5),
            _make_metrics(50, float("-inf")),
        ]
        assert tuner.check_early_anomaly(metrics) is True

    def test_abnormal_growth_detected(self, tuner):
        metrics = [
            _make_metrics(0, 1.0),
            _make_metrics(50, 5.0),
            _make_metrics(80, 15.0),  # 15x growth from initial 1.0
        ]
        assert tuner.check_early_anomaly(metrics) is True

    def test_normal_training_no_anomaly(self, tuner):
        metrics = [
            _make_metrics(0, 2.5),
            _make_metrics(20, 2.3),
            _make_metrics(40, 2.1),
            _make_metrics(60, 1.9),
            _make_metrics(80, 1.7),
            _make_metrics(100, 1.5),
        ]
        assert tuner.check_early_anomaly(metrics) is False

    def test_empty_metrics_no_anomaly(self, tuner):
        assert tuner.check_early_anomaly([]) is False

    def test_only_considers_early_steps(self, tuner):
        metrics = [
            _make_metrics(0, 2.5),
            _make_metrics(50, 2.3),
            _make_metrics(100, 2.1),
            # Step 200 is beyond threshold, NaN should be ignored
            _make_metrics(200, float("nan")),
        ]
        assert tuner.check_early_anomaly(metrics) is False

    def test_custom_step_threshold(self, tuner):
        metrics = [
            _make_metrics(0, 2.5),
            _make_metrics(50, float("nan")),
        ]
        # With threshold=30, step 50 is excluded
        assert tuner.check_early_anomaly(metrics, step_threshold=30) is False
        # With threshold=50, step 50 is included
        assert tuner.check_early_anomaly(metrics, step_threshold=50) is True

    def test_moderate_growth_no_anomaly(self, tuner):
        metrics = [
            _make_metrics(0, 1.0),
            _make_metrics(50, 3.0),
            _make_metrics(100, 5.0),  # 5x growth, below 10x threshold
        ]
        assert tuner.check_early_anomaly(metrics) is False

    def test_single_metric_no_anomaly(self, tuner):
        metrics = [_make_metrics(0, 2.5)]
        assert tuner.check_early_anomaly(metrics) is False

    def test_nan_at_step_zero(self, tuner):
        metrics = [_make_metrics(0, float("nan"))]
        assert tuner.check_early_anomaly(metrics) is True


# =============================================================================
# Rollback Tests
# =============================================================================


class TestRollback:
    def test_rollback_restores_previous(self, tuner):
        original = tuner.current_config
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        tuner.apply_config(new_config)
        restored = tuner.rollback()
        assert restored.learning_rate == original.learning_rate
        assert restored.batch_size == original.batch_size
        assert restored.lr_schedule == original.lr_schedule

    def test_rollback_empty_history_raises(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        pt = ParamTuner(config_dir=config_dir)
        with pytest.raises(ValueError, match="No previous configuration"):
            pt.rollback()

    def test_multi_step_rollback(self, tuner):
        configs = [tuner.current_config]
        for lr in [5e-5, 2e-5, 1e-5]:
            config = TrainingConfig(
                learning_rate=lr,
                batch_size=32,
                warmup_steps=2000,
                weight_decay=0.01,
                lr_schedule=LRScheduleType.WARMUP_DECAY,
                max_epochs=50,
                gradient_clip=5.0,
            )
            tuner.apply_config(config)
            configs.append(tuner.current_config)

        # Roll back 3 times
        for i in range(3, 0, -1):
            restored = tuner.rollback()
            expected = configs[i - 1]
            assert restored.learning_rate == expected.learning_rate

    def test_rollback_persists_versioned_yaml(self, tuner, config_dir):
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        tuner.apply_config(new_config)
        versions_before = list_versions(config_dir, "train")
        assert len(versions_before) == 1

        tuner.rollback()
        versions_after = list_versions(config_dir, "train")
        assert len(versions_after) == 2

        # Latest version should contain the rolled-back config
        latest = read_training_config(versions_after[-1])
        assert latest.learning_rate == 1e-4  # original value

    def test_rollback_records_change_history(self, tuner):
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        tuner.apply_config(new_config)
        tuner.rollback()
        history = tuner.get_change_history()
        assert len(history) == 2
        assert history[1]["operation"] == "rollback"

    def test_rollback_exhausts_history(self, tuner):
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.WARMUP_DECAY,
            max_epochs=50,
            gradient_clip=5.0,
        )
        tuner.apply_config(new_config)
        tuner.rollback()
        assert len(tuner.history) == 0
        with pytest.raises(ValueError, match="No previous configuration"):
            tuner.rollback()

    def test_rollback_then_new_apply(self, tuner, config_dir):
        new_config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.WARMUP_DECAY,
            max_epochs=50,
            gradient_clip=5.0,
        )
        tuner.apply_config(new_config)
        tuner.rollback()

        another_config = TrainingConfig(
            learning_rate=2e-4,
            batch_size=16,
            warmup_steps=500,
            weight_decay=0.005,
            lr_schedule=LRScheduleType.STEP_DECAY,
            max_epochs=80,
            gradient_clip=3.0,
        )
        tuner.apply_config(another_config)
        current = tuner.current_config
        assert current.learning_rate == 2e-4
        assert current.lr_schedule == LRScheduleType.STEP_DECAY

        versions = list_versions(config_dir, "train")
        assert len(versions) == 3  # apply + rollback + apply


# =============================================================================
# get_param_history Tests
# =============================================================================


class TestGetParamHistory:
    def test_empty_history_without_db(self, tuner):
        assert tuner.get_param_history() == []

    def test_history_from_db(self, tuner_with_db, db):
        # Insert iteration records to satisfy FK constraints
        db.insert("iterations", {
            "iteration_id": "iter_001",
            "status": "running",
            "created_at": "2024-01-01T00:00:00",
        })
        db.insert("iterations", {
            "iteration_id": "iter_002",
            "status": "running",
            "created_at": "2024-01-01T01:00:00",
        })
        config1 = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        config2 = TrainingConfig(
            learning_rate=2e-5,
            batch_size=128,
            warmup_steps=500,
            weight_decay=0.01,
            lr_schedule=LRScheduleType.STEP_DECAY,
            max_epochs=80,
            gradient_clip=2.0,
        )
        tuner_with_db.apply_config(config1, iteration_id="iter_001")
        tuner_with_db.apply_config(config2, iteration_id="iter_002")

        history = tuner_with_db.get_param_history()
        assert len(history) == 2
        assert history[0].config.learning_rate == 5e-5
        assert history[0].iteration_id == "iter_001"
        assert history[1].config.learning_rate == 2e-5
        assert history[1].iteration_id == "iter_002"

    def test_history_preserves_lr_schedule(self, tuner_with_db, db):
        # Insert iteration record to satisfy FK constraint
        db.insert("iterations", {
            "iteration_id": "iter_001",
            "status": "running",
            "created_at": "2024-01-01T00:00:00",
        })
        config = TrainingConfig(
            learning_rate=5e-5,
            batch_size=64,
            warmup_steps=1000,
            weight_decay=0.02,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=100,
            gradient_clip=3.0,
        )
        tuner_with_db.apply_config(config, iteration_id="iter_001")
        history = tuner_with_db.get_param_history()
        assert history[0].config.lr_schedule == LRScheduleType.COSINE_ANNEALING


# =============================================================================
# LR Schedule Strategy Tests
# =============================================================================


class TestLRScheduleStrategies:
    """Verify all three LR schedule strategies are supported."""

    def test_warmup_decay_roundtrip(self, tuner, config_dir):
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=32,
            warmup_steps=2000,
            weight_decay=0.01,
            lr_schedule=LRScheduleType.WARMUP_DECAY,
            max_epochs=50,
            gradient_clip=5.0,
        )
        tuner.apply_config(config)
        versions = list_versions(config_dir, "train")
        loaded = read_training_config(versions[-1])
        assert loaded.lr_schedule == LRScheduleType.WARMUP_DECAY

    def test_cosine_annealing_roundtrip(self, tuner, config_dir):
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=32,
            warmup_steps=2000,
            weight_decay=0.01,
            lr_schedule=LRScheduleType.COSINE_ANNEALING,
            max_epochs=50,
            gradient_clip=5.0,
        )
        tuner.apply_config(config)
        versions = list_versions(config_dir, "train")
        loaded = read_training_config(versions[-1])
        assert loaded.lr_schedule == LRScheduleType.COSINE_ANNEALING

    def test_step_decay_roundtrip(self, tuner, config_dir):
        config = TrainingConfig(
            learning_rate=1e-4,
            batch_size=32,
            warmup_steps=2000,
            weight_decay=0.01,
            lr_schedule=LRScheduleType.STEP_DECAY,
            max_epochs=50,
            gradient_clip=5.0,
        )
        tuner.apply_config(config)
        versions = list_versions(config_dir, "train")
        loaded = read_training_config(versions[-1])
        assert loaded.lr_schedule == LRScheduleType.STEP_DECAY


# =============================================================================
# Default Config Tests
# =============================================================================


class TestDefaultConfig:
    def test_default_config_values(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        pt = ParamTuner(config_dir=config_dir)
        config = pt.current_config
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.warmup_steps == 2000
        assert config.weight_decay == 0.01
        assert config.lr_schedule == LRScheduleType.WARMUP_DECAY
        assert config.max_epochs == 50
        assert config.gradient_clip == 5.0

    def test_current_config_is_independent_copy(self, tuner):
        config = tuner.current_config
        config.learning_rate = 999.0
        assert tuner.current_config.learning_rate == 1e-4  # unchanged


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_generate_then_apply(self, tuner, config_dir):
        """Full workflow: generate config from suggestions, then apply."""
        suggestions = [
            _make_suggestion({
                "learning_rate": 5e-5,
                "batch_size": 64,
                "lr_schedule": "cosine_annealing",
            })
        ]
        new_config = tuner.generate_config(suggestions)
        filepath = tuner.apply_config(new_config)

        assert os.path.exists(filepath)
        current = tuner.current_config
        assert current.learning_rate == 5e-5
        assert current.batch_size == 64
        assert current.lr_schedule == LRScheduleType.COSINE_ANNEALING

    def test_apply_detect_anomaly_rollback(self, tuner):
        """Apply config, detect anomaly, then rollback."""
        original = tuner.current_config
        new_config = TrainingConfig(
            learning_rate=0.1,  # Very high LR
            batch_size=32,
            warmup_steps=0,
            weight_decay=0.01,
            lr_schedule=LRScheduleType.WARMUP_DECAY,
            max_epochs=50,
            gradient_clip=5.0,
        )
        tuner.apply_config(new_config)

        # Simulate anomalous training metrics
        metrics = [
            _make_metrics(0, 2.5),
            _make_metrics(10, 5.0),
            _make_metrics(20, float("nan")),
        ]
        assert tuner.check_early_anomaly(metrics) is True

        # Rollback
        restored = tuner.rollback()
        assert restored.learning_rate == original.learning_rate


# =============================================================================
# CLI Tests (task 9.4)
# =============================================================================


class TestCLI:
    """Tests for the apply_training_config.py CLI script."""

    def _capture_main(self, argv: list[str]) -> dict:
        """Run CLI main and capture JSON output."""
        from scripts.apply_training_config import main

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(argv)
        finally:
            sys.stdout = old_stdout
        return json.loads(captured.getvalue())

    def test_apply_learning_rate(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--learning-rate", "5e-5",
        ])
        assert output["status"] == "success"
        assert output["config"]["learning_rate"] == 5e-5
        assert "learning_rate" in output["changes_applied"]

    def test_apply_batch_size(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--batch-size", "64",
        ])
        assert output["status"] == "success"
        assert output["config"]["batch_size"] == 64
        assert "batch_size" in output["changes_applied"]

    def test_apply_lr_schedule(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--lr-schedule", "cosine_annealing",
        ])
        assert output["status"] == "success"
        assert output["config"]["lr_schedule"] == "cosine_annealing"
        assert "lr_schedule" in output["changes_applied"]

    def test_apply_multiple_params(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--learning-rate", "2e-4",
            "--batch-size", "16",
            "--warmup-steps", "1000",
            "--weight-decay", "0.05",
            "--lr-schedule", "step_decay",
            "--max-epochs", "100",
            "--gradient-clip", "3.0",
        ])
        assert output["status"] == "success"
        assert output["config"]["learning_rate"] == 2e-4
        assert output["config"]["batch_size"] == 16
        assert output["config"]["warmup_steps"] == 1000
        assert output["config"]["weight_decay"] == 0.05
        assert output["config"]["lr_schedule"] == "step_decay"
        assert output["config"]["max_epochs"] == 100
        assert output["config"]["gradient_clip"] == 3.0

    def test_apply_no_changes(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "apply",
            "--config-dir", config_dir,
        ])
        assert output["status"] == "noop"

    def test_show_empty(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "show",
            "--config-dir", config_dir,
        ])
        assert output["status"] in ("success", "empty")

    def test_show_after_apply(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--learning-rate", "5e-5",
            "--batch-size", "64",
        ])
        output = self._capture_main([
            "show",
            "--config-dir", config_dir,
        ])
        assert output["status"] == "success"
        assert output["config"]["learning_rate"] == 5e-5
        assert output["config"]["batch_size"] == 64

    def test_history_empty(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "history",
            "--config-dir", config_dir,
        ])
        assert output["status"] == "success"
        assert output["total_versions"] == 0

    def test_history_after_multiple_applies(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--learning-rate", "1e-4",
        ])
        self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--learning-rate", "5e-5",
        ])
        output = self._capture_main([
            "history",
            "--config-dir", config_dir,
        ])
        assert output["status"] == "success"
        assert output["total_versions"] == 2

    def test_rollback_not_enough_versions(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "rollback",
            "--config-dir", config_dir,
        ])
        assert output["status"] == "error"
        assert "Not enough" in output["message"]

    def test_rollback_success(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        # Apply two configs
        self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--learning-rate", "1e-4",
            "--batch-size", "32",
        ])
        self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--learning-rate", "5e-5",
            "--batch-size", "64",
        ])
        # Rollback
        output = self._capture_main([
            "rollback",
            "--config-dir", config_dir,
        ])
        assert output["status"] == "success"
        assert output["config"]["learning_rate"] == 1e-4
        assert output["config"]["batch_size"] == 32

    def test_apply_persists_config_file(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "training")
        output = self._capture_main([
            "apply",
            "--config-dir", config_dir,
            "--learning-rate", "5e-5",
        ])
        assert output["status"] == "success"
        assert "config_file" in output
        assert os.path.exists(output["config_file"])
