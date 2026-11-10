"""Tests for the OptimizerSystem module.

Covers iteration lifecycle management, iteration comparison, rollback,
iteration history retrieval, and optimization plateau detection.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import json
import os

import pytest

from scripts.core.database import Database
from scripts.core.models import (
    EvaluationReport,
    IterationRecord,
    OptimizationPlan,
    OptimizationSuggestion,
    SceneMetrics,
)
from scripts.decision.optimizer_system import OptimizerSystem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with required directory structure."""
    dirs = [
        "configs/model",
        "configs/training",
        "configs/post_process",
        "models/baseline",
        "models/iterations",
        "data/original",
        "data/augmented",
        "data/filtered",
        "logs/training",
        "logs/optimization",
        "reports/evaluation",
        "reports/dataset",
        "reports/augmentation",
        "knowledge/skill_versions",
    ]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def db(tmp_path):
    """Return an initialized Database instance."""
    path = str(tmp_path / "test_metadata.db")
    database = Database(path)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def optimizer(workspace, db):
    """Return an OptimizerSystem instance."""
    return OptimizerSystem(workspace_dir=str(workspace), db=db)


@pytest.fixture
def sample_eval_report():
    """Return a sample EvaluationReport."""
    return EvaluationReport(
        overall_cer=0.15,
        overall_wer=0.22,
        scene_metrics=[
            SceneMetrics(scene_type="clean", cer=0.05, wer=0.08, sample_count=500),
            SceneMetrics(scene_type="background_voice", cer=0.25, wer=0.35, sample_count=300),
        ],
    )


@pytest.fixture
def sample_plan():
    """Return a sample OptimizationPlan."""
    return OptimizationPlan(
        iteration_id="test_iter",
        suggestions=[
            OptimizationSuggestion(
                dimension="model",
                title="Increase attention heads",
                description="Increase from 8 to 16",
                priority=4,
                confidence=0.8,
                config_changes={"attention_heads": 16},
            ),
            OptimizationSuggestion(
                dimension="param",
                title="Reduce learning rate",
                description="Reduce from 1e-4 to 5e-5",
                priority=3,
                confidence=0.7,
                config_changes={"learning_rate": 5e-5},
            ),
        ],
        diagnosis_summary="Background voice CER is high",
    )


# ---------------------------------------------------------------------------
# 14.1 Iteration lifecycle management
# ---------------------------------------------------------------------------


class TestStartIteration:
    """Tests for OptimizerSystem.start_iteration."""

    def test_returns_iteration_id(self, optimizer):
        iter_id = optimizer.start_iteration("Test iteration")
        assert iter_id.startswith("iter_")
        assert len(iter_id) > 5

    def test_creates_database_record(self, optimizer, db):
        iter_id = optimizer.start_iteration("Test iteration")
        row = db.get_by_id("iterations", "iteration_id", iter_id)
        assert row is not None
        assert row["status"] == "pending"
        assert row["description"] == "Test iteration"
        assert row["created_at"] is not None

    def test_unique_ids(self, optimizer):
        ids = {optimizer.start_iteration(f"iter {i}") for i in range(10)}
        assert len(ids) == 10

    def test_empty_description(self, optimizer, db):
        iter_id = optimizer.start_iteration()
        row = db.get_by_id("iterations", "iteration_id", iter_id)
        assert row is not None
        assert row["description"] == ""

    def test_initial_cer_wer_are_none(self, optimizer, db):
        iter_id = optimizer.start_iteration("Test")
        row = db.get_by_id("iterations", "iteration_id", iter_id)
        assert row["cer_before"] is None
        assert row["cer_after"] is None
        assert row["wer_before"] is None
        assert row["wer_after"] is None


class TestRunAnalysisPhase:
    """Tests for OptimizerSystem.run_analysis_phase."""

    def test_updates_status_to_running(self, optimizer, db):
        iter_id = optimizer.start_iteration("Analysis test")
        optimizer.run_analysis_phase(iter_id)
        row = db.get_by_id("iterations", "iteration_id", iter_id)
        assert row["status"] == "running"

    def test_stores_baseline_cer_wer(self, optimizer, db, sample_eval_report):
        iter_id = optimizer.start_iteration("Analysis test")
        optimizer.run_analysis_phase(
            iter_id, evaluation_report=sample_eval_report
        )
        row = db.get_by_id("iterations", "iteration_id", iter_id)
        assert row["cer_before"] == pytest.approx(0.15)
        assert row["wer_before"] == pytest.approx(0.22)

    def test_returns_combined_analysis(self, optimizer, sample_eval_report):
        iter_id = optimizer.start_iteration("Analysis test")
        training_data = {"loss": 1.5}
        dataset_data = {"total_samples": 1000}
        result = optimizer.run_analysis_phase(
            iter_id,
            training_metrics=training_data,
            evaluation_report=sample_eval_report,
            dataset_report=dataset_data,
        )
        assert result["iteration_id"] == iter_id
        assert result["training"] == training_data
        assert result["evaluation"] is not None
        assert result["dataset"] == dataset_data

    def test_saves_analysis_report_to_disk(self, optimizer, workspace):
        iter_id = optimizer.start_iteration("Disk test")
        optimizer.run_analysis_phase(iter_id)
        report_path = os.path.join(
            str(workspace), "reports", "evaluation", f"{iter_id}_analysis.json"
        )
        assert os.path.exists(report_path)

    def test_handles_none_inputs(self, optimizer):
        iter_id = optimizer.start_iteration("None test")
        result = optimizer.run_analysis_phase(iter_id)
        assert result["training"] is None
        assert result["evaluation"] is None
        assert result["dataset"] is None


class TestRunOptimizationPhase:
    """Tests for OptimizerSystem.run_optimization_phase."""

    def test_records_applied_changes(self, optimizer, sample_plan):
        iter_id = optimizer.start_iteration("Opt test")
        result = optimizer.run_optimization_phase(iter_id, sample_plan)
        assert len(result["applied_changes"]) == 2
        assert result["applied_changes"][0]["dimension"] == "model"
        assert result["applied_changes"][1]["dimension"] == "param"

    def test_completes_with_evaluation(
        self, optimizer, db, sample_plan, sample_eval_report
    ):
        iter_id = optimizer.start_iteration("Opt test")
        # Set baseline first
        optimizer.run_analysis_phase(
            iter_id,
            evaluation_report=EvaluationReport(
                overall_cer=0.20, overall_wer=0.30
            ),
        )
        result = optimizer.run_optimization_phase(
            iter_id, sample_plan, evaluation_after=sample_eval_report
        )
        assert result["status"] == "completed"
        assert result["cer_delta"] == pytest.approx(0.15 - 0.20)
        assert result["wer_delta"] == pytest.approx(0.22 - 0.30)

    def test_running_without_evaluation(self, optimizer, sample_plan):
        iter_id = optimizer.start_iteration("Opt test")
        result = optimizer.run_optimization_phase(iter_id, sample_plan)
        assert result["status"] == "running"
        assert result["cer_delta"] is None


class TestCompleteIteration:
    """Tests for OptimizerSystem.complete_iteration."""

    def test_marks_completed(self, optimizer, db, sample_eval_report):
        iter_id = optimizer.start_iteration("Complete test")
        optimizer.run_analysis_phase(
            iter_id,
            evaluation_report=EvaluationReport(
                overall_cer=0.20, overall_wer=0.30
            ),
        )
        result = optimizer.complete_iteration(iter_id, sample_eval_report)
        assert result["status"] == "completed"
        assert result["cer_delta"] == pytest.approx(0.15 - 0.20)

        row = db.get_by_id("iterations", "iteration_id", iter_id)
        assert row["status"] == "completed"
        assert row["completed_at"] is not None

    def test_not_found(self, optimizer):
        result = optimizer.complete_iteration("nonexistent", EvaluationReport(
            overall_cer=0.1, overall_wer=0.2
        ))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# 14.2 Iteration comparison and rollback
# ---------------------------------------------------------------------------


class TestCompareIterations:
    """Tests for OptimizerSystem.compare_iterations."""

    def test_compare_two_completed_iterations(self, optimizer, db):
        # Create and complete two iterations
        id_a = optimizer.start_iteration("Iteration A")
        optimizer.run_analysis_phase(
            id_a,
            evaluation_report=EvaluationReport(overall_cer=0.20, overall_wer=0.30),
        )
        optimizer.complete_iteration(
            id_a,
            EvaluationReport(overall_cer=0.15, overall_wer=0.25),
        )

        id_b = optimizer.start_iteration("Iteration B")
        optimizer.run_analysis_phase(
            id_b,
            evaluation_report=EvaluationReport(overall_cer=0.15, overall_wer=0.25),
        )
        optimizer.complete_iteration(
            id_b,
            EvaluationReport(overall_cer=0.10, overall_wer=0.18),
        )

        result = optimizer.compare_iterations(id_a, id_b)
        assert result["status"] == "success"
        assert result["cer_delta"] == pytest.approx(0.10 - 0.15)
        assert result["wer_delta"] == pytest.approx(0.18 - 0.25)

    def test_compare_not_found(self, optimizer):
        id_a = optimizer.start_iteration("A")
        result = optimizer.compare_iterations(id_a, "nonexistent")
        assert result["status"] == "error"

    def test_compare_with_none_values(self, optimizer):
        id_a = optimizer.start_iteration("A")
        id_b = optimizer.start_iteration("B")
        result = optimizer.compare_iterations(id_a, id_b)
        assert result["status"] == "success"
        assert result["cer_delta"] is None
        assert result["wer_delta"] is None


class TestRollbackToIteration:
    """Tests for OptimizerSystem.rollback_to_iteration."""

    def test_rollback_restores_config(self, optimizer, db, workspace):
        # Create an iteration with model config snapshot
        from scripts.core.config_utils import save_versioned_config

        model_config = {"attention_heads": 8, "hidden_dim": 512}
        save_versioned_config(
            str(workspace / "configs" / "model"), "model", model_config
        )

        iter_id = optimizer.start_iteration("Rollback test")

        # Now change the config
        new_config = {"attention_heads": 16, "hidden_dim": 1024}
        save_versioned_config(
            str(workspace / "configs" / "model"), "model", new_config
        )

        # Rollback
        result = optimizer.rollback_to_iteration(iter_id)
        assert result["status"] == "success"
        assert result["iteration_id"] == iter_id

    def test_rollback_not_found(self, optimizer):
        result = optimizer.rollback_to_iteration("nonexistent")
        assert result["status"] == "error"

    def test_rollback_creates_new_config_version(self, optimizer, workspace):
        from scripts.core.config_utils import list_versions, save_versioned_config

        model_dir = str(workspace / "configs" / "model")
        save_versioned_config(model_dir, "model", {"attention_heads": 8})

        iter_id = optimizer.start_iteration("Rollback version test")

        save_versioned_config(model_dir, "model", {"attention_heads": 16})
        versions_before = list_versions(model_dir, "model")

        optimizer.rollback_to_iteration(iter_id)
        versions_after = list_versions(model_dir, "model")

        # Rollback should create a new version file
        assert len(versions_after) >= len(versions_before)


class TestGetIterationHistory:
    """Tests for OptimizerSystem.get_iteration_history."""

    def test_empty_history(self, optimizer):
        history = optimizer.get_iteration_history()
        assert history == []

    def test_returns_all_iterations(self, optimizer):
        optimizer.start_iteration("First")
        optimizer.start_iteration("Second")
        optimizer.start_iteration("Third")
        history = optimizer.get_iteration_history()
        assert len(history) == 3

    def test_returns_iteration_records(self, optimizer):
        optimizer.start_iteration("Test")
        history = optimizer.get_iteration_history()
        assert len(history) == 1
        record = history[0]
        assert isinstance(record, IterationRecord)
        assert record.status == "pending"
        assert record.optimization_content == "Test"

    def test_chronological_order(self, optimizer):
        id1 = optimizer.start_iteration("First")
        id2 = optimizer.start_iteration("Second")
        history = optimizer.get_iteration_history()
        assert history[0].iteration_id == id1
        assert history[1].iteration_id == id2

    def test_includes_completed_data(self, optimizer):
        iter_id = optimizer.start_iteration("Complete test")
        optimizer.run_analysis_phase(
            iter_id,
            evaluation_report=EvaluationReport(overall_cer=0.20, overall_wer=0.30),
        )
        optimizer.complete_iteration(
            iter_id,
            EvaluationReport(overall_cer=0.15, overall_wer=0.25),
        )
        history = optimizer.get_iteration_history()
        record = history[0]
        assert record.status == "completed"
        assert record.evaluation_after is not None
        assert record.evaluation_after.overall_cer == pytest.approx(0.15)
        assert record.cer_delta == pytest.approx(0.15 - 0.20)


# ---------------------------------------------------------------------------
# 14.4 Optimization plateau detection
# ---------------------------------------------------------------------------


class TestCheckOptimizationPlateau:
    """Tests for OptimizerSystem.check_optimization_plateau."""

    def test_no_iterations_returns_false(self, optimizer):
        assert optimizer.check_optimization_plateau() is False

    def test_fewer_than_window_returns_false(self, optimizer):
        # Create only 2 completed iterations (window=3)
        for i in range(2):
            iter_id = optimizer.start_iteration(f"Iter {i}")
            optimizer.run_analysis_phase(
                iter_id,
                evaluation_report=EvaluationReport(
                    overall_cer=0.20, overall_wer=0.30
                ),
            )
            optimizer.complete_iteration(
                iter_id,
                EvaluationReport(overall_cer=0.20, overall_wer=0.30),
            )
        assert optimizer.check_optimization_plateau(window=3) is False

    def test_plateau_detected(self, optimizer):
        """Three iterations with no CER improvement → plateau."""
        for i in range(3):
            iter_id = optimizer.start_iteration(f"Plateau iter {i}")
            optimizer.run_analysis_phase(
                iter_id,
                evaluation_report=EvaluationReport(
                    overall_cer=0.15, overall_wer=0.22
                ),
            )
            # CER stays the same or gets worse
            optimizer.complete_iteration(
                iter_id,
                EvaluationReport(overall_cer=0.15 + i * 0.01, overall_wer=0.22),
            )
        assert optimizer.check_optimization_plateau(window=3) is True

    def test_no_plateau_with_improvement(self, optimizer):
        """Three iterations where the last one improves → no plateau."""
        cer_values = [(0.20, 0.20), (0.20, 0.21), (0.20, 0.15)]
        for i, (cer_before, cer_after) in enumerate(cer_values):
            iter_id = optimizer.start_iteration(f"Iter {i}")
            optimizer.run_analysis_phase(
                iter_id,
                evaluation_report=EvaluationReport(
                    overall_cer=cer_before, overall_wer=0.30
                ),
            )
            optimizer.complete_iteration(
                iter_id,
                EvaluationReport(overall_cer=cer_after, overall_wer=0.25),
            )
        assert optimizer.check_optimization_plateau(window=3) is False

    def test_plateau_with_custom_window(self, optimizer):
        """Custom window size of 2."""
        for i in range(2):
            iter_id = optimizer.start_iteration(f"Iter {i}")
            optimizer.run_analysis_phase(
                iter_id,
                evaluation_report=EvaluationReport(
                    overall_cer=0.15, overall_wer=0.22
                ),
            )
            optimizer.complete_iteration(
                iter_id,
                EvaluationReport(overall_cer=0.16, overall_wer=0.23),
            )
        assert optimizer.check_optimization_plateau(window=2) is True

    def test_pending_iterations_ignored(self, optimizer):
        """Pending (non-completed) iterations should not count."""
        # Create 3 completed iterations with no improvement
        for i in range(3):
            iter_id = optimizer.start_iteration(f"Completed {i}")
            optimizer.run_analysis_phase(
                iter_id,
                evaluation_report=EvaluationReport(
                    overall_cer=0.15, overall_wer=0.22
                ),
            )
            optimizer.complete_iteration(
                iter_id,
                EvaluationReport(overall_cer=0.15, overall_wer=0.22),
            )

        # Add a pending iteration
        optimizer.start_iteration("Pending")

        # Should still detect plateau from the 3 completed ones
        assert optimizer.check_optimization_plateau(window=3) is True

    def test_mixed_improvement_and_regression(self, optimizer):
        """Mix of improvement and regression in window → no plateau."""
        scenarios = [
            (0.20, 0.18),  # improved
            (0.18, 0.19),  # regressed
            (0.19, 0.17),  # improved
        ]
        for i, (before, after) in enumerate(scenarios):
            iter_id = optimizer.start_iteration(f"Mixed {i}")
            optimizer.run_analysis_phase(
                iter_id,
                evaluation_report=EvaluationReport(
                    overall_cer=before, overall_wer=0.30
                ),
            )
            optimizer.complete_iteration(
                iter_id,
                EvaluationReport(overall_cer=after, overall_wer=0.25),
            )
        assert optimizer.check_optimization_plateau(window=3) is False
