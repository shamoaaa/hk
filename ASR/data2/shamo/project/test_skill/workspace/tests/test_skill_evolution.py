"""Tests for the SkillEvolutionEngine module.

Covers effect feedback recording, feedback labeling (positive/negative),
and edge cases such as equal CER values and missing iteration data.

Requirements: 10.1, 10.2
"""

import json
import os

import pytest

from scripts.core.database import Database
from scripts.core.models import EffectFeedback
from scripts.decision.skill_evolution import SkillEvolutionEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    """Create a temporary workspace with required directory structure."""
    dirs = [
        ".kiro/skills",
        "configs/model",
        "configs/training",
        "configs/post_process",
        "knowledge/skill_versions",
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
def engine(workspace, db):
    """Return a SkillEvolutionEngine instance."""
    return SkillEvolutionEngine(workspace_dir=str(workspace), db=db)


@pytest.fixture
def sample_feedback(db):
    """Return a sample EffectFeedback instance.

    Also creates the parent iteration record to satisfy the foreign key
    constraint on the ``effect_feedbacks`` table.
    """
    _create_iteration(db, "iter_test001")
    return EffectFeedback(
        iteration_id="iter_test001",
        suggestion_id="sug_test001",
        suggestion_content="Increase attention heads from 8 to 16",
        input_analysis={"overall_cer": 0.15, "scene": "background_voice"},
        param_changes={"attention_heads": 16},
        cer_before=0.15,
        cer_after=0.12,
        wer_before=0.22,
        wer_after=0.18,
        is_positive=True,
        timestamp="2024-01-01T00:00:00",
    )


def _create_iteration(db, iteration_id, cer_before=0.20, cer_after=0.15,
                       wer_before=0.30, wer_after=0.25, description="Test iteration",
                       param_changes=None, model_changes=None):
    """Helper to insert an iteration record into the database."""
    db.insert("iterations", {
        "iteration_id": iteration_id,
        "description": description,
        "status": "completed",
        "cer_before": cer_before,
        "cer_after": cer_after,
        "wer_before": wer_before,
        "wer_after": wer_after,
        "param_changes_json": json.dumps(param_changes or {}),
        "model_changes_json": json.dumps(model_changes or {}),
        "data_changes_json": json.dumps({}),
        "created_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T01:00:00",
    })


# ---------------------------------------------------------------------------
# 15.1 record_feedback tests
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    """Tests for SkillEvolutionEngine.record_feedback."""

    def test_stores_feedback_in_database(self, engine, db, sample_feedback):
        """record_feedback should persist the feedback to effect_feedbacks table."""
        engine.record_feedback(sample_feedback)

        rows = db.get_all("effect_feedbacks")
        assert len(rows) == 1

    def test_stored_fields_match_input(self, engine, db, sample_feedback):
        """All feedback fields should be correctly stored in the database."""
        engine.record_feedback(sample_feedback)

        rows = db.get_all("effect_feedbacks")
        row = rows[0]

        assert row["iteration_id"] == "iter_test001"
        assert row["suggestion_id"] == "sug_test001"
        assert row["suggestion_content"] == "Increase attention heads from 8 to 16"
        assert row["cer_before"] == pytest.approx(0.15)
        assert row["cer_after"] == pytest.approx(0.12)
        assert row["wer_before"] == pytest.approx(0.22)
        assert row["wer_after"] == pytest.approx(0.18)
        assert row["is_positive"] == 1

    def test_serializes_input_analysis_as_json(self, engine, db, sample_feedback):
        """input_analysis dict should be serialized as JSON string."""
        engine.record_feedback(sample_feedback)

        rows = db.get_all("effect_feedbacks")
        row = rows[0]

        analysis = json.loads(row["input_analysis_json"])
        assert analysis == {"overall_cer": 0.15, "scene": "background_voice"}

    def test_serializes_param_changes_as_json(self, engine, db, sample_feedback):
        """param_changes dict should be serialized as JSON string."""
        engine.record_feedback(sample_feedback)

        rows = db.get_all("effect_feedbacks")
        row = rows[0]

        changes = json.loads(row["param_changes_json"])
        assert changes == {"attention_heads": 16}

    def test_generates_unique_feedback_id(self, engine, db, sample_feedback):
        """Each recorded feedback should get a unique feedback_id."""
        engine.record_feedback(sample_feedback)
        engine.record_feedback(sample_feedback)

        rows = db.get_all("effect_feedbacks")
        assert len(rows) == 2

        ids = {row["feedback_id"] for row in rows}
        assert len(ids) == 2
        for fid in ids:
            assert fid.startswith("fb_")

    def test_stores_timestamp(self, engine, db, sample_feedback):
        """record_feedback should store a created_at timestamp."""
        engine.record_feedback(sample_feedback)

        rows = db.get_all("effect_feedbacks")
        row = rows[0]

        assert row["created_at"] is not None
        assert len(row["created_at"]) > 0

    def test_negative_feedback_stored_correctly(self, engine, db):
        """Negative feedback (is_positive=False) should store is_positive=0."""
        _create_iteration(db, "iter_neg")
        feedback = EffectFeedback(
            iteration_id="iter_neg",
            suggestion_id="sug_neg",
            suggestion_content="Bad suggestion",
            input_analysis={},
            param_changes={},
            cer_before=0.10,
            cer_after=0.15,
            wer_before=0.20,
            wer_after=0.25,
            is_positive=False,
            timestamp="2024-01-01T00:00:00",
        )
        engine.record_feedback(feedback)

        rows = db.get_all("effect_feedbacks")
        assert rows[0]["is_positive"] == 0

    def test_empty_dicts_stored_as_json(self, engine, db):
        """Empty input_analysis and param_changes should serialize as '{}'."""
        _create_iteration(db, "iter_empty")
        feedback = EffectFeedback(
            iteration_id="iter_empty",
            suggestion_id="sug_empty",
            suggestion_content="",
            input_analysis={},
            param_changes={},
            cer_before=0.10,
            cer_after=0.10,
            wer_before=0.20,
            wer_after=0.20,
            is_positive=False,
            timestamp="2024-01-01T00:00:00",
        )
        engine.record_feedback(feedback)

        rows = db.get_all("effect_feedbacks")
        row = rows[0]

        assert json.loads(row["input_analysis_json"]) == {}
        assert json.loads(row["param_changes_json"]) == {}


# ---------------------------------------------------------------------------
# 15.1 label_feedback tests
# ---------------------------------------------------------------------------


class TestLabelFeedback:
    """Tests for SkillEvolutionEngine.label_feedback."""

    def test_positive_feedback_when_cer_improves(self, engine, db):
        """label_feedback should set is_positive=True when cer_after < cer_before."""
        _create_iteration(db, "iter_pos", cer_before=0.20, cer_after=0.15)

        feedback = engine.label_feedback("iter_pos", cer_before=0.20, cer_after=0.15)

        assert feedback.is_positive is True
        assert feedback.cer_before == pytest.approx(0.20)
        assert feedback.cer_after == pytest.approx(0.15)

    def test_negative_feedback_when_cer_worsens(self, engine, db):
        """label_feedback should set is_positive=False when cer_after > cer_before."""
        _create_iteration(db, "iter_neg", cer_before=0.15, cer_after=0.20)

        feedback = engine.label_feedback("iter_neg", cer_before=0.15, cer_after=0.20)

        assert feedback.is_positive is False

    def test_negative_feedback_when_cer_equal(self, engine, db):
        """label_feedback should set is_positive=False when cer_after == cer_before."""
        _create_iteration(db, "iter_eq", cer_before=0.15, cer_after=0.15)

        feedback = engine.label_feedback("iter_eq", cer_before=0.15, cer_after=0.15)

        assert feedback.is_positive is False

    def test_returns_effect_feedback_instance(self, engine, db):
        """label_feedback should return an EffectFeedback dataclass."""
        _create_iteration(db, "iter_type")

        feedback = engine.label_feedback("iter_type", cer_before=0.20, cer_after=0.15)

        assert isinstance(feedback, EffectFeedback)

    def test_populates_iteration_context(self, engine, db):
        """label_feedback should populate context from the iteration record."""
        _create_iteration(
            db, "iter_ctx",
            description="Adjust attention heads",
            param_changes={"learning_rate": 5e-5},
            model_changes={"attention_heads": 16},
            wer_before=0.30,
            wer_after=0.25,
        )

        feedback = engine.label_feedback("iter_ctx", cer_before=0.20, cer_after=0.15)

        assert feedback.iteration_id == "iter_ctx"
        assert feedback.suggestion_content == "Adjust attention heads"
        assert feedback.param_changes == {"learning_rate": 5e-5}
        assert feedback.input_analysis == {"attention_heads": 16}
        assert feedback.wer_before == pytest.approx(0.30)
        assert feedback.wer_after == pytest.approx(0.25)

    def test_missing_iteration_returns_defaults(self, engine, db):
        """label_feedback should handle missing iteration gracefully."""
        feedback = engine.label_feedback("nonexistent", cer_before=0.20, cer_after=0.15)

        assert feedback.is_positive is True
        assert feedback.iteration_id == "nonexistent"
        assert feedback.suggestion_content == ""
        assert feedback.input_analysis == {}
        assert feedback.param_changes == {}
        assert feedback.wer_before == pytest.approx(0.0)
        assert feedback.wer_after == pytest.approx(0.0)

    def test_generates_suggestion_id(self, engine, db):
        """label_feedback should generate a unique suggestion_id."""
        _create_iteration(db, "iter_sugid")

        feedback = engine.label_feedback("iter_sugid", cer_before=0.20, cer_after=0.15)

        assert feedback.suggestion_id.startswith("sug_")
        assert len(feedback.suggestion_id) > 4

    def test_sets_timestamp(self, engine, db):
        """label_feedback should set a timestamp on the feedback."""
        _create_iteration(db, "iter_ts")

        feedback = engine.label_feedback("iter_ts", cer_before=0.20, cer_after=0.15)

        assert feedback.timestamp is not None
        assert len(feedback.timestamp) > 0

    def test_very_small_cer_improvement(self, engine, db):
        """Even a tiny CER improvement should be labeled positive."""
        _create_iteration(db, "iter_tiny")

        feedback = engine.label_feedback(
            "iter_tiny", cer_before=0.100001, cer_after=0.100000
        )

        assert feedback.is_positive is True

    def test_zero_cer_values(self, engine, db):
        """Both CER values at zero should be labeled negative (no improvement)."""
        _create_iteration(db, "iter_zero", cer_before=0.0, cer_after=0.0)

        feedback = engine.label_feedback("iter_zero", cer_before=0.0, cer_after=0.0)

        assert feedback.is_positive is False

    def test_iteration_with_null_wer(self, engine, db):
        """label_feedback should handle iteration with NULL wer values."""
        db.insert("iterations", {
            "iteration_id": "iter_nullwer",
            "description": "No WER data",
            "status": "completed",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": None,
            "wer_after": None,
            "param_changes_json": json.dumps({}),
            "model_changes_json": json.dumps({}),
            "data_changes_json": json.dumps({}),
            "created_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T01:00:00",
        })

        feedback = engine.label_feedback("iter_nullwer", cer_before=0.20, cer_after=0.15)

        assert feedback.is_positive is True
        assert feedback.wer_before == pytest.approx(0.0)
        assert feedback.wer_after == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Helpers for 15.3 tests
# ---------------------------------------------------------------------------


def _record_feedback_directly(db, engine, iteration_id, suggestion_content,
                               cer_before, cer_after, is_positive,
                               wer_before=0.20, wer_after=0.18):
    """Helper to record a feedback with specific suggestion_content."""
    _create_iteration(db, iteration_id, cer_before=cer_before,
                       cer_after=cer_after, wer_before=wer_before,
                       wer_after=wer_after, description=suggestion_content)
    feedback = EffectFeedback(
        iteration_id=iteration_id,
        suggestion_id=f"sug_{iteration_id}",
        suggestion_content=suggestion_content,
        input_analysis={},
        param_changes={},
        cer_before=cer_before,
        cer_after=cer_after,
        wer_before=wer_before,
        wer_after=wer_after,
        is_positive=is_positive,
        timestamp="2024-01-01T00:00:00",
    )
    engine.record_feedback(feedback)


# ---------------------------------------------------------------------------
# 15.3 analyze_patterns tests
# ---------------------------------------------------------------------------


class TestAnalyzePatterns:
    """Tests for SkillEvolutionEngine.analyze_patterns."""

    def test_returns_empty_when_below_threshold(self, engine, db):
        """analyze_patterns should return {} when feedback count < min_feedback_count."""
        # Record 2 feedbacks, but require 5
        _record_feedback_directly(db, engine, "iter_ap1", "Strategy A",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_ap2", "Strategy A",
                                   0.20, 0.18, True)

        result = engine.analyze_patterns(min_feedback_count=5)
        assert result == {}

    def test_groups_feedbacks_by_strategy(self, engine, db):
        """analyze_patterns should group feedbacks by suggestion_content."""
        _record_feedback_directly(db, engine, "iter_g1", "Increase LR",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_g2", "Increase LR",
                                   0.20, 0.18, True)
        _record_feedback_directly(db, engine, "iter_g3", "Add augmentation",
                                   0.20, 0.22, False)

        result = engine.analyze_patterns(min_feedback_count=1)

        assert "strategies" in result
        assert "Increase LR" in result["strategies"]
        assert "Add augmentation" in result["strategies"]
        assert result["strategies"]["Increase LR"]["total_count"] == 2
        assert result["strategies"]["Add augmentation"]["total_count"] == 1

    def test_computes_success_rates_correctly(self, engine, db):
        """analyze_patterns should compute success_rate = positive_count / total_count."""
        _record_feedback_directly(db, engine, "iter_sr1", "Strategy X",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_sr2", "Strategy X",
                                   0.20, 0.22, False)
        _record_feedback_directly(db, engine, "iter_sr3", "Strategy X",
                                   0.20, 0.18, True)
        _record_feedback_directly(db, engine, "iter_sr4", "Strategy X",
                                   0.20, 0.25, False)

        result = engine.analyze_patterns(min_feedback_count=1)
        stats = result["strategies"]["Strategy X"]

        assert stats["total_count"] == 4
        assert stats["positive_count"] == 2
        assert stats["success_rate"] == pytest.approx(0.5)

    def test_identifies_high_and_low_success_strategies(self, engine, db):
        """analyze_patterns should classify strategies as high/low success."""
        # Strategy A: 100% success rate
        _record_feedback_directly(db, engine, "iter_hl1", "Strategy A",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_hl2", "Strategy A",
                                   0.20, 0.18, True)
        # Strategy B: 0% success rate
        _record_feedback_directly(db, engine, "iter_hl3", "Strategy B",
                                   0.20, 0.22, False)
        _record_feedback_directly(db, engine, "iter_hl4", "Strategy B",
                                   0.20, 0.25, False)

        result = engine.analyze_patterns(min_feedback_count=1)

        # Average success rate = (1.0 + 0.0) / 2 = 0.5
        assert result["average_success_rate"] == pytest.approx(0.5)
        assert "Strategy A" in result["high_success_strategies"]
        assert "Strategy B" in result["low_success_strategies"]
        assert "Strategy A" not in result["low_success_strategies"]
        assert "Strategy B" not in result["high_success_strategies"]

    def test_all_positive_feedbacks(self, engine, db):
        """analyze_patterns should handle all-positive feedbacks correctly."""
        _record_feedback_directly(db, engine, "iter_allp1", "Good Strategy",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_allp2", "Good Strategy",
                                   0.20, 0.10, True)

        result = engine.analyze_patterns(min_feedback_count=1)
        stats = result["strategies"]["Good Strategy"]

        assert stats["success_rate"] == pytest.approx(1.0)
        assert stats["positive_count"] == 2
        assert stats["avg_cer_improvement"] > 0

    def test_all_negative_feedbacks(self, engine, db):
        """analyze_patterns should handle all-negative feedbacks correctly."""
        _record_feedback_directly(db, engine, "iter_alln1", "Bad Strategy",
                                   0.20, 0.25, False)
        _record_feedback_directly(db, engine, "iter_alln2", "Bad Strategy",
                                   0.20, 0.30, False)

        result = engine.analyze_patterns(min_feedback_count=1)
        stats = result["strategies"]["Bad Strategy"]

        assert stats["success_rate"] == pytest.approx(0.0)
        assert stats["positive_count"] == 0
        assert stats["avg_cer_improvement"] == pytest.approx(0.0)

    def test_returns_total_feedbacks_count(self, engine, db):
        """analyze_patterns should return the total feedback count."""
        _record_feedback_directly(db, engine, "iter_tc1", "S1",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_tc2", "S2",
                                   0.20, 0.22, False)
        _record_feedback_directly(db, engine, "iter_tc3", "S1",
                                   0.20, 0.18, True)

        result = engine.analyze_patterns(min_feedback_count=1)
        assert result["total_feedbacks"] == 3

    def test_avg_cer_improvement_only_from_positive(self, engine, db):
        """avg_cer_improvement should only consider positive feedbacks."""
        # Positive: cer_before=0.20, cer_after=0.10 → improvement=0.10
        _record_feedback_directly(db, engine, "iter_aci1", "Mixed",
                                   0.20, 0.10, True)
        # Negative: cer_before=0.20, cer_after=0.25 → not counted
        _record_feedback_directly(db, engine, "iter_aci2", "Mixed",
                                   0.20, 0.25, False)
        # Positive: cer_before=0.20, cer_after=0.14 → improvement=0.06
        _record_feedback_directly(db, engine, "iter_aci3", "Mixed",
                                   0.20, 0.14, True)

        result = engine.analyze_patterns(min_feedback_count=1)
        stats = result["strategies"]["Mixed"]

        # avg improvement = (0.10 + 0.06) / 2 = 0.08
        assert stats["avg_cer_improvement"] == pytest.approx(0.08)

    def test_exact_threshold_boundary(self, engine, db):
        """analyze_patterns should return empty when count == min_feedback_count - 1."""
        _record_feedback_directly(db, engine, "iter_bt1", "S",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_bt2", "S",
                                   0.20, 0.18, True)

        # Exactly at threshold should work
        result = engine.analyze_patterns(min_feedback_count=2)
        assert result != {}
        assert result["total_feedbacks"] == 2

        # One above threshold should return empty
        result = engine.analyze_patterns(min_feedback_count=3)
        assert result == {}


# ---------------------------------------------------------------------------
# 15.3 update_strategy_weights tests
# ---------------------------------------------------------------------------


class TestUpdateStrategyWeights:
    """Tests for SkillEvolutionEngine.update_strategy_weights."""

    def test_persists_weights_to_database(self, engine, db):
        """update_strategy_weights should persist weights to strategy_weights table."""
        # Build feedbacks and analyze
        _record_feedback_directly(db, engine, "iter_pw1", "Strategy A",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_pw2", "Strategy B",
                                   0.20, 0.22, False)

        analysis = engine.analyze_patterns(min_feedback_count=1)
        weights = engine.update_strategy_weights(analysis)

        assert len(weights) == 2

        # Check database
        rows = db.get_all("strategy_weights")
        assert len(rows) == 2

        names = {row["strategy_name"] for row in rows}
        assert "Strategy A" in names
        assert "Strategy B" in names

    def test_high_success_gets_higher_weight(self, engine, db):
        """Strategies with higher success_rate should get >= weight than lower ones."""
        # Strategy A: 100% success
        _record_feedback_directly(db, engine, "iter_hw1", "High Success",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_hw2", "High Success",
                                   0.20, 0.12, True)
        # Strategy B: 0% success
        _record_feedback_directly(db, engine, "iter_hw3", "Low Success",
                                   0.20, 0.25, False)
        _record_feedback_directly(db, engine, "iter_hw4", "Low Success",
                                   0.20, 0.30, False)

        analysis = engine.analyze_patterns(min_feedback_count=1)
        weights = engine.update_strategy_weights(analysis)

        weight_map = {w.strategy_name: w.weight for w in weights}
        assert weight_map["High Success"] >= weight_map["Low Success"]

    def test_handles_empty_analysis(self, engine, db):
        """update_strategy_weights should return [] for empty analysis."""
        result = engine.update_strategy_weights({})
        assert result == []

    def test_handles_analysis_without_strategies(self, engine, db):
        """update_strategy_weights should return [] when strategies key is missing."""
        result = engine.update_strategy_weights({"total_feedbacks": 0})
        assert result == []

    def test_upsert_updates_existing_weights(self, engine, db):
        """update_strategy_weights should update existing rows (upsert behavior)."""
        # First round: Strategy A with 100% success
        _record_feedback_directly(db, engine, "iter_up1", "Strategy A",
                                   0.20, 0.15, True)

        analysis1 = engine.analyze_patterns(min_feedback_count=1)
        weights1 = engine.update_strategy_weights(analysis1)
        first_weight = weights1[0].weight

        # Add a negative feedback for Strategy A
        _record_feedback_directly(db, engine, "iter_up2", "Strategy A",
                                   0.20, 0.25, False)

        analysis2 = engine.analyze_patterns(min_feedback_count=1)
        weights2 = engine.update_strategy_weights(analysis2)

        # Weight should have changed (lower now since success rate dropped)
        second_weight = [w for w in weights2 if w.strategy_name == "Strategy A"][0].weight

        # Only 1 row in DB for Strategy A (upsert, not duplicate)
        rows = db.get_all("strategy_weights")
        strategy_a_rows = [r for r in rows if r["strategy_name"] == "Strategy A"]
        assert len(strategy_a_rows) == 1

        # The weight should reflect the updated success rate
        assert strategy_a_rows[0]["success_rate"] == pytest.approx(0.5)

    def test_returns_strategy_weight_objects(self, engine, db):
        """update_strategy_weights should return StrategyWeight dataclass instances."""
        from scripts.core.models import StrategyWeight

        _record_feedback_directly(db, engine, "iter_sw1", "Test Strategy",
                                   0.20, 0.15, True)

        analysis = engine.analyze_patterns(min_feedback_count=1)
        weights = engine.update_strategy_weights(analysis)

        assert len(weights) == 1
        assert isinstance(weights[0], StrategyWeight)
        assert weights[0].strategy_name == "Test Strategy"
        assert weights[0].success_rate == pytest.approx(1.0)
        assert weights[0].sample_count == 1

    def test_weight_values_stored_in_db(self, engine, db):
        """Weights stored in DB should match the returned StrategyWeight objects."""
        _record_feedback_directly(db, engine, "iter_dbw1", "S1",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_dbw2", "S2",
                                   0.20, 0.25, False)

        analysis = engine.analyze_patterns(min_feedback_count=1)
        weights = engine.update_strategy_weights(analysis)

        weight_map = {w.strategy_name: w for w in weights}

        for row in db.get_all("strategy_weights"):
            name = row["strategy_name"]
            assert row["weight"] == pytest.approx(weight_map[name].weight)
            assert row["success_rate"] == pytest.approx(weight_map[name].success_rate)
            assert row["sample_count"] == weight_map[name].sample_count

    def test_multiple_strategies_mixed_success(self, engine, db):
        """Test with multiple strategies having varying success rates."""
        # Strategy A: 2/3 positive = 66.7%
        _record_feedback_directly(db, engine, "iter_ms1", "Strategy A",
                                   0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_ms2", "Strategy A",
                                   0.20, 0.18, True)
        _record_feedback_directly(db, engine, "iter_ms3", "Strategy A",
                                   0.20, 0.22, False)
        # Strategy B: 1/3 positive = 33.3%
        _record_feedback_directly(db, engine, "iter_ms4", "Strategy B",
                                   0.20, 0.18, True)
        _record_feedback_directly(db, engine, "iter_ms5", "Strategy B",
                                   0.20, 0.22, False)
        _record_feedback_directly(db, engine, "iter_ms6", "Strategy B",
                                   0.20, 0.25, False)

        analysis = engine.analyze_patterns(min_feedback_count=1)
        weights = engine.update_strategy_weights(analysis)

        weight_map = {w.strategy_name: w for w in weights}

        # Strategy A has higher success rate, should get higher weight
        assert weight_map["Strategy A"].weight >= weight_map["Strategy B"].weight


# ---------------------------------------------------------------------------
# Helpers for 15.5 tests
# ---------------------------------------------------------------------------


def _create_skill_files(skills_dir, filenames=None):
    """Create sample skill .md files in the given directory."""
    if filenames is None:
        filenames = [
            "asr-training-diagnostics.md",
            "asr-badcase-analysis.md",
            "asr-data-optimization.md",
            "asr-model-tuning.md",
            "asr-postprocess-rules.md",
        ]
    for fname in filenames:
        filepath = os.path.join(str(skills_dir), fname)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {fname.replace('.md', '').replace('-', ' ').title()}\n\n")
            f.write("## 能力描述\n\nSample skill content.\n")


def _make_strategy_weights(names_and_weights):
    """Build a list of StrategyWeight from (name, weight, success_rate) tuples."""
    from scripts.core.models import StrategyWeight
    return [
        StrategyWeight(
            strategy_name=name,
            weight=weight,
            success_rate=sr,
            sample_count=10,
        )
        for name, weight, sr in names_and_weights
    ]


# ---------------------------------------------------------------------------
# 15.5 update_skill_files tests
# ---------------------------------------------------------------------------


class TestUpdateSkillFiles:
    """Tests for SkillEvolutionEngine.update_skill_files."""

    def test_updates_skill_files_with_strategy_section(self, engine, workspace):
        """update_skill_files should append a Strategy Weights section to each skill file."""
        _create_skill_files(engine.skills_dir)
        weights = _make_strategy_weights([
            ("Increase LR", 1.5, 0.8),
            ("Add augmentation", 0.8, 0.4),
        ])

        result = engine.update_skill_files(weights)

        assert len(result["updated_files"]) == 5
        assert result["strategy_count"] == 2

        # Verify content of one file
        with open(result["updated_files"][0], "r", encoding="utf-8") as f:
            content = f.read()
        assert "## Strategy Weights" in content
        assert "Increase LR" in content
        assert "Add augmentation" in content

    def test_strategies_sorted_by_weight_descending(self, engine, workspace):
        """Strategies in the updated section should be sorted by weight descending."""
        _create_skill_files(engine.skills_dir)
        weights = _make_strategy_weights([
            ("Low Weight", 0.5, 0.2),
            ("High Weight", 2.0, 0.9),
            ("Mid Weight", 1.0, 0.5),
        ])

        result = engine.update_skill_files(weights)

        with open(result["updated_files"][0], "r", encoding="utf-8") as f:
            content = f.read()

        # Find positions of each strategy name after the marker
        marker_pos = content.find("## Strategy Weights")
        section = content[marker_pos:]
        high_pos = section.find("High Weight")
        mid_pos = section.find("Mid Weight")
        low_pos = section.find("Low Weight")

        assert high_pos < mid_pos < low_pos

    def test_handles_empty_strategy_weights(self, engine, workspace):
        """update_skill_files should handle an empty strategy weights list."""
        _create_skill_files(engine.skills_dir)

        result = engine.update_skill_files([])

        assert len(result["updated_files"]) == 5
        assert result["strategy_count"] == 0

        # Section header should still be present
        with open(result["updated_files"][0], "r", encoding="utf-8") as f:
            content = f.read()
        assert "## Strategy Weights" in content

    def test_handles_missing_skills_directory(self, engine, workspace):
        """update_skill_files should return empty result when skills dir doesn't exist."""
        import shutil
        if os.path.isdir(engine.skills_dir):
            shutil.rmtree(engine.skills_dir)

        weights = _make_strategy_weights([("S1", 1.0, 0.5)])
        result = engine.update_skill_files(weights)

        assert result == {"updated_files": [], "strategy_count": 0}

    def test_returns_correct_summary_dict(self, engine, workspace):
        """update_skill_files should return a dict with updated_files and strategy_count."""
        _create_skill_files(engine.skills_dir, ["skill-a.md", "skill-b.md"])
        weights = _make_strategy_weights([
            ("S1", 1.5, 0.8),
            ("S2", 0.5, 0.3),
            ("S3", 1.0, 0.5),
        ])

        result = engine.update_skill_files(weights)

        assert "updated_files" in result
        assert "strategy_count" in result
        assert len(result["updated_files"]) == 2
        assert result["strategy_count"] == 3

    def test_replaces_existing_strategy_section(self, engine, workspace):
        """Calling update_skill_files twice should replace the section, not duplicate it."""
        _create_skill_files(engine.skills_dir, ["test-skill.md"])

        weights_v1 = _make_strategy_weights([("Old Strategy", 1.0, 0.5)])
        engine.update_skill_files(weights_v1)

        weights_v2 = _make_strategy_weights([("New Strategy", 1.5, 0.8)])
        result = engine.update_skill_files(weights_v2)

        with open(result["updated_files"][0], "r", encoding="utf-8") as f:
            content = f.read()

        assert content.count("## Strategy Weights") == 1
        assert "New Strategy" in content
        assert "Old Strategy" not in content

    def test_preserves_original_content(self, engine, workspace):
        """update_skill_files should preserve the original file content above the section."""
        _create_skill_files(engine.skills_dir, ["test-skill.md"])
        weights = _make_strategy_weights([("S1", 1.0, 0.5)])

        result = engine.update_skill_files(weights)

        with open(result["updated_files"][0], "r", encoding="utf-8") as f:
            content = f.read()

        assert "# Test Skill" in content
        assert "## 能力描述" in content
        assert "Sample skill content." in content


# ---------------------------------------------------------------------------
# 15.5 create_version_snapshot tests
# ---------------------------------------------------------------------------


class TestCreateVersionSnapshot:
    """Tests for SkillEvolutionEngine.create_version_snapshot."""

    def test_creates_version_record_in_database(self, engine, db, workspace):
        """create_version_snapshot should persist a record to skill_versions table."""
        version = engine.create_version_snapshot("Initial version")

        row = db.get_by_id("skill_versions", "version_id", version.version_id)
        assert row is not None
        assert row["change_description"] == "Initial version"

    def test_returns_skill_version_dataclass(self, engine, db, workspace):
        """create_version_snapshot should return a SkillVersion dataclass."""
        from scripts.core.models import SkillVersion

        version = engine.create_version_snapshot("Test version")

        assert isinstance(version, SkillVersion)
        assert version.version_id.startswith("sv_")
        assert version.change_description == "Test version"
        assert version.created_at != ""

    def test_copies_skill_files_to_version_directory(self, engine, db, workspace):
        """create_version_snapshot should copy skill files to version directory."""
        _create_skill_files(engine.skills_dir, ["skill-a.md", "skill-b.md"])

        version = engine.create_version_snapshot("Copy test")

        version_dir = os.path.join(engine.skill_versions_dir, version.version_id)
        assert os.path.isdir(version_dir)
        assert os.path.isfile(os.path.join(version_dir, "skill-a.md"))
        assert os.path.isfile(os.path.join(version_dir, "skill-b.md"))

    def test_captures_strategy_rules_from_db(self, engine, db, workspace):
        """create_version_snapshot should capture current strategy weights."""
        # Insert strategy weights
        db.insert("strategy_weights", {
            "strategy_name": "Increase LR",
            "weight": 1.5,
            "success_rate": 0.8,
            "sample_count": 10,
            "version_id": None,
            "updated_at": "2024-01-01T00:00:00",
        })
        db.insert("strategy_weights", {
            "strategy_name": "Add data",
            "weight": 0.8,
            "success_rate": 0.4,
            "sample_count": 5,
            "version_id": None,
            "updated_at": "2024-01-01T00:00:00",
        })

        version = engine.create_version_snapshot("Strategy capture test")

        assert "Increase LR" in version.strategy_rules
        assert version.strategy_rules["Increase LR"]["weight"] == 1.5
        assert version.strategy_rules["Increase LR"]["success_rate"] == 0.8
        assert "Add data" in version.strategy_rules
        assert version.strategy_rules["Add data"]["weight"] == 0.8

    def test_captures_knowledge_entries_from_db(self, engine, db, workspace):
        """create_version_snapshot should capture knowledge entries."""
        db.insert("knowledge_entries", {
            "entry_id": "ke_001",
            "knowledge_type": "scene_solution",
            "content_json": json.dumps({"scene": "background_voice", "solution": "augment"}),
            "confidence": 0.85,
            "source_iteration_id": "iter_001",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        })

        version = engine.create_version_snapshot("Knowledge capture test")

        entries = version.knowledge_base_snapshot.get("entries", [])
        assert len(entries) == 1
        assert entries[0]["entry_id"] == "ke_001"
        assert entries[0]["knowledge_type"] == "scene_solution"
        assert entries[0]["confidence"] == 0.85
        assert entries[0]["content"]["scene"] == "background_voice"

    def test_computes_positive_feedback_rate(self, engine, db, workspace):
        """create_version_snapshot should compute positive_feedback_rate from feedbacks."""
        # Record 3 feedbacks: 2 positive, 1 negative
        _record_feedback_directly(db, engine, "iter_pfr1", "S1", 0.20, 0.15, True)
        _record_feedback_directly(db, engine, "iter_pfr2", "S1", 0.20, 0.18, True)
        _record_feedback_directly(db, engine, "iter_pfr3", "S1", 0.20, 0.25, False)

        version = engine.create_version_snapshot("Feedback rate test")

        assert version.positive_feedback_rate == pytest.approx(2.0 / 3.0)

    def test_computes_avg_cer_improvement(self, engine, db, workspace):
        """create_version_snapshot should compute avg_cer_improvement from positive feedbacks."""
        # Positive: improvement = 0.20 - 0.15 = 0.05
        _record_feedback_directly(db, engine, "iter_aci_v1", "S1", 0.20, 0.15, True)
        # Positive: improvement = 0.20 - 0.10 = 0.10
        _record_feedback_directly(db, engine, "iter_aci_v2", "S1", 0.20, 0.10, True)
        # Negative: not counted
        _record_feedback_directly(db, engine, "iter_aci_v3", "S1", 0.20, 0.25, False)

        version = engine.create_version_snapshot("CER improvement test")

        # avg = (0.05 + 0.10) / 2 = 0.075
        assert version.avg_cer_improvement == pytest.approx(0.075)

    def test_no_feedbacks_gives_zero_metrics(self, engine, db, workspace):
        """create_version_snapshot with no feedbacks should have zero metrics."""
        version = engine.create_version_snapshot("Empty feedbacks")

        assert version.positive_feedback_rate == pytest.approx(0.0)
        assert version.avg_cer_improvement == pytest.approx(0.0)

    def test_empty_strategy_and_knowledge(self, engine, db, workspace):
        """create_version_snapshot with empty DB should have empty rules and snapshot."""
        version = engine.create_version_snapshot("Empty state")

        assert version.strategy_rules == {}
        assert version.knowledge_base_snapshot == {"entries": []}

    def test_saves_knowledge_snapshot_json_file(self, engine, db, workspace):
        """create_version_snapshot should save a knowledge_snapshot.json in version dir."""
        db.insert("knowledge_entries", {
            "entry_id": "ke_json",
            "knowledge_type": "param_effect",
            "content_json": json.dumps({"param": "lr", "effect": "positive"}),
            "confidence": 0.9,
            "source_iteration_id": "iter_json",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        })

        version = engine.create_version_snapshot("JSON file test")

        version_dir = os.path.join(engine.skill_versions_dir, version.version_id)
        snapshot_path = os.path.join(version_dir, "knowledge_snapshot.json")
        assert os.path.isfile(snapshot_path)

        with open(snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["entries"]) == 1
        assert data["entries"][0]["entry_id"] == "ke_json"

    def test_version_id_is_unique(self, engine, db, workspace):
        """Each call to create_version_snapshot should produce a unique version_id."""
        v1 = engine.create_version_snapshot("Version 1")
        v2 = engine.create_version_snapshot("Version 2")

        assert v1.version_id != v2.version_id
        assert v1.version_id.startswith("sv_")
        assert v2.version_id.startswith("sv_")


# ---------------------------------------------------------------------------
# 15.5 compare_versions tests
# ---------------------------------------------------------------------------


class TestCompareVersions:
    """Tests for SkillEvolutionEngine.compare_versions."""

    def test_compares_two_existing_versions(self, engine, db, workspace):
        """compare_versions should return metrics and deltas for two existing versions."""
        # Create two versions with different feedback states
        _record_feedback_directly(db, engine, "iter_cv1", "S1", 0.20, 0.15, True)
        v1 = engine.create_version_snapshot("Version A")

        _record_feedback_directly(db, engine, "iter_cv2", "S1", 0.20, 0.25, False)
        v2 = engine.create_version_snapshot("Version B")

        result = engine.compare_versions(v1.version_id, v2.version_id)

        assert "version_a" in result
        assert "version_b" in result
        assert "deltas" in result
        assert result["version_a"]["version_id"] == v1.version_id
        assert result["version_b"]["version_id"] == v2.version_id

    def test_handles_missing_version_gracefully(self, engine, db, workspace):
        """compare_versions should return {} when a version is not found."""
        v1 = engine.create_version_snapshot("Existing version")

        result = engine.compare_versions(v1.version_id, "nonexistent_version")
        assert result == {}

        result = engine.compare_versions("nonexistent_version", v1.version_id)
        assert result == {}

    def test_handles_both_versions_missing(self, engine, db, workspace):
        """compare_versions should return {} when both versions are missing."""
        result = engine.compare_versions("missing_a", "missing_b")
        assert result == {}

    def test_computes_deltas_correctly(self, engine, db, workspace):
        """compare_versions should compute deltas as version_b - version_a."""
        # Version A: 1 positive feedback, rate=1.0, improvement=0.05
        _record_feedback_directly(db, engine, "iter_d1", "S1", 0.20, 0.15, True)
        v1 = engine.create_version_snapshot("Version A")

        # Version B: 1 positive + 1 negative, rate=0.5
        _record_feedback_directly(db, engine, "iter_d2", "S1", 0.20, 0.25, False)
        v2 = engine.create_version_snapshot("Version B")

        result = engine.compare_versions(v1.version_id, v2.version_id)

        rate_a = result["version_a"]["positive_feedback_rate"]
        rate_b = result["version_b"]["positive_feedback_rate"]
        delta_rate = result["deltas"]["positive_feedback_rate"]

        assert delta_rate == pytest.approx(rate_b - rate_a)

        cer_a = result["version_a"]["avg_cer_improvement"]
        cer_b = result["version_b"]["avg_cer_improvement"]
        delta_cer = result["deltas"]["avg_cer_improvement"]

        assert delta_cer == pytest.approx(cer_b - cer_a)

    def test_same_version_gives_zero_deltas(self, engine, db, workspace):
        """Comparing a version with itself should give zero deltas."""
        _record_feedback_directly(db, engine, "iter_sv1", "S1", 0.20, 0.15, True)
        v1 = engine.create_version_snapshot("Same version")

        result = engine.compare_versions(v1.version_id, v1.version_id)

        assert result["deltas"]["positive_feedback_rate"] == pytest.approx(0.0)
        assert result["deltas"]["avg_cer_improvement"] == pytest.approx(0.0)

    def test_returns_correct_metric_values(self, engine, db, workspace):
        """compare_versions should return the actual stored metric values."""
        # All positive feedbacks
        _record_feedback_directly(db, engine, "iter_rm1", "S1", 0.20, 0.10, True)
        _record_feedback_directly(db, engine, "iter_rm2", "S1", 0.20, 0.12, True)
        v1 = engine.create_version_snapshot("All positive")

        result = engine.compare_versions(v1.version_id, v1.version_id)

        assert result["version_a"]["positive_feedback_rate"] == pytest.approx(1.0)
        # avg improvement = ((0.20-0.10) + (0.20-0.12)) / 2 = (0.10 + 0.08) / 2 = 0.09
        assert result["version_a"]["avg_cer_improvement"] == pytest.approx(0.09)


# ---------------------------------------------------------------------------
# Helpers for 15.7 tests
# ---------------------------------------------------------------------------


def _create_version_with_rate(db, engine, workspace, version_id_hint, rate,
                               skill_filenames=None, created_at=None):
    """Create a skill version with a specific positive_feedback_rate.

    Inserts a version record directly into the DB and copies skill files
    to the version directory so rollback can restore them.
    """
    import uuid
    if version_id_hint.startswith("sv_"):
        vid = version_id_hint
    else:
        vid = f"sv_{version_id_hint}"

    now = created_at or "2024-01-01T00:00:00"

    version_dir = os.path.join(str(workspace), "knowledge", "skill_versions", vid)
    os.makedirs(version_dir, exist_ok=True)

    # Create skill file copies in the version directory
    if skill_filenames is None:
        skill_filenames = ["skill-a.md"]
    for fname in skill_filenames:
        filepath = os.path.join(version_dir, fname)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {fname} content for version {vid}\n")

    db.insert("skill_versions", {
        "version_id": vid,
        "strategy_rules_json": json.dumps({}),
        "knowledge_snapshot_path": os.path.join(version_dir, "knowledge_snapshot.json"),
        "change_description": f"Version {vid}",
        "positive_feedback_rate": rate,
        "avg_cer_improvement": 0.0,
        "created_at": now,
    })

    return vid


# ---------------------------------------------------------------------------
# 15.7 rollback_version tests
# ---------------------------------------------------------------------------


class TestRollbackVersion:
    """Tests for SkillEvolutionEngine.rollback_version."""

    def test_restores_skill_files_from_version_directory(self, engine, db, workspace):
        """rollback_version should copy .md files from version dir to skills dir."""
        # Create a version with specific skill file content
        vid = _create_version_with_rate(
            db, engine, workspace, "rollback1", 0.8,
            skill_filenames=["skill-a.md", "skill-b.md"],
        )

        # Ensure skills dir exists but has different content
        os.makedirs(engine.skills_dir, exist_ok=True)
        with open(os.path.join(engine.skills_dir, "skill-a.md"), "w") as f:
            f.write("# Current content\n")

        result = engine.rollback_version(vid)

        assert result is True

        # Verify the files were restored from the version directory
        with open(os.path.join(engine.skills_dir, "skill-a.md"), "r") as f:
            content = f.read()
        assert f"version {vid}" in content

        # skill-b.md should also be copied
        assert os.path.isfile(os.path.join(engine.skills_dir, "skill-b.md"))

    def test_returns_false_for_nonexistent_version(self, engine, db, workspace):
        """rollback_version should return False when version_id is not in the DB."""
        result = engine.rollback_version("sv_nonexistent")
        assert result is False

    def test_returns_true_on_successful_rollback(self, engine, db, workspace):
        """rollback_version should return True when rollback succeeds."""
        vid = _create_version_with_rate(
            db, engine, workspace, "success1", 0.9,
        )
        result = engine.rollback_version(vid)
        assert result is True

    def test_handles_missing_version_directory(self, engine, db, workspace):
        """rollback_version should return False when version dir doesn't exist."""
        # Insert a version record but don't create the directory
        db.insert("skill_versions", {
            "version_id": "sv_nodir",
            "strategy_rules_json": json.dumps({}),
            "knowledge_snapshot_path": "",
            "change_description": "No directory",
            "positive_feedback_rate": 0.5,
            "avg_cer_improvement": 0.0,
            "created_at": "2024-01-01T00:00:00",
        })

        # Remove the directory if it was auto-created
        version_dir = os.path.join(engine.skill_versions_dir, "sv_nodir")
        if os.path.isdir(version_dir):
            import shutil
            shutil.rmtree(version_dir)

        result = engine.rollback_version("sv_nodir")
        assert result is False

    def test_creates_skills_dir_if_missing(self, engine, db, workspace):
        """rollback_version should create .kiro/skills/ if it doesn't exist."""
        vid = _create_version_with_rate(
            db, engine, workspace, "createdir1", 0.7,
        )

        # Remove skills dir
        import shutil
        if os.path.isdir(engine.skills_dir):
            shutil.rmtree(engine.skills_dir)

        result = engine.rollback_version(vid)
        assert result is True
        assert os.path.isdir(engine.skills_dir)


# ---------------------------------------------------------------------------
# 15.7 check_auto_rollback tests
# ---------------------------------------------------------------------------


class TestCheckAutoRollback:
    """Tests for SkillEvolutionEngine.check_auto_rollback."""

    def test_returns_none_when_fewer_than_two_versions(self, engine, db, workspace):
        """check_auto_rollback should return None with < 2 versions."""
        # No versions at all
        result = engine.check_auto_rollback()
        assert result is None

        # Only one version
        _create_version_with_rate(
            db, engine, workspace, "single1", 0.8,
            created_at="2024-01-01T00:00:00",
        )
        result = engine.check_auto_rollback()
        assert result is None

    def test_returns_none_when_latest_has_higher_rate(self, engine, db, workspace):
        """check_auto_rollback should return None when latest version is better."""
        _create_version_with_rate(
            db, engine, workspace, "older1", 0.5,
            created_at="2024-01-01T00:00:00",
        )
        _create_version_with_rate(
            db, engine, workspace, "newer1", 0.8,
            created_at="2024-01-02T00:00:00",
        )

        result = engine.check_auto_rollback()
        assert result is None

    def test_triggers_rollback_when_latest_has_lower_rate(self, engine, db, workspace):
        """check_auto_rollback should rollback when latest version has lower rate."""
        prev_vid = _create_version_with_rate(
            db, engine, workspace, "prev1", 0.8,
            skill_filenames=["skill-a.md"],
            created_at="2024-01-01T00:00:00",
        )
        _create_version_with_rate(
            db, engine, workspace, "latest1", 0.3,
            skill_filenames=["skill-a.md"],
            created_at="2024-01-02T00:00:00",
        )

        result = engine.check_auto_rollback()
        assert result == prev_vid

    def test_restores_skill_files_after_auto_rollback(self, engine, db, workspace):
        """check_auto_rollback should restore skill files from the previous version."""
        prev_vid = _create_version_with_rate(
            db, engine, workspace, "restore_prev", 0.9,
            skill_filenames=["skill-a.md"],
            created_at="2024-01-01T00:00:00",
        )
        _create_version_with_rate(
            db, engine, workspace, "restore_latest", 0.2,
            skill_filenames=["skill-a.md"],
            created_at="2024-01-02T00:00:00",
        )

        # Write different content to current skills
        os.makedirs(engine.skills_dir, exist_ok=True)
        with open(os.path.join(engine.skills_dir, "skill-a.md"), "w") as f:
            f.write("# Latest bad content\n")

        result = engine.check_auto_rollback()
        assert result == prev_vid

        # Verify skill files were restored from previous version
        with open(os.path.join(engine.skills_dir, "skill-a.md"), "r") as f:
            content = f.read()
        assert f"version {prev_vid}" in content

    def test_returns_none_when_rates_are_equal(self, engine, db, workspace):
        """check_auto_rollback should return None when rates are equal."""
        _create_version_with_rate(
            db, engine, workspace, "eq1", 0.5,
            created_at="2024-01-01T00:00:00",
        )
        _create_version_with_rate(
            db, engine, workspace, "eq2", 0.5,
            created_at="2024-01-02T00:00:00",
        )

        result = engine.check_auto_rollback()
        assert result is None


# ---------------------------------------------------------------------------
# 15.7 trigger_learning tests
# ---------------------------------------------------------------------------


class TestTriggerLearning:
    """Tests for SkillEvolutionEngine.trigger_learning."""

    def test_returns_empty_result_when_no_feedbacks_in_range(self, engine, db, workspace):
        """trigger_learning should return zero counts when no feedbacks match."""
        # Record a feedback outside the range
        _record_feedback_directly(
            db, engine, "iter_tl_out", "Strategy A", 0.20, 0.15, True,
        )

        result = engine.trigger_learning(("2025-01-01T00:00:00", "2025-12-31T23:59:59"))

        assert result["feedbacks_analyzed"] == 0
        assert result["strategies_updated"] == 0
        assert result["version_id"] is None

    def test_analyzes_feedbacks_within_time_range(self, engine, db, workspace):
        """trigger_learning should analyze only feedbacks within the time range."""
        _create_skill_files(engine.skills_dir)

        # Record feedbacks with known timestamps via direct DB insert
        _create_iteration(db, "iter_tl1")
        engine.db.insert("effect_feedbacks", {
            "feedback_id": "fb_tl1",
            "iteration_id": "iter_tl1",
            "suggestion_id": "sug_tl1",
            "suggestion_content": "Strategy A",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": 0.20,
            "wer_after": 0.18,
            "is_positive": 1,
            "created_at": "2024-06-15T12:00:00",
        })

        # Feedback outside range
        _create_iteration(db, "iter_tl2")
        engine.db.insert("effect_feedbacks", {
            "feedback_id": "fb_tl2",
            "iteration_id": "iter_tl2",
            "suggestion_id": "sug_tl2",
            "suggestion_content": "Strategy B",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.25,
            "wer_before": 0.20,
            "wer_after": 0.22,
            "is_positive": 0,
            "created_at": "2024-01-01T00:00:00",
        })

        result = engine.trigger_learning(("2024-06-01T00:00:00", "2024-06-30T23:59:59"))

        assert result["feedbacks_analyzed"] == 1
        assert result["strategies_updated"] >= 1

    def test_creates_new_version_snapshot(self, engine, db, workspace):
        """trigger_learning should create a new version snapshot."""
        _create_skill_files(engine.skills_dir)

        _create_iteration(db, "iter_tl_snap")
        engine.db.insert("effect_feedbacks", {
            "feedback_id": "fb_tl_snap",
            "iteration_id": "iter_tl_snap",
            "suggestion_id": "sug_tl_snap",
            "suggestion_content": "Strategy X",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": 0.20,
            "wer_after": 0.18,
            "is_positive": 1,
            "created_at": "2024-06-15T12:00:00",
        })

        result = engine.trigger_learning(("2024-06-01T00:00:00", "2024-06-30T23:59:59"))

        assert result["version_id"] is not None
        assert result["version_id"].startswith("sv_")

        # Verify version exists in DB
        row = db.get_by_id("skill_versions", "version_id", result["version_id"])
        assert row is not None

    def test_returns_correct_summary_dict(self, engine, db, workspace):
        """trigger_learning should return a dict with the expected keys."""
        _create_skill_files(engine.skills_dir)

        # Insert two feedbacks in range with different strategies
        _create_iteration(db, "iter_tl_sum1")
        engine.db.insert("effect_feedbacks", {
            "feedback_id": "fb_tl_sum1",
            "iteration_id": "iter_tl_sum1",
            "suggestion_id": "sug_tl_sum1",
            "suggestion_content": "Strategy A",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": 0.20,
            "wer_after": 0.18,
            "is_positive": 1,
            "created_at": "2024-06-10T12:00:00",
        })
        _create_iteration(db, "iter_tl_sum2")
        engine.db.insert("effect_feedbacks", {
            "feedback_id": "fb_tl_sum2",
            "iteration_id": "iter_tl_sum2",
            "suggestion_id": "sug_tl_sum2",
            "suggestion_content": "Strategy B",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.22,
            "wer_before": 0.20,
            "wer_after": 0.25,
            "is_positive": 0,
            "created_at": "2024-06-20T12:00:00",
        })

        result = engine.trigger_learning(("2024-06-01T00:00:00", "2024-06-30T23:59:59"))

        assert "feedbacks_analyzed" in result
        assert "strategies_updated" in result
        assert "version_id" in result
        assert result["feedbacks_analyzed"] == 2
        assert result["strategies_updated"] == 2
        assert result["version_id"] is not None

    def test_updates_strategy_weights_in_db(self, engine, db, workspace):
        """trigger_learning should persist updated strategy weights to the DB."""
        _create_skill_files(engine.skills_dir)

        _create_iteration(db, "iter_tl_wt")
        engine.db.insert("effect_feedbacks", {
            "feedback_id": "fb_tl_wt",
            "iteration_id": "iter_tl_wt",
            "suggestion_id": "sug_tl_wt",
            "suggestion_content": "Strategy W",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": 0.20,
            "wer_after": 0.18,
            "is_positive": 1,
            "created_at": "2024-06-15T12:00:00",
        })

        engine.trigger_learning(("2024-06-01T00:00:00", "2024-06-30T23:59:59"))

        rows = db.get_all("strategy_weights")
        names = {row["strategy_name"] for row in rows}
        assert "Strategy W" in names
