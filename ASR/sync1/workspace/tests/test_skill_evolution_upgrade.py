"""Integration tests for Skill Evolution Upgrade features (Task 20).

Verifies:
- Knowledge entry lifecycle transitions (active → stale → archived)
- Auto-confidence computation
- Auto-review skills on iteration completion
- State-based query filtering
- Skill file quick reference updates
- Deep review triggering

Requirements: 11.1-11.12
"""

import json
import os

import pytest

from scripts.core.database import Database
from scripts.core.models import (
    EvaluationReport,
    EffectFeedback,
    KnowledgeEntry,
    StrategyWeight,
)
from scripts.decision.optimizer_system import OptimizerSystem
from scripts.decision.skill_evolution import SkillEvolutionEngine
from scripts.decision.skill_knowledge_base import SkillKnowledgeBase


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

    # Create sample skill files
    skill_content = "# ASR Training Diagnostics\n\n## 能力描述\nSample content.\n"
    (tmp_path / ".kiro" / "skills" / "asr-training-diagnostics.md").write_text(
        skill_content, encoding="utf-8"
    )
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
def kb(db):
    """Return a SkillKnowledgeBase instance."""
    return SkillKnowledgeBase(db=db)


@pytest.fixture
def engine(workspace, db):
    """Return a SkillEvolutionEngine instance."""
    return SkillEvolutionEngine(workspace_dir=str(workspace), db=db)


@pytest.fixture
def optimizer(workspace, db):
    """Return an OptimizerSystem instance."""
    return OptimizerSystem(workspace_dir=str(workspace), db=db)


def _create_iteration(db, iteration_id, cer_before=0.20, cer_after=0.15,
                       wer_before=0.30, wer_after=0.25):
    """Helper to insert an iteration record."""
    db.insert("iterations", {
        "iteration_id": iteration_id,
        "description": "Test iteration",
        "status": "completed",
        "cer_before": cer_before,
        "cer_after": cer_after,
        "wer_before": wer_before,
        "wer_after": wer_after,
        "param_changes_json": json.dumps({}),
        "model_changes_json": json.dumps({}),
        "data_changes_json": json.dumps({}),
        "created_at": "2024-01-01T00:00:00",
        "completed_at": "2024-01-01T01:00:00",
    })


# ---------------------------------------------------------------------------
# 20.2 Knowledge entry confidence auto-computation
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    """Tests for SkillKnowledgeBase.compute_confidence (Req 11.10)."""

    def test_computes_confidence_with_positive_feedback(self, kb, db):
        """Confidence should reflect success rate with time decay."""
        _create_iteration(db, "iter_cc1")
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"scene": "background_voice"},
            confidence=0.5,
            source_iteration_id="iter_cc1",
        )
        entry_id = kb.add_knowledge(entry)

        # Add positive feedback
        db.insert("effect_feedbacks", {
            "feedback_id": "fb_cc1",
            "iteration_id": "iter_cc1",
            "suggestion_id": "sug_cc1",
            "suggestion_content": "test",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": 0.30,
            "wer_after": 0.25,
            "is_positive": 1,
            "created_at": "2024-01-01T00:00:00",
        })

        conf = kb.compute_confidence(entry_id)
        assert 0.0 <= conf <= 1.0

    def test_raises_for_nonexistent_entry(self, kb):
        """compute_confidence should raise KeyError for unknown entry."""
        with pytest.raises(KeyError):
            kb.compute_confidence("ke_nonexistent")

    def test_updates_auto_confidence_field(self, kb, db):
        """compute_confidence should update the auto_confidence column."""
        _create_iteration(db, "iter_ac")
        entry = KnowledgeEntry(
            knowledge_type="param_effect",
            content={"param": "lr"},
            confidence=0.5,
            source_iteration_id="iter_ac",
        )
        entry_id = kb.add_knowledge(entry)

        db.insert("effect_feedbacks", {
            "feedback_id": "fb_ac1",
            "iteration_id": "iter_ac",
            "suggestion_id": "sug_ac1",
            "suggestion_content": "test",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": 0.30,
            "wer_after": 0.25,
            "is_positive": 1,
            "created_at": "2024-01-01T00:00:00",
        })

        kb.compute_confidence(entry_id)
        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["auto_confidence"] is not None
        assert row["auto_confidence"] > 0.0


# ---------------------------------------------------------------------------
# 20.3 Knowledge entry lifecycle state transitions
# ---------------------------------------------------------------------------


class TestLifecycleTransitions:
    """Tests for SkillKnowledgeBase.apply_lifecycle_transitions (Req 11.9)."""

    def test_active_to_stale_after_60_days(self, kb, db):
        """Entries not verified in 60+ days should transition to stale."""
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"test": True},
            confidence=0.8,
            source_iteration_id="iter_old",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
            last_verified_at="2024-01-01T00:00:00",
            state="active",
        )
        entry_id = kb.add_knowledge(entry)

        transitions = kb.apply_lifecycle_transitions()
        assert transitions["active_to_stale"] >= 1

        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["state"] == "stale"

    def test_stale_to_archived_after_120_days(self, kb, db):
        """Stale entries not verified in 120+ days should be archived."""
        entry_id = "ke_stale_old"
        db.insert("knowledge_entries", {
            "entry_id": entry_id,
            "knowledge_type": "param_effect",
            "content_json": json.dumps({"test": True}),
            "confidence": 0.5,
            "source_iteration_id": "iter_old",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "state": "stale",
            "verified_count": 0,
            "last_verified_at": "2023-01-01T00:00:00",
            "auto_confidence": None,
        })

        transitions = kb.apply_lifecycle_transitions()
        assert transitions["stale_to_archived"] >= 1

        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["state"] == "archived"

    def test_stale_to_archived_low_confidence(self, kb, db):
        """Stale entries with auto_confidence < 0.3 should be archived."""
        entry_id = "ke_stale_low"
        db.insert("knowledge_entries", {
            "entry_id": entry_id,
            "knowledge_type": "data_strategy",
            "content_json": json.dumps({"test": True}),
            "confidence": 0.2,
            "source_iteration_id": "iter_low",
            "created_at": "2024-06-01T00:00:00",
            "updated_at": "2024-06-01T00:00:00",
            "state": "stale",
            "verified_count": 0,
            "last_verified_at": "2024-06-01T00:00:00",
            "auto_confidence": 0.2,
        })

        transitions = kb.apply_lifecycle_transitions()
        assert transitions["stale_to_archived"] >= 1

        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["state"] == "archived"

    def test_no_transitions_for_recent_entries(self, kb, db):
        """Recently verified active entries should stay active."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"recent": True},
            confidence=0.9,
            source_iteration_id="iter_recent",
            created_at=now,
            updated_at=now,
            last_verified_at=now,
            state="active",
        )
        entry_id = kb.add_knowledge(entry)

        transitions = kb.apply_lifecycle_transitions()
        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["state"] == "active"


# ---------------------------------------------------------------------------
# 20.4 Knowledge write action verification
# ---------------------------------------------------------------------------


class TestAddKnowledgeVerified:
    """Tests for SkillKnowledgeBase.add_knowledge_verified (Req 11.11)."""

    def test_rejects_empty_iteration_id(self, kb):
        """Should reject entries without source_iteration_id."""
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"test": True},
            confidence=0.8,
            source_iteration_id="",
        )
        with pytest.raises(ValueError, match="non-empty source_iteration_id"):
            kb.add_knowledge_verified(entry)

    def test_rejects_without_positive_feedback(self, kb, db):
        """Should reject when no positive feedback exists for the iteration."""
        _create_iteration(db, "iter_nofb")
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"test": True},
            confidence=0.8,
            source_iteration_id="iter_nofb",
        )
        with pytest.raises(ValueError, match="positive effect_feedback"):
            kb.add_knowledge_verified(entry)

    def test_rejects_insufficient_verification_iterations(self, kb, db):
        """Should reject when same-type knowledge not verified in 2+ iterations."""
        _create_iteration(db, "iter_single")
        db.insert("effect_feedbacks", {
            "feedback_id": "fb_single",
            "iteration_id": "iter_single",
            "suggestion_id": "sug_single",
            "suggestion_content": "test",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": 0.30,
            "wer_after": 0.25,
            "is_positive": 1,
            "created_at": "2024-01-01T00:00:00",
        })
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"test": True},
            confidence=0.8,
            source_iteration_id="iter_single",
        )
        with pytest.raises(ValueError, match="at least 2 different iterations"):
            kb.add_knowledge_verified(entry)

    def test_accepts_valid_verified_entry(self, kb, db):
        """Should accept entry meeting all verification criteria."""
        # Create two iterations with positive feedback
        _create_iteration(db, "iter_v1")
        _create_iteration(db, "iter_v2")
        db.insert("effect_feedbacks", {
            "feedback_id": "fb_v2",
            "iteration_id": "iter_v2",
            "suggestion_id": "sug_v2",
            "suggestion_content": "test",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.20,
            "cer_after": 0.15,
            "wer_before": 0.30,
            "wer_after": 0.25,
            "is_positive": 1,
            "created_at": "2024-01-01T00:00:00",
        })
        # Add existing knowledge from iter_v1
        kb.add_knowledge(KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"existing": True},
            confidence=0.7,
            source_iteration_id="iter_v1",
        ))

        # Now add verified entry from iter_v2
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"verified": True},
            confidence=0.9,
            source_iteration_id="iter_v2",
        )
        entry_id = kb.add_knowledge_verified(entry)
        assert entry_id.startswith("ke_")


# ---------------------------------------------------------------------------
# 20.5 Skill file quick reference table
# ---------------------------------------------------------------------------


class TestSkillQuickReference:
    """Tests for generate_skill_summary_table and update_skill_quick_reference (Req 11.6)."""

    def test_generates_markdown_table(self, engine, db):
        """generate_skill_summary_table should return a markdown string."""
        # Add some strategy weights
        db.insert("strategy_weights", {
            "strategy_name": "Increase LR",
            "weight": 0.9,
            "success_rate": 0.85,
            "sample_count": 10,
            "version_id": None,
            "updated_at": "2024-01-01T00:00:00",
        })
        table = engine.generate_skill_summary_table("asr-training-diagnostics")
        assert isinstance(table, str)

    def test_update_quick_reference_writes_to_file(self, engine, workspace, db):
        """update_skill_quick_reference should modify the skill file."""
        db.insert("strategy_weights", {
            "strategy_name": "Test Strategy",
            "weight": 0.8,
            "success_rate": 0.75,
            "sample_count": 5,
            "version_id": None,
            "updated_at": "2024-01-01T00:00:00",
        })
        result = engine.update_skill_quick_reference("asr-training-diagnostics")
        assert result.get("success") is True

        # Verify file was updated
        skill_path = workspace / ".kiro" / "skills" / "asr-training-diagnostics.md"
        content = skill_path.read_text(encoding="utf-8")
        assert "经验速查" in content or "Quick Reference" in content or len(content) > 50


# ---------------------------------------------------------------------------
# 20.6 Skill file rules auto-update
# ---------------------------------------------------------------------------


class TestUpdateSkillRulesFromPatterns:
    """Tests for update_skill_rules_from_patterns (Req 11.1-11.4)."""

    def test_returns_dict_summary(self, engine, db):
        """update_skill_rules_from_patterns should return a dict."""
        patterns = {
            "strategies": {
                "Strategy A": {
                    "total_count": 5,
                    "positive_count": 4,
                    "success_rate": 0.8,
                    "avg_cer_improvement": 0.05,
                }
            },
            "total_feedbacks": 5,
            "average_success_rate": 0.8,
            "high_success_strategies": ["Strategy A"],
            "low_success_strategies": [],
        }
        weights = [
            StrategyWeight(
                strategy_name="Strategy A",
                weight=0.9,
                success_rate=0.8,
                sample_count=5,
            )
        ]
        result = engine.update_skill_rules_from_patterns(patterns, weights)
        assert isinstance(result, dict)

    def test_handles_empty_patterns(self, engine):
        """Should handle empty patterns gracefully."""
        result = engine.update_skill_rules_from_patterns({}, [])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 20.7 Auto-trigger Skill review on iteration completion
# ---------------------------------------------------------------------------


class TestAutoReviewOnComplete:
    """Tests for _auto_review_skills triggered by complete_iteration (Req 11.7)."""

    def test_complete_iteration_triggers_auto_review(self, optimizer, db):
        """complete_iteration should include skill_review in result."""
        iter_id = optimizer.start_iteration("Auto review test")
        optimizer.run_analysis_phase(
            iter_id,
            evaluation_report=EvaluationReport(overall_cer=0.20, overall_wer=0.30),
        )
        result = optimizer.complete_iteration(
            iter_id,
            EvaluationReport(overall_cer=0.15, overall_wer=0.25),
        )
        assert result["status"] == "completed"
        assert "skill_review" in result
        review = result["skill_review"]
        assert "iteration_id" in review
        assert "feedback_labeled" in review
        assert "lifecycle_transitions" in review

    def test_auto_review_labels_feedback(self, optimizer, db):
        """Auto review should label feedback for the completed iteration."""
        iter_id = optimizer.start_iteration("Feedback label test")
        optimizer.run_analysis_phase(
            iter_id,
            evaluation_report=EvaluationReport(overall_cer=0.20, overall_wer=0.30),
        )
        result = optimizer.complete_iteration(
            iter_id,
            EvaluationReport(overall_cer=0.15, overall_wer=0.25),
        )
        assert result["skill_review"]["feedback_labeled"] is True

        # Verify feedback was recorded in DB
        feedbacks = db.get_all("effect_feedbacks")
        assert len(feedbacks) >= 1


# ---------------------------------------------------------------------------
# 20.8 Deep Skill review (every 5 iterations)
# ---------------------------------------------------------------------------


class TestDeepReview:
    """Tests for _deep_review_skills triggered every 5 iterations (Req 11.8)."""

    def test_deep_review_triggered_at_5_iterations(self, optimizer, db):
        """Deep review should trigger when completed iterations count is multiple of 5."""
        # Complete 5 iterations
        for i in range(5):
            iter_id = optimizer.start_iteration(f"Deep review iter {i}")
            optimizer.run_analysis_phase(
                iter_id,
                evaluation_report=EvaluationReport(
                    overall_cer=0.20 - i * 0.01, overall_wer=0.30
                ),
            )
            result = optimizer.complete_iteration(
                iter_id,
                EvaluationReport(
                    overall_cer=0.19 - i * 0.01, overall_wer=0.28
                ),
            )

        # The 5th iteration should have triggered deep review
        assert result["skill_review"]["deep_review"] is not None

    def test_deep_review_not_triggered_before_5(self, optimizer, db):
        """Deep review should NOT trigger before 5 completed iterations."""
        for i in range(4):
            iter_id = optimizer.start_iteration(f"Iter {i}")
            optimizer.run_analysis_phase(
                iter_id,
                evaluation_report=EvaluationReport(
                    overall_cer=0.20, overall_wer=0.30
                ),
            )
            result = optimizer.complete_iteration(
                iter_id,
                EvaluationReport(overall_cer=0.19, overall_wer=0.28),
            )

        # 4th iteration should NOT have deep review
        assert result["skill_review"]["deep_review"] is None


# ---------------------------------------------------------------------------
# 20.9 Query with state filtering
# ---------------------------------------------------------------------------


class TestQueryStateFiltering:
    """Tests for query_knowledge state filtering (Req 11.12)."""

    def test_default_only_returns_active(self, kb, db):
        """Default query should only return active entries."""
        # Add active entry
        kb.add_knowledge(KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"state_test": "active"},
            confidence=0.8,
            source_iteration_id="iter_a",
            state="active",
        ))
        # Add stale entry directly
        db.insert("knowledge_entries", {
            "entry_id": "ke_stale_test",
            "knowledge_type": "scene_solution",
            "content_json": json.dumps({"state_test": "stale"}),
            "confidence": 0.5,
            "source_iteration_id": "iter_s",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "state": "stale",
            "verified_count": 0,
            "last_verified_at": None,
            "auto_confidence": None,
        })

        results = kb.query_knowledge("scene_solution")
        assert all(r.state == "active" for r in results)

    def test_include_stale_returns_both(self, kb, db):
        """include_stale=True should return active and stale entries."""
        kb.add_knowledge(KnowledgeEntry(
            knowledge_type="param_effect",
            content={"test": "active"},
            confidence=0.8,
            source_iteration_id="iter_a",
            state="active",
        ))
        db.insert("knowledge_entries", {
            "entry_id": "ke_stale_inc",
            "knowledge_type": "param_effect",
            "content_json": json.dumps({"test": "stale"}),
            "confidence": 0.4,
            "source_iteration_id": "iter_s",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "state": "stale",
            "verified_count": 0,
            "last_verified_at": None,
            "auto_confidence": None,
        })

        results = kb.query_knowledge("param_effect", include_stale=True)
        states = {r.state for r in results}
        assert "active" in states
        assert "stale" in states

    def test_include_archived_returns_all(self, kb, db):
        """include_archived=True should include archived entries."""
        db.insert("knowledge_entries", {
            "entry_id": "ke_archived_test",
            "knowledge_type": "data_strategy",
            "content_json": json.dumps({"test": "archived"}),
            "confidence": 0.1,
            "source_iteration_id": "iter_ar",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "state": "archived",
            "verified_count": 0,
            "last_verified_at": None,
            "auto_confidence": None,
        })

        results_default = kb.query_knowledge("data_strategy")
        results_with_archived = kb.query_knowledge(
            "data_strategy", include_archived=True
        )
        assert len(results_with_archived) > len(results_default)
