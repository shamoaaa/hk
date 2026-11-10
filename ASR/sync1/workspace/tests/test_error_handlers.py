"""Tests for the global error handlers module.

Covers ``TrainingErrorHandler``, ``ConfigErrorHandler``, and
``SkillOutputHandler`` with unit tests for core functionality,
edge cases, and error logging transparency.
"""

import json
import os

import pytest

from scripts.core.error_handlers import (
    ConfigErrorHandler,
    SkillOutputHandler,
    TrainingErrorHandler,
)
from scripts.decision.llm_skill_engine import LLMSkillEngine
from scripts.execution.model_optimizer import ModelOptimizer
from scripts.core.models import AlertType, ModelConfig, TrainingAlert
from scripts.decision.skill_evolution import SkillEvolutionEngine


# =====================================================================
# TrainingErrorHandler tests
# =====================================================================


class TestTrainingErrorHandler:
    """Tests for TrainingErrorHandler."""

    def test_handle_nan_inf_returns_alert(self):
        """handle_nan_inf returns a NAN_INF TrainingAlert."""
        handler = TrainingErrorHandler()
        context = {
            "recent_losses": [1.0, 1.1, float("nan")],
            "learning_rate": 1e-4,
            "gradient_norm": 2.5,
        }
        alert = handler.handle_nan_inf(step=500, epoch=3, context=context)

        assert isinstance(alert, TrainingAlert)
        assert alert.alert_type == AlertType.NAN_INF
        assert alert.step == 500
        assert alert.epoch == 3
        assert "NaN/Inf" in alert.message
        assert alert.context["learning_rate"] == 1e-4

    def test_handle_nan_inf_logs_event(self):
        """handle_nan_inf appends an event to the error log."""
        handler = TrainingErrorHandler()
        handler.handle_nan_inf(step=10, epoch=0, context={})

        assert len(handler.error_log) == 1
        event = handler.error_log[0]
        assert event["event_type"] == "nan_inf"
        assert event["step"] == 10
        assert event["epoch"] == 0
        assert event["action_taken"] == "training_paused"
        assert "timestamp" in event

    def test_handle_nan_inf_preserves_context(self):
        """handle_nan_inf stores a copy of the context, not a reference."""
        handler = TrainingErrorHandler()
        context = {"learning_rate": 0.001}
        handler.handle_nan_inf(step=1, epoch=0, context=context)

        # Mutate original context
        context["learning_rate"] = 999
        assert handler.error_log[0]["context"]["learning_rate"] == 0.001

    def test_handle_gradient_explosion_clipping(self):
        """First gradient explosion triggers gradient clipping."""
        handler = TrainingErrorHandler()
        alert = handler.handle_gradient_explosion(
            gradient_norm=50.0, threshold=5.0
        )

        assert isinstance(alert, TrainingAlert)
        assert alert.alert_type == AlertType.GRADIENT_EXPLOSION
        assert alert.context["action"] == "gradient_clipping"
        assert "50.0" in alert.message

    def test_handle_gradient_explosion_lr_reduction_after_max_attempts(self):
        """After max_gradient_clip_attempts, action switches to LR reduction."""
        handler = TrainingErrorHandler(max_gradient_clip_attempts=2)

        # First two attempts: clipping
        alert1 = handler.handle_gradient_explosion(100.0, 5.0)
        assert alert1.context["action"] == "gradient_clipping"

        alert2 = handler.handle_gradient_explosion(80.0, 5.0)
        assert alert2.context["action"] == "gradient_clipping"

        # Third attempt: LR reduction
        alert3 = handler.handle_gradient_explosion(60.0, 5.0)
        assert alert3.context["action"] == "reduce_learning_rate"
        assert "reducing learning rate" in alert3.message.lower()

    def test_handle_gradient_explosion_logs_events(self):
        """Each gradient explosion call appends to the error log."""
        handler = TrainingErrorHandler()
        handler.handle_gradient_explosion(10.0, 5.0)
        handler.handle_gradient_explosion(20.0, 5.0)

        assert len(handler.error_log) == 2
        assert all(
            e["event_type"] == "gradient_explosion"
            for e in handler.error_log
        )

    def test_handle_gradient_explosion_lr_factor_in_context(self):
        """The LR reduction factor is included in the alert context."""
        handler = TrainingErrorHandler(lr_reduction_factor=0.25)
        alert = handler.handle_gradient_explosion(50.0, 5.0)
        assert alert.context["lr_reduction_factor"] == 0.25

    def test_multiple_nan_inf_events_accumulate(self):
        """Multiple NaN/Inf events all get logged."""
        handler = TrainingErrorHandler()
        handler.handle_nan_inf(step=1, epoch=0, context={})
        handler.handle_nan_inf(step=2, epoch=0, context={})
        handler.handle_nan_inf(step=3, epoch=1, context={})

        assert len(handler.error_log) == 3


# =====================================================================
# ConfigErrorHandler tests
# =====================================================================


class TestConfigErrorHandler:
    """Tests for ConfigErrorHandler."""

    def _make_optimizer_with_history(self, tmp_path):
        """Create a ModelOptimizer with one config in history."""
        config_dir = str(tmp_path / "configs" / "model")
        os.makedirs(config_dir, exist_ok=True)

        initial = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"co_decoder_layers": 2},
        )
        optimizer = ModelOptimizer(config_dir, initial_config=initial)
        # Apply a change so there's history to roll back to
        optimizer.apply_attention_config({"attention_heads": 16, "hidden_dim": 1024})
        return optimizer

    def test_handle_invalid_config_rolls_back(self, tmp_path):
        """handle_invalid_config rolls back and returns previous config."""
        optimizer = self._make_optimizer_with_history(tmp_path)
        handler = ConfigErrorHandler(optimizer)

        bad_config = ModelConfig(attention_heads=0, hidden_dim=-1)
        errors = ["attention_heads must be positive", "hidden_dim must be positive"]

        result = handler.handle_invalid_config(bad_config, errors)

        # Should have rolled back to the initial config (8 heads, 512 dim)
        assert result.attention_heads == 8
        assert result.hidden_dim == 512

    def test_handle_invalid_config_logs_event(self, tmp_path):
        """handle_invalid_config records the event in error_log."""
        optimizer = self._make_optimizer_with_history(tmp_path)
        handler = ConfigErrorHandler(optimizer)

        bad_config = ModelConfig(attention_heads=3, hidden_dim=100)
        handler.handle_invalid_config(bad_config, ["not divisible"])

        assert len(handler.error_log) == 1
        event = handler.error_log[0]
        assert event["event_type"] == "invalid_config"
        assert event["errors"] == ["not divisible"]
        assert event["action_taken"] == "rollback"

    def test_handle_invalid_config_no_history_raises(self, tmp_path):
        """handle_invalid_config raises ValueError when no history exists."""
        config_dir = str(tmp_path / "configs" / "model")
        os.makedirs(config_dir, exist_ok=True)
        optimizer = ModelOptimizer(config_dir, initial_config=ModelConfig(
            attention_heads=8, hidden_dim=512
        ))
        handler = ConfigErrorHandler(optimizer)

        with pytest.raises(ValueError, match="No previous configuration"):
            handler.handle_invalid_config(ModelConfig(), ["error"])

    def test_handle_init_failure_rolls_back(self, tmp_path):
        """handle_init_failure rolls back and returns previous config."""
        optimizer = self._make_optimizer_with_history(tmp_path)
        handler = ConfigErrorHandler(optimizer)

        bad_config = ModelConfig(attention_heads=64, hidden_dim=8192)
        error = MemoryError("Out of memory")

        result = handler.handle_init_failure(bad_config, error)

        assert result.attention_heads == 8
        assert result.hidden_dim == 512

    def test_handle_init_failure_logs_event(self, tmp_path):
        """handle_init_failure records error type and message."""
        optimizer = self._make_optimizer_with_history(tmp_path)
        handler = ConfigErrorHandler(optimizer)

        error = RuntimeError("Incompatible parameters")
        handler.handle_init_failure(ModelConfig(), error)

        assert len(handler.error_log) == 1
        event = handler.error_log[0]
        assert event["event_type"] == "init_failure"
        assert event["error_type"] == "RuntimeError"
        assert "Incompatible" in event["error_message"]

    def test_handle_init_failure_no_history_raises(self, tmp_path):
        """handle_init_failure raises ValueError when no history exists."""
        config_dir = str(tmp_path / "configs" / "model")
        os.makedirs(config_dir, exist_ok=True)
        optimizer = ModelOptimizer(config_dir, initial_config=ModelConfig(
            attention_heads=4, hidden_dim=256
        ))
        handler = ConfigErrorHandler(optimizer)

        with pytest.raises(ValueError, match="No previous configuration"):
            handler.handle_init_failure(ModelConfig(), ValueError("bad"))


# =====================================================================
# SkillOutputHandler tests
# =====================================================================


class TestSkillOutputHandler:
    """Tests for SkillOutputHandler."""

    @pytest.fixture
    def engine(self):
        """Return a fresh LLMSkillEngine."""
        return LLMSkillEngine()

    @pytest.fixture
    def handler(self, engine):
        """Return a SkillOutputHandler with no evolution engine."""
        return SkillOutputHandler(
            llm_skill_engine=engine,
            skill_evolution_engine=None,
            workspace_dir="workspace",
        )

    # ------------------------------------------------------------------
    # parse_optimization_output
    # ------------------------------------------------------------------

    def test_parse_valid_json_output(self, handler):
        """Valid JSON output is parsed successfully."""
        output = json.dumps({
            "iteration_id": "iter_001",
            "diagnosis_summary": "All good",
            "suggestions": [
                {
                    "dimension": "param",
                    "title": "Reduce LR",
                    "description": "Lower learning rate",
                    "priority": 3,
                    "confidence": 0.8,
                    "config_changes": {"learning_rate": 1e-5},
                    "rationale": "Loss plateau",
                },
            ],
        })

        result = handler.parse_optimization_output(output)

        assert result["status"] == "success"
        assert len(result["suggestions"]) == 1
        assert result["suggestions"][0]["dimension"] == "param"
        assert result["diagnosis_summary"] == "All good"

    def test_parse_json_in_code_block(self, handler):
        """JSON wrapped in ```json ... ``` is parsed successfully."""
        output = (
            "Here is my analysis:\n"
            "```json\n"
            '{"iteration_id": "iter_002", "suggestions": ['
            '{"dimension": "model", "title": "Add heads", '
            '"description": "More attention", "priority": 2, '
            '"confidence": 0.7, "config_changes": {}, "rationale": "test"}'
            "], \"diagnosis_summary\": \"ok\"}\n"
            "```\n"
            "Let me know if you need more."
        )

        result = handler.parse_optimization_output(output)
        assert result["status"] == "success"
        assert len(result["suggestions"]) == 1

    def test_parse_keyword_fallback(self, handler):
        """When JSON parsing fails, keyword fallback extracts dimensions."""
        output = "建议调整模型的注意力头数，同时优化数据增强策略。"

        result = handler.parse_optimization_output(output)

        assert result["status"] == "fallback"
        dimensions = {s["dimension"] for s in result["suggestions"]}
        assert "model" in dimensions
        assert "data" in dimensions

    def test_parse_complete_failure(self, handler):
        """Completely unparseable output returns manual_review_needed."""
        output = "This is just random text with no useful content."

        result = handler.parse_optimization_output(output)

        assert result["status"] == "manual_review_needed"
        assert result["suggestions"] == []
        assert result["raw_output"] == output

    def test_parse_failure_logs_event(self, handler):
        """Parse failure is recorded in the error log."""
        handler.parse_optimization_output("unparseable gibberish")

        assert len(handler.error_log) == 1
        assert handler.error_log[0]["event_type"] == "parse_failure"

    def test_parse_fallback_logs_event(self, handler):
        """Keyword fallback parsing is recorded in the error log."""
        handler.parse_optimization_output("需要调整学习率参数")

        assert len(handler.error_log) == 1
        assert handler.error_log[0]["event_type"] == "parse_fallback"

    def test_parse_empty_output(self, handler):
        """Empty string triggers manual review."""
        result = handler.parse_optimization_output("")
        assert result["status"] == "manual_review_needed"

    # ------------------------------------------------------------------
    # validate_skill_file
    # ------------------------------------------------------------------

    def test_validate_valid_skill_file(self, handler, tmp_path):
        """A well-formed Skill file passes validation."""
        skill_file = tmp_path / "test_skill.md"
        skill_file.write_text(
            "# ASR Training Diagnostics\n\n"
            "## 能力描述\n\n"
            "This skill analyzes training metrics.\n\n"
            "## 分析规则\n\n"
            "1. Check loss trends\n",
            encoding="utf-8",
        )

        is_valid, errors = handler.validate_skill_file(str(skill_file))

        assert is_valid is True
        assert errors == []

    def test_validate_missing_file(self, handler):
        """A non-existent file fails validation."""
        is_valid, errors = handler.validate_skill_file("/nonexistent/skill.md")

        assert is_valid is False
        assert any("not found" in e for e in errors)

    def test_validate_no_headings(self, handler, tmp_path):
        """A file with no Markdown headings fails validation."""
        skill_file = tmp_path / "no_headings.md"
        skill_file.write_text(
            "This file has no headings at all.\n"
            "Just plain text.\n",
            encoding="utf-8",
        )

        is_valid, errors = handler.validate_skill_file(str(skill_file))

        assert is_valid is False
        assert any("no headings" in e.lower() for e in errors)

    def test_validate_missing_required_sections(self, handler, tmp_path):
        """A file missing required sections fails validation."""
        skill_file = tmp_path / "incomplete.md"
        skill_file.write_text(
            "# My Skill\n\n"
            "## Some Other Section\n\n"
            "Content here.\n",
            encoding="utf-8",
        )

        is_valid, errors = handler.validate_skill_file(str(skill_file))

        assert is_valid is False
        assert any("能力描述" in e for e in errors)
        assert any("分析规则" in e for e in errors)

    def test_validate_invalid_knowledge_ref(self, handler, tmp_path):
        """Invalid knowledge base references are flagged."""
        skill_file = tmp_path / "bad_ref.md"
        skill_file.write_text(
            "# Skill\n\n"
            "## 能力描述\n\nDesc\n\n"
            "## 分析规则\n\nRules\n\n"
            "See knowledge/nonexistent_file.json for details.\n",
            encoding="utf-8",
        )

        is_valid, errors = handler.validate_skill_file(str(skill_file))

        assert is_valid is False
        assert any("knowledge/nonexistent_file.json" in e for e in errors)

    def test_validate_logs_event(self, handler, tmp_path):
        """Validation results are logged."""
        skill_file = tmp_path / "skill.md"
        skill_file.write_text(
            "# Skill\n## 能力描述\nDesc\n## 分析规则\nRules\n",
            encoding="utf-8",
        )

        handler.validate_skill_file(str(skill_file))

        assert len(handler.error_log) == 1
        assert handler.error_log[0]["event_type"] == "skill_validation"
        assert handler.error_log[0]["is_valid"] is True

    # ------------------------------------------------------------------
    # handle_skill_evolution_failure
    # ------------------------------------------------------------------

    def test_evolution_failure_no_engine(self, handler):
        """Without an evolution engine, rollback returns False."""
        result = handler.handle_skill_evolution_failure("v001")

        assert result is False
        assert len(handler.error_log) == 1
        assert handler.error_log[0]["result"] == "no_evolution_engine"

    def test_evolution_failure_successful_rollback(self, tmp_path, db):
        """Successful rollback returns True."""
        workspace_dir = str(tmp_path)
        skills_dir = os.path.join(workspace_dir, ".kiro", "skills")
        versions_dir = os.path.join(
            workspace_dir, "knowledge", "skill_versions", "v_test"
        )
        os.makedirs(skills_dir, exist_ok=True)
        os.makedirs(versions_dir, exist_ok=True)

        # Create a version snapshot file
        with open(os.path.join(versions_dir, "skill.md"), "w") as f:
            f.write("# Old Skill\n## 能力描述\nOld\n## 分析规则\nOld\n")

        # Register version in DB
        db.insert("skill_versions", {
            "version_id": "v_test",
            "strategy_rules_json": "{}",
            "knowledge_snapshot_path": versions_dir,
            "change_description": "test",
            "positive_feedback_rate": 0.5,
            "avg_cer_improvement": 0.01,
            "created_at": "2024-01-01T00:00:00",
        })

        engine = SkillEvolutionEngine(workspace_dir, db)
        llm_engine = LLMSkillEngine()
        handler = SkillOutputHandler(llm_engine, engine, workspace_dir)

        result = handler.handle_skill_evolution_failure("v_test")

        assert result is True
        assert len(handler.error_log) == 1
        assert handler.error_log[0]["result"] == "success"

        # Verify the file was copied
        assert os.path.isfile(os.path.join(skills_dir, "skill.md"))

    def test_evolution_failure_version_not_found(self, tmp_path, db):
        """Rollback to a non-existent version returns False."""
        workspace_dir = str(tmp_path)
        os.makedirs(os.path.join(workspace_dir, "knowledge", "skill_versions"), exist_ok=True)

        engine = SkillEvolutionEngine(workspace_dir, db)
        llm_engine = LLMSkillEngine()
        handler = SkillOutputHandler(llm_engine, engine, workspace_dir)

        result = handler.handle_skill_evolution_failure("nonexistent_version")

        assert result is False
        assert handler.error_log[0]["result"] == "failed"
