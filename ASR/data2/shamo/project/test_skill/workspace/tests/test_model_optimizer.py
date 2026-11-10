"""Tests for the model_optimizer module.

Covers:
- apply_attention_config: adjusting attention heads and hidden dim
- add_speech_enhance_module: adding/updating speech enhancement config
- adjust_seaco_params: adjusting SeACo module parameters
- validate_config: validation rules for attention, hidden_dim, SeACo params
- Configuration history and rollback basics
- Config persistence via versioned YAML files
"""

import os
import sys

import pytest

from scripts.core.config_utils import list_versions, read_model_config
from scripts.execution.model_optimizer import ModelOptimizer
from scripts.core.models import ModelConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def optimizer(tmp_workspace):
    """Return a ModelOptimizer with a valid initial config."""
    config_dir = str(tmp_workspace / "configs" / "model")
    initial = ModelConfig(
        attention_heads=8,
        hidden_dim=512,
        seaco_params={
            "co_decoder_layers": 2,
            "attention_weight": 0.5,
            "vocab_size": 8404,
        },
        speech_enhance_config=None,
        extra_params={"dropout": 0.1},
    )
    return ModelOptimizer(config_dir=config_dir, initial_config=initial)


@pytest.fixture
def config_dir(tmp_workspace):
    """Return the model config directory path."""
    return str(tmp_workspace / "configs" / "model")


# =============================================================================
# apply_attention_config Tests
# =============================================================================


class TestApplyAttentionConfig:
    def test_update_attention_heads(self, optimizer):
        result = optimizer.apply_attention_config({"attention_heads": 16})
        assert result.attention_heads == 16
        assert result.hidden_dim == 512  # unchanged

    def test_update_hidden_dim(self, optimizer):
        result = optimizer.apply_attention_config({"hidden_dim": 1024})
        assert result.hidden_dim == 1024
        assert result.attention_heads == 8  # unchanged

    def test_update_both(self, optimizer):
        result = optimizer.apply_attention_config(
            {"attention_heads": 16, "hidden_dim": 1024}
        )
        assert result.attention_heads == 16
        assert result.hidden_dim == 1024

    def test_preserves_other_fields(self, optimizer):
        result = optimizer.apply_attention_config({"attention_heads": 4})
        assert result.seaco_params == {
            "co_decoder_layers": 2,
            "attention_weight": 0.5,
            "vocab_size": 8404,
        }
        assert result.extra_params == {"dropout": 0.1}

    def test_saves_to_history(self, optimizer):
        optimizer.apply_attention_config({"attention_heads": 16})
        history = optimizer.history
        assert len(history) == 1
        assert history[0].attention_heads == 8  # original value

    def test_persists_versioned_config(self, optimizer, config_dir):
        optimizer.apply_attention_config({"attention_heads": 16})
        versions = list_versions(config_dir, "model")
        assert len(versions) == 1
        loaded = read_model_config(versions[0])
        assert loaded.attention_heads == 16

    def test_empty_config_no_change(self, optimizer):
        result = optimizer.apply_attention_config({})
        assert result.attention_heads == 8
        assert result.hidden_dim == 512


# =============================================================================
# add_speech_enhance_module Tests
# =============================================================================


class TestAddSpeechEnhanceModule:
    def test_add_module(self, optimizer):
        module_config = {"type": "conv_tasnet", "channels": 64, "layers": 4}
        result = optimizer.add_speech_enhance_module(module_config)
        assert result.speech_enhance_config == module_config

    def test_replace_existing_module(self, optimizer):
        optimizer.add_speech_enhance_module({"type": "conv_tasnet", "channels": 64})
        result = optimizer.add_speech_enhance_module(
            {"type": "dprnn", "channels": 128, "layers": 6}
        )
        assert result.speech_enhance_config == {
            "type": "dprnn",
            "channels": 128,
            "layers": 6,
        }

    def test_preserves_other_fields(self, optimizer):
        result = optimizer.add_speech_enhance_module({"type": "conv_tasnet"})
        assert result.attention_heads == 8
        assert result.hidden_dim == 512

    def test_saves_to_history(self, optimizer):
        optimizer.add_speech_enhance_module({"type": "conv_tasnet"})
        history = optimizer.history
        assert len(history) == 1
        assert history[0].speech_enhance_config is None  # was None before

    def test_persists_versioned_config(self, optimizer, config_dir):
        optimizer.add_speech_enhance_module({"type": "conv_tasnet", "channels": 64})
        versions = list_versions(config_dir, "model")
        assert len(versions) == 1
        loaded = read_model_config(versions[0])
        assert loaded.speech_enhance_config == {"type": "conv_tasnet", "channels": 64}


# =============================================================================
# adjust_seaco_params Tests
# =============================================================================


class TestAdjustSeacoParams:
    def test_update_single_param(self, optimizer):
        result = optimizer.adjust_seaco_params({"co_decoder_layers": 4})
        assert result.seaco_params["co_decoder_layers"] == 4
        # Other SeACo params preserved
        assert result.seaco_params["attention_weight"] == 0.5
        assert result.seaco_params["vocab_size"] == 8404

    def test_update_multiple_params(self, optimizer):
        result = optimizer.adjust_seaco_params(
            {"co_decoder_layers": 6, "attention_weight": 0.8}
        )
        assert result.seaco_params["co_decoder_layers"] == 6
        assert result.seaco_params["attention_weight"] == 0.8

    def test_add_new_param(self, optimizer):
        result = optimizer.adjust_seaco_params({"new_param": 42})
        assert result.seaco_params["new_param"] == 42
        # Existing params preserved
        assert result.seaco_params["co_decoder_layers"] == 2

    def test_preserves_other_fields(self, optimizer):
        result = optimizer.adjust_seaco_params({"attention_weight": 0.9})
        assert result.attention_heads == 8
        assert result.hidden_dim == 512

    def test_saves_to_history(self, optimizer):
        optimizer.adjust_seaco_params({"co_decoder_layers": 4})
        history = optimizer.history
        assert len(history) == 1
        assert history[0].seaco_params["co_decoder_layers"] == 2  # original

    def test_persists_versioned_config(self, optimizer, config_dir):
        optimizer.adjust_seaco_params({"attention_weight": 0.9})
        versions = list_versions(config_dir, "model")
        assert len(versions) == 1


# =============================================================================
# validate_config Tests
# =============================================================================


class TestValidateConfig:
    """Tests for validate_config covering all validation rules."""

    # --- Valid configurations ---

    def test_valid_basic_config(self, optimizer):
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={
                "co_decoder_layers": 2,
                "attention_weight": 0.5,
                "vocab_size": 8404,
            },
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is True
        assert msg == ""

    def test_valid_minimal_config(self, optimizer):
        config = ModelConfig(attention_heads=1, hidden_dim=1, seaco_params={})
        valid, msg = optimizer.validate_config(config)
        assert valid is True

    def test_valid_large_config(self, optimizer):
        config = ModelConfig(
            attention_heads=16,
            hidden_dim=1024,
            seaco_params={
                "co_decoder_layers": 6,
                "attention_weight": 1.0,
                "vocab_size": 100000,
            },
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is True

    def test_valid_seaco_boundary_min(self, optimizer):
        config = ModelConfig(
            attention_heads=4,
            hidden_dim=256,
            seaco_params={
                "co_decoder_layers": 1,
                "attention_weight": 0.0,
                "vocab_size": 1,
            },
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is True

    def test_valid_no_seaco_params(self, optimizer):
        config = ModelConfig(attention_heads=4, hidden_dim=256, seaco_params={})
        valid, msg = optimizer.validate_config(config)
        assert valid is True

    # --- Invalid: attention_heads ---

    def test_invalid_attention_heads_zero(self, optimizer):
        config = ModelConfig(attention_heads=0, hidden_dim=512)
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "attention_heads" in msg

    def test_invalid_attention_heads_negative(self, optimizer):
        config = ModelConfig(attention_heads=-1, hidden_dim=512)
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "attention_heads" in msg

    # --- Invalid: hidden_dim ---

    def test_invalid_hidden_dim_zero(self, optimizer):
        config = ModelConfig(attention_heads=8, hidden_dim=0)
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "hidden_dim" in msg

    def test_invalid_hidden_dim_negative(self, optimizer):
        config = ModelConfig(attention_heads=8, hidden_dim=-256)
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "hidden_dim" in msg

    # --- Invalid: divisibility ---

    def test_invalid_not_divisible(self, optimizer):
        config = ModelConfig(attention_heads=8, hidden_dim=513)
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "divisible" in msg

    def test_invalid_not_divisible_prime(self, optimizer):
        config = ModelConfig(attention_heads=7, hidden_dim=100)
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "divisible" in msg

    # --- Invalid: SeACo co_decoder_layers ---

    def test_invalid_seaco_layers_zero(self, optimizer):
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"co_decoder_layers": 0},
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "co_decoder_layers" in msg

    def test_invalid_seaco_layers_too_high(self, optimizer):
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"co_decoder_layers": 7},
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "co_decoder_layers" in msg

    def test_invalid_seaco_layers_negative(self, optimizer):
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"co_decoder_layers": -1},
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "co_decoder_layers" in msg

    # --- Invalid: SeACo attention_weight ---

    def test_invalid_seaco_weight_negative(self, optimizer):
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"attention_weight": -0.1},
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "attention_weight" in msg

    def test_invalid_seaco_weight_above_one(self, optimizer):
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"attention_weight": 1.1},
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "attention_weight" in msg

    # --- Invalid: SeACo vocab_size ---

    def test_invalid_seaco_vocab_zero(self, optimizer):
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"vocab_size": 0},
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "vocab_size" in msg

    def test_invalid_seaco_vocab_negative(self, optimizer):
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"vocab_size": -100},
        )
        valid, msg = optimizer.validate_config(config)
        assert valid is False
        assert "vocab_size" in msg


# =============================================================================
# Rollback Tests (basic – full rollback in task 8.3)
# =============================================================================


class TestRollback:
    def test_rollback_restores_previous(self, optimizer):
        original = optimizer.current_config
        optimizer.apply_attention_config({"attention_heads": 16, "hidden_dim": 1024})
        restored = optimizer.rollback()
        assert restored.attention_heads == original.attention_heads
        assert restored.hidden_dim == original.hidden_dim

    def test_rollback_empty_history_raises(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "model")
        opt = ModelOptimizer(config_dir=config_dir)
        with pytest.raises(ValueError, match="No previous configuration"):
            opt.rollback()

    def test_multi_step_rollback(self, optimizer):
        optimizer.apply_attention_config({"attention_heads": 16})
        optimizer.apply_attention_config({"attention_heads": 32})
        # First rollback: back to heads=16
        restored = optimizer.rollback()
        assert restored.attention_heads == 16
        # Second rollback: back to original heads=8
        restored = optimizer.rollback()
        assert restored.attention_heads == 8


# =============================================================================
# Integration: multiple operations and persistence
# =============================================================================


class TestIntegration:
    def test_multiple_operations_persist(self, optimizer, config_dir):
        optimizer.apply_attention_config({"attention_heads": 16, "hidden_dim": 1024})
        optimizer.add_speech_enhance_module({"type": "conv_tasnet", "channels": 64})
        optimizer.adjust_seaco_params({"co_decoder_layers": 4})

        versions = list_versions(config_dir, "model")
        assert len(versions) == 3

        # Latest version should have all changes
        latest = read_model_config(versions[-1])
        assert latest.attention_heads == 16
        assert latest.hidden_dim == 1024
        assert latest.speech_enhance_config == {"type": "conv_tasnet", "channels": 64}
        assert latest.seaco_params["co_decoder_layers"] == 4

    def test_current_config_is_independent_copy(self, optimizer):
        config = optimizer.current_config
        config.attention_heads = 999
        assert optimizer.current_config.attention_heads == 8  # unchanged

    def test_default_initial_config(self, tmp_workspace):
        config_dir = str(tmp_workspace / "configs" / "model")
        opt = ModelOptimizer(config_dir=config_dir)
        config = opt.current_config
        assert config.attention_heads == 0
        assert config.hidden_dim == 0
        assert config.seaco_params == {}
        assert config.speech_enhance_config is None


# =============================================================================
# Enhanced Rollback and Change History Tests (task 8.3)
# =============================================================================


class TestRollbackEnhanced:
    """Comprehensive rollback tests covering different operation types,
    multi-level rollback, persistence, and edge cases."""

    def test_rollback_after_speech_enhance(self, optimizer):
        """Rollback restores config after add_speech_enhance_module."""
        original = optimizer.current_config
        optimizer.add_speech_enhance_module({"type": "conv_tasnet", "channels": 64})
        restored = optimizer.rollback()
        assert restored.speech_enhance_config == original.speech_enhance_config

    def test_rollback_after_seaco_adjust(self, optimizer):
        """Rollback restores config after adjust_seaco_params."""
        original = optimizer.current_config
        optimizer.adjust_seaco_params({"co_decoder_layers": 6, "attention_weight": 0.9})
        restored = optimizer.rollback()
        assert restored.seaco_params == original.seaco_params

    def test_rollback_after_mixed_operations(self, optimizer):
        """Multi-level rollback across different operation types."""
        s0 = optimizer.current_config

        optimizer.apply_attention_config({"attention_heads": 16})
        s1 = optimizer.current_config

        optimizer.add_speech_enhance_module({"type": "dprnn"})
        s2 = optimizer.current_config

        optimizer.adjust_seaco_params({"co_decoder_layers": 5})

        # Rollback 1: undo seaco change → back to s2
        r1 = optimizer.rollback()
        assert r1.attention_heads == s2.attention_heads
        assert r1.speech_enhance_config == s2.speech_enhance_config
        assert r1.seaco_params == s2.seaco_params

        # Rollback 2: undo speech enhance → back to s1
        r2 = optimizer.rollback()
        assert r2.speech_enhance_config == s1.speech_enhance_config

        # Rollback 3: undo attention change → back to s0
        r3 = optimizer.rollback()
        assert r3.attention_heads == s0.attention_heads

    def test_rollback_persists_versioned_yaml(self, optimizer, config_dir):
        """Rollback creates a new versioned YAML file with the restored config."""
        optimizer.apply_attention_config({"attention_heads": 16})
        versions_before = list_versions(config_dir, "model")
        assert len(versions_before) == 1

        optimizer.rollback()
        versions_after = list_versions(config_dir, "model")
        assert len(versions_after) == 2

        # The latest version should contain the rolled-back config
        latest = read_model_config(versions_after[-1])
        assert latest.attention_heads == 8  # original value

    def test_rollback_persisted_config_matches_current(self, optimizer, config_dir):
        """The persisted YAML after rollback matches current_config exactly."""
        optimizer.apply_attention_config({"attention_heads": 32, "hidden_dim": 1024})
        optimizer.add_speech_enhance_module({"type": "conv_tasnet"})
        optimizer.rollback()

        versions = list_versions(config_dir, "model")
        loaded = read_model_config(versions[-1])
        current = optimizer.current_config

        assert loaded.attention_heads == current.attention_heads
        assert loaded.hidden_dim == current.hidden_dim
        assert loaded.speech_enhance_config == current.speech_enhance_config
        assert loaded.seaco_params == current.seaco_params

    def test_rollback_exhausts_history(self, optimizer):
        """Rolling back all changes exhausts the history stack."""
        optimizer.apply_attention_config({"attention_heads": 16})
        optimizer.apply_attention_config({"attention_heads": 32})
        optimizer.rollback()
        optimizer.rollback()

        assert len(optimizer.history) == 0
        with pytest.raises(ValueError, match="No previous configuration"):
            optimizer.rollback()

    def test_rollback_then_new_change(self, optimizer, config_dir):
        """After rollback, new changes work correctly and persist."""
        optimizer.apply_attention_config({"attention_heads": 16})
        optimizer.rollback()

        # Apply a new change after rollback
        result = optimizer.apply_attention_config({"attention_heads": 4, "hidden_dim": 256})
        assert result.attention_heads == 4
        assert result.hidden_dim == 256

        # History should have one entry (the state before the new change)
        assert len(optimizer.history) == 1

        # Persistence should include all versions
        versions = list_versions(config_dir, "model")
        assert len(versions) == 3  # apply + rollback + apply

    def test_five_level_rollback(self, optimizer):
        """Five successive changes can all be rolled back one by one."""
        configs = []
        configs.append(optimizer.current_config)

        for heads in [16, 32, 64, 128, 256]:
            optimizer.apply_attention_config({"attention_heads": heads})
            configs.append(optimizer.current_config)

        # Roll back all 5 changes
        for i in range(5, 0, -1):
            restored = optimizer.rollback()
            expected = configs[i - 1]
            assert restored.attention_heads == expected.attention_heads


# =============================================================================
# Change History Tests (task 8.3)
# =============================================================================


class TestChangeHistory:
    """Tests for the get_change_history() method and change metadata recording."""

    def test_empty_history_initially(self, optimizer):
        """No change history before any operations."""
        assert optimizer.get_change_history() == []

    def test_attention_change_recorded(self, optimizer):
        """apply_attention_config records a change entry."""
        optimizer.apply_attention_config({"attention_heads": 16})
        history = optimizer.get_change_history()
        assert len(history) == 1

        entry = history[0]
        assert entry["operation"] == "apply_attention_config"
        assert "timestamp" in entry
        assert "attention_heads" in entry["changes"]
        assert entry["changes"]["attention_heads"]["old"] == 8
        assert entry["changes"]["attention_heads"]["new"] == 16

    def test_speech_enhance_change_recorded(self, optimizer):
        """add_speech_enhance_module records a change entry."""
        optimizer.add_speech_enhance_module({"type": "conv_tasnet", "channels": 64})
        history = optimizer.get_change_history()
        assert len(history) == 1

        entry = history[0]
        assert entry["operation"] == "add_speech_enhance_module"
        assert "speech_enhance_config" in entry["changes"]
        assert entry["changes"]["speech_enhance_config"]["old"] is None
        assert entry["changes"]["speech_enhance_config"]["new"] == {
            "type": "conv_tasnet",
            "channels": 64,
        }

    def test_seaco_change_recorded(self, optimizer):
        """adjust_seaco_params records a change entry with per-key diffs."""
        optimizer.adjust_seaco_params({"co_decoder_layers": 4})
        history = optimizer.get_change_history()
        assert len(history) == 1

        entry = history[0]
        assert entry["operation"] == "adjust_seaco_params"
        assert "seaco_params.co_decoder_layers" in entry["changes"]
        assert entry["changes"]["seaco_params.co_decoder_layers"]["old"] == 2
        assert entry["changes"]["seaco_params.co_decoder_layers"]["new"] == 4

    def test_rollback_change_recorded(self, optimizer):
        """rollback() records a change entry with operation='rollback'."""
        optimizer.apply_attention_config({"attention_heads": 16})
        optimizer.rollback()
        history = optimizer.get_change_history()
        assert len(history) == 2

        rollback_entry = history[1]
        assert rollback_entry["operation"] == "rollback"
        assert "attention_heads" in rollback_entry["changes"]
        assert rollback_entry["changes"]["attention_heads"]["old"] == 16
        assert rollback_entry["changes"]["attention_heads"]["new"] == 8

    def test_multiple_operations_chronological(self, optimizer):
        """Change history is in chronological order."""
        optimizer.apply_attention_config({"attention_heads": 16})
        optimizer.add_speech_enhance_module({"type": "conv_tasnet"})
        optimizer.adjust_seaco_params({"co_decoder_layers": 4})
        optimizer.rollback()

        history = optimizer.get_change_history()
        assert len(history) == 4

        operations = [e["operation"] for e in history]
        assert operations == [
            "apply_attention_config",
            "add_speech_enhance_module",
            "adjust_seaco_params",
            "rollback",
        ]

        # Timestamps should be non-decreasing
        timestamps = [e["timestamp"] for e in history]
        assert timestamps == sorted(timestamps)

    def test_change_history_has_iso_timestamps(self, optimizer):
        """Timestamps are valid ISO-8601 strings."""
        from datetime import datetime

        optimizer.apply_attention_config({"attention_heads": 16})
        history = optimizer.get_change_history()
        ts = history[0]["timestamp"]
        # Should parse without error
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_change_history_is_independent_copy(self, optimizer):
        """Mutating the returned history does not affect internal state."""
        optimizer.apply_attention_config({"attention_heads": 16})
        history = optimizer.get_change_history()
        history[0]["operation"] = "tampered"
        history.clear()

        # Internal state should be unaffected
        assert len(optimizer.get_change_history()) == 1
        assert optimizer.get_change_history()[0]["operation"] == "apply_attention_config"

    def test_empty_attention_config_records_no_field_changes(self, optimizer):
        """Applying an empty attention config records an entry with no field changes."""
        optimizer.apply_attention_config({})
        history = optimizer.get_change_history()
        assert len(history) == 1
        assert history[0]["changes"] == {}

    def test_rollback_records_all_changed_fields(self, optimizer):
        """Rollback after changing multiple fields records all diffs."""
        optimizer.apply_attention_config({"attention_heads": 16, "hidden_dim": 1024})
        optimizer.rollback()

        history = optimizer.get_change_history()
        rollback_changes = history[1]["changes"]
        assert "attention_heads" in rollback_changes
        assert "hidden_dim" in rollback_changes
        assert rollback_changes["attention_heads"]["old"] == 16
        assert rollback_changes["attention_heads"]["new"] == 8
        assert rollback_changes["hidden_dim"]["old"] == 1024
        assert rollback_changes["hidden_dim"]["new"] == 512


# =============================================================================
# CLI Tests (task 8.5)
# =============================================================================


class TestCLI:
    """Tests for the apply_model_config.py CLI script."""

    def test_apply_attention_heads(self, tmp_workspace):
        """apply subcommand updates attention heads."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([
                "apply",
                "--config-dir", config_dir,
                "--attention-heads", "8",
                "--hidden-dim", "512",
            ])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "success"
        assert output["config"]["attention_heads"] == 8
        assert output["config"]["hidden_dim"] == 512
        assert "attention_config" in output["changes_applied"]

    def test_apply_speech_enhance(self, tmp_workspace):
        """apply subcommand adds speech enhancement module."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([
                "apply",
                "--config-dir", config_dir,
                "--attention-heads", "4",
                "--hidden-dim", "256",
                "--speech-enhance", '{"type": "conv_tasnet", "channels": 64}',
            ])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "success"
        assert output["config"]["speech_enhance_config"]["type"] == "conv_tasnet"
        assert "speech_enhance_module" in output["changes_applied"]

    def test_apply_seaco_params(self, tmp_workspace):
        """apply subcommand adjusts SeACo parameters."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        # First create a valid base config
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([
                "apply",
                "--config-dir", config_dir,
                "--attention-heads", "8",
                "--hidden-dim", "512",
                "--seaco-params", '{"co_decoder_layers": 4, "attention_weight": 0.7}',
            ])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "success"
        assert output["config"]["seaco_params"]["co_decoder_layers"] == 4
        assert output["config"]["seaco_params"]["attention_weight"] == 0.7
        assert "seaco_params" in output["changes_applied"]

    def test_apply_invalid_json_speech_enhance(self, tmp_workspace):
        """apply subcommand reports error for invalid JSON in --speech-enhance."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([
                "apply",
                "--config-dir", config_dir,
                "--speech-enhance", "not-valid-json",
            ])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "error"
        assert "Invalid JSON" in output["message"]

    def test_apply_invalid_json_seaco_params(self, tmp_workspace):
        """apply subcommand reports error for invalid JSON in --seaco-params."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([
                "apply",
                "--config-dir", config_dir,
                "--seaco-params", "{bad json}",
            ])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "error"
        assert "Invalid JSON" in output["message"]

    def test_apply_no_changes(self, tmp_workspace):
        """apply subcommand with no change flags returns noop."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["apply", "--config-dir", config_dir])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "noop"

    def test_apply_validation_failure_auto_rollback(self, tmp_workspace):
        """apply subcommand auto-rolls back when validation fails."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        # Apply an invalid config: hidden_dim not divisible by attention_heads
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([
                "apply",
                "--config-dir", config_dir,
                "--attention-heads", "7",
                "--hidden-dim", "100",
            ])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "error"
        assert "Validation failed" in output["message"]
        assert "rolled back" in output["message"].lower()

    def test_validate_valid_config(self, tmp_workspace):
        """validate subcommand reports valid for a correct config."""
        from scripts.apply_model_config import main
        from scripts.core.config_utils import save_versioned_config
        from dataclasses import asdict
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        config = ModelConfig(
            attention_heads=8, hidden_dim=512,
            seaco_params={"co_decoder_layers": 2, "attention_weight": 0.5},
        )
        path = save_versioned_config(config_dir, "model", asdict(config))

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["validate", "--config-file", path])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "valid"

    def test_validate_invalid_config(self, tmp_workspace):
        """validate subcommand reports invalid for a bad config."""
        from scripts.apply_model_config import main
        from scripts.core.config_utils import save_versioned_config
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        bad_data = {
            "attention_heads": 0,
            "hidden_dim": 512,
            "seaco_params": {},
            "speech_enhance_config": None,
            "extra_params": {},
        }
        path = save_versioned_config(config_dir, "model", bad_data)

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["validate", "--config-file", path])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "invalid"
        assert "attention_heads" in output["message"]

    def test_validate_missing_file(self, tmp_workspace):
        """validate subcommand reports error for missing file."""
        from scripts.apply_model_config import main
        import io
        import json

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["validate", "--config-file", "/nonexistent/path.yaml"])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "error"
        assert "not found" in output["message"].lower()

    def test_rollback_success(self, tmp_workspace):
        """rollback subcommand restores the previous config."""
        from scripts.apply_model_config import main
        from scripts.core.config_utils import save_versioned_config
        from dataclasses import asdict
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        # Create two versions
        config_v1 = ModelConfig(attention_heads=8, hidden_dim=512, seaco_params={})
        config_v2 = ModelConfig(attention_heads=16, hidden_dim=1024, seaco_params={})
        save_versioned_config(config_dir, "model", asdict(config_v1))
        save_versioned_config(config_dir, "model", asdict(config_v2))

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["rollback", "--config-dir", config_dir])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "success"
        assert output["config"]["attention_heads"] == 8
        assert output["config"]["hidden_dim"] == 512

    def test_rollback_not_enough_versions(self, tmp_workspace):
        """rollback subcommand reports error when only one version exists."""
        from scripts.apply_model_config import main
        from scripts.core.config_utils import save_versioned_config
        from dataclasses import asdict
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        config_v1 = ModelConfig(attention_heads=8, hidden_dim=512, seaco_params={})
        save_versioned_config(config_dir, "model", asdict(config_v1))

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["rollback", "--config-dir", config_dir])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "error"
        assert "Not enough" in output["message"]

    def test_rollback_empty_dir(self, tmp_workspace):
        """rollback subcommand reports error when no versions exist."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["rollback", "--config-dir", config_dir])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "error"

    def test_history_with_versions(self, tmp_workspace):
        """history subcommand lists all versioned configs."""
        from scripts.apply_model_config import main
        from scripts.core.config_utils import save_versioned_config
        from dataclasses import asdict
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        for heads in [4, 8, 16]:
            config = ModelConfig(attention_heads=heads, hidden_dim=heads * 64, seaco_params={})
            save_versioned_config(config_dir, "model", asdict(config))

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["history", "--config-dir", config_dir])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "success"
        assert output["total_versions"] == 3
        assert len(output["versions"]) == 3
        assert output["versions"][0]["config"]["attention_heads"] == 4
        assert output["versions"][2]["config"]["attention_heads"] == 16

    def test_history_empty(self, tmp_workspace):
        """history subcommand returns empty list when no versions exist."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["history", "--config-dir", config_dir])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "success"
        assert output["total_versions"] == 0

    def test_show_current_config(self, tmp_workspace):
        """show subcommand displays the latest config."""
        from scripts.apply_model_config import main
        from scripts.core.config_utils import save_versioned_config
        from dataclasses import asdict
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        config = ModelConfig(
            attention_heads=16, hidden_dim=1024,
            seaco_params={"co_decoder_layers": 4},
        )
        save_versioned_config(config_dir, "model", asdict(config))

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["show", "--config-dir", config_dir])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "success"
        assert output["config"]["attention_heads"] == 16
        assert output["config"]["hidden_dim"] == 1024

    def test_show_empty_dir(self, tmp_workspace):
        """show subcommand returns config from main config fallback when no versioned config exists."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main(["show", "--config-dir", config_dir])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        # With main config fallback, status is either "success" or "empty"
        assert output["status"] in ("success", "empty")

    def test_no_subcommand_exits(self):
        """CLI with no subcommand exits with code 1."""
        from scripts.apply_model_config import main

        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_apply_all_three_changes(self, tmp_workspace):
        """apply subcommand can apply attention, speech enhance, and seaco in one call."""
        from scripts.apply_model_config import main
        import io
        import json

        config_dir = str(tmp_workspace / "configs" / "model")
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            main([
                "apply",
                "--config-dir", config_dir,
                "--attention-heads", "8",
                "--hidden-dim", "512",
                "--speech-enhance", '{"type": "conv_tasnet", "channels": 64}',
                "--seaco-params", '{"co_decoder_layers": 3, "attention_weight": 0.6}',
            ])
        finally:
            sys.stdout = old_stdout

        output = json.loads(captured.getvalue())
        assert output["status"] == "success"
        assert set(output["changes_applied"]) == {
            "attention_config", "speech_enhance_module", "seaco_params"
        }
        assert output["config"]["attention_heads"] == 8
        assert output["config"]["speech_enhance_config"]["type"] == "conv_tasnet"
        assert output["config"]["seaco_params"]["co_decoder_layers"] == 3
