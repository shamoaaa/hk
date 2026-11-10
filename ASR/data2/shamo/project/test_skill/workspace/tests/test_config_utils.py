"""Tests for config_utils module.

Covers:
- YAML round-trip for ModelConfig, TrainingConfig, PostProcessConfig
- Version auto-increment logic
- JSON/JSONL read/write/append
- Edge cases: empty files, missing files, enum serialization
"""

import json
import os
import pytest

from scripts.core.config_utils import (
    append_jsonl,
    get_latest_version,
    list_versions,
    read_json,
    read_jsonl,
    read_model_config,
    read_post_process_config,
    read_training_config,
    read_yaml,
    save_versioned_config,
    write_json,
    write_jsonl,
    write_model_config,
    write_post_process_config,
    write_training_config,
    write_yaml,
)
from scripts.core.models import (
    HotwordEntry,
    LRScheduleType,
    ModelConfig,
    PostProcessConfig,
    RegexRule,
    TrainingConfig,
)


# =============================================================================
# YAML Read/Write Tests
# =============================================================================


class TestYamlReadWrite:
    def test_write_and_read_yaml(self, tmp_path):
        path = str(tmp_path / "test.yaml")
        data = {"key": "value", "number": 42, "nested": {"a": 1}}
        write_yaml(path, data)
        result = read_yaml(path)
        assert result == data

    def test_read_empty_yaml(self, tmp_path):
        path = str(tmp_path / "empty.yaml")
        with open(path, "w") as f:
            f.write("")
        result = read_yaml(path)
        assert result == {}

    def test_read_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_yaml("/nonexistent/path.yaml")

    def test_write_yaml_creates_directories(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "test.yaml")
        write_yaml(path, {"key": "value"})
        assert os.path.exists(path)
        assert read_yaml(path) == {"key": "value"}

    def test_yaml_unicode(self, tmp_path):
        path = str(tmp_path / "unicode.yaml")
        data = {"中文": "测试", "日本語": "テスト"}
        write_yaml(path, data)
        result = read_yaml(path)
        assert result == data


# =============================================================================
# ModelConfig Round-Trip Tests
# =============================================================================


class TestModelConfig:
    def test_round_trip_default(self, tmp_path):
        path = str(tmp_path / "model.yaml")
        config = ModelConfig()
        write_model_config(config, path)
        loaded = read_model_config(path)
        assert loaded == config

    def test_round_trip_with_values(self, tmp_path):
        path = str(tmp_path / "model.yaml")
        config = ModelConfig(
            attention_heads=8,
            hidden_dim=512,
            seaco_params={"weight": 0.5, "layers": 3},
            speech_enhance_config={"type": "conv", "channels": 64},
            extra_params={"dropout": 0.1},
        )
        write_model_config(config, path)
        loaded = read_model_config(path)
        assert loaded == config

    def test_round_trip_none_speech_enhance(self, tmp_path):
        path = str(tmp_path / "model.yaml")
        config = ModelConfig(
            attention_heads=4,
            hidden_dim=256,
            speech_enhance_config=None,
        )
        write_model_config(config, path)
        loaded = read_model_config(path)
        assert loaded.speech_enhance_config is None


# =============================================================================
# TrainingConfig Round-Trip Tests
# =============================================================================


class TestTrainingConfig:
    def test_round_trip_default(self, tmp_path):
        path = str(tmp_path / "train.yaml")
        config = TrainingConfig()
        write_training_config(config, path)
        loaded = read_training_config(path)
        assert loaded == config

    def test_round_trip_warmup_decay(self, tmp_path):
        path = str(tmp_path / "train.yaml")
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=32,
            warmup_steps=1000,
            weight_decay=0.01,
            lr_schedule=LRScheduleType.WARMUP_DECAY,
            max_epochs=50,
            gradient_clip=5.0,
            extra_params={"label_smoothing": 0.1},
        )
        write_training_config(config, path)
        loaded = read_training_config(path)
        assert loaded == config
        assert loaded.lr_schedule == LRScheduleType.WARMUP_DECAY

    def test_round_trip_cosine_annealing(self, tmp_path):
        path = str(tmp_path / "train.yaml")
        config = TrainingConfig(lr_schedule=LRScheduleType.COSINE_ANNEALING)
        write_training_config(config, path)
        loaded = read_training_config(path)
        assert loaded.lr_schedule == LRScheduleType.COSINE_ANNEALING

    def test_round_trip_step_decay(self, tmp_path):
        path = str(tmp_path / "train.yaml")
        config = TrainingConfig(lr_schedule=LRScheduleType.STEP_DECAY)
        write_training_config(config, path)
        loaded = read_training_config(path)
        assert loaded.lr_schedule == LRScheduleType.STEP_DECAY

    def test_enum_serialized_as_string(self, tmp_path):
        path = str(tmp_path / "train.yaml")
        config = TrainingConfig(lr_schedule=LRScheduleType.COSINE_ANNEALING)
        write_training_config(config, path)
        raw = read_yaml(path)
        assert raw["lr_schedule"] == "cosine_annealing"


# =============================================================================
# PostProcessConfig Round-Trip Tests
# =============================================================================


class TestPostProcessConfig:
    def test_round_trip_default(self, tmp_path):
        path = str(tmp_path / "pp.yaml")
        config = PostProcessConfig()
        write_post_process_config(config, path)
        loaded = read_post_process_config(path)
        assert loaded == config

    def test_round_trip_with_hotwords_and_regex(self, tmp_path):
        path = str(tmp_path / "pp.yaml")
        config = PostProcessConfig(
            hotword_entries=[
                HotwordEntry(wrong_text="错误词", correct_text="正确词", priority=1),
                HotwordEntry(wrong_text="typo", correct_text="type", priority=2),
            ],
            regex_rules=[
                RegexRule(
                    pattern=r"\d+年",
                    replacement="YEAR",
                    description="Replace year pattern",
                    priority=1,
                ),
            ],
            enable_lm_rerank=True,
            lm_model_path="/path/to/lm",
            pipeline_order=["hotword", "regex", "lm_rerank"],
        )
        write_post_process_config(config, path)
        loaded = read_post_process_config(path)
        assert loaded == config
        assert len(loaded.hotword_entries) == 2
        assert loaded.hotword_entries[0].wrong_text == "错误词"
        assert loaded.hotword_entries[0].correct_text == "正确词"
        assert len(loaded.regex_rules) == 1
        assert loaded.regex_rules[0].pattern == r"\d+年"

    def test_round_trip_empty_lists(self, tmp_path):
        path = str(tmp_path / "pp.yaml")
        config = PostProcessConfig(
            hotword_entries=[],
            regex_rules=[],
            enable_lm_rerank=False,
            lm_model_path=None,
        )
        write_post_process_config(config, path)
        loaded = read_post_process_config(path)
        assert loaded.hotword_entries == []
        assert loaded.regex_rules == []
        assert loaded.enable_lm_rerank is False
        assert loaded.lm_model_path is None


# =============================================================================
# Versioned Config Storage Tests
# =============================================================================


class TestVersionedConfig:
    def test_save_first_version(self, tmp_path):
        config_dir = str(tmp_path / "configs")
        path = save_versioned_config(config_dir, "model", {"key": "v1"})
        assert os.path.basename(path) == "v001_model.yaml"
        assert os.path.exists(path)
        assert read_yaml(path) == {"key": "v1"}

    def test_auto_increment(self, tmp_path):
        config_dir = str(tmp_path / "configs")
        p1 = save_versioned_config(config_dir, "model", {"v": 1})
        p2 = save_versioned_config(config_dir, "model", {"v": 2})
        p3 = save_versioned_config(config_dir, "model", {"v": 3})
        assert os.path.basename(p1) == "v001_model.yaml"
        assert os.path.basename(p2) == "v002_model.yaml"
        assert os.path.basename(p3) == "v003_model.yaml"

    def test_different_prefixes(self, tmp_path):
        config_dir = str(tmp_path / "configs")
        p1 = save_versioned_config(config_dir, "model", {"type": "model"})
        p2 = save_versioned_config(config_dir, "train", {"type": "train"})
        p3 = save_versioned_config(config_dir, "model", {"type": "model2"})
        assert os.path.basename(p1) == "v001_model.yaml"
        assert os.path.basename(p2) == "v001_train.yaml"
        assert os.path.basename(p3) == "v002_model.yaml"

    def test_get_latest_version(self, tmp_path):
        config_dir = str(tmp_path / "configs")
        assert get_latest_version(config_dir, "model") is None
        save_versioned_config(config_dir, "model", {"v": 1})
        save_versioned_config(config_dir, "model", {"v": 2})
        latest = get_latest_version(config_dir, "model")
        assert os.path.basename(latest) == "v002_model.yaml"

    def test_list_versions_empty(self, tmp_path):
        config_dir = str(tmp_path / "configs")
        assert list_versions(config_dir, "model") == []

    def test_list_versions_sorted(self, tmp_path):
        config_dir = str(tmp_path / "configs")
        save_versioned_config(config_dir, "model", {"v": 1})
        save_versioned_config(config_dir, "model", {"v": 2})
        save_versioned_config(config_dir, "model", {"v": 3})
        versions = list_versions(config_dir, "model")
        assert len(versions) == 3
        basenames = [os.path.basename(v) for v in versions]
        assert basenames == ["v001_model.yaml", "v002_model.yaml", "v003_model.yaml"]

    def test_list_versions_nonexistent_dir(self):
        assert list_versions("/nonexistent/dir", "model") == []

    def test_get_latest_version_nonexistent_dir(self):
        assert get_latest_version("/nonexistent/dir", "model") is None


# =============================================================================
# JSON Read/Write Tests
# =============================================================================


class TestJsonReadWrite:
    def test_write_and_read_dict(self, tmp_path):
        path = str(tmp_path / "data.json")
        data = {"key": "value", "number": 42}
        write_json(path, data)
        result = read_json(path)
        assert result == data

    def test_write_and_read_list(self, tmp_path):
        path = str(tmp_path / "data.json")
        data = [{"a": 1}, {"b": 2}]
        write_json(path, data)
        result = read_json(path)
        assert result == data

    def test_json_indent(self, tmp_path):
        path = str(tmp_path / "data.json")
        write_json(path, {"key": "value"}, indent=4)
        with open(path, "r") as f:
            content = f.read()
        assert "    " in content  # 4-space indent

    def test_json_unicode(self, tmp_path):
        path = str(tmp_path / "data.json")
        data = {"中文": "测试"}
        write_json(path, data)
        result = read_json(path)
        assert result == data

    def test_json_creates_directories(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "data.json")
        write_json(path, {"key": "value"})
        assert os.path.exists(path)

    def test_read_json_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_json("/nonexistent/path.json")


# =============================================================================
# JSONL Read/Write Tests
# =============================================================================


class TestJsonlReadWrite:
    def test_write_and_read(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        records = [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.4}]
        write_jsonl(path, records)
        result = read_jsonl(path)
        assert result == records

    def test_append(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        write_jsonl(path, [{"step": 1}])
        append_jsonl(path, {"step": 2})
        result = read_jsonl(path)
        assert len(result) == 2
        assert result[0] == {"step": 1}
        assert result[1] == {"step": 2}

    def test_append_to_new_file(self, tmp_path):
        path = str(tmp_path / "new.jsonl")
        append_jsonl(path, {"first": True})
        result = read_jsonl(path)
        assert result == [{"first": True}]

    def test_write_empty_list(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        write_jsonl(path, [])
        result = read_jsonl(path)
        assert result == []

    def test_jsonl_unicode(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        records = [{"text": "你好世界"}, {"text": "テスト"}]
        write_jsonl(path, records)
        result = read_jsonl(path)
        assert result == records

    def test_jsonl_creates_directories(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "data.jsonl")
        write_jsonl(path, [{"key": "value"}])
        assert os.path.exists(path)

    def test_read_jsonl_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_jsonl("/nonexistent/path.jsonl")

    def test_read_jsonl_with_blank_lines(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        with open(path, "w") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n\n')
        result = read_jsonl(path)
        assert result == [{"a": 1}, {"b": 2}]
