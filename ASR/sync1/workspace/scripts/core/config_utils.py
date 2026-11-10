"""Configuration file read/write utilities for the ASR Training Optimizer system.

Provides YAML config read/write (ModelConfig, TrainingConfig, PostProcessConfig),
versioned config storage with auto-incrementing version numbers,
and JSON/JSONL file read/write utilities.
"""

import json
import os
import re
from dataclasses import asdict, fields
from typing import Optional, Union

import yaml

from scripts.core.models import (
    HotwordEntry,
    LRScheduleType,
    ModelConfig,
    PostProcessConfig,
    RegexRule,
    TrainingConfig,
)


# =============================================================================
# YAML Read/Write
# =============================================================================


def read_yaml(path: str) -> dict:
    """Read a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def write_yaml(path: str, data: dict) -> None:
    """Write a dict to a YAML file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# =============================================================================
# ModelConfig Read/Write
# =============================================================================


def read_model_config(path: str) -> ModelConfig:
    """Read a YAML file and convert to ModelConfig dataclass."""
    data = read_yaml(path)
    return ModelConfig(**{k: v for k, v in data.items() if k in {f.name for f in fields(ModelConfig)}})


def write_model_config(config: ModelConfig, path: str) -> None:
    """Write a ModelConfig dataclass to a YAML file."""
    data = asdict(config)
    write_yaml(path, data)


# =============================================================================
# TrainingConfig Read/Write
# =============================================================================


def read_training_config(path: str) -> TrainingConfig:
    """Read a YAML file and convert to TrainingConfig dataclass.

    Handles LRScheduleType enum deserialization from string value.
    """
    data = read_yaml(path)
    if "lr_schedule" in data and isinstance(data["lr_schedule"], str):
        data["lr_schedule"] = LRScheduleType(data["lr_schedule"])
    return TrainingConfig(**{k: v for k, v in data.items() if k in {f.name for f in fields(TrainingConfig)}})


def write_training_config(config: TrainingConfig, path: str) -> None:
    """Write a TrainingConfig dataclass to a YAML file.

    Serializes LRScheduleType enum as its string value.
    """
    data = asdict(config)
    if isinstance(data.get("lr_schedule"), LRScheduleType):
        data["lr_schedule"] = data["lr_schedule"].value
    else:
        # asdict converts enums to their value already in Python 3.10+,
        # but we ensure it's a string regardless
        data["lr_schedule"] = config.lr_schedule.value
    write_yaml(path, data)


# =============================================================================
# PostProcessConfig Read/Write
# =============================================================================


def read_post_process_config(path: str) -> PostProcessConfig:
    """Read a YAML file and convert to PostProcessConfig dataclass.

    Handles nested HotwordEntry and RegexRule list deserialization.
    """
    data = read_yaml(path)

    hotword_entries = [
        HotwordEntry(**entry) for entry in data.get("hotword_entries", [])
    ]
    regex_rules = [
        RegexRule(**rule) for rule in data.get("regex_rules", [])
    ]

    return PostProcessConfig(
        hotword_entries=hotword_entries,
        regex_rules=regex_rules,
        enable_lm_rerank=data.get("enable_lm_rerank", False),
        lm_model_path=data.get("lm_model_path"),
        pipeline_order=data.get("pipeline_order", ["hotword", "regex", "lm_rerank"]),
    )


def write_post_process_config(config: PostProcessConfig, path: str) -> None:
    """Write a PostProcessConfig dataclass to a YAML file.

    Serializes nested HotwordEntry and RegexRule lists properly.
    """
    data = asdict(config)
    write_yaml(path, data)


# =============================================================================
# Versioned Config Storage
# =============================================================================

_VERSION_PATTERN = re.compile(r"^v(\d{3})_(.+)\.yaml$")


def save_versioned_config(config_dir: str, prefix: str, data: dict) -> str:
    """Save a config with auto-incrementing version number.

    Files are named v001_{prefix}.yaml, v002_{prefix}.yaml, etc.
    Returns the path of the saved file.
    """
    os.makedirs(config_dir, exist_ok=True)
    existing = list_versions(config_dir, prefix)
    if existing:
        last_file = os.path.basename(existing[-1])
        match = _VERSION_PATTERN.match(last_file)
        if match:
            next_version = int(match.group(1)) + 1
        else:
            next_version = 1
    else:
        next_version = 1

    filename = f"v{next_version:03d}_{prefix}.yaml"
    filepath = os.path.join(config_dir, filename)
    write_yaml(filepath, data)
    return filepath


def get_latest_version(config_dir: str, prefix: str) -> Optional[str]:
    """Return the path to the latest versioned config file, or None if none exist."""
    versions = list_versions(config_dir, prefix)
    if versions:
        return versions[-1]
    return None


def list_versions(config_dir: str, prefix: str) -> list[str]:
    """Return a sorted list of all versioned config file paths for the given prefix."""
    if not os.path.isdir(config_dir):
        return []

    results = []
    for filename in os.listdir(config_dir):
        match = _VERSION_PATTERN.match(filename)
        if match and match.group(2) == prefix:
            results.append(os.path.join(config_dir, filename))

    # Sort by version number
    results.sort(key=lambda p: int(_VERSION_PATTERN.match(os.path.basename(p)).group(1)))
    return results


# =============================================================================
# Unified Config Loader
# =============================================================================

_DEFAULT_CONFIG_PATH = "configs/funasr_config.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    """Load the unified project config.

    Tries in order:
    1. Explicit *config_path* if provided
    2. ``workspace/configs/funasr_config.yaml``

    Returns an empty dict if nothing is found.
    """
    from pathlib import Path

    if config_path and Path(config_path).exists():
        return read_yaml(config_path)

    if Path(_DEFAULT_CONFIG_PATH).exists():
        return read_yaml(_DEFAULT_CONFIG_PATH)

    return {}


def save_config(config: dict, config_path: Optional[str] = None) -> str:
    """Write the unified config back and save versioned snapshots.

    Writes to *config_path* (or the default location) and saves
    versioned copies under ``configs/model/``, ``configs/training/``,
    ``configs/augmentation/``, ``configs/post_process/`` for history.

    Returns the path written to.
    """
    from pathlib import Path

    path = config_path or _DEFAULT_CONFIG_PATH
    write_yaml(path, config)

    base = str(Path(path).parent)
    if config.get("model"):
        save_versioned_config(
            os.path.join(base, "model"), "model", config["model"]
        )
    if config.get("training"):
        save_versioned_config(
            os.path.join(base, "training"), "train", config["training"]
        )
    if config.get("augmentation"):
        save_versioned_config(
            os.path.join(base, "augmentation"), "augment", config["augmentation"]
        )
    if config.get("post_process"):
        save_versioned_config(
            os.path.join(base, "post_process"), "postprocess", config["post_process"]
        )

    return path


# =============================================================================
# JSON Read/Write
# =============================================================================


def read_json(path: str) -> Union[dict, list]:
    """Read a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Union[dict, list], indent: int = 2) -> None:
    """Write data to a JSON file with indentation."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# =============================================================================
# JSONL Read/Write
# =============================================================================


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file (one JSON object per line)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: str, record: dict) -> None:
    """Append a single record to a JSONL file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
