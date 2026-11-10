"""CLI script for applying, validating, and rolling back model configuration changes.

Usage:
    python scripts/apply_model_config.py apply --config-dir configs/model/ --attention-heads 16 --hidden-dim 1024
    python scripts/apply_model_config.py apply --config-dir configs/model/ --speech-enhance '{"type": "conv_tasnet", "channels": 64}'
    python scripts/apply_model_config.py apply --config-dir configs/model/ --seaco-params '{"co_decoder_layers": 4}'
    python scripts/apply_model_config.py validate --config-file configs/model/v001_model.yaml
    python scripts/apply_model_config.py rollback --config-dir configs/model/
    python scripts/apply_model_config.py history --config-dir configs/model/
    python scripts/apply_model_config.py show --config-dir configs/model/

Manages model configuration changes via the ModelOptimizer, with automatic
validation and rollback on failure.  Outputs structured JSON to stdout.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional

from scripts.core.config_utils import (
    get_latest_version,
    list_versions,
    load_config,
    read_model_config,
    write_yaml,
)
from scripts.execution.model_optimizer import ModelOptimizer
from scripts.core.models import ModelConfig

_MAIN_CONFIG = "configs/funasr_config.yaml"


def _sync_model_to_main_config(config: ModelConfig) -> None:
    """Write model params back to funasr_config.yaml."""
    main = load_config(_MAIN_CONFIG)
    model_section = main.setdefault("model", {})
    model_section["attention_heads"] = config.attention_heads
    model_section["hidden_dim"] = config.hidden_dim
    model_section["seaco_params"] = config.seaco_params
    model_section["speech_enhance_config"] = config.speech_enhance_config
    write_yaml(_MAIN_CONFIG, main)


def _load_latest_config(config_dir: str) -> Optional[ModelConfig]:
    """Load the latest versioned model config, falling back to main config."""
    latest = get_latest_version(config_dir, "model")
    if latest is not None:
        return read_model_config(latest)
    # Fall back to main config's model section
    main = load_config(_MAIN_CONFIG)
    model = main.get("model", {})
    if model.get("attention_heads"):
        return ModelConfig(
            attention_heads=model.get("attention_heads", 8),
            hidden_dim=model.get("hidden_dim", 512),
            seaco_params=model.get("seaco_params", {}),
            speech_enhance_config=model.get("speech_enhance_config"),
            extra_params={k: v for k, v in model.items()
                          if k not in ("name", "hub", "attention_heads", "hidden_dim",
                                       "seaco_params", "speech_enhance_config")},
        )
    return None


def _run_apply(args: argparse.Namespace) -> dict:
    """Execute the ``apply`` subcommand."""
    config_dir: str = args.config_dir

    # Load existing config or use defaults
    initial = _load_latest_config(config_dir)
    optimizer = ModelOptimizer(config_dir=config_dir, initial_config=initial)

    changes_applied: list[str] = []

    # Apply attention config changes
    attention_changes: dict = {}
    if args.attention_heads is not None:
        attention_changes["attention_heads"] = args.attention_heads
    if args.hidden_dim is not None:
        attention_changes["hidden_dim"] = args.hidden_dim
    if attention_changes:
        optimizer.apply_attention_config(attention_changes)
        changes_applied.append("attention_config")

    # Apply speech enhance module
    if args.speech_enhance is not None:
        try:
            module_config = json.loads(args.speech_enhance)
        except json.JSONDecodeError as exc:
            return {
                "status": "error",
                "message": f"Invalid JSON for --speech-enhance: {exc}",
            }
        optimizer.add_speech_enhance_module(module_config)
        changes_applied.append("speech_enhance_module")

    # Apply SeACo params
    if args.seaco_params is not None:
        try:
            seaco_config = json.loads(args.seaco_params)
        except json.JSONDecodeError as exc:
            return {
                "status": "error",
                "message": f"Invalid JSON for --seaco-params: {exc}",
            }
        optimizer.adjust_seaco_params(seaco_config)
        changes_applied.append("seaco_params")

    if not changes_applied:
        return {
            "status": "noop",
            "message": "No changes specified.",
            "config": asdict(optimizer.current_config),
        }

    # Validate the resulting config
    valid, msg = optimizer.validate_config(optimizer.current_config)
    if not valid:
        # Auto-rollback all changes
        for _ in changes_applied:
            try:
                optimizer.rollback()
            except ValueError:
                break
        return {
            "status": "error",
            "message": f"Validation failed: {msg}. Changes have been rolled back.",
            "config": asdict(optimizer.current_config),
        }

    # Sync back to main config
    _sync_model_to_main_config(optimizer.current_config)

    return {
        "status": "success",
        "message": f"Applied changes: {', '.join(changes_applied)}",
        "config": asdict(optimizer.current_config),
        "changes_applied": changes_applied,
    }


def _run_validate(args: argparse.Namespace) -> dict:
    """Execute the ``validate`` subcommand."""
    config_file: str = args.config_file
    try:
        config = read_model_config(config_file)
    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"Config file not found: {config_file}",
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Failed to read config file: {exc}",
        }

    optimizer = ModelOptimizer(config_dir="")
    valid, msg = optimizer.validate_config(config)

    return {
        "status": "valid" if valid else "invalid",
        "message": msg if msg else "Configuration is valid.",
        "config": asdict(config),
    }


def _run_rollback(args: argparse.Namespace) -> dict:
    """Execute the ``rollback`` subcommand."""
    config_dir: str = args.config_dir

    versions = list_versions(config_dir, "model")
    if len(versions) < 2:
        return {
            "status": "error",
            "message": "Not enough configuration versions to roll back.",
        }

    # Load the two most recent configs to build an optimizer with history
    previous_config = read_model_config(versions[-2])
    current_config = read_model_config(versions[-1])

    optimizer = ModelOptimizer(config_dir=config_dir, initial_config=previous_config)
    # Simulate the last change so rollback works
    optimizer._config = current_config
    optimizer._history = [previous_config]

    restored = optimizer.rollback()

    # Sync back to main config
    _sync_model_to_main_config(restored)

    return {
        "status": "success",
        "message": "Rolled back to previous configuration.",
        "config": asdict(restored),
    }


def _run_history(args: argparse.Namespace) -> dict:
    """Execute the ``history`` subcommand."""
    config_dir: str = args.config_dir
    versions = list_versions(config_dir, "model")

    history_entries: list[dict] = []
    for version_path in versions:
        config = read_model_config(version_path)
        history_entries.append({
            "file": version_path,
            "config": asdict(config),
        })

    return {
        "status": "success",
        "total_versions": len(history_entries),
        "versions": history_entries,
    }


def _run_show(args: argparse.Namespace) -> dict:
    """Execute the ``show`` subcommand."""
    config_dir: str = args.config_dir
    config = _load_latest_config(config_dir)

    if config is None:
        return {
            "status": "empty",
            "message": "No model configuration found.",
            "config": asdict(ModelConfig()),
        }

    return {
        "status": "success",
        "config": asdict(config),
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point for the apply_model_config CLI."""
    parser = argparse.ArgumentParser(
        description="Apply, validate, and manage model configuration changes."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- apply ---
    apply_parser = subparsers.add_parser(
        "apply", help="Apply model configuration changes."
    )
    apply_parser.add_argument(
        "--config-dir",
        default="configs/model/",
        help="Directory for versioned model configs (default: configs/model/).",
    )
    apply_parser.add_argument(
        "--attention-heads",
        type=int,
        default=None,
        help="Number of attention heads.",
    )
    apply_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden layer dimension.",
    )
    apply_parser.add_argument(
        "--speech-enhance",
        type=str,
        default=None,
        help="Speech enhancement module config as JSON string.",
    )
    apply_parser.add_argument(
        "--seaco-params",
        type=str,
        default=None,
        help="SeACo module parameters as JSON string.",
    )

    # --- validate ---
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a model configuration file."
    )
    validate_parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the model configuration YAML file to validate.",
    )

    # --- rollback ---
    rollback_parser = subparsers.add_parser(
        "rollback", help="Roll back to the previous model configuration."
    )
    rollback_parser.add_argument(
        "--config-dir",
        default="configs/model/",
        help="Directory for versioned model configs (default: configs/model/).",
    )

    # --- history ---
    history_parser = subparsers.add_parser(
        "history", help="Show configuration change history."
    )
    history_parser.add_argument(
        "--config-dir",
        default="configs/model/",
        help="Directory for versioned model configs (default: configs/model/).",
    )

    # --- show ---
    show_parser = subparsers.add_parser(
        "show", help="Show the current model configuration."
    )
    show_parser.add_argument(
        "--config-dir",
        default="configs/model/",
        help="Directory for versioned model configs (default: configs/model/).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "apply": _run_apply,
        "validate": _run_validate,
        "rollback": _run_rollback,
        "history": _run_history,
        "show": _run_show,
    }

    result = handlers[args.command](args)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
