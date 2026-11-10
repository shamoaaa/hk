"""CLI script for applying, viewing, and rolling back training parameter configurations.

Usage:
    python scripts/apply_training_config.py apply --config-dir configs/training/ --learning-rate 5e-5 --batch-size 64
    python scripts/apply_training_config.py apply --config-dir configs/training/ --lr-schedule cosine_annealing
    python scripts/apply_training_config.py show --config-dir configs/training/
    python scripts/apply_training_config.py history --config-dir configs/training/
    python scripts/apply_training_config.py rollback --config-dir configs/training/

Manages training parameter configurations via the ParamTuner, with versioned
persistence and rollback support.  Outputs structured JSON to stdout.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
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
    read_training_config,
    write_yaml,
)
from scripts.core.models import LRScheduleType, TrainingConfig
from scripts.execution.param_tuner import ParamTuner

_MAIN_CONFIG = "configs/funasr_config.yaml"


def _sync_training_to_main_config(config: TrainingConfig) -> None:
    """Write training params back to funasr_config.yaml."""
    main = load_config(_MAIN_CONFIG)
    main.setdefault("training", {}).update({
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "warmup_steps": config.warmup_steps,
        "weight_decay": config.weight_decay,
        "lr_schedule": config.lr_schedule.value,
        "max_epoch": config.max_epochs,
        "gradient_clip": config.gradient_clip,
    })
    write_yaml(_MAIN_CONFIG, main)


def _load_latest_config(config_dir: str) -> Optional[TrainingConfig]:
    """Load the latest versioned training config, falling back to main config."""
    latest = get_latest_version(config_dir, "train")
    if latest is not None:
        return read_training_config(latest)
    # Fall back to main config's training section
    main = load_config(_MAIN_CONFIG)
    t = main.get("training", {})
    if t.get("learning_rate"):
        lr_str = t.get("lr_schedule", "warmup_decay")
        return TrainingConfig(
            learning_rate=t.get("learning_rate", 0.0002),
            batch_size=t.get("batch_size", 5000),
            warmup_steps=t.get("warmup_steps", 1000),
            weight_decay=t.get("weight_decay", 0.01),
            lr_schedule=LRScheduleType(lr_str) if isinstance(lr_str, str) else LRScheduleType.WARMUP_DECAY,
            max_epochs=t.get("max_epoch", 50),
            gradient_clip=t.get("gradient_clip", 5.0),
        )
    return None


def _serialize_config(config: TrainingConfig) -> dict:
    """Serialize a TrainingConfig to a JSON-safe dict."""
    data = asdict(config)
    data["lr_schedule"] = config.lr_schedule.value
    return data


def _run_apply(args: argparse.Namespace) -> dict:
    """Execute the ``apply`` subcommand."""
    config_dir: str = args.config_dir

    # Load existing config or use defaults
    initial = _load_latest_config(config_dir)
    tuner = ParamTuner(config_dir=config_dir, initial_config=initial)

    current = tuner.current_config
    changes_made: list[str] = []

    # Apply individual parameter changes
    if args.learning_rate is not None:
        current.learning_rate = args.learning_rate
        changes_made.append("learning_rate")
    if args.batch_size is not None:
        current.batch_size = args.batch_size
        changes_made.append("batch_size")
    if args.warmup_steps is not None:
        current.warmup_steps = args.warmup_steps
        changes_made.append("warmup_steps")
    if args.weight_decay is not None:
        current.weight_decay = args.weight_decay
        changes_made.append("weight_decay")
    if args.lr_schedule is not None:
        try:
            current.lr_schedule = LRScheduleType(args.lr_schedule)
        except ValueError:
            valid = [e.value for e in LRScheduleType]
            return {
                "status": "error",
                "message": (
                    f"Invalid lr_schedule '{args.lr_schedule}'. "
                    f"Valid values: {valid}"
                ),
            }
        changes_made.append("lr_schedule")
    if args.max_epochs is not None:
        current.max_epochs = args.max_epochs
        changes_made.append("max_epochs")
    if args.gradient_clip is not None:
        current.gradient_clip = args.gradient_clip
        changes_made.append("gradient_clip")

    if not changes_made:
        return {
            "status": "noop",
            "message": "No changes specified.",
            "config": _serialize_config(tuner.current_config),
        }

    filepath = tuner.apply_config(current)

    # Sync back to main config
    _sync_training_to_main_config(tuner.current_config)

    return {
        "status": "success",
        "message": f"Applied changes: {', '.join(changes_made)}",
        "config": _serialize_config(tuner.current_config),
        "changes_applied": changes_made,
        "config_file": filepath,
    }


def _run_show(args: argparse.Namespace) -> dict:
    """Execute the ``show`` subcommand."""
    config_dir: str = args.config_dir
    config = _load_latest_config(config_dir)

    if config is None:
        return {
            "status": "empty",
            "message": "No training configuration found.",
            "config": _serialize_config(TrainingConfig()),
        }

    return {
        "status": "success",
        "config": _serialize_config(config),
    }


def _run_history(args: argparse.Namespace) -> dict:
    """Execute the ``history`` subcommand."""
    config_dir: str = args.config_dir
    versions = list_versions(config_dir, "train")

    history_entries: list[dict] = []
    for version_path in versions:
        config = read_training_config(version_path)
        history_entries.append({
            "file": version_path,
            "config": _serialize_config(config),
        })

    return {
        "status": "success",
        "total_versions": len(history_entries),
        "versions": history_entries,
    }


def _run_rollback(args: argparse.Namespace) -> dict:
    """Execute the ``rollback`` subcommand."""
    config_dir: str = args.config_dir

    versions = list_versions(config_dir, "train")
    if len(versions) < 2:
        return {
            "status": "error",
            "message": "Not enough configuration versions to roll back.",
        }

    # Load the two most recent configs to build a tuner with history
    previous_config = read_training_config(versions[-2])
    current_config = read_training_config(versions[-1])

    tuner = ParamTuner(config_dir=config_dir, initial_config=previous_config)
    # Simulate the last change so rollback works
    tuner._config = current_config
    tuner._history = [previous_config]

    restored = tuner.rollback()

    # Sync back to main config
    _sync_training_to_main_config(restored)

    return {
        "status": "success",
        "message": "Rolled back to previous configuration.",
        "config": _serialize_config(restored),
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point for the apply_training_config CLI."""
    parser = argparse.ArgumentParser(
        description="Apply, view, and manage training parameter configurations."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- apply ---
    apply_parser = subparsers.add_parser(
        "apply", help="Apply training parameter changes."
    )
    apply_parser.add_argument(
        "--config-dir",
        default="configs/training/",
        help="Directory for versioned training configs (default: configs/training/).",
    )
    apply_parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate.",
    )
    apply_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size.",
    )
    apply_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Number of warmup steps.",
    )
    apply_parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay.",
    )
    apply_parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        choices=["warmup_decay", "cosine_annealing", "step_decay"],
        help="Learning rate schedule strategy.",
    )
    apply_parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum number of training epochs.",
    )
    apply_parser.add_argument(
        "--gradient-clip",
        type=float,
        default=None,
        help="Gradient clipping value.",
    )

    # --- show ---
    show_parser = subparsers.add_parser(
        "show", help="Show the current training configuration."
    )
    show_parser.add_argument(
        "--config-dir",
        default="configs/training/",
        help="Directory for versioned training configs (default: configs/training/).",
    )

    # --- history ---
    history_parser = subparsers.add_parser(
        "history", help="Show training configuration version history."
    )
    history_parser.add_argument(
        "--config-dir",
        default="configs/training/",
        help="Directory for versioned training configs (default: configs/training/).",
    )

    # --- rollback ---
    rollback_parser = subparsers.add_parser(
        "rollback", help="Roll back to the previous training configuration."
    )
    rollback_parser.add_argument(
        "--config-dir",
        default="configs/training/",
        help="Directory for versioned training configs (default: configs/training/).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "apply": _run_apply,
        "show": _run_show,
        "history": _run_history,
        "rollback": _run_rollback,
    }

    result = handlers[args.command](args)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
