"""Training parameter tuning module for the ASR Training Optimizer system.

Provides the ``ParamTuner`` class which manages training hyperparameter
configuration, including learning rate, batch size, warmup steps, weight
decay, and learning rate scheduling strategy.  Supports configuration
generation from optimization suggestions, versioned persistence, early
anomaly detection, and rollback.

Configuration persistence uses versioned YAML files in ``configs/training/``
via :mod:`scripts.core.config_utils`.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import copy
import math
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from scripts.core.config_utils import (
    get_latest_version,
    read_training_config,
    save_versioned_config,
    write_training_config,
)
from scripts.core.database import Database
from scripts.core.models import (
    EvaluationReport,
    LRScheduleType,
    OptimizationSuggestion,
    ParamHistory,
    TrainingConfig,
    TrainingMetrics,
)


# =====================================================================
# Default training configuration values
# =====================================================================

_DEFAULT_LEARNING_RATE = 1e-4
_DEFAULT_BATCH_SIZE = 32
_DEFAULT_WARMUP_STEPS = 2000
_DEFAULT_WEIGHT_DECAY = 0.01
_DEFAULT_LR_SCHEDULE = LRScheduleType.WARMUP_DECAY
_DEFAULT_MAX_EPOCHS = 50
_DEFAULT_GRADIENT_CLIP = 5.0

# Early anomaly detection thresholds
_EARLY_STEP_THRESHOLD = 100
_LOSS_GROWTH_FACTOR = 10.0  # loss grew by 10x from initial → anomaly


# =====================================================================
# ParamTuner
# =====================================================================


class ParamTuner:
    """Manages training hyperparameter configuration and tuning.

    The tuner maintains a current :class:`TrainingConfig` and a history
    of previous configurations.  All configuration changes are persisted
    as versioned YAML files in the specified config directory.

    Parameters
    ----------
    config_dir : str
        Directory for versioned training configuration files
        (e.g. ``configs/training/``).
    initial_config : TrainingConfig | None
        Starting configuration.  If ``None``, a default
        :class:`TrainingConfig` is used.
    db : Database | None
        Optional database instance for persisting param history.
    """

    def __init__(
        self,
        config_dir: str,
        initial_config: Optional[TrainingConfig] = None,
        db: Optional[Database] = None,
    ) -> None:
        self.config_dir = config_dir
        self._db = db

        if initial_config is not None:
            self._config = copy.deepcopy(initial_config)
        else:
            self._config = TrainingConfig(
                learning_rate=_DEFAULT_LEARNING_RATE,
                batch_size=_DEFAULT_BATCH_SIZE,
                warmup_steps=_DEFAULT_WARMUP_STEPS,
                weight_decay=_DEFAULT_WEIGHT_DECAY,
                lr_schedule=_DEFAULT_LR_SCHEDULE,
                max_epochs=_DEFAULT_MAX_EPOCHS,
                gradient_clip=_DEFAULT_GRADIENT_CLIP,
            )

        self._history: list[TrainingConfig] = []
        self._change_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_config(self) -> TrainingConfig:
        """Return a copy of the current training configuration."""
        return copy.deepcopy(self._config)

    @property
    def history(self) -> list[TrainingConfig]:
        """Return a copy of the configuration history stack."""
        return [copy.deepcopy(c) for c in self._history]

    # ------------------------------------------------------------------
    # Configuration generation
    # ------------------------------------------------------------------

    def generate_config(
        self, suggestions: list[OptimizationSuggestion]
    ) -> TrainingConfig:
        """Generate a new TrainingConfig from optimization suggestions.

        Processes suggestions with ``dimension == "param"`` and applies
        their ``config_changes`` to the current configuration.  Supported
        config_changes keys:

        - ``learning_rate`` (float)
        - ``batch_size`` (int)
        - ``warmup_steps`` (int)
        - ``weight_decay`` (float)
        - ``lr_schedule`` (str): one of ``"warmup_decay"``,
          ``"cosine_annealing"``, ``"step_decay"``
        - ``max_epochs`` (int)
        - ``gradient_clip`` (float)

        Parameters
        ----------
        suggestions : list[OptimizationSuggestion]
            Optimization suggestions.  Only those with
            ``dimension == "param"`` are considered.

        Returns
        -------
        TrainingConfig
            The newly generated configuration (not yet applied).
        """
        new_config = copy.deepcopy(self._config)

        # Filter and sort by priority (highest first)
        param_suggestions = sorted(
            [s for s in suggestions if s.dimension == "param"],
            key=lambda s: s.priority,
            reverse=True,
        )

        for suggestion in param_suggestions:
            changes = suggestion.config_changes
            if not changes:
                continue

            if "learning_rate" in changes:
                new_config.learning_rate = float(changes["learning_rate"])
            if "batch_size" in changes:
                new_config.batch_size = int(changes["batch_size"])
            if "warmup_steps" in changes:
                new_config.warmup_steps = int(changes["warmup_steps"])
            if "weight_decay" in changes:
                new_config.weight_decay = float(changes["weight_decay"])
            if "lr_schedule" in changes:
                schedule_str = str(changes["lr_schedule"])
                new_config.lr_schedule = LRScheduleType(schedule_str)
            if "max_epochs" in changes:
                new_config.max_epochs = int(changes["max_epochs"])
            if "gradient_clip" in changes:
                new_config.gradient_clip = float(changes["gradient_clip"])

            # Merge any extra params
            extra = {
                k: v
                for k, v in changes.items()
                if k
                not in {
                    "learning_rate",
                    "batch_size",
                    "warmup_steps",
                    "weight_decay",
                    "lr_schedule",
                    "max_epochs",
                    "gradient_clip",
                }
            }
            if extra:
                new_config.extra_params.update(extra)

        return new_config

    # ------------------------------------------------------------------
    # Configuration application
    # ------------------------------------------------------------------

    def apply_config(
        self,
        config: TrainingConfig,
        iteration_id: str = "",
    ) -> str:
        """Apply a training configuration, saving it as a versioned YAML file.

        The current configuration is pushed to the history stack before
        the new configuration is applied.

        Parameters
        ----------
        config : TrainingConfig
            The new training configuration to apply.
        iteration_id : str
            Optional iteration identifier for tracking.

        Returns
        -------
        str
            The file path of the saved versioned configuration.
        """
        self._push_history()
        self._config = copy.deepcopy(config)

        filepath = self._persist()

        self._record_change("apply_config", {
            "learning_rate": {
                "old": self._history[-1].learning_rate if self._history else None,
                "new": config.learning_rate,
            },
            "batch_size": {
                "old": self._history[-1].batch_size if self._history else None,
                "new": config.batch_size,
            },
            "lr_schedule": {
                "old": (
                    self._history[-1].lr_schedule.value if self._history else None
                ),
                "new": config.lr_schedule.value,
            },
        })

        # Persist to database if available
        if self._db is not None:
            self._save_param_history(iteration_id)

        return filepath

    # ------------------------------------------------------------------
    # Early anomaly detection
    # ------------------------------------------------------------------

    def check_early_anomaly(
        self,
        metrics: list[TrainingMetrics],
        step_threshold: int = _EARLY_STEP_THRESHOLD,
    ) -> bool:
        """Check whether early training steps exhibit loss anomalies.

        An anomaly is detected if any of the following conditions hold
        for the metrics within the first *step_threshold* steps:

        1. Any ``train_loss`` is NaN or Inf.
        2. The loss grows by more than ``_LOSS_GROWTH_FACTOR`` times
           relative to the first recorded loss value.

        Parameters
        ----------
        metrics : list[TrainingMetrics]
            Training metrics collected so far.
        step_threshold : int
            Only metrics with ``step <= step_threshold`` are examined.

        Returns
        -------
        bool
            ``True`` if an anomaly is detected, ``False`` otherwise.
        """
        early_metrics = [m for m in metrics if m.step <= step_threshold]
        if not early_metrics:
            return False

        # Sort by step to get the initial loss
        early_metrics_sorted = sorted(early_metrics, key=lambda m: m.step)
        initial_loss = early_metrics_sorted[0].train_loss

        for m in early_metrics_sorted:
            loss = m.train_loss

            # Check for NaN / Inf
            if math.isnan(loss) or math.isinf(loss):
                return True

            # Check for abnormal growth (only if initial loss is finite and positive)
            if (
                math.isfinite(initial_loss)
                and initial_loss > 0
                and loss > initial_loss * _LOSS_GROWTH_FACTOR
            ):
                return True

        return False

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self) -> TrainingConfig:
        """Roll back to the previous valid training configuration.

        The restored configuration is persisted as a new versioned YAML
        file.

        Returns
        -------
        TrainingConfig
            The restored configuration.

        Raises
        ------
        ValueError
            If there is no previous configuration to roll back to.
        """
        if not self._history:
            raise ValueError("No previous configuration to roll back to")

        before = copy.deepcopy(self._config)
        self._config = self._history.pop()
        self._persist()

        self._record_change("rollback", {
            "learning_rate": {
                "old": before.learning_rate,
                "new": self._config.learning_rate,
            },
            "batch_size": {
                "old": before.batch_size,
                "new": self._config.batch_size,
            },
            "lr_schedule": {
                "old": before.lr_schedule.value,
                "new": self._config.lr_schedule.value,
            },
        })

        return copy.deepcopy(self._config)

    # ------------------------------------------------------------------
    # Parameter history
    # ------------------------------------------------------------------

    def get_param_history(self) -> list[ParamHistory]:
        """Retrieve the parameter history from the database.

        If no database is configured, returns an empty list.

        Returns
        -------
        list[ParamHistory]
            Parameter history entries ordered by creation time.
        """
        if self._db is None:
            return []

        rows = self._db.get_all(
            "param_history", order_by="created_at ASC"
        )

        result: list[ParamHistory] = []
        for row in rows:
            config_dict = {}
            if row.get("config_json"):
                import json

                config_dict = json.loads(row["config_json"])

            # Reconstruct TrainingConfig from stored JSON
            lr_schedule_val = config_dict.get(
                "lr_schedule", _DEFAULT_LR_SCHEDULE.value
            )
            if isinstance(lr_schedule_val, str):
                lr_schedule = LRScheduleType(lr_schedule_val)
            else:
                lr_schedule = _DEFAULT_LR_SCHEDULE

            tc = TrainingConfig(
                learning_rate=config_dict.get(
                    "learning_rate", _DEFAULT_LEARNING_RATE
                ),
                batch_size=config_dict.get("batch_size", _DEFAULT_BATCH_SIZE),
                warmup_steps=config_dict.get(
                    "warmup_steps", _DEFAULT_WARMUP_STEPS
                ),
                weight_decay=config_dict.get(
                    "weight_decay", _DEFAULT_WEIGHT_DECAY
                ),
                lr_schedule=lr_schedule,
                max_epochs=config_dict.get("max_epochs", _DEFAULT_MAX_EPOCHS),
                gradient_clip=config_dict.get(
                    "gradient_clip", _DEFAULT_GRADIENT_CLIP
                ),
                extra_params=config_dict.get("extra_params", {}),
            )

            result.append(
                ParamHistory(
                    config=tc,
                    evaluation_result=None,
                    timestamp=row.get("created_at", ""),
                    iteration_id=row.get("iteration_id", ""),
                )
            )

        return result

    # ------------------------------------------------------------------
    # Change history (in-memory)
    # ------------------------------------------------------------------

    def get_change_history(self) -> list[dict[str, Any]]:
        """Return the full configuration change history.

        Each entry is a dict with keys:

        - ``operation`` (str): The method that triggered the change.
        - ``timestamp`` (str): ISO-8601 UTC timestamp.
        - ``changes`` (dict): Field-level diffs.

        Returns
        -------
        list[dict[str, Any]]
            Change records in chronological order.
        """
        return [copy.deepcopy(entry) for entry in self._change_history]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_history(self) -> None:
        """Save the current config to the history stack."""
        self._history.append(copy.deepcopy(self._config))

    def _persist(self) -> str:
        """Persist the current config as a versioned YAML file.

        Returns
        -------
        str
            The file path of the saved configuration.
        """
        data = asdict(self._config)
        # Serialize LRScheduleType enum to its string value
        data["lr_schedule"] = self._config.lr_schedule.value
        return save_versioned_config(self.config_dir, "train", data)

    def _record_change(
        self,
        operation: str,
        changes: dict[str, dict],
    ) -> None:
        """Append a change record to the internal change history."""
        self._change_history.append({
            "operation": operation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "changes": changes,
        })

    def _save_param_history(self, iteration_id: str) -> None:
        """Save the current config to the param_history database table."""
        import json

        if self._db is None:
            return

        data = asdict(self._config)
        data["lr_schedule"] = self._config.lr_schedule.value

        history_id = f"ph_{uuid.uuid4().hex[:12]}"
        self._db.insert("param_history", {
            "history_id": history_id,
            "iteration_id": iteration_id,
            "config_json": json.dumps(data, ensure_ascii=False),
            "cer_result": None,
            "wer_result": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
