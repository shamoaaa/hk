"""Model optimizer module for the ASR Training Optimizer system.

Provides the ``ModelOptimizer`` class which manages model configuration
changes including attention mechanism adjustments, speech enhancement
module configuration, SeACo parameter tuning, and configuration validation.

Configuration persistence uses versioned YAML files in ``configs/model/``
via :mod:`scripts.core.config_utils`.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from scripts.core.config_utils import (
    get_latest_version,
    read_model_config,
    save_versioned_config,
    write_model_config,
)
from scripts.core.models import ModelConfig


# =====================================================================
# Validation constants
# =====================================================================

_SEACO_CO_DECODER_LAYERS_MIN = 1
_SEACO_CO_DECODER_LAYERS_MAX = 6
_SEACO_ATTENTION_WEIGHT_MIN = 0.0
_SEACO_ATTENTION_WEIGHT_MAX = 1.0


# =====================================================================
# ModelOptimizer
# =====================================================================


class ModelOptimizer:
    """Manages ASR model structure configuration and optimization.

    The optimizer maintains a current :class:`ModelConfig` and a history
    of previous configurations.  All configuration changes are persisted
    as versioned YAML files in the specified config directory.

    Parameters
    ----------
    config_dir : str
        Directory for versioned model configuration files
        (e.g. ``configs/model/``).
    initial_config : ModelConfig | None
        Starting configuration.  If ``None``, a default
        :class:`ModelConfig` is used.
    """

    def __init__(
        self,
        config_dir: str,
        initial_config: Optional[ModelConfig] = None,
    ) -> None:
        self.config_dir = config_dir
        self._config = copy.deepcopy(initial_config) if initial_config else ModelConfig()
        self._history: list[ModelConfig] = []
        self._change_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_config(self) -> ModelConfig:
        """Return a copy of the current model configuration."""
        return copy.deepcopy(self._config)

    @property
    def history(self) -> list[ModelConfig]:
        """Return a copy of the configuration history."""
        return [copy.deepcopy(c) for c in self._history]

    # ------------------------------------------------------------------
    # Configuration methods
    # ------------------------------------------------------------------

    def apply_attention_config(self, config: dict) -> ModelConfig:
        """Adjust attention mechanism parameters.

        Supported keys in *config*:

        - ``attention_heads`` (int): Number of attention heads.
        - ``hidden_dim`` (int): Hidden layer dimension.

        The current configuration is saved to history before applying
        changes.  The updated configuration is persisted as a new
        versioned YAML file.

        Parameters
        ----------
        config : dict
            Dictionary with attention configuration values.

        Returns
        -------
        ModelConfig
            The updated model configuration.
        """
        self._push_history()

        changes: dict[str, dict] = {}
        if "attention_heads" in config:
            changes["attention_heads"] = {
                "old": self._config.attention_heads,
                "new": config["attention_heads"],
            }
            self._config.attention_heads = config["attention_heads"]
        if "hidden_dim" in config:
            changes["hidden_dim"] = {
                "old": self._config.hidden_dim,
                "new": config["hidden_dim"],
            }
            self._config.hidden_dim = config["hidden_dim"]

        self._persist()
        self._record_change("apply_attention_config", changes)
        return copy.deepcopy(self._config)

    def add_speech_enhance_module(self, module_config: dict) -> ModelConfig:
        """Add or update the speech enhancement front-end module configuration.

        Parameters
        ----------
        module_config : dict
            Speech enhancement module configuration.  Typical keys
            include ``type`` (e.g. ``"conv_tasnet"``), ``channels``,
            and ``layers``.

        Returns
        -------
        ModelConfig
            The updated model configuration.
        """
        self._push_history()
        old_config = copy.deepcopy(self._config.speech_enhance_config)
        self._config.speech_enhance_config = dict(module_config)
        self._persist()
        self._record_change("add_speech_enhance_module", {
            "speech_enhance_config": {
                "old": old_config,
                "new": dict(module_config),
            },
        })
        return copy.deepcopy(self._config)

    def adjust_seaco_params(self, seaco_config: dict) -> ModelConfig:
        """Adjust SeACo (Semantic-Aware Co-decoder) module parameters.

        Only keys present in *seaco_config* are updated; existing keys
        not mentioned are preserved.

        Parameters
        ----------
        seaco_config : dict
            Dictionary with SeACo parameter updates.  Common keys:
            ``co_decoder_layers``, ``attention_weight``, ``vocab_size``.

        Returns
        -------
        ModelConfig
            The updated model configuration.
        """
        self._push_history()
        changes: dict[str, dict] = {}
        for key, new_val in seaco_config.items():
            old_val = self._config.seaco_params.get(key)
            changes[f"seaco_params.{key}"] = {"old": old_val, "new": new_val}
        self._config.seaco_params.update(seaco_config)
        self._persist()
        self._record_change("adjust_seaco_params", changes)
        return copy.deepcopy(self._config)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_config(self, config: ModelConfig) -> tuple[bool, str]:
        """Validate a model configuration for correctness.

        Checks performed:

        1. ``attention_heads`` must be a positive integer.
        2. ``hidden_dim`` must be a positive integer.
        3. ``hidden_dim`` must be divisible by ``attention_heads``.
        4. SeACo ``co_decoder_layers`` (if present) must be in [1, 6].
        5. SeACo ``attention_weight`` (if present) must be in [0.0, 1.0].
        6. SeACo ``vocab_size`` (if present) must be a positive integer.

        Parameters
        ----------
        config : ModelConfig
            The configuration to validate.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when valid, ``(False, reason)`` otherwise.
        """
        # --- attention_heads ---
        if not isinstance(config.attention_heads, int) or config.attention_heads <= 0:
            return False, (
                f"attention_heads must be a positive integer, "
                f"got {config.attention_heads!r}"
            )

        # --- hidden_dim ---
        if not isinstance(config.hidden_dim, int) or config.hidden_dim <= 0:
            return False, (
                f"hidden_dim must be a positive integer, "
                f"got {config.hidden_dim!r}"
            )

        # --- divisibility ---
        if config.hidden_dim % config.attention_heads != 0:
            return False, (
                f"hidden_dim ({config.hidden_dim}) must be divisible by "
                f"attention_heads ({config.attention_heads})"
            )

        # --- SeACo params ---
        seaco = config.seaco_params
        if seaco:
            # co_decoder_layers
            if "co_decoder_layers" in seaco:
                layers = seaco["co_decoder_layers"]
                if (
                    not isinstance(layers, int)
                    or layers < _SEACO_CO_DECODER_LAYERS_MIN
                    or layers > _SEACO_CO_DECODER_LAYERS_MAX
                ):
                    return False, (
                        f"seaco_params.co_decoder_layers must be an integer "
                        f"in [{_SEACO_CO_DECODER_LAYERS_MIN}, "
                        f"{_SEACO_CO_DECODER_LAYERS_MAX}], got {layers!r}"
                    )

            # attention_weight
            if "attention_weight" in seaco:
                weight = seaco["attention_weight"]
                if (
                    not isinstance(weight, (int, float))
                    or weight < _SEACO_ATTENTION_WEIGHT_MIN
                    or weight > _SEACO_ATTENTION_WEIGHT_MAX
                ):
                    return False, (
                        f"seaco_params.attention_weight must be a float "
                        f"in [{_SEACO_ATTENTION_WEIGHT_MIN}, "
                        f"{_SEACO_ATTENTION_WEIGHT_MAX}], got {weight!r}"
                    )

            # vocab_size
            if "vocab_size" in seaco:
                vocab = seaco["vocab_size"]
                if not isinstance(vocab, int) or vocab <= 0:
                    return False, (
                        f"seaco_params.vocab_size must be a positive integer, "
                        f"got {vocab!r}"
                    )

        return True, ""

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self) -> ModelConfig:
        """Roll back to the previous valid configuration.

        The restored configuration is persisted as a new versioned YAML
        file and a change-history record is appended.

        Returns
        -------
        ModelConfig
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

        # Build a diff summary between the rolled-back config and the
        # config that was active before the rollback.
        changes: dict[str, dict] = {}
        restored = self._config
        if before.attention_heads != restored.attention_heads:
            changes["attention_heads"] = {
                "old": before.attention_heads,
                "new": restored.attention_heads,
            }
        if before.hidden_dim != restored.hidden_dim:
            changes["hidden_dim"] = {
                "old": before.hidden_dim,
                "new": restored.hidden_dim,
            }
        if before.speech_enhance_config != restored.speech_enhance_config:
            changes["speech_enhance_config"] = {
                "old": before.speech_enhance_config,
                "new": restored.speech_enhance_config,
            }
        if before.seaco_params != restored.seaco_params:
            changes["seaco_params"] = {
                "old": before.seaco_params,
                "new": restored.seaco_params,
            }
        if before.extra_params != restored.extra_params:
            changes["extra_params"] = {
                "old": before.extra_params,
                "new": restored.extra_params,
            }

        self._record_change("rollback", changes)
        return copy.deepcopy(self._config)

    # ------------------------------------------------------------------
    # Change history
    # ------------------------------------------------------------------

    def get_change_history(self) -> list[dict[str, Any]]:
        """Return the full configuration change history.

        Each entry is a dict with the following keys:

        - ``operation`` (str): The method that triggered the change
          (e.g. ``"apply_attention_config"``, ``"rollback"``).
        - ``timestamp`` (str): ISO-8601 UTC timestamp of the change.
        - ``changes`` (dict): Mapping of field names to
          ``{"old": ..., "new": ...}`` dicts describing what changed.

        Returns
        -------
        list[dict[str, Any]]
            A list of change records in chronological order.
        """
        return [copy.deepcopy(entry) for entry in self._change_history]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_history(self) -> None:
        """Save the current config to history before a change."""
        self._history.append(copy.deepcopy(self._config))

    def _persist(self) -> None:
        """Persist the current config as a versioned YAML file."""
        save_versioned_config(
            self.config_dir,
            "model",
            asdict(self._config),
        )

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
