"""Global error handlers for the ASR Training Optimizer system.

Provides three error handler classes:

- ``TrainingErrorHandler``: Handles NaN/Inf anomalies and gradient explosion
  during training.
- ``ConfigErrorHandler``: Handles invalid model configurations and model
  initialization failures with rollback support.
- ``SkillOutputHandler``: Handles Kiro Skill output parsing exceptions and
  Skill file validation.

Each handler maintains an error log (list of dicts) for transparency,
following the global error handling principles:

1. No data loss
2. Recoverability (rollback support)
3. Transparency (all errors logged)
4. Graceful degradation

Requirements: 1.5, 5.4, 5.5, 6.5
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

from scripts.decision.llm_skill_engine import LLMSkillEngine
from scripts.execution.model_optimizer import ModelOptimizer
from scripts.core.models import (
    AlertType,
    ModelConfig,
    TrainingAlert,
)
from scripts.decision.skill_evolution import SkillEvolutionEngine

logger = logging.getLogger(__name__)


# =====================================================================
# TrainingErrorHandler
# =====================================================================


class TrainingErrorHandler:
    """Handle training anomalies such as NaN/Inf loss and gradient explosion.

    Maintains an error log of all handled events for transparency.
    Generates ``TrainingAlert`` objects to notify downstream components.

    Parameters
    ----------
    max_gradient_clip_attempts : int
        Maximum number of gradient clipping attempts before falling back
        to learning rate reduction.
    lr_reduction_factor : float
        Factor by which the learning rate is reduced when gradient
        clipping alone cannot resolve gradient explosion.
    """

    def __init__(
        self,
        max_gradient_clip_attempts: int = 3,
        lr_reduction_factor: float = 0.5,
    ) -> None:
        self.max_gradient_clip_attempts = max_gradient_clip_attempts
        self.lr_reduction_factor = lr_reduction_factor
        self.error_log: list[dict] = []

    def handle_nan_inf(
        self, step: int, epoch: int, context: dict
    ) -> TrainingAlert:
        """Handle NaN/Inf anomaly detected during training.

        Actions performed:

        1. Record anomaly context (step, epoch, recent loss history,
           learning rate, gradient norm).
        2. Generate a ``NAN_INF`` ``TrainingAlert`` to notify downstream
           components (e.g. LLM_Skill_Engine).

        Parameters
        ----------
        step : int
            The training step at which the anomaly occurred.
        epoch : int
            The training epoch at which the anomaly occurred.
        context : dict
            Additional context about the anomaly.  Expected keys include
            ``recent_losses`` (list of recent loss values),
            ``learning_rate`` (current LR), and ``gradient_norm``
            (current gradient norm).

        Returns
        -------
        TrainingAlert
            A ``NAN_INF`` alert with full anomaly context.
        """
        event = {
            "event_type": "nan_inf",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "epoch": epoch,
            "context": copy.deepcopy(context),
            "action_taken": "training_paused",
        }
        self.error_log.append(event)
        logger.warning(
            "NaN/Inf anomaly at step=%d epoch=%d", step, epoch
        )

        alert = TrainingAlert(
            alert_type=AlertType.NAN_INF,
            message=(
                f"NaN/Inf anomaly detected at step {step}, epoch {epoch}. "
                f"Training paused."
            ),
            context={
                "step": step,
                "epoch": epoch,
                "recent_losses": context.get("recent_losses", []),
                "learning_rate": context.get("learning_rate"),
                "gradient_norm": context.get("gradient_norm"),
            },
            epoch=epoch,
            step=step,
        )
        return alert

    def handle_gradient_explosion(
        self, gradient_norm: float, threshold: float
    ) -> TrainingAlert:
        """Handle gradient explosion during training.

        Actions performed:

        1. Apply gradient clipping (conceptually — the caller should
           actually clip gradients using the returned threshold).
        2. If the gradient norm still exceeds the threshold after
           accounting for repeated attempts, recommend learning rate
           reduction.
        3. Record the event.

        Parameters
        ----------
        gradient_norm : float
            The observed gradient norm.
        threshold : float
            The gradient norm threshold above which explosion is declared.

        Returns
        -------
        TrainingAlert
            A ``GRADIENT_EXPLOSION`` alert with recommended action.
        """
        # Count recent gradient explosion events to decide action
        recent_explosions = [
            e for e in self.error_log
            if e["event_type"] == "gradient_explosion"
        ]

        if len(recent_explosions) >= self.max_gradient_clip_attempts:
            action = "reduce_learning_rate"
            message = (
                f"Gradient explosion (norm={gradient_norm:.4f}, "
                f"threshold={threshold:.4f}). Clipping insufficient after "
                f"{self.max_gradient_clip_attempts} attempts — reducing "
                f"learning rate by factor {self.lr_reduction_factor}."
            )
        else:
            action = "gradient_clipping"
            message = (
                f"Gradient explosion (norm={gradient_norm:.4f}, "
                f"threshold={threshold:.4f}). Applying gradient clipping."
            )

        event = {
            "event_type": "gradient_explosion",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gradient_norm": gradient_norm,
            "threshold": threshold,
            "action_taken": action,
        }
        self.error_log.append(event)
        logger.warning(
            "Gradient explosion: norm=%.4f threshold=%.4f action=%s",
            gradient_norm,
            threshold,
            action,
        )

        alert = TrainingAlert(
            alert_type=AlertType.GRADIENT_EXPLOSION,
            message=message,
            context={
                "gradient_norm": gradient_norm,
                "threshold": threshold,
                "action": action,
                "lr_reduction_factor": self.lr_reduction_factor,
                "clip_attempts": len(recent_explosions) + 1,
            },
        )
        return alert


# =====================================================================
# ConfigErrorHandler
# =====================================================================


class ConfigErrorHandler:
    """Handle invalid model configurations and initialization failures.

    Works with a ``ModelOptimizer`` instance to perform configuration
    rollback when errors are detected.

    Parameters
    ----------
    model_optimizer : ModelOptimizer
        The model optimizer instance used for rollback operations.
    """

    def __init__(self, model_optimizer: ModelOptimizer) -> None:
        self.model_optimizer = model_optimizer
        self.error_log: list[dict] = []

    def handle_invalid_config(
        self, config: ModelConfig, errors: list[str]
    ) -> ModelConfig:
        """Handle an invalid model configuration.

        Actions performed:

        1. Record validation error details.
        2. Roll back to the previous valid configuration via
           ``ModelOptimizer.rollback()``.
        3. Return the rolled-back configuration.

        Parameters
        ----------
        config : ModelConfig
            The invalid configuration that failed validation.
        errors : list[str]
            List of validation error messages.

        Returns
        -------
        ModelConfig
            The rolled-back (previous valid) configuration.

        Raises
        ------
        ValueError
            If rollback is not possible (no history available).
        """
        event = {
            "event_type": "invalid_config",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_snapshot": {
                "attention_heads": config.attention_heads,
                "hidden_dim": config.hidden_dim,
            },
            "errors": list(errors),
            "action_taken": "rollback",
        }
        self.error_log.append(event)
        logger.warning(
            "Invalid config detected (%d errors). Rolling back.",
            len(errors),
        )

        rolled_back = self.model_optimizer.rollback()
        return rolled_back

    def handle_init_failure(
        self, config: ModelConfig, error: Exception
    ) -> ModelConfig:
        """Handle a model initialization failure.

        Actions performed:

        1. Catch and record the initialization exception details
           (e.g. out of memory, incompatible parameters).
        2. Roll back to the previous valid configuration.
        3. Return the rolled-back configuration.

        Parameters
        ----------
        config : ModelConfig
            The configuration that caused the initialization failure.
        error : Exception
            The exception raised during model initialization.

        Returns
        -------
        ModelConfig
            The rolled-back (previous valid) configuration.

        Raises
        ------
        ValueError
            If rollback is not possible (no history available).
        """
        event = {
            "event_type": "init_failure",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_snapshot": {
                "attention_heads": config.attention_heads,
                "hidden_dim": config.hidden_dim,
                "seaco_params": copy.deepcopy(config.seaco_params),
            },
            "error_type": type(error).__name__,
            "error_message": str(error),
            "action_taken": "rollback",
        }
        self.error_log.append(event)
        logger.error(
            "Model init failure: %s: %s. Rolling back.",
            type(error).__name__,
            error,
        )

        rolled_back = self.model_optimizer.rollback()
        return rolled_back


# =====================================================================
# SkillOutputHandler
# =====================================================================

# Required sections in a Skill Markdown file
_REQUIRED_SKILL_SECTIONS = [
    "能力描述",
    "分析规则",
]

# Regex for Markdown headings (levels 1-6)
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)

# Regex for knowledge base reference paths (e.g. `knowledge/xxx.json`)
_KNOWLEDGE_REF_RE = re.compile(r"knowledge/[\w/.-]+")


class SkillOutputHandler:
    """Handle Kiro Skill output parsing and Skill file validation.

    Works with ``LLMSkillEngine`` for parsing optimization output and
    ``SkillEvolutionEngine`` for version rollback on evolution failure.

    Parameters
    ----------
    llm_skill_engine : LLMSkillEngine
        Engine used for parsing optimization plan output.
    skill_evolution_engine : SkillEvolutionEngine or None
        Engine used for Skill version rollback.  May be ``None`` if
        evolution features are not available.
    workspace_dir : str
        Root workspace directory for resolving knowledge base paths.
    """

    def __init__(
        self,
        llm_skill_engine: LLMSkillEngine,
        skill_evolution_engine: Optional[SkillEvolutionEngine] = None,
        workspace_dir: str = "workspace",
    ) -> None:
        self.llm_skill_engine = llm_skill_engine
        self.skill_evolution_engine = skill_evolution_engine
        self.workspace_dir = workspace_dir
        self.error_log: list[dict] = []

    def parse_optimization_output(self, kiro_output: str) -> dict:
        """Parse Kiro Skill optimization output into a structured dict.

        Parsing strategies (tried in order):

        1. Delegate to ``LLMSkillEngine.parse_optimization_plan()`` which
           extracts a JSON block from the output.
        2. If the plan has no suggestions (parse failure), attempt
           keyword/template matching as a fallback.
        3. On complete failure, record the raw output and return a
           fallback dict indicating manual confirmation is needed.

        Parameters
        ----------
        kiro_output : str
            Raw text output from Kiro Skill.

        Returns
        -------
        dict
            Parsed optimization output with at minimum the keys
            ``"status"`` (``"success"`` or ``"fallback"`` or
            ``"manual_review_needed"``), ``"suggestions"`` (list), and
            ``"diagnosis_summary"`` (str).
        """
        # Strategy 1: structured JSON parsing via LLMSkillEngine
        plan = self.llm_skill_engine.parse_optimization_plan(kiro_output)

        if plan.suggestions:
            return {
                "status": "success",
                "iteration_id": plan.iteration_id,
                "suggestions": [
                    {
                        "dimension": s.dimension,
                        "title": s.title,
                        "description": s.description,
                        "priority": s.priority,
                        "confidence": s.confidence,
                        "config_changes": s.config_changes,
                        "rationale": s.rationale,
                    }
                    for s in plan.suggestions
                ],
                "diagnosis_summary": plan.diagnosis_summary,
            }

        # Strategy 2: keyword/template matching fallback
        fallback_result = self._keyword_fallback_parse(kiro_output)
        if fallback_result["suggestions"]:
            event = {
                "event_type": "parse_fallback",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": "keyword_matching",
                "suggestion_count": len(fallback_result["suggestions"]),
            }
            self.error_log.append(event)
            logger.info(
                "Parsed %d suggestions via keyword fallback.",
                len(fallback_result["suggestions"]),
            )
            return fallback_result

        # Strategy 3: complete failure — record raw output
        event = {
            "event_type": "parse_failure",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_output_length": len(kiro_output),
            "raw_output_preview": kiro_output[:500],
            "action_taken": "manual_review_needed",
        }
        self.error_log.append(event)
        logger.warning(
            "Failed to parse Skill output (%d chars). Manual review needed.",
            len(kiro_output),
        )

        return {
            "status": "manual_review_needed",
            "suggestions": [],
            "diagnosis_summary": plan.diagnosis_summary or "Parse failure",
            "raw_output": kiro_output,
        }

    def validate_skill_file(
        self, skill_path: str
    ) -> tuple[bool, list[str]]:
        """Validate a Skill file's format and content.

        Checks performed:

        1. File exists and is readable.
        2. Markdown format is correct (contains at least one heading).
        3. Required sections exist (能力描述, 分析规则).
        4. Knowledge base reference paths are valid (files exist).

        Parameters
        ----------
        skill_path : str
            Path to the Skill Markdown file.

        Returns
        -------
        tuple[bool, list[str]]
            ``(is_valid, error_list)`` where ``is_valid`` is ``True``
            when no errors are found.
        """
        errors: list[str] = []

        # Check 1: file exists
        if not os.path.isfile(skill_path):
            errors.append(f"Skill file not found: {skill_path}")
            self._log_validation(skill_path, errors)
            return False, errors

        # Read file content
        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            errors.append(f"Cannot read Skill file: {exc}")
            self._log_validation(skill_path, errors)
            return False, errors

        # Check 2: Markdown format — at least one heading
        headings = _HEADING_RE.findall(content)
        if not headings:
            errors.append(
                "Invalid Markdown format: no headings found"
            )

        # Check 3: required sections
        headings_lower = [h.strip().lower() for h in headings]
        for section in _REQUIRED_SKILL_SECTIONS:
            if not any(section.lower() in h for h in headings_lower):
                errors.append(
                    f"Missing required section: '{section}'"
                )

        # Check 4: knowledge base reference paths
        refs = _KNOWLEDGE_REF_RE.findall(content)
        for ref_path in refs:
            full_path = os.path.join(self.workspace_dir, ref_path)
            if not os.path.exists(full_path):
                errors.append(
                    f"Invalid knowledge base reference: '{ref_path}' "
                    f"(resolved to '{full_path}')"
                )

        self._log_validation(skill_path, errors)
        return len(errors) == 0, errors

    def handle_skill_evolution_failure(
        self, version_id: str
    ) -> bool:
        """Handle a Skill evolution failure by rolling back.

        Actions performed:

        1. Record the failed evolution attempt.
        2. Roll back to the specified previous Skill file version via
           ``SkillEvolutionEngine.rollback_version()``.
        3. Log the outcome.

        Parameters
        ----------
        version_id : str
            The version ID to roll back to.

        Returns
        -------
        bool
            ``True`` if rollback succeeded, ``False`` otherwise (e.g.
            no evolution engine configured or version not found).
        """
        event = {
            "event_type": "skill_evolution_failure",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version_id": version_id,
            "action_taken": "rollback_attempted",
        }

        if self.skill_evolution_engine is None:
            event["result"] = "no_evolution_engine"
            event["action_taken"] = "rollback_skipped"
            self.error_log.append(event)
            logger.warning(
                "Skill evolution failure: no evolution engine configured."
            )
            return False

        success = self.skill_evolution_engine.rollback_version(version_id)
        event["result"] = "success" if success else "failed"
        self.error_log.append(event)

        if success:
            logger.info(
                "Skill evolution failure handled: rolled back to %s.",
                version_id,
            )
        else:
            logger.warning(
                "Skill evolution rollback to %s failed.", version_id
            )

        return success

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_fallback_parse(text: str) -> dict:
        """Attempt to extract suggestions via keyword/template matching.

        Looks for dimension keywords (模型/model, 参数/param, 数据/data,
        后处理/post_process) in the text and creates minimal suggestion
        entries for each detected dimension.

        Parameters
        ----------
        text : str
            Raw Kiro output text.

        Returns
        -------
        dict
            A result dict with ``"status"`` set to ``"fallback"`` and
            any extracted suggestions.
        """
        dimension_keywords = {
            "model": ["模型", "model", "attention", "注意力"],
            "param": ["参数", "param", "learning_rate", "学习率", "batch"],
            "data": ["数据", "data", "augment", "增强"],
            "post_process": ["后处理", "post_process", "热词", "hotword"],
        }

        suggestions: list[dict] = []
        text_lower = text.lower()

        for dimension, keywords in dimension_keywords.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    suggestions.append({
                        "dimension": dimension,
                        "title": f"Keyword-matched suggestion ({dimension})",
                        "description": f"Detected keyword '{kw}' in output",
                        "priority": 1,
                        "confidence": 0.3,
                        "config_changes": {},
                        "rationale": "Extracted via keyword fallback",
                    })
                    break  # one suggestion per dimension

        return {
            "status": "fallback",
            "suggestions": suggestions,
            "diagnosis_summary": "Parsed via keyword fallback",
        }

    def _log_validation(
        self, skill_path: str, errors: list[str]
    ) -> None:
        """Record a Skill file validation result in the error log."""
        event = {
            "event_type": "skill_validation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "skill_path": skill_path,
            "is_valid": len(errors) == 0,
            "error_count": len(errors),
            "errors": list(errors),
        }
        self.error_log.append(event)
