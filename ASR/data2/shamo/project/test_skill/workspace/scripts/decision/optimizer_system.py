"""Optimizer System module for the ASR Training Optimizer.

Provides the ``OptimizerSystem`` class which manages the full optimization
iteration lifecycle: starting iterations, coordinating analysis and
optimization phases, comparing iterations, rolling back to previous states,
and detecting optimization plateaus.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

from scripts.core.config_utils import (
    get_latest_version,
    list_versions,
    read_model_config,
    read_training_config,
    read_post_process_config,
    save_versioned_config,
    write_json,
    read_json,
)
from scripts.core.database import Database
from scripts.core.models import (
    EvaluationReport,
    IterationRecord,
    ModelConfig,
    OptimizationPlan,
    OptimizationSuggestion,
    SceneMetrics,
    TrainingConfig,
)


class OptimizerSystem:
    """Optimization loop coordinator managing iteration lifecycles.

    Coordinates the analysis layer (Training_Monitor, Result_Analyzer,
    Dataset_Analyzer) and execution layer (Model_Optimizer, Param_Tuner,
    Data_Pipeline, Post_Processor) to drive iterative ASR model
    optimization.

    Parameters
    ----------
    workspace_dir : str
        Root workspace directory containing ``configs/``, ``models/``,
        ``data/``, ``logs/``, ``reports/``, and ``knowledge/``.
    db : Database
        Initialized database instance for persisting iteration records.
    """

    def __init__(self, workspace_dir: str, db: Database) -> None:
        self.workspace_dir = workspace_dir
        self.db = db

        # Derived paths
        self.configs_dir = os.path.join(workspace_dir, "configs")
        self.model_config_dir = os.path.join(self.configs_dir, "model")
        self.training_config_dir = os.path.join(self.configs_dir, "training")
        self.post_process_config_dir = os.path.join(self.configs_dir, "post_process")
        self.models_dir = os.path.join(workspace_dir, "models")
        self.data_dir = os.path.join(workspace_dir, "data")
        self.logs_dir = os.path.join(workspace_dir, "logs")
        self.reports_dir = os.path.join(workspace_dir, "reports")

    # ------------------------------------------------------------------
    # Iteration lifecycle
    # ------------------------------------------------------------------

    def start_iteration(self, description: str = "") -> str:
        """Create a new optimization iteration record.

        Generates a unique ``iteration_id``, captures the current
        configuration snapshot, and persists the record to the database
        with status ``"pending"``.

        Parameters
        ----------
        description : str
            Human-readable description of the iteration's purpose.

        Returns
        -------
        str
            The generated ``iteration_id``.
        """
        iteration_id = f"iter_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        # Capture current configuration state
        model_changes = self._snapshot_model_config()
        param_changes = self._snapshot_training_config()
        data_changes: dict = {}

        self.db.insert("iterations", {
            "iteration_id": iteration_id,
            "description": description,
            "status": "pending",
            "cer_before": None,
            "cer_after": None,
            "wer_before": None,
            "wer_after": None,
            "param_changes_json": json.dumps(param_changes, ensure_ascii=False),
            "model_changes_json": json.dumps(model_changes, ensure_ascii=False),
            "data_changes_json": json.dumps(data_changes, ensure_ascii=False),
            "created_at": now,
            "completed_at": None,
        })

        return iteration_id

    def run_analysis_phase(
        self,
        iteration_id: str,
        training_metrics: Optional[dict] = None,
        evaluation_report: Optional[EvaluationReport] = None,
        dataset_report: Optional[dict] = None,
    ) -> dict:
        """Execute the analysis phase for an iteration.

        Coordinates Training_Monitor, Result_Analyzer, and
        Dataset_Analyzer outputs.  Callers provide pre-computed analysis
        data (or ``None`` for unavailable analyses).  The method updates
        the iteration status to ``"running"`` and stores baseline CER/WER
        from the evaluation report.

        Parameters
        ----------
        iteration_id : str
            The iteration to run analysis for.
        training_metrics : dict or None
            Pre-computed training analysis data.
        evaluation_report : EvaluationReport or None
            Pre-computed evaluation report.
        dataset_report : dict or None
            Pre-computed dataset quality analysis data.

        Returns
        -------
        dict
            Combined analysis results with keys ``training``,
            ``evaluation``, and ``dataset``.
        """
        # Update status to running
        self.db.update(
            "iterations", "iteration_id", iteration_id,
            {"status": "running"},
        )

        # Store baseline CER/WER from evaluation report
        if evaluation_report is not None:
            self.db.update(
                "iterations", "iteration_id", iteration_id,
                {
                    "cer_before": evaluation_report.overall_cer,
                    "wer_before": evaluation_report.overall_wer,
                },
            )

        analysis_result = {
            "iteration_id": iteration_id,
            "training": training_metrics,
            "evaluation": asdict(evaluation_report) if evaluation_report else None,
            "dataset": dataset_report,
        }

        # Save analysis report to disk
        report_dir = os.path.join(self.reports_dir, "evaluation")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{iteration_id}_analysis.json")
        write_json(report_path, analysis_result)

        return analysis_result

    def run_optimization_phase(
        self,
        iteration_id: str,
        plan: OptimizationPlan,
        evaluation_after: Optional[EvaluationReport] = None,
    ) -> dict:
        """Execute the optimization phase for an iteration.

        Processes each suggestion in the optimization plan, records the
        changes applied, and updates the iteration record with post-
        optimization evaluation results.

        Parameters
        ----------
        iteration_id : str
            The iteration to run optimization for.
        plan : OptimizationPlan
            The optimization plan containing suggestions to execute.
        evaluation_after : EvaluationReport or None
            Post-optimization evaluation report.  If provided, the
            iteration record is updated with the new CER/WER and
            marked as ``"completed"``.

        Returns
        -------
        dict
            Summary of the optimization phase including applied changes
            and evaluation deltas.
        """
        applied_changes: list[dict] = []
        model_changes: dict = {}
        param_changes: dict = {}
        data_changes: dict = {}

        for suggestion in plan.suggestions:
            change_record = {
                "dimension": suggestion.dimension,
                "title": suggestion.title,
                "priority": suggestion.priority,
                "confidence": suggestion.confidence,
                "config_changes": suggestion.config_changes,
            }
            applied_changes.append(change_record)

            # Categorize changes by dimension
            if suggestion.dimension == "model":
                model_changes.update(suggestion.config_changes)
            elif suggestion.dimension == "param":
                param_changes.update(suggestion.config_changes)
            elif suggestion.dimension == "data":
                data_changes.update(suggestion.config_changes)

        # Update iteration record with changes
        # Store optimization content summary in the description field
        opt_summary = plan.diagnosis_summary or ""
        if plan.suggestions:
            titles = [s.title for s in plan.suggestions]
            opt_summary += f" | Applied: {', '.join(titles)}"

        update_data: dict[str, Any] = {
            "model_changes_json": json.dumps(model_changes, ensure_ascii=False),
            "param_changes_json": json.dumps(param_changes, ensure_ascii=False),
            "data_changes_json": json.dumps(data_changes, ensure_ascii=False),
            "description": opt_summary.strip(),
        }

        # Compute deltas if we have post-optimization evaluation
        cer_delta = None
        wer_delta = None
        if evaluation_after is not None:
            update_data["cer_after"] = evaluation_after.overall_cer
            update_data["wer_after"] = evaluation_after.overall_wer
            update_data["status"] = "completed"
            update_data["completed_at"] = datetime.now(timezone.utc).isoformat()

            # Retrieve baseline CER/WER for delta computation
            row = self.db.get_by_id(
                "iterations", "iteration_id", iteration_id
            )
            if row and row.get("cer_before") is not None:
                cer_delta = evaluation_after.overall_cer - row["cer_before"]
                wer_delta = evaluation_after.overall_wer - row["wer_before"]
                update_data["cer_after"] = evaluation_after.overall_cer
                update_data["wer_after"] = evaluation_after.overall_wer

        self.db.update("iterations", "iteration_id", iteration_id, update_data)

        return {
            "iteration_id": iteration_id,
            "applied_changes": applied_changes,
            "cer_delta": cer_delta,
            "wer_delta": wer_delta,
            "status": "completed" if evaluation_after else "running",
        }

    def complete_iteration(
        self,
        iteration_id: str,
        evaluation_after: EvaluationReport,
    ) -> dict:
        """Mark an iteration as completed with final evaluation results.

        Parameters
        ----------
        iteration_id : str
            The iteration to complete.
        evaluation_after : EvaluationReport
            Post-optimization evaluation report.

        Returns
        -------
        dict
            Summary with CER/WER deltas.
        """
        row = self.db.get_by_id("iterations", "iteration_id", iteration_id)
        if row is None:
            return {"status": "error", "message": f"Iteration {iteration_id} not found"}

        cer_before = row.get("cer_before")
        wer_before = row.get("wer_before")
        cer_delta = None
        wer_delta = None

        if cer_before is not None:
            cer_delta = evaluation_after.overall_cer - cer_before
        if wer_before is not None:
            wer_delta = evaluation_after.overall_wer - wer_before

        self.db.update("iterations", "iteration_id", iteration_id, {
            "cer_after": evaluation_after.overall_cer,
            "wer_after": evaluation_after.overall_wer,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "iteration_id": iteration_id,
            "cer_before": cer_before,
            "cer_after": evaluation_after.overall_cer,
            "wer_before": wer_before,
            "wer_after": evaluation_after.overall_wer,
            "cer_delta": cer_delta,
            "wer_delta": wer_delta,
            "status": "completed",
            "skill_review": skill_review,
        }

    # ------------------------------------------------------------------
    # Iteration comparison
    # ------------------------------------------------------------------

    def compare_iterations(self, iter_id_a: str, iter_id_b: str) -> dict:
        """Compare two iterations by their CER/WER metrics.

        Parameters
        ----------
        iter_id_a : str
            First iteration ID (typically the earlier one).
        iter_id_b : str
            Second iteration ID (typically the later one).

        Returns
        -------
        dict
            Comparison result with CER/WER values and deltas for both
            iterations.  Returns an error dict if either iteration is
            not found.
        """
        row_a = self.db.get_by_id("iterations", "iteration_id", iter_id_a)
        row_b = self.db.get_by_id("iterations", "iteration_id", iter_id_b)

        if row_a is None:
            return {"status": "error", "message": f"Iteration {iter_id_a} not found"}
        if row_b is None:
            return {"status": "error", "message": f"Iteration {iter_id_b} not found"}

        cer_a_before = row_a.get("cer_before")
        cer_a_after = row_a.get("cer_after")
        wer_a_before = row_a.get("wer_before")
        wer_a_after = row_a.get("wer_after")

        cer_b_before = row_b.get("cer_before")
        cer_b_after = row_b.get("cer_after")
        wer_b_before = row_b.get("wer_before")
        wer_b_after = row_b.get("wer_after")

        # Compute cross-iteration deltas using after values
        cer_cross_delta = None
        wer_cross_delta = None
        if cer_a_after is not None and cer_b_after is not None:
            cer_cross_delta = cer_b_after - cer_a_after
        if wer_a_after is not None and wer_b_after is not None:
            wer_cross_delta = wer_b_after - wer_a_after

        return {
            "status": "success",
            "iteration_a": {
                "iteration_id": iter_id_a,
                "cer_before": cer_a_before,
                "cer_after": cer_a_after,
                "wer_before": wer_a_before,
                "wer_after": wer_a_after,
                "description": row_a.get("description", ""),
                "status": row_a.get("status", ""),
            },
            "iteration_b": {
                "iteration_id": iter_id_b,
                "cer_before": cer_b_before,
                "cer_after": cer_b_after,
                "wer_before": wer_b_before,
                "wer_after": wer_b_after,
                "description": row_b.get("description", ""),
                "status": row_b.get("status", ""),
            },
            "cer_delta": cer_cross_delta,
            "wer_delta": wer_cross_delta,
        }

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback_to_iteration(self, iteration_id: str) -> dict:
        """Roll back system state to a specific iteration's configuration.

        Restores the model configuration, training configuration, and
        data changes that were captured at the start of the target
        iteration.

        Parameters
        ----------
        iteration_id : str
            The iteration to roll back to.

        Returns
        -------
        dict
            Summary of the rollback operation including restored
            configurations.
        """
        row = self.db.get_by_id("iterations", "iteration_id", iteration_id)
        if row is None:
            return {"status": "error", "message": f"Iteration {iteration_id} not found"}

        restored_model = {}
        restored_params = {}
        restored_data = {}

        # Restore model configuration
        model_json = row.get("model_changes_json", "{}")
        if model_json:
            try:
                restored_model = json.loads(model_json)
            except (json.JSONDecodeError, TypeError):
                restored_model = {}

        if restored_model:
            save_versioned_config(
                self.model_config_dir, "model", restored_model
            )

        # Restore training configuration
        param_json = row.get("param_changes_json", "{}")
        if param_json:
            try:
                restored_params = json.loads(param_json)
            except (json.JSONDecodeError, TypeError):
                restored_params = {}

        if restored_params:
            save_versioned_config(
                self.training_config_dir, "train", restored_params
            )

        # Restore data changes
        data_json = row.get("data_changes_json", "{}")
        if data_json:
            try:
                restored_data = json.loads(data_json)
            except (json.JSONDecodeError, TypeError):
                restored_data = {}

        return {
            "status": "success",
            "iteration_id": iteration_id,
            "restored_model_config": restored_model,
            "restored_training_config": restored_params,
            "restored_data_changes": restored_data,
            "message": f"Rolled back to iteration {iteration_id}",
        }

    # ------------------------------------------------------------------
    # Iteration history
    # ------------------------------------------------------------------

    def get_iteration_history(self) -> list[IterationRecord]:
        """Retrieve all iteration records ordered by creation time.

        Returns
        -------
        list[IterationRecord]
            Chronologically ordered iteration records.
        """
        rows = self.db.get_all("iterations", order_by="created_at ASC")
        records: list[IterationRecord] = []

        for row in rows:
            model_changes = _safe_json_loads(row.get("model_changes_json", "{}"))
            param_changes = _safe_json_loads(row.get("param_changes_json", "{}"))
            data_changes = _safe_json_loads(row.get("data_changes_json", "{}"))

            record = IterationRecord(
                iteration_id=row.get("iteration_id", ""),
                optimization_content=row.get("description", ""),
                param_changes=param_changes,
                data_changes=data_changes,
                model_changes=model_changes,
                evaluation_before=EvaluationReport(
                    overall_cer=row.get("cer_before") or 0.0,
                    overall_wer=row.get("wer_before") or 0.0,
                ),
                evaluation_after=(
                    EvaluationReport(
                        overall_cer=row["cer_after"],
                        overall_wer=row["wer_after"],
                    )
                    if row.get("cer_after") is not None
                    else None
                ),
                cer_delta=(
                    row["cer_after"] - row["cer_before"]
                    if row.get("cer_after") is not None
                    and row.get("cer_before") is not None
                    else None
                ),
                wer_delta=(
                    row["wer_after"] - row["wer_before"]
                    if row.get("wer_after") is not None
                    and row.get("wer_before") is not None
                    else None
                ),
                timestamp=row.get("created_at", ""),
                status=row.get("status", "pending"),
            )
            records.append(record)

        return records

    # ------------------------------------------------------------------
    # Optimization plateau detection
    # ------------------------------------------------------------------

    def check_optimization_plateau(self, window: int = 3) -> bool:
        """Detect whether the last *window* iterations show no CER improvement.

        An optimization plateau is detected when the last *window*
        completed iterations all have ``cer_after >= cer_before``
        (i.e. CER did not improve in any of them).

        Parameters
        ----------
        window : int
            Number of recent completed iterations to examine
            (default ``3``).

        Returns
        -------
        bool
            ``True`` if a plateau is detected, ``False`` otherwise.
        """
        rows = self.db.get_all(
            "iterations",
            filters={"status": "completed"},
            order_by="created_at ASC",
        )

        if len(rows) < window:
            return False

        recent = rows[-window:]

        for row in recent:
            cer_before = row.get("cer_before")
            cer_after = row.get("cer_after")

            # If either value is missing, we can't determine improvement
            if cer_before is None or cer_after is None:
                return False

            # If CER improved (decreased), no plateau
            if cer_after < cer_before:
                return False

        return True

    # ------------------------------------------------------------------
    # Skill auto-review (Requirement 11.7)
    # ------------------------------------------------------------------

    def _auto_review_skills(self, iteration_id: str) -> dict:
        """Lightweight Skill review triggered after each iteration completes.

        Performs:
        1. Updates feedback labels for this iteration's suggestions
        2. Checks if rules need threshold/confidence updates
        3. Executes knowledge entry lifecycle transitions
        4. Updates Skill file quick reference tables
        5. Every 5 completed iterations, triggers a deep review

        Parameters
        ----------
        iteration_id : str
            The completed iteration ID.

        Returns
        -------
        dict
            Summary of the review actions taken.
        """
        from scripts.decision.skill_evolution import SkillEvolutionEngine
        from scripts.decision.skill_knowledge_base import SkillKnowledgeBase

        engine = SkillEvolutionEngine(self.workspace_dir, self.db)
        kb = SkillKnowledgeBase(self.db)

        review_result: dict = {
            "iteration_id": iteration_id,
            "feedback_labeled": False,
            "lifecycle_transitions": {},
            "quick_reference_updated": [],
            "rules_updated": {},
            "deep_review": None,
        }

        # 1. Label feedback for this iteration
        row = self.db.get_by_id("iterations", "iteration_id", iteration_id)
        if row and row.get("cer_before") is not None and row.get("cer_after") is not None:
            feedback = engine.label_feedback(
                iteration_id,
                cer_before=row["cer_before"],
                cer_after=row["cer_after"],
            )
            engine.record_feedback(feedback)
            review_result["feedback_labeled"] = True

        # 2. Check if rules need updating (run pattern analysis)
        patterns = engine.analyze_patterns(min_feedback_count=3)
        if patterns:
            weights = engine.update_strategy_weights(patterns)
            rules_result = engine.update_skill_rules_from_patterns(patterns, weights)
            review_result["rules_updated"] = rules_result

        # 3. Execute knowledge entry lifecycle transitions
        transitions = kb.apply_lifecycle_transitions()
        review_result["lifecycle_transitions"] = transitions

        # 4. Update quick reference tables for all skill files
        skills_dir = os.path.join(self.workspace_dir, ".kiro", "skills")
        if os.path.isdir(skills_dir):
            for filename in sorted(os.listdir(skills_dir)):
                if filename.endswith(".md"):
                    skill_name = filename[:-3]  # Remove .md extension
                    result = engine.update_skill_quick_reference(skill_name)
                    if result.get("success"):
                        review_result["quick_reference_updated"].append(skill_name)

        # 5. Check if deep review is needed (every 5 completed iterations)
        completed_rows = self.db.get_all(
            "iterations", filters={"status": "completed"}
        )
        if len(completed_rows) > 0 and len(completed_rows) % 5 == 0:
            deep_result = self._deep_review_skills()
            review_result["deep_review"] = deep_result

        return review_result

    def _deep_review_skills(self) -> dict:
        """Deep Skill review triggered every 5 iterations.

        Performs:
        1. Merges similar knowledge entries (same type + similar content)
        2. Deprecates long-term ineffective rules
        3. Recalculates all strategy weights
        4. Updates all Skill file quick reference tables and rule params

        Returns
        -------
        dict
            Summary of the deep review actions taken.
        """
        from scripts.decision.skill_evolution import SkillEvolutionEngine
        from scripts.decision.skill_knowledge_base import SkillKnowledgeBase

        engine = SkillEvolutionEngine(self.workspace_dir, self.db)
        kb = SkillKnowledgeBase(self.db)

        result: dict = {
            "merged_entries": 0,
            "deprecated_rules": [],
            "strategies_recalculated": 0,
            "quick_references_updated": [],
            "rules_updated": {},
        }

        # 1. Merge similar knowledge entries
        all_entries = self.db.get_all(
            "knowledge_entries", filters={"state": "active"}
        )
        merged = self._merge_similar_entries(all_entries)
        result["merged_entries"] = merged

        # 2. Deprecate long-term ineffective rules
        deprecated = engine.deprecate_failed_rules(min_failures=3)
        result["deprecated_rules"] = deprecated

        # 3. Recalculate all strategy weights from full feedback history
        patterns = engine.analyze_patterns(min_feedback_count=1)
        if patterns:
            weights = engine.update_strategy_weights(patterns)
            result["strategies_recalculated"] = len(weights)

            # Update rules based on patterns
            rules_result = engine.update_skill_rules_from_patterns(patterns, weights)
            result["rules_updated"] = rules_result

        # 4. Update all Skill file quick reference tables
        skills_dir = os.path.join(self.workspace_dir, ".kiro", "skills")
        if os.path.isdir(skills_dir):
            for filename in sorted(os.listdir(skills_dir)):
                if filename.endswith(".md"):
                    skill_name = filename[:-3]
                    update_result = engine.update_skill_quick_reference(skill_name)
                    if update_result.get("success"):
                        result["quick_references_updated"].append(skill_name)

        # Create a version snapshot after deep review
        engine.create_version_snapshot("Deep review after 5 iterations")

        return result

    def _merge_similar_entries(self, entries: list[dict]) -> int:
        """Merge knowledge entries with the same type and similar content.

        When two entries of the same knowledge_type have identical
        content keys (ignoring confidence/metadata), the one with lower
        confidence is archived.

        Parameters
        ----------
        entries : list[dict]
            Active knowledge entries from the database.

        Returns
        -------
        int
            Number of entries archived due to merging.
        """
        from scripts.decision.skill_knowledge_base import _safe_json_loads

        merged_count = 0
        seen: dict[str, str] = {}  # (type, content_key) -> entry_id

        for entry in entries:
            ktype = entry.get("knowledge_type", "")
            content = _safe_json_loads(entry.get("content_json", "{}"))
            # Create a content key for deduplication
            content_key = json.dumps(content, sort_keys=True, ensure_ascii=False)
            lookup_key = f"{ktype}::{content_key}"

            entry_id = entry.get("entry_id", "")
            if lookup_key in seen:
                # Duplicate found - archive the one with lower confidence
                existing_id = seen[lookup_key]
                existing = self.db.get_by_id(
                    "knowledge_entries", "entry_id", existing_id
                )
                existing_conf = (existing.get("confidence", 0.0) or 0.0) if existing else 0.0
                current_conf = entry.get("confidence", 0.0) or 0.0

                if current_conf >= existing_conf:
                    # Archive the existing one, keep current
                    self.db.update(
                        "knowledge_entries", "entry_id", existing_id,
                        {"state": "archived"},
                    )
                    seen[lookup_key] = entry_id
                else:
                    # Archive current one
                    self.db.update(
                        "knowledge_entries", "entry_id", entry_id,
                        {"state": "archived"},
                    )
                merged_count += 1
            else:
                seen[lookup_key] = entry_id

        return merged_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_model_config(self) -> dict:
        """Capture the current model configuration as a dict."""
        latest = get_latest_version(self.model_config_dir, "model")
        if latest is not None:
            try:
                config = read_model_config(latest)
                return asdict(config)
            except Exception:
                return {}
        return {}

    def _snapshot_training_config(self) -> dict:
        """Capture the current training configuration as a dict."""
        latest = get_latest_version(self.training_config_dir, "train")
        if latest is not None:
            try:
                config = read_training_config(latest)
                data = asdict(config)
                # Serialize enum
                data["lr_schedule"] = config.lr_schedule.value
                return data
            except Exception:
                return {}
        return {}


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_json_loads(value: Any) -> dict:
    """Parse a JSON string, returning an empty dict on failure."""
    if not value:
        return {}
    try:
        result = json.loads(value)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
