"""LLM Skill Engine module for the ASR Training Optimizer system.

Provides the ``LLMSkillEngine`` class which prepares structured analysis
data from training metrics, badcase results, and dataset quality reports
for Kiro to read and reason about.  All methods return plain dicts
(JSON-serializable data) that can be written to files or passed to
Kiro Skill files for intelligent decision-making.
"""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict
from typing import Optional

from scripts.core.models import (
    BadCase,
    DatasetQualityReport,
    EvaluationReport,
    OptimizationPlan,
    OptimizationSuggestion,
    TrainingAlert,
    TrainingMetrics,
)


class LLMSkillEngine:
    """LLM Skill Engine for preparing analysis data and parsing results.

    This class encapsulates the data preparation logic for the Kiro Skill
    decision layer.  Python scripts produce structured analysis data which
    Kiro reads alongside Skill files (steering + skills) to generate
    optimization suggestions.

    The engine converts internal dataclass instances into plain dicts
    suitable for JSON serialization, adding summary statistics and
    contextual information that help Kiro reason about the training state.
    """

    # ------------------------------------------------------------------
    # Training analysis data preparation
    # ------------------------------------------------------------------

    def prepare_training_analysis_data(
        self,
        metrics_history: list[TrainingMetrics],
        alert: Optional[TrainingAlert] = None,
    ) -> dict:
        """Prepare training metrics and alert data for Kiro to read.

        Converts a history of training metrics and an optional alert
        into a structured dict containing:

        - A summary of the latest training metrics (loss, learning rate,
          gradient norm).
        - Epoch-level trends (loss trend, CER/WER trend).
        - Alert details if present.
        - Recommendations context derived from the metrics.

        Parameters
        ----------
        metrics_history : list[TrainingMetrics]
            Chronologically ordered list of collected training metrics.
        alert : TrainingAlert or None
            An optional training alert (e.g. stagnation, NaN/Inf).

        Returns
        -------
        dict
            Structured JSON-serializable analysis data.
        """
        # --- Latest metrics summary ---
        summary: dict = {}
        if metrics_history:
            latest = metrics_history[-1]
            summary = {
                "latest_train_loss": latest.train_loss,
                "latest_val_loss": latest.val_loss,
                "latest_learning_rate": latest.learning_rate,
                "latest_gradient_norm": latest.gradient_norm,
                "latest_cer": latest.cer,
                "latest_wer": latest.wer,
                "latest_epoch": latest.epoch,
                "latest_step": latest.step,
                "total_metrics_collected": len(metrics_history),
            }

        # --- Epoch-level trends ---
        epoch_trends = self._compute_epoch_trends(metrics_history)

        # --- Alert details ---
        alert_data: Optional[dict] = None
        if alert is not None:
            alert_data = {
                "alert_type": alert.alert_type.value,
                "message": alert.message,
                "context": alert.context,
                "epoch": alert.epoch,
                "step": alert.step,
            }

        # --- Recommendations context ---
        recommendations_context = self._build_recommendations_context(
            metrics_history, alert
        )

        return {
            "analysis_type": "training",
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": summary,
            "epoch_trends": epoch_trends,
            "alert": alert_data,
            "recommendations_context": recommendations_context,
        }

    # ------------------------------------------------------------------
    # Badcase analysis data preparation
    # ------------------------------------------------------------------

    def prepare_badcase_analysis_data(
        self,
        badcases: list[BadCase],
        error_distribution: dict[str, int],
    ) -> dict:
        """Prepare badcase and error distribution data for Kiro to read.

        Converts badcase instances and an error-type distribution into a
        structured dict containing:

        - Summary statistics (total badcases, counts by scene type).
        - Error type distribution.
        - Top badcases sorted by CER (descending).
        - Pattern analysis data.

        Parameters
        ----------
        badcases : list[BadCase]
            List of badcase instances from the result analyzer.
        error_distribution : dict[str, int]
            Mapping of error type names to their counts.

        Returns
        -------
        dict
            Structured JSON-serializable analysis data.
        """
        # --- Summary statistics ---
        scene_counts: dict[str, int] = {}
        for bc in badcases:
            scene = bc.scene_type or "unknown"
            scene_counts[scene] = scene_counts.get(scene, 0) + 1

        summary = {
            "total_badcases": len(badcases),
            "by_scene_type": scene_counts,
        }

        # --- Top badcases by CER ---
        sorted_badcases = sorted(badcases, key=lambda b: b.cer, reverse=True)
        top_badcases = [asdict(bc) for bc in sorted_badcases[:10]]

        # --- Pattern analysis ---
        pattern_analysis = self._analyze_badcase_patterns(badcases)

        return {
            "analysis_type": "badcase",
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": summary,
            "error_type_distribution": error_distribution,
            "top_badcases_by_cer": top_badcases,
            "pattern_analysis": pattern_analysis,
        }

    # ------------------------------------------------------------------
    # Data quality analysis preparation
    # ------------------------------------------------------------------

    def prepare_data_quality_analysis(
        self,
        report: DatasetQualityReport,
    ) -> dict:
        """Prepare a DatasetQualityReport for Kiro to read.

        Converts the quality report into a structured dict containing:

        - Basic stats summary.
        - SNR analysis.
        - Text stats summary.
        - Anomaly summary.
        - Scene distribution.
        - Issues sorted by severity.

        Parameters
        ----------
        report : DatasetQualityReport
            A complete dataset quality report.

        Returns
        -------
        dict
            Structured JSON-serializable analysis data.
        """
        # --- Basic stats ---
        stats_data = asdict(report.stats)

        # --- SNR analysis ---
        snr_analysis = {
            "snr_distribution": report.stats.snr_distribution,
            "low_quality_count": len(report.stats.low_quality_samples),
            "low_quality_samples": report.stats.low_quality_samples[:20],
        }

        # --- Text stats ---
        text_stats_data = asdict(report.text_stats)

        # --- Anomaly summary ---
        anomaly_types: dict[str, int] = {}
        for a in report.anomalies:
            anomaly_types[a.anomaly_type] = (
                anomaly_types.get(a.anomaly_type, 0) + 1
            )
        anomaly_summary = {
            "total_anomalies": len(report.anomalies),
            "by_type": anomaly_types,
            "details": [asdict(a) for a in report.anomalies[:20]],
        }

        # --- Scene distribution ---
        scene_data: Optional[dict] = None
        if report.scene_distribution is not None:
            scene_data = asdict(report.scene_distribution)

        # --- Issues by severity ---
        issues_data = report.issues_by_severity

        return {
            "analysis_type": "data_quality",
            "timestamp": datetime.datetime.now().isoformat(),
            "basic_stats": stats_data,
            "snr_analysis": snr_analysis,
            "text_stats": text_stats_data,
            "anomaly_summary": anomaly_summary,
            "scene_distribution": scene_data,
            "issues_by_severity": issues_data,
        }

    # ------------------------------------------------------------------
    # Comprehensive analysis
    # ------------------------------------------------------------------

    def prepare_comprehensive_analysis(
        self,
        training_data: dict,
        result_data: EvaluationReport,
        dataset_data: DatasetQualityReport,
    ) -> dict:
        """Combine all three analyses into a complete analysis JSON.

        Merges training analysis data, evaluation report data, and
        dataset quality data into a single comprehensive dict.  Also
        generates cross-analysis insights that correlate findings across
        the three domains.

        Parameters
        ----------
        training_data : dict
            Pre-prepared training analysis data (output of
            :meth:`prepare_training_analysis_data`).
        result_data : EvaluationReport
            Model evaluation report.
        dataset_data : DatasetQualityReport
            Dataset quality report.

        Returns
        -------
        dict
            Structured JSON-serializable comprehensive analysis.
        """
        # Convert result_data to a dict section
        result_section = {
            "overall_cer": result_data.overall_cer,
            "overall_wer": result_data.overall_wer,
            "scene_metrics": [asdict(sm) for sm in result_data.scene_metrics],
            "badcase_count": len(result_data.badcases),
            "error_type_distribution": result_data.error_type_distribution,
        }

        # Convert dataset_data via the existing method
        dataset_section = self.prepare_data_quality_analysis(dataset_data)

        # Cross-analysis insights
        cross_insights = self._generate_cross_insights(
            training_data, result_data, dataset_data
        )

        return {
            "analysis_type": "comprehensive",
            "timestamp": datetime.datetime.now().isoformat(),
            "training_analysis": training_data,
            "result_analysis": result_section,
            "dataset_analysis": dataset_section,
            "cross_analysis_insights": cross_insights,
        }

    # ------------------------------------------------------------------
    # Optimization plan parsing
    # ------------------------------------------------------------------

    _JSON_BLOCK_RE = re.compile(
        r"```json\s*\n(.*?)```", re.DOTALL
    )

    def parse_optimization_plan(self, kiro_output: str) -> OptimizationPlan:
        """Parse Kiro-generated optimization plan text into a structured OptimizationPlan.

        The input string may contain a JSON block delimited by
        ````` ```json ... ``` ````` markers, or may be raw JSON.  The method
        extracts the JSON, parses it, and returns an ``OptimizationPlan``
        populated with ``OptimizationSuggestion`` instances.

        If the input is malformed or cannot be parsed, an empty plan with
        a diagnostic ``diagnosis_summary`` is returned.

        Parameters
        ----------
        kiro_output : str
            Raw text from Kiro that may contain a JSON optimization plan.

        Returns
        -------
        OptimizationPlan
            Parsed plan, or an empty plan with a diagnostic message on failure.
        """
        json_str = self._extract_json_block(kiro_output)
        if json_str is None:
            return OptimizationPlan(
                iteration_id="",
                suggestions=[],
                diagnosis_summary="Failed to extract JSON block from input",
                created_at=datetime.datetime.now().isoformat(),
            )

        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as exc:
            return OptimizationPlan(
                iteration_id="",
                suggestions=[],
                diagnosis_summary=f"JSON parse error: {exc}",
                created_at=datetime.datetime.now().isoformat(),
            )

        if not isinstance(data, dict):
            return OptimizationPlan(
                iteration_id="",
                suggestions=[],
                diagnosis_summary="Parsed JSON is not a dict",
                created_at=datetime.datetime.now().isoformat(),
            )

        suggestions: list[OptimizationSuggestion] = []
        for raw in data.get("suggestions", []):
            if not isinstance(raw, dict):
                continue
            suggestions.append(
                OptimizationSuggestion(
                    dimension=str(raw.get("dimension", "")),
                    title=str(raw.get("title", "")),
                    description=str(raw.get("description", "")),
                    priority=int(raw.get("priority", 1)),
                    confidence=float(raw.get("confidence", 0.0)),
                    config_changes=raw.get("config_changes", {}),
                    rationale=str(raw.get("rationale", "")),
                )
            )

        return OptimizationPlan(
            iteration_id=str(data.get("iteration_id", "")),
            suggestions=suggestions,
            diagnosis_summary=str(data.get("diagnosis_summary", "")),
            created_at=datetime.datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # Executable config generation
    # ------------------------------------------------------------------

    _DIMENSION_CONFIG_TYPE_MAP: dict[str, str] = {
        "model": "model_config",
        "param": "training_config",
        "data": "data_pipeline_config",
        "post_process": "post_process_config",
    }

    _DIMENSION_DEFAULT_KEYS: dict[str, list[str]] = {
        "model": ["attention_heads", "hidden_dim", "seaco_params", "speech_enhance_config"],
        "param": ["learning_rate", "batch_size", "warmup_steps", "weight_decay",
                   "lr_schedule", "max_epochs", "gradient_clip"],
        "data": ["augmentation_types", "snr_threshold", "sampling_strategy"],
        "post_process": ["hotword_entries", "regex_rules", "lm_rerank"],
    }

    def generate_executable_config(
        self, suggestion: OptimizationSuggestion
    ) -> dict:
        """Convert an OptimizationSuggestion into an executable configuration change.

        Maps the suggestion's ``dimension`` to the appropriate config type
        and produces a dict with the following keys:

        - ``dimension`` – the suggestion dimension.
        - ``config_type`` – the mapped config type string.
        - ``changes`` – the concrete configuration changes.
        - ``metadata`` – additional context (dimension, title, timestamp).

        Parameters
        ----------
        suggestion : OptimizationSuggestion
            A single optimization suggestion.

        Returns
        -------
        dict
            Executable configuration change dict.
        """
        dimension = suggestion.dimension
        config_type = self._DIMENSION_CONFIG_TYPE_MAP.get(
            dimension, f"{dimension}_config"
        )

        changes = self._build_changes_for_dimension(suggestion)

        metadata = {
            "dimension": dimension,
            "title": suggestion.title,
            "timestamp": datetime.datetime.now().isoformat(),
            "priority": suggestion.priority,
            "confidence": suggestion.confidence,
        }

        return {
            "dimension": dimension,
            "config_type": config_type,
            "changes": changes,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """Extract a JSON string from *text*.

        Tries the following strategies in order:

        1. Look for a fenced ````` ```json ... ``` ````` code block.
        2. Try to parse the entire (stripped) text as JSON.

        Returns ``None`` if no valid JSON can be located.
        """
        if not text or not text.strip():
            return None

        # Strategy 1: fenced code block
        match = LLMSkillEngine._JSON_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()

        # Strategy 2: raw JSON (the whole string)
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped

        return None

    @staticmethod
    def _build_changes_for_dimension(
        suggestion: OptimizationSuggestion,
    ) -> dict:
        """Build the ``changes`` dict for a given suggestion dimension.

        Uses the suggestion's ``config_changes`` as the base and adds
        dimension-specific default structure where keys are missing.
        """
        changes: dict = dict(suggestion.config_changes)
        dimension = suggestion.dimension

        if dimension == "model":
            # Ensure model-specific keys exist
            changes.setdefault("attention_heads", None)
            changes.setdefault("hidden_dim", None)
            changes.setdefault("seaco_params", {})
            changes.setdefault("speech_enhance_config", {})
        elif dimension == "param":
            changes.setdefault("learning_rate", None)
            changes.setdefault("batch_size", None)
            changes.setdefault("warmup_steps", None)
            changes.setdefault("weight_decay", None)
            changes.setdefault("lr_schedule", None)
            changes.setdefault("max_epochs", None)
            changes.setdefault("gradient_clip", None)
        elif dimension == "data":
            changes.setdefault("augmentation_types", [])
            changes.setdefault("snr_threshold", None)
            changes.setdefault("sampling_strategy", {})
        elif dimension == "post_process":
            changes.setdefault("hotword_entries", [])
            changes.setdefault("regex_rules", [])
            changes.setdefault("lm_rerank", {})

        return changes

    @staticmethod
    def _compute_epoch_trends(
        metrics_history: list[TrainingMetrics],
    ) -> list[dict]:
        """Compute per-epoch trend summaries from metrics history.

        Groups metrics by epoch and computes average loss, CER, and WER
        for each epoch.

        Parameters
        ----------
        metrics_history : list[TrainingMetrics]
            Chronologically ordered metrics.

        Returns
        -------
        list[dict]
            One dict per epoch with keys ``epoch``, ``avg_train_loss``,
            ``avg_val_loss``, ``avg_cer``, ``avg_wer``, and
            ``sample_count``.
        """
        epoch_data: dict[int, dict] = {}
        for m in metrics_history:
            if m.epoch not in epoch_data:
                epoch_data[m.epoch] = {
                    "train_losses": [],
                    "val_losses": [],
                    "cers": [],
                    "wers": [],
                }
            epoch_data[m.epoch]["train_losses"].append(m.train_loss)
            if m.val_loss is not None:
                epoch_data[m.epoch]["val_losses"].append(m.val_loss)
            if m.cer is not None:
                epoch_data[m.epoch]["cers"].append(m.cer)
            if m.wer is not None:
                epoch_data[m.epoch]["wers"].append(m.wer)

        trends: list[dict] = []
        for epoch in sorted(epoch_data.keys()):
            d = epoch_data[epoch]
            tl = d["train_losses"]
            vl = d["val_losses"]
            cers = d["cers"]
            wers = d["wers"]
            trends.append({
                "epoch": epoch,
                "avg_train_loss": sum(tl) / len(tl) if tl else None,
                "avg_val_loss": sum(vl) / len(vl) if vl else None,
                "avg_cer": sum(cers) / len(cers) if cers else None,
                "avg_wer": sum(wers) / len(wers) if wers else None,
                "sample_count": len(tl),
            })
        return trends

    @staticmethod
    def _build_recommendations_context(
        metrics_history: list[TrainingMetrics],
        alert: Optional[TrainingAlert],
    ) -> dict:
        """Build contextual hints for Kiro's recommendation generation.

        Parameters
        ----------
        metrics_history : list[TrainingMetrics]
            Metrics history.
        alert : TrainingAlert or None
            Optional alert.

        Returns
        -------
        dict
        """
        context: dict = {
            "has_alert": alert is not None,
            "alert_type": alert.alert_type.value if alert else None,
            "total_epochs": 0,
            "total_steps": 0,
            "loss_trend": "unknown",
        }

        if not metrics_history:
            return context

        epochs = {m.epoch for m in metrics_history}
        context["total_epochs"] = len(epochs)
        context["total_steps"] = max(m.step for m in metrics_history)

        # Determine loss trend from first to last metric
        first_loss = metrics_history[0].train_loss
        last_loss = metrics_history[-1].train_loss
        if last_loss < first_loss:
            context["loss_trend"] = "decreasing"
        elif last_loss > first_loss:
            context["loss_trend"] = "increasing"
        else:
            context["loss_trend"] = "flat"

        return context

    @staticmethod
    def _analyze_badcase_patterns(badcases: list[BadCase]) -> dict:
        """Analyze patterns in badcases for Kiro's pattern recognition.

        Parameters
        ----------
        badcases : list[BadCase]
            Badcase list.

        Returns
        -------
        dict
            Pattern analysis with average CER per scene and error type
            counts per scene.
        """
        scene_cer: dict[str, list[float]] = {}
        scene_error_types: dict[str, dict[str, int]] = {}

        for bc in badcases:
            scene = bc.scene_type or "unknown"

            # CER per scene
            scene_cer.setdefault(scene, []).append(bc.cer)

            # Error types per scene
            if scene not in scene_error_types:
                scene_error_types[scene] = {}
            for err in bc.error_details:
                err_type = err.get("type", "unknown")
                scene_error_types[scene][err_type] = (
                    scene_error_types[scene].get(err_type, 0) + 1
                )

        avg_cer_by_scene: dict[str, float] = {}
        for scene, cers in scene_cer.items():
            avg_cer_by_scene[scene] = sum(cers) / len(cers) if cers else 0.0

        return {
            "avg_cer_by_scene": avg_cer_by_scene,
            "error_types_by_scene": scene_error_types,
        }

    @staticmethod
    def _generate_cross_insights(
        training_data: dict,
        result_data: EvaluationReport,
        dataset_data: DatasetQualityReport,
    ) -> list[dict]:
        """Generate cross-analysis insights correlating all three domains.

        Parameters
        ----------
        training_data : dict
            Training analysis data.
        result_data : EvaluationReport
            Evaluation report.
        dataset_data : DatasetQualityReport
            Dataset quality report.

        Returns
        -------
        list[dict]
            List of insight dicts with ``type`` and ``description`` keys.
        """
        insights: list[dict] = []

        # Insight: high CER scenes vs dataset distribution
        if (
            result_data.scene_metrics
            and dataset_data.scene_distribution is not None
        ):
            scene_counts = dataset_data.scene_distribution.scene_counts
            for sm in result_data.scene_metrics:
                count = scene_counts.get(sm.scene_type, 0)
                if sm.cer > 0.1 and count < 200:
                    insights.append({
                        "type": "data_insufficiency",
                        "description": (
                            f"Scene '{sm.scene_type}' has high CER "
                            f"({sm.cer:.3f}) with only {count} training "
                            f"samples. Consider data augmentation."
                        ),
                    })

        # Insight: training alert correlation
        alert_info = training_data.get("alert")
        if alert_info is not None:
            insights.append({
                "type": "training_alert_correlation",
                "description": (
                    f"Training alert ({alert_info['alert_type']}): "
                    f"{alert_info['message']}. Review training parameters "
                    f"and data quality together."
                ),
            })

        # Insight: data quality issues affecting results
        critical_issues = [
            i for i in dataset_data.issues_by_severity
            if i.get("severity") == "critical"
        ]
        if critical_issues:
            insights.append({
                "type": "data_quality_impact",
                "description": (
                    f"Found {len(critical_issues)} critical data quality "
                    f"issue(s) that may affect model performance."
                ),
            })

        return insights
