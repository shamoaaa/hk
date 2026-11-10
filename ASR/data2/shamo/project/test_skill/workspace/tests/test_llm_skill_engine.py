"""Tests for the LLM Skill Engine module.

Covers each ``prepare_*`` method with sample data, edge cases (empty
lists, None values), and verifies output structure has expected keys.
"""

from __future__ import annotations

import pytest

from scripts.decision.llm_skill_engine import LLMSkillEngine
from scripts.core.models import (
    AlertType,
    AnomalySample,
    BadCase,
    DatasetQualityReport,
    DatasetStats,
    EvaluationReport,
    SceneDistribution,
    SceneMetrics,
    TextStats,
    TrainingAlert,
    TrainingMetrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Return a fresh LLMSkillEngine instance."""
    return LLMSkillEngine()


@pytest.fixture
def metrics_history():
    """Return a sample metrics history spanning two epochs."""
    return [
        TrainingMetrics(
            step=0, epoch=0, train_loss=2.5, val_loss=2.8,
            learning_rate=1e-4, gradient_norm=1.2,
            cer=0.45, wer=0.55, timestamp="2024-01-01T00:00:00",
        ),
        TrainingMetrics(
            step=100, epoch=0, train_loss=1.8, val_loss=2.1,
            learning_rate=1e-4, gradient_norm=0.9,
            cer=0.35, wer=0.45, timestamp="2024-01-01T00:10:00",
        ),
        TrainingMetrics(
            step=200, epoch=1, train_loss=1.2, val_loss=1.5,
            learning_rate=5e-5, gradient_norm=0.7,
            cer=0.25, wer=0.35, timestamp="2024-01-01T00:20:00",
        ),
    ]


@pytest.fixture
def stagnation_alert():
    """Return a sample stagnation alert."""
    return TrainingAlert(
        alert_type=AlertType.STAGNATION,
        message="Training stagnation detected",
        context={"window": 5, "threshold": 0.001},
        epoch=10,
        step=1000,
    )


@pytest.fixture
def badcases():
    """Return a sample list of badcases."""
    return [
        BadCase(
            audio_path="/data/audio_001.wav",
            reference_text="今天天气很好",
            hypothesis_text="今天天气很号",
            cer=0.167, wer=0.2,
            error_details=[
                {"type": "substitution", "ref": "好", "hyp": "号", "position": 5},
            ],
            scene_type="clean",
        ),
        BadCase(
            audio_path="/data/audio_002.wav",
            reference_text="请打开空调",
            hypothesis_text="请打开空",
            cer=0.2, wer=0.25,
            error_details=[
                {"type": "deletion", "ref": "调", "hyp": "", "position": 4},
            ],
            scene_type="background_voice",
        ),
        BadCase(
            audio_path="/data/audio_003.wav",
            reference_text="明天下午开会",
            hypothesis_text="明天下午来开会",
            cer=0.167, wer=0.2,
            error_details=[
                {"type": "insertion", "ref": "", "hyp": "来", "position": 4},
            ],
            scene_type="background_noise",
        ),
    ]


@pytest.fixture
def error_distribution():
    """Return a sample error type distribution."""
    return {"substitution": 45, "insertion": 20, "deletion": 35}


@pytest.fixture
def quality_report():
    """Return a sample DatasetQualityReport."""
    stats = DatasetStats(
        total_samples=1000,
        total_duration_hours=50.0,
        avg_duration_seconds=5.0,
        duration_distribution={"0-5s": 600, "5-10s": 300, "10-20s": 100},
        snr_distribution={"<0dB": 5, "0-10dB": 50, "10-20dB": 500, "20-30dB": 400, "30+dB": 45},
        low_quality_samples=["/data/low_001.wav", "/data/low_002.wav"],
    )
    text_stats = TextStats(
        char_distribution={"你": 100, "好": 80},
        word_frequency={"你好": 50},
        sentence_length_distribution={"0-10": 200, "10-20": 500, "20-50": 300},
    )
    anomalies = [
        AnomalySample(
            audio_path="/data/bad_001.wav",
            anomaly_type="empty_label",
            details="Text label is empty",
        ),
        AnomalySample(
            audio_path="/data/bad_002.wav",
            anomaly_type="corrupted_audio",
            details="Duration is 0",
        ),
    ]
    scene_dist = SceneDistribution(
        scene_counts={"clean": 600, "background_voice": 300, "background_noise": 100},
        scene_ratios={"clean": 0.6, "background_voice": 0.3, "background_noise": 0.1},
        underrepresented_scenes=["background_noise"],
    )
    issues = [
        {"severity": "critical", "type": "corrupted_audio", "count": 1,
         "message": "Found 1 corrupted audio sample(s)"},
        {"severity": "critical", "type": "empty_label", "count": 1,
         "message": "Found 1 sample(s) with empty labels"},
        {"severity": "info", "type": "underrepresented_scenes",
         "scenes": ["background_noise"],
         "message": "Found 1 underrepresented scene(s)"},
    ]
    return DatasetQualityReport(
        stats=stats,
        text_stats=text_stats,
        anomalies=anomalies,
        scene_distribution=scene_dist,
        issues_by_severity=issues,
    )


@pytest.fixture
def evaluation_report():
    """Return a sample EvaluationReport."""
    return EvaluationReport(
        overall_cer=0.09,
        overall_wer=0.14,
        scene_metrics=[
            SceneMetrics(scene_type="clean", cer=0.05, wer=0.08, sample_count=500),
            SceneMetrics(scene_type="background_voice", cer=0.15, wer=0.22, sample_count=300),
            SceneMetrics(scene_type="background_noise", cer=0.10, wer=0.15, sample_count=200),
        ],
        badcases=[],
        error_type_distribution={"substitution": 45, "insertion": 20, "deletion": 35},
    )


# =========================================================================
# Tests for prepare_training_analysis_data
# =========================================================================


class TestPrepareTrainingAnalysisData:
    """Tests for ``prepare_training_analysis_data``."""

    def test_basic_structure(self, engine, metrics_history, stagnation_alert):
        """Output contains all expected top-level keys."""
        result = engine.prepare_training_analysis_data(
            metrics_history, stagnation_alert
        )
        assert "analysis_type" in result
        assert result["analysis_type"] == "training"
        assert "timestamp" in result
        assert "summary" in result
        assert "epoch_trends" in result
        assert "alert" in result
        assert "recommendations_context" in result

    def test_summary_fields(self, engine, metrics_history):
        """Summary contains latest metric values."""
        result = engine.prepare_training_analysis_data(metrics_history)
        summary = result["summary"]
        assert summary["latest_train_loss"] == 1.2
        assert summary["latest_learning_rate"] == 5e-5
        assert summary["latest_gradient_norm"] == 0.7
        assert summary["latest_cer"] == 0.25
        assert summary["latest_wer"] == 0.35
        assert summary["latest_epoch"] == 1
        assert summary["latest_step"] == 200
        assert summary["total_metrics_collected"] == 3

    def test_epoch_trends(self, engine, metrics_history):
        """Epoch trends are computed correctly for two epochs."""
        result = engine.prepare_training_analysis_data(metrics_history)
        trends = result["epoch_trends"]
        assert len(trends) == 2
        # Epoch 0 has 2 metrics
        assert trends[0]["epoch"] == 0
        assert trends[0]["sample_count"] == 2
        assert trends[0]["avg_train_loss"] == pytest.approx((2.5 + 1.8) / 2)
        # Epoch 1 has 1 metric
        assert trends[1]["epoch"] == 1
        assert trends[1]["sample_count"] == 1
        assert trends[1]["avg_train_loss"] == pytest.approx(1.2)

    def test_alert_present(self, engine, metrics_history, stagnation_alert):
        """Alert data is included when an alert is provided."""
        result = engine.prepare_training_analysis_data(
            metrics_history, stagnation_alert
        )
        alert = result["alert"]
        assert alert is not None
        assert alert["alert_type"] == "stagnation"
        assert alert["message"] == "Training stagnation detected"
        assert alert["epoch"] == 10
        assert alert["step"] == 1000

    def test_alert_none(self, engine, metrics_history):
        """Alert is None when no alert is provided."""
        result = engine.prepare_training_analysis_data(metrics_history)
        assert result["alert"] is None

    def test_recommendations_context(self, engine, metrics_history):
        """Recommendations context has expected fields."""
        result = engine.prepare_training_analysis_data(metrics_history)
        ctx = result["recommendations_context"]
        assert ctx["has_alert"] is False
        assert ctx["alert_type"] is None
        assert ctx["total_epochs"] == 2
        assert ctx["total_steps"] == 200
        assert ctx["loss_trend"] == "decreasing"

    def test_empty_metrics(self, engine):
        """Empty metrics list produces empty summary and trends."""
        result = engine.prepare_training_analysis_data([])
        assert result["summary"] == {}
        assert result["epoch_trends"] == []
        assert result["alert"] is None
        ctx = result["recommendations_context"]
        assert ctx["total_epochs"] == 0
        assert ctx["total_steps"] == 0
        assert ctx["loss_trend"] == "unknown"

    def test_metrics_with_none_values(self, engine):
        """Metrics with None CER/WER are handled gracefully."""
        metrics = [
            TrainingMetrics(
                step=0, epoch=0, train_loss=2.0, val_loss=None,
                learning_rate=1e-4, gradient_norm=1.0,
                cer=None, wer=None, timestamp="",
            ),
        ]
        result = engine.prepare_training_analysis_data(metrics)
        summary = result["summary"]
        assert summary["latest_cer"] is None
        assert summary["latest_wer"] is None
        assert summary["latest_val_loss"] is None
        trends = result["epoch_trends"]
        assert len(trends) == 1
        assert trends[0]["avg_cer"] is None
        assert trends[0]["avg_wer"] is None
        assert trends[0]["avg_val_loss"] is None


# =========================================================================
# Tests for prepare_badcase_analysis_data
# =========================================================================


class TestPrepareBadcaseAnalysisData:
    """Tests for ``prepare_badcase_analysis_data``."""

    def test_basic_structure(self, engine, badcases, error_distribution):
        """Output contains all expected top-level keys."""
        result = engine.prepare_badcase_analysis_data(badcases, error_distribution)
        assert result["analysis_type"] == "badcase"
        assert "timestamp" in result
        assert "summary" in result
        assert "error_type_distribution" in result
        assert "top_badcases_by_cer" in result
        assert "pattern_analysis" in result

    def test_summary_counts(self, engine, badcases, error_distribution):
        """Summary has correct total and per-scene counts."""
        result = engine.prepare_badcase_analysis_data(badcases, error_distribution)
        summary = result["summary"]
        assert summary["total_badcases"] == 3
        assert summary["by_scene_type"]["clean"] == 1
        assert summary["by_scene_type"]["background_voice"] == 1
        assert summary["by_scene_type"]["background_noise"] == 1

    def test_error_distribution_passthrough(self, engine, badcases, error_distribution):
        """Error distribution is passed through unchanged."""
        result = engine.prepare_badcase_analysis_data(badcases, error_distribution)
        assert result["error_type_distribution"] == error_distribution

    def test_top_badcases_sorted(self, engine, badcases, error_distribution):
        """Top badcases are sorted by CER descending."""
        result = engine.prepare_badcase_analysis_data(badcases, error_distribution)
        top = result["top_badcases_by_cer"]
        assert len(top) == 3
        # Highest CER first (0.2)
        assert top[0]["cer"] == 0.2

    def test_pattern_analysis(self, engine, badcases, error_distribution):
        """Pattern analysis contains per-scene CER and error types."""
        result = engine.prepare_badcase_analysis_data(badcases, error_distribution)
        pa = result["pattern_analysis"]
        assert "avg_cer_by_scene" in pa
        assert "error_types_by_scene" in pa
        assert "clean" in pa["avg_cer_by_scene"]
        assert pa["avg_cer_by_scene"]["clean"] == pytest.approx(0.167)

    def test_empty_badcases(self, engine):
        """Empty badcase list produces zero counts."""
        result = engine.prepare_badcase_analysis_data([], {})
        assert result["summary"]["total_badcases"] == 0
        assert result["summary"]["by_scene_type"] == {}
        assert result["top_badcases_by_cer"] == []
        assert result["pattern_analysis"]["avg_cer_by_scene"] == {}

    def test_badcase_without_scene_type(self, engine):
        """Badcases with empty scene_type are grouped as 'unknown'."""
        bcs = [
            BadCase(
                audio_path="/data/x.wav",
                reference_text="abc",
                hypothesis_text="ab",
                cer=0.33, wer=0.5,
                error_details=[],
                scene_type="",
            ),
        ]
        result = engine.prepare_badcase_analysis_data(bcs, {})
        assert result["summary"]["by_scene_type"]["unknown"] == 1


# =========================================================================
# Tests for prepare_data_quality_analysis
# =========================================================================


class TestPrepareDataQualityAnalysis:
    """Tests for ``prepare_data_quality_analysis``."""

    def test_basic_structure(self, engine, quality_report):
        """Output contains all expected top-level keys."""
        result = engine.prepare_data_quality_analysis(quality_report)
        assert result["analysis_type"] == "data_quality"
        assert "timestamp" in result
        assert "basic_stats" in result
        assert "snr_analysis" in result
        assert "text_stats" in result
        assert "anomaly_summary" in result
        assert "scene_distribution" in result
        assert "issues_by_severity" in result

    def test_basic_stats(self, engine, quality_report):
        """Basic stats are serialized correctly."""
        result = engine.prepare_data_quality_analysis(quality_report)
        stats = result["basic_stats"]
        assert stats["total_samples"] == 1000
        assert stats["total_duration_hours"] == 50.0
        assert stats["avg_duration_seconds"] == 5.0

    def test_snr_analysis(self, engine, quality_report):
        """SNR analysis includes distribution and low quality count."""
        result = engine.prepare_data_quality_analysis(quality_report)
        snr = result["snr_analysis"]
        assert snr["low_quality_count"] == 2
        assert len(snr["low_quality_samples"]) == 2

    def test_anomaly_summary(self, engine, quality_report):
        """Anomaly summary has correct total and type breakdown."""
        result = engine.prepare_data_quality_analysis(quality_report)
        anomaly = result["anomaly_summary"]
        assert anomaly["total_anomalies"] == 2
        assert anomaly["by_type"]["empty_label"] == 1
        assert anomaly["by_type"]["corrupted_audio"] == 1

    def test_scene_distribution(self, engine, quality_report):
        """Scene distribution is serialized correctly."""
        result = engine.prepare_data_quality_analysis(quality_report)
        sd = result["scene_distribution"]
        assert sd is not None
        assert sd["scene_counts"]["clean"] == 600
        assert "background_noise" in sd["underrepresented_scenes"]

    def test_issues_by_severity(self, engine, quality_report):
        """Issues are included in the output."""
        result = engine.prepare_data_quality_analysis(quality_report)
        issues = result["issues_by_severity"]
        assert len(issues) == 3
        assert issues[0]["severity"] == "critical"

    def test_empty_report(self, engine):
        """Default empty report produces valid output."""
        report = DatasetQualityReport()
        result = engine.prepare_data_quality_analysis(report)
        assert result["basic_stats"]["total_samples"] == 0
        assert result["anomaly_summary"]["total_anomalies"] == 0
        assert result["scene_distribution"] is None
        assert result["issues_by_severity"] == []

    def test_report_without_scene_distribution(self, engine):
        """Report with scene_distribution=None produces None in output."""
        report = DatasetQualityReport(
            stats=DatasetStats(
                total_samples=10,
                total_duration_hours=1.0,
                avg_duration_seconds=6.0,
            ),
            scene_distribution=None,
        )
        result = engine.prepare_data_quality_analysis(report)
        assert result["scene_distribution"] is None


# =========================================================================
# Tests for prepare_comprehensive_analysis
# =========================================================================


class TestPrepareComprehensiveAnalysis:
    """Tests for ``prepare_comprehensive_analysis``."""

    def test_basic_structure(
        self, engine, metrics_history, evaluation_report, quality_report
    ):
        """Output contains all expected top-level keys."""
        training_data = engine.prepare_training_analysis_data(metrics_history)
        result = engine.prepare_comprehensive_analysis(
            training_data, evaluation_report, quality_report
        )
        assert result["analysis_type"] == "comprehensive"
        assert "timestamp" in result
        assert "training_analysis" in result
        assert "result_analysis" in result
        assert "dataset_analysis" in result
        assert "cross_analysis_insights" in result

    def test_result_analysis_section(
        self, engine, metrics_history, evaluation_report, quality_report
    ):
        """Result analysis section has correct CER/WER and scene metrics."""
        training_data = engine.prepare_training_analysis_data(metrics_history)
        result = engine.prepare_comprehensive_analysis(
            training_data, evaluation_report, quality_report
        )
        ra = result["result_analysis"]
        assert ra["overall_cer"] == 0.09
        assert ra["overall_wer"] == 0.14
        assert len(ra["scene_metrics"]) == 3
        assert ra["error_type_distribution"]["substitution"] == 45

    def test_dataset_analysis_section(
        self, engine, metrics_history, evaluation_report, quality_report
    ):
        """Dataset analysis section is a valid data quality analysis."""
        training_data = engine.prepare_training_analysis_data(metrics_history)
        result = engine.prepare_comprehensive_analysis(
            training_data, evaluation_report, quality_report
        )
        da = result["dataset_analysis"]
        assert da["analysis_type"] == "data_quality"
        assert da["basic_stats"]["total_samples"] == 1000

    def test_cross_insights_data_insufficiency(
        self, engine, metrics_history, quality_report
    ):
        """Cross insights detect high-CER scenes with few training samples."""
        # Create an evaluation report with a high-CER scene that has
        # few samples in the dataset.
        eval_report = EvaluationReport(
            overall_cer=0.15,
            overall_wer=0.20,
            scene_metrics=[
                SceneMetrics(
                    scene_type="background_noise",
                    cer=0.25,
                    wer=0.30,
                    sample_count=50,
                ),
            ],
        )
        training_data = engine.prepare_training_analysis_data(metrics_history)
        result = engine.prepare_comprehensive_analysis(
            training_data, eval_report, quality_report
        )
        insights = result["cross_analysis_insights"]
        types = [i["type"] for i in insights]
        assert "data_insufficiency" in types

    def test_cross_insights_with_alert(
        self, engine, metrics_history, stagnation_alert,
        evaluation_report, quality_report,
    ):
        """Cross insights include training alert correlation."""
        training_data = engine.prepare_training_analysis_data(
            metrics_history, stagnation_alert
        )
        result = engine.prepare_comprehensive_analysis(
            training_data, evaluation_report, quality_report
        )
        insights = result["cross_analysis_insights"]
        types = [i["type"] for i in insights]
        assert "training_alert_correlation" in types

    def test_cross_insights_critical_issues(
        self, engine, metrics_history, evaluation_report, quality_report
    ):
        """Cross insights detect critical data quality issues."""
        training_data = engine.prepare_training_analysis_data(metrics_history)
        result = engine.prepare_comprehensive_analysis(
            training_data, evaluation_report, quality_report
        )
        insights = result["cross_analysis_insights"]
        types = [i["type"] for i in insights]
        assert "data_quality_impact" in types

    def test_comprehensive_with_empty_data(self, engine):
        """Comprehensive analysis works with minimal/empty data."""
        training_data = engine.prepare_training_analysis_data([])
        eval_report = EvaluationReport(overall_cer=0.0, overall_wer=0.0)
        dataset_report = DatasetQualityReport()
        result = engine.prepare_comprehensive_analysis(
            training_data, eval_report, dataset_report
        )
        assert result["analysis_type"] == "comprehensive"
        assert result["result_analysis"]["overall_cer"] == 0.0
        assert result["dataset_analysis"]["basic_stats"]["total_samples"] == 0
        assert result["cross_analysis_insights"] == []

# =========================================================================
# Imports for new tests
# =========================================================================

import json

from scripts.core.models import OptimizationPlan, OptimizationSuggestion


# =========================================================================
# Tests for parse_optimization_plan
# =========================================================================


class TestParseOptimizationPlan:
    """Tests for ``parse_optimization_plan``."""

    def test_parse_json_from_markdown_code_block(self, engine):
        """Parse valid JSON wrapped in ```json ... ``` markers."""
        plan_json = {
            "iteration_id": "iter_001",
            "diagnosis_summary": "High CER in background voice scenes",
            "suggestions": [
                {
                    "dimension": "model",
                    "title": "Increase attention heads",
                    "description": "Add more attention heads for better context",
                    "priority": 4,
                    "confidence": 0.85,
                    "config_changes": {"attention_heads": 16},
                    "rationale": "More heads improve multi-speaker separation",
                },
            ],
        }
        kiro_output = (
            "Here is my optimization plan:\n\n"
            "```json\n"
            + json.dumps(plan_json, indent=2)
            + "\n```\n\n"
            "Let me know if you'd like to proceed."
        )
        result = engine.parse_optimization_plan(kiro_output)

        assert isinstance(result, OptimizationPlan)
        assert result.iteration_id == "iter_001"
        assert result.diagnosis_summary == "High CER in background voice scenes"
        assert len(result.suggestions) == 1
        s = result.suggestions[0]
        assert s.dimension == "model"
        assert s.title == "Increase attention heads"
        assert s.priority == 4
        assert s.confidence == 0.85
        assert s.config_changes == {"attention_heads": 16}

    def test_parse_raw_json_string(self, engine):
        """Parse a raw JSON string without markdown delimiters."""
        plan_json = {
            "iteration_id": "iter_002",
            "diagnosis_summary": "Training stagnation detected",
            "suggestions": [
                {
                    "dimension": "param",
                    "title": "Reduce learning rate",
                    "description": "Lower LR to escape plateau",
                    "priority": 5,
                    "confidence": 0.9,
                    "config_changes": {"learning_rate": 1e-5},
                    "rationale": "LR too high for fine-tuning stage",
                },
                {
                    "dimension": "data",
                    "title": "Add noise augmentation",
                    "description": "Augment with background voice",
                    "priority": 3,
                    "confidence": 0.7,
                    "config_changes": {"augmentation_types": ["background_voice"]},
                    "rationale": "Insufficient noisy training data",
                },
            ],
        }
        result = engine.parse_optimization_plan(json.dumps(plan_json))

        assert result.iteration_id == "iter_002"
        assert len(result.suggestions) == 2
        assert result.suggestions[0].dimension == "param"
        assert result.suggestions[1].dimension == "data"

    def test_parse_empty_string(self, engine):
        """Empty input returns an empty plan with diagnostic message."""
        result = engine.parse_optimization_plan("")
        assert isinstance(result, OptimizationPlan)
        assert result.iteration_id == ""
        assert result.suggestions == []
        assert "Failed to extract" in result.diagnosis_summary

    def test_parse_none_like_whitespace(self, engine):
        """Whitespace-only input returns an empty plan."""
        result = engine.parse_optimization_plan("   \n\t  ")
        assert result.suggestions == []
        assert result.diagnosis_summary != ""

    def test_parse_malformed_json(self, engine):
        """Malformed JSON returns an empty plan with parse error."""
        result = engine.parse_optimization_plan('```json\n{bad json}\n```')
        assert result.suggestions == []
        assert "JSON parse error" in result.diagnosis_summary

    def test_parse_non_dict_json(self, engine):
        """JSON that is not a dict returns an empty plan."""
        result = engine.parse_optimization_plan("[1, 2, 3]")
        assert result.suggestions == []
        assert "not a dict" in result.diagnosis_summary

    def test_parse_json_missing_suggestions(self, engine):
        """JSON without suggestions key returns plan with empty list."""
        plan_json = {
            "iteration_id": "iter_003",
            "diagnosis_summary": "All good",
        }
        result = engine.parse_optimization_plan(json.dumps(plan_json))
        assert result.iteration_id == "iter_003"
        assert result.suggestions == []

    def test_parse_suggestion_with_missing_fields(self, engine):
        """Suggestions with missing optional fields get defaults."""
        plan_json = {
            "iteration_id": "iter_004",
            "diagnosis_summary": "Partial data",
            "suggestions": [
                {
                    "dimension": "post_process",
                    "title": "Add hotwords",
                },
            ],
        }
        result = engine.parse_optimization_plan(json.dumps(plan_json))
        assert len(result.suggestions) == 1
        s = result.suggestions[0]
        assert s.dimension == "post_process"
        assert s.title == "Add hotwords"
        assert s.description == ""
        assert s.priority == 1
        assert s.confidence == 0.0
        assert s.config_changes == {}
        assert s.rationale == ""

    def test_parse_ignores_non_dict_suggestions(self, engine):
        """Non-dict items in suggestions list are skipped."""
        plan_json = {
            "iteration_id": "iter_005",
            "diagnosis_summary": "Mixed",
            "suggestions": [
                "not a dict",
                42,
                {"dimension": "model", "title": "Valid", "description": "ok",
                 "priority": 2, "confidence": 0.5},
            ],
        }
        result = engine.parse_optimization_plan(json.dumps(plan_json))
        assert len(result.suggestions) == 1
        assert result.suggestions[0].title == "Valid"

    def test_parse_created_at_is_set(self, engine):
        """Parsed plan has a non-empty created_at timestamp."""
        plan_json = {"iteration_id": "iter_006", "suggestions": []}
        result = engine.parse_optimization_plan(json.dumps(plan_json))
        assert result.created_at != ""


# =========================================================================
# Tests for generate_executable_config
# =========================================================================


class TestGenerateExecutableConfig:
    """Tests for ``generate_executable_config``."""

    def test_model_dimension(self, engine):
        """Model dimension generates model_config with expected keys."""
        suggestion = OptimizationSuggestion(
            dimension="model",
            title="Increase attention heads",
            description="More heads for multi-speaker",
            priority=4,
            confidence=0.85,
            config_changes={"attention_heads": 16, "hidden_dim": 1024},
            rationale="Better separation",
        )
        result = engine.generate_executable_config(suggestion)

        assert result["dimension"] == "model"
        assert result["config_type"] == "model_config"
        assert result["changes"]["attention_heads"] == 16
        assert result["changes"]["hidden_dim"] == 1024
        # Default keys should be present
        assert "seaco_params" in result["changes"]
        assert "speech_enhance_config" in result["changes"]
        # Metadata
        assert result["metadata"]["dimension"] == "model"
        assert result["metadata"]["title"] == "Increase attention heads"
        assert "timestamp" in result["metadata"]

    def test_param_dimension(self, engine):
        """Param dimension generates training_config with expected keys."""
        suggestion = OptimizationSuggestion(
            dimension="param",
            title="Reduce learning rate",
            description="Lower LR",
            priority=5,
            confidence=0.9,
            config_changes={"learning_rate": 1e-5, "batch_size": 16},
        )
        result = engine.generate_executable_config(suggestion)

        assert result["dimension"] == "param"
        assert result["config_type"] == "training_config"
        assert result["changes"]["learning_rate"] == 1e-5
        assert result["changes"]["batch_size"] == 16
        assert "warmup_steps" in result["changes"]
        assert "weight_decay" in result["changes"]

    def test_data_dimension(self, engine):
        """Data dimension generates data_pipeline_config with expected keys."""
        suggestion = OptimizationSuggestion(
            dimension="data",
            title="Add background voice augmentation",
            description="Augment noisy data",
            priority=3,
            confidence=0.7,
            config_changes={
                "augmentation_types": ["background_voice", "reverb"],
                "snr_threshold": 10.0,
            },
        )
        result = engine.generate_executable_config(suggestion)

        assert result["dimension"] == "data"
        assert result["config_type"] == "data_pipeline_config"
        assert result["changes"]["augmentation_types"] == ["background_voice", "reverb"]
        assert result["changes"]["snr_threshold"] == 10.0
        assert "sampling_strategy" in result["changes"]

    def test_post_process_dimension(self, engine):
        """Post-process dimension generates post_process_config with expected keys."""
        suggestion = OptimizationSuggestion(
            dimension="post_process",
            title="Add hotword corrections",
            description="Fix common errors",
            priority=2,
            confidence=0.6,
            config_changes={
                "hotword_entries": [
                    {"wrong_text": "号", "correct_text": "好"}
                ],
            },
        )
        result = engine.generate_executable_config(suggestion)

        assert result["dimension"] == "post_process"
        assert result["config_type"] == "post_process_config"
        assert len(result["changes"]["hotword_entries"]) == 1
        assert "regex_rules" in result["changes"]
        assert "lm_rerank" in result["changes"]

    def test_unknown_dimension(self, engine):
        """Unknown dimension still produces a valid config dict."""
        suggestion = OptimizationSuggestion(
            dimension="custom",
            title="Custom optimization",
            description="Something custom",
            priority=1,
            confidence=0.5,
            config_changes={"key": "value"},
        )
        result = engine.generate_executable_config(suggestion)

        assert result["dimension"] == "custom"
        assert result["config_type"] == "custom_config"
        assert result["changes"]["key"] == "value"

    def test_empty_config_changes(self, engine):
        """Suggestion with empty config_changes gets dimension defaults."""
        suggestion = OptimizationSuggestion(
            dimension="model",
            title="Default model config",
            description="No specific changes",
            priority=1,
            confidence=0.3,
            config_changes={},
        )
        result = engine.generate_executable_config(suggestion)

        assert result["changes"]["attention_heads"] is None
        assert result["changes"]["hidden_dim"] is None
        assert result["changes"]["seaco_params"] == {}
        assert result["changes"]["speech_enhance_config"] == {}

    def test_metadata_contains_priority_and_confidence(self, engine):
        """Metadata includes priority and confidence from the suggestion."""
        suggestion = OptimizationSuggestion(
            dimension="param",
            title="Test",
            description="Test",
            priority=3,
            confidence=0.75,
        )
        result = engine.generate_executable_config(suggestion)

        assert result["metadata"]["priority"] == 3
        assert result["metadata"]["confidence"] == 0.75

    def test_config_changes_not_overwritten(self, engine):
        """Existing config_changes values are preserved, not overwritten by defaults."""
        suggestion = OptimizationSuggestion(
            dimension="data",
            title="Custom augmentation",
            description="Specific config",
            priority=4,
            confidence=0.8,
            config_changes={
                "augmentation_types": ["speed_perturb"],
                "snr_threshold": 5.0,
                "sampling_strategy": {"clean": 0.3, "noisy": 0.7},
            },
        )
        result = engine.generate_executable_config(suggestion)

        assert result["changes"]["augmentation_types"] == ["speed_perturb"]
        assert result["changes"]["snr_threshold"] == 5.0
        assert result["changes"]["sampling_strategy"] == {"clean": 0.3, "noisy": 0.7}
