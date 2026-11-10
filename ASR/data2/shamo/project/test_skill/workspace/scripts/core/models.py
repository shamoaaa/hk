"""Core data models for the ASR Training Optimizer system.

Defines all dataclasses and enums used across the system modules:
Training_Monitor, Result_Analyzer, Dataset_Analyzer, LLM_Skill_Engine,
Model_Optimizer, Param_Tuner, Data_Pipeline, Post_Processor,
Optimizer_System, and Skill_Evolution_Engine.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# =============================================================================
# Training_Monitor Models
# =============================================================================

class AlertType(Enum):
    """Types of training alerts."""
    STAGNATION = "stagnation"
    NAN_INF = "nan_inf"
    GRADIENT_EXPLOSION = "gradient_explosion"


@dataclass
class TrainingMetrics:
    """Metrics collected at each training step."""
    step: int
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    gradient_norm: float
    cer: Optional[float] = None
    wer: Optional[float] = None
    timestamp: str = ""


@dataclass
class TrainingAlert:
    """Alert generated when training anomalies are detected."""
    alert_type: AlertType
    message: str
    context: dict = field(default_factory=dict)
    epoch: int = 0
    step: int = 0


# =============================================================================
# Result_Analyzer Models
# =============================================================================

@dataclass
class SceneMetrics:
    """Evaluation metrics for a specific audio scene type."""
    scene_type: str  # "clean" | "background_voice" | "background_noise"
    cer: float
    wer: float
    sample_count: int


class ErrorType(Enum):
    """Types of ASR recognition errors."""
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"


@dataclass
class BadCase:
    """A sample with recognition errors exceeding the threshold."""
    audio_path: str
    reference_text: str
    hypothesis_text: str
    cer: float
    wer: float
    error_details: list[dict] = field(default_factory=list)
    scene_type: str = ""


@dataclass
class EvaluationReport:
    """Complete evaluation report for a model."""
    overall_cer: float
    overall_wer: float
    scene_metrics: list[SceneMetrics] = field(default_factory=list)
    badcases: list[BadCase] = field(default_factory=list)
    error_type_distribution: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Dataset_Analyzer Models
# =============================================================================

@dataclass
class DatasetStats:
    """Basic statistics for a dataset."""
    total_samples: int
    total_duration_hours: float
    avg_duration_seconds: float
    duration_distribution: dict = field(default_factory=dict)
    snr_distribution: dict = field(default_factory=dict)
    low_quality_samples: list[str] = field(default_factory=list)


@dataclass
class TextStats:
    """Text annotation statistics."""
    char_distribution: dict[str, int] = field(default_factory=dict)
    word_frequency: dict[str, int] = field(default_factory=dict)
    sentence_length_distribution: dict = field(default_factory=dict)


@dataclass
class AnomalySample:
    """A sample flagged as anomalous."""
    audio_path: str
    anomaly_type: str  # "empty_label" | "duration_mismatch" | "corrupted_audio"
    details: str


@dataclass
class SceneDistribution:
    """Distribution of samples across scene types."""
    scene_counts: dict[str, int] = field(default_factory=dict)
    scene_ratios: dict[str, float] = field(default_factory=dict)
    underrepresented_scenes: list[str] = field(default_factory=list)


@dataclass
class DatasetQualityReport:
    """Complete dataset quality report."""
    stats: DatasetStats = field(default_factory=lambda: DatasetStats(
        total_samples=0, total_duration_hours=0.0, avg_duration_seconds=0.0
    ))
    text_stats: TextStats = field(default_factory=TextStats)
    anomalies: list[AnomalySample] = field(default_factory=list)
    scene_distribution: Optional[SceneDistribution] = None
    issues_by_severity: list[dict] = field(default_factory=list)


# =============================================================================
# LLM_Skill_Engine Models
# =============================================================================

@dataclass
class OptimizationSuggestion:
    """A single optimization suggestion from the LLM Skill Engine."""
    dimension: str  # "model" | "param" | "data" | "post_process"
    title: str
    description: str
    priority: int  # 1-5, 5 highest
    confidence: float  # 0.0-1.0
    config_changes: dict = field(default_factory=dict)
    rationale: str = ""


@dataclass
class OptimizationPlan:
    """A complete optimization plan for one iteration."""
    iteration_id: str
    suggestions: list[OptimizationSuggestion] = field(default_factory=list)
    diagnosis_summary: str = ""
    created_at: str = ""


@dataclass
class OptimizationLog:
    """Log entry for an executed optimization suggestion."""
    iteration_id: str
    suggestion: OptimizationSuggestion = field(default_factory=lambda: OptimizationSuggestion(
        dimension="", title="", description="", priority=0, confidence=0.0
    ))
    executed: bool = False
    result_cer_before: float = 0.0
    result_cer_after: Optional[float] = None
    result_wer_before: float = 0.0
    result_wer_after: Optional[float] = None


# =============================================================================
# Model_Optimizer Models
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the ASR model structure."""
    attention_heads: int = 0
    hidden_dim: int = 0
    seaco_params: dict = field(default_factory=dict)
    speech_enhance_config: Optional[dict] = None
    extra_params: dict = field(default_factory=dict)


# =============================================================================
# Param_Tuner Models
# =============================================================================

class LRScheduleType(Enum):
    """Learning rate schedule types."""
    WARMUP_DECAY = "warmup_decay"
    COSINE_ANNEALING = "cosine_annealing"
    STEP_DECAY = "step_decay"


@dataclass
class TrainingConfig:
    """Training hyperparameter configuration."""
    learning_rate: float = 0.0
    batch_size: int = 0
    warmup_steps: int = 0
    weight_decay: float = 0.0
    lr_schedule: LRScheduleType = LRScheduleType.WARMUP_DECAY
    max_epochs: int = 0
    gradient_clip: float = 0.0
    extra_params: dict = field(default_factory=dict)


@dataclass
class ParamHistory:
    """History entry for a training parameter configuration."""
    config: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation_result: Optional[EvaluationReport] = None
    timestamp: str = ""
    iteration_id: str = ""


# =============================================================================
# Data_Pipeline Models
# =============================================================================

class AugmentationType(Enum):
    """Types of data augmentation."""
    BACKGROUND_VOICE = "background_voice"
    SPEED_PERTURB = "speed_perturb"
    VOLUME_PERTURB = "volume_perturb"
    REVERB = "reverb"


@dataclass
class AugmentationConfig:
    """Configuration for a data augmentation operation."""
    aug_type: AugmentationType = AugmentationType.BACKGROUND_VOICE
    params: dict = field(default_factory=dict)
    target_scene: Optional[str] = None


@dataclass
class AugmentationReport:
    """Report generated after data augmentation."""
    original_count: int = 0
    augmented_count: int = 0
    distribution_before: dict = field(default_factory=dict)
    distribution_after: dict = field(default_factory=dict)
    augmentation_details: list[dict] = field(default_factory=list)


@dataclass
class SamplingStrategy:
    """Dynamic sampling strategy based on scene performance."""
    scene_weights: dict[str, float] = field(default_factory=dict)


# =============================================================================
# Post_Processor Models
# =============================================================================

@dataclass
class HotwordEntry:
    """A hotword correction entry."""
    wrong_text: str = ""
    correct_text: str = ""
    priority: int = 0


@dataclass
class RegexRule:
    """A regex-based text replacement rule."""
    pattern: str = ""
    replacement: str = ""
    description: str = ""
    priority: int = 0


@dataclass
class PostProcessConfig:
    """Configuration for the post-processing pipeline."""
    hotword_entries: list[HotwordEntry] = field(default_factory=list)
    regex_rules: list[RegexRule] = field(default_factory=list)
    enable_lm_rerank: bool = False
    lm_model_path: Optional[str] = None
    pipeline_order: list[str] = field(
        default_factory=lambda: ["hotword", "regex", "lm_rerank"]
    )


@dataclass
class PostProcessResult:
    """Result of applying post-processing to a text."""
    original_text: str = ""
    processed_text: str = ""
    applied_rules: list[str] = field(default_factory=list)
    cer_before: float = 0.0
    cer_after: float = 0.0


# =============================================================================
# Optimizer_System Models
# =============================================================================

@dataclass
class IterationRecord:
    """Record of a single optimization iteration."""
    iteration_id: str = ""
    optimization_content: str = ""
    param_changes: dict = field(default_factory=dict)
    data_changes: dict = field(default_factory=dict)
    model_changes: dict = field(default_factory=dict)
    evaluation_before: EvaluationReport = field(
        default_factory=lambda: EvaluationReport(overall_cer=0.0, overall_wer=0.0)
    )
    evaluation_after: Optional[EvaluationReport] = None
    cer_delta: Optional[float] = None
    wer_delta: Optional[float] = None
    timestamp: str = ""
    status: str = "pending"  # "pending" | "running" | "completed" | "failed"


# =============================================================================
# Skill_Evolution_Engine Models
# =============================================================================

@dataclass
class EffectFeedback:
    """Feedback record for an optimization suggestion's effect."""
    iteration_id: str = ""
    suggestion_id: str = ""
    suggestion_content: str = ""
    input_analysis: dict = field(default_factory=dict)
    param_changes: dict = field(default_factory=dict)
    cer_before: float = 0.0
    cer_after: float = 0.0
    wer_before: float = 0.0
    wer_after: float = 0.0
    is_positive: bool = False
    reflection_text: str = ""
    timestamp: str = ""


@dataclass
class KnowledgeEntry:
    """A knowledge entry in the Skill Knowledge Base."""
    knowledge_type: str = ""  # "scene_solution" | "param_effect" | "data_strategy"
    content: dict = field(default_factory=dict)
    confidence: float = 0.0  # 0.0-1.0
    source_iteration_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    state: str = "active"  # "active" | "stale" | "archived"
    verified_count: int = 0
    last_verified_at: str = ""
    auto_confidence: Optional[float] = None


@dataclass
class StrategyWeight:
    """Weight assigned to an optimization strategy."""
    strategy_name: str = ""
    weight: float = 0.0
    success_rate: float = 0.0
    sample_count: int = 0


@dataclass
class SkillVersion:
    """A versioned snapshot of the Skill Engine's state."""
    version_id: str = ""
    strategy_rules: dict = field(default_factory=dict)
    knowledge_base_snapshot: dict = field(default_factory=dict)
    change_description: str = ""
    positive_feedback_rate: float = 0.0
    avg_cer_improvement: float = 0.0
    created_at: str = ""
