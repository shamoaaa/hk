"""asr_evolve.pipeline — 业务 ASR 训练流水线子包.

导出核心组件供外部使用。
"""

from asr_evolve.pipeline.orchestrator import PipelineOrchestrator, CANONICAL_STAGES
from asr_evolve.pipeline.models import (
    StageContext,
    ArtifactStatus,
    StageResult,
    PipelineRunReport,
    QualityThresholds,
    GoodBadLists,
    KaldiEntry,
    Sample,
    RecallReport,
    DomainReport,
    ComparisonReport,
)
from asr_evolve.pipeline.stage import Stage

__all__ = [
    "PipelineOrchestrator",
    "CANONICAL_STAGES",
    "Stage",
    "StageContext",
    "ArtifactStatus",
    "StageResult",
    "PipelineRunReport",
    "QualityThresholds",
    "GoodBadLists",
    "KaldiEntry",
    "Sample",
    "RecallReport",
    "DomainReport",
    "ComparisonReport",
]
