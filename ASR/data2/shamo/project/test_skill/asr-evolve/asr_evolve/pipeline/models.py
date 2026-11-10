"""流水线运行时数据模型.

包含所有数据类定义，阶段间传递上下文与结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageContext:
    """传给每个 Stage 的执行上下文。"""
    base_dir: str
    stages_root: str
    mode: str
    datasets: list[str]
    config: dict
    force: bool = False
    dry_run: bool = False


@dataclass
class ArtifactStatus:
    """阶段产物状态。"""
    exists: bool
    entry_count: int = 0
    detail: str = ""


@dataclass
class StageResult:
    """阶段执行结果。"""
    stage: str
    status: str  # "completed" | "skipped" | "failed" | "planned"
    reason: str = ""
    entry_count: int = 0
    artifacts: list[str] = field(default_factory=list)


@dataclass
class PipelineRunReport:
    """一次流水线运行的完整报告。"""
    mode: str
    planned: list[str]
    results: list[StageResult] = field(default_factory=list)
    halted_at: str | None = None
    decision_points: list[str] = field(default_factory=list)


@dataclass
class QualityThresholds:
    """质量判定阈值配置。"""
    min_duration_s: float = 0.5
    max_duration_s: float = 20.0
    expected_sample_rate: int = 16000
    expected_channels: int = 1
    min_rms: float = 0.001
    max_clip_rate: float = 0.01


@dataclass
class GoodBadLists:
    """质量判定输出：good/bad 划分。"""
    good: list[str] = field(default_factory=list)  # utt_id 列表
    bad: dict[str, list[str]] = field(default_factory=dict)  # utt_id -> 违反的问题类型列表


@dataclass
class KaldiEntry:
    """Kaldi 格式条目。"""
    utt_id: str
    audio_path: str
    text: str = ""


@dataclass
class Sample:
    """评估样本。"""
    audio_path: str
    reference_text: str
    hypothesis_text: str
    scene_type: str = ""


@dataclass
class RecallReport:
    """指令召回率报告。"""
    test_set: str = ""
    recall: float = 0.0
    numerator: int = 0    # 命中数（仅 strip(ref) 非空）
    denominator: int = 0  # strip(ref) 非空样本数


@dataclass
class DomainReport:
    """单测试集领域 KPI 报告——独立，不聚合。"""
    test_set: str
    cer: float = 0.0           # 剥离前 CER
    stripped_cer: float = 0.0  # 剥离后 Stripped_CER
    recall: RecallReport = field(default_factory=RecallReport)
    missing: bool = False      # 明细缺失标记


@dataclass
class ComparisonReport:
    """基线对比报告。"""
    version: str = ""                            # v0.X.0
    rows: list[dict] = field(default_factory=list)  # 每模型: recall + stripped_cer
