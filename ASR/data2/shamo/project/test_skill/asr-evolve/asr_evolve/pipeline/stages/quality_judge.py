"""QualityJudge — 质量判定（与 Scanner 解耦）(R5.2-5.4, R19.1).

基于既有 metrics 与可配置阈值判定 good/bad，不重跑扫描。
bad 条目记录违反的具体问题类型。
阈值由 Agent 决定：编排器将 metrics 分布作为决策点交给 Kiro。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from asr_evolve.pipeline.models import GoodBadLists, QualityThresholds


class QualityJudge:
    """质量判定器 — 纯依赖既有指标，不重跑扫描 (R5.4)。"""

    def __init__(self, thresholds: dict[str, Any] | None = None):
        thresholds = thresholds or {}
        self.min_dur = thresholds.get("min_duration_s", 0.5)
        self.max_dur = thresholds.get("max_duration_s", 20.0)
        self.expected_sr = thresholds.get("expected_sample_rate", 16000)
        self.expected_ch = thresholds.get("expected_channels", 1)
        self.min_rms = thresholds.get("min_rms", 0.001)
        self.max_clip_rate = thresholds.get("max_clip_rate", 0.01)

    def judge(self, metrics: dict) -> tuple[bool, list[str]]:
        """判定单条音频质量 (R5.2, R5.3).

        Returns:
            (is_good, problems) — problems 为违反的具体问题类型列表
        """
        problems: list[str] = []

        # 有错误直接判 bad
        if "error" in metrics:
            problems.append(f"read_error:{metrics['error']}")
            return False, problems

        # 时长检查
        dur = metrics.get("duration_s", 0)
        if dur < self.min_dur:
            problems.append("too_short")
        if dur > self.max_dur:
            problems.append("too_long")

        # 采样率检查
        sr = metrics.get("sample_rate", 0)
        if sr != self.expected_sr:
            problems.append("wrong_sample_rate")

        # 声道检查
        ch = metrics.get("channels", 0)
        if ch != self.expected_ch:
            problems.append("wrong_channels")

        # RMS 检查（静音检测）
        rms = metrics.get("rms", 0)
        if rms < self.min_rms:
            problems.append("too_quiet")

        # 削波率检查
        clip_rate = metrics.get("clip_rate", 0)
        if clip_rate > self.max_clip_rate:
            problems.append("clipping")

        is_good = len(problems) == 0
        return is_good, problems

    def judge_from_metrics_file(
        self, metrics_path: str | Path
    ) -> GoodBadLists:
        """从 metrics.jsonl 批量判定 (R5.4 — 调阈值不重扫)。"""
        metrics_path = Path(metrics_path)
        result = GoodBadLists()

        if not metrics_path.exists():
            return result

        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            metrics = json.loads(line)
            utt_id = metrics.get("utt_id", "")
            is_good, problems = self.judge(metrics)
            if is_good:
                result.good.append(utt_id)
            else:
                result.bad[utt_id] = problems

        return result
