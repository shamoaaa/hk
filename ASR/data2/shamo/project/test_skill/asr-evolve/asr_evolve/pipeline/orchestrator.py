"""PipelineOrchestrator — 流水线编排层.

按规范阶段顺序执行，支持：
- 子集执行（--only）
- 幂等跳过（artifact 存在则跳过）
- 强制重跑（--force）
- dry-run 预览
- 失败即终止（非破坏性阶段）
- 决策点持久化（R19.5）
- Loop-engineering 融合（STATE.md 暂停检测、DEAD_ENDS 冲突检查、预算）
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asr_evolve.pipeline.models import (
    ArtifactStatus,
    PipelineRunReport,
    StageContext,
    StageResult,
)
from asr_evolve.pipeline.stage import Stage

# 规范阶段顺序（不可变）— 8 阶段流水线
CANONICAL_STAGES: list[str] = [
    "prepare", "quality", "clean", "split",
    "augment", "merge", "jsonl", "train",
]


class PipelineOrchestrator:
    """编排层顶层组件。

    R1-R2: 阶段化编排与幂等执行
    R19.5: 决策点持久化
    R20.3: 模式切换不改变编排逻辑
    """

    def __init__(self, config: dict, base_dir: str = "."):
        self.config = config
        self.base_dir = base_dir
        self._stages: dict[str, Stage] = {}

    def register_stage(self, stage: Stage) -> None:
        """注册一个阶段实现。"""
        self._stages[stage.name] = stage

    def run(
        self,
        only: list[str] | None = None,
        *,
        dry_run: bool = False,
        force: set[str] | None = None,
        mode: str = "aishell",
    ) -> PipelineRunReport:
        """按规范顺序执行 only 指定的阶段子集。

        - only: None 表示全部，否则按 CANONICAL_STAGES 相对顺序过滤
        - dry_run: 输出每阶段动作，不写盘 (R1.4)
        - force: 需强制重跑的阶段名集合 (R2.2)
        - mode: aishell | business (R20.1-20.3)
        """
        force = force or set()
        planned = self._plan(only)

        # 读取数据集配置
        pipeline_cfg = self.config.get("pipeline", {})
        datasets_map = pipeline_cfg.get("datasets", {})
        datasets = datasets_map.get(mode, [])
        stages_root = pipeline_cfg.get("stages_root", "data/pipeline")

        report = PipelineRunReport(mode=mode, planned=planned)

        # Loop-engineering: 检查 STATE.md 暂停标记
        if not dry_run:
            if self._is_paused():
                report.halted_at = planned[0] if planned else None
                report.results.append(StageResult(
                    stage=planned[0] if planned else "unknown",
                    status="failed",
                    reason="LOOPS PAUSED — STATE.md 含暂停标记",
                ))
                return report

        for stage_name in planned:
            stage = self._stages.get(stage_name)
            if stage is None:
                report.results.append(StageResult(
                    stage=stage_name, status="failed",
                    reason=f"Stage '{stage_name}' not registered",
                ))
                report.halted_at = stage_name
                break

            ctx = StageContext(
                base_dir=self.base_dir,
                stages_root=stages_root,
                mode=mode,
                datasets=datasets,
                config=self.config,
                force=(stage_name in force),
                dry_run=dry_run,
            )

            # dry-run: 只输出计划，不写盘 (R1.4)
            if dry_run:
                actions = stage.plan(ctx)
                report.results.append(StageResult(
                    stage=stage_name, status="planned",
                    reason="; ".join(actions),
                ))
                continue

            # 跳过检测 (R2.1)
            should_skip, skip_reason = self._should_skip(stage, ctx)
            if should_skip:
                report.results.append(StageResult(
                    stage=stage_name, status="skipped", reason=skip_reason,
                ))
                continue

            # 执行阶段
            result = stage.execute(ctx)
            report.results.append(result)

            # 非破坏性阶段失败即终止 (R1.5)
            if result.status == "failed":
                report.halted_at = stage_name
                break

        return report

    def _plan(self, only: list[str] | None) -> list[str]:
        """过滤并按 CANONICAL_STAGES 排序 (R1.2, R1.3)。"""
        if only is None:
            return list(CANONICAL_STAGES)
        only_set = set(only)
        return [s for s in CANONICAL_STAGES if s in only_set]

    def _should_skip(self, stage: Stage, ctx: StageContext) -> tuple[bool, str]:
        """artifact 存在且未 force → 跳过 (R2.1, R2.2)。"""
        if ctx.force:
            return False, ""
        status = stage.artifact_exists(ctx)
        if status.exists:
            return True, f"Artifact exists ({status.entry_count} entries): {status.detail}"
        return False, ""

    def record_decision_point(self, kind: str, payload: dict) -> str:
        """将决策点输入以结构化形式持久化 (R19.5)。

        写入 loop-run-log.md，供 Agent 通过记忆系统追溯。
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": kind,
            "payload": payload,
        }
        line = json.dumps(entry, ensure_ascii=False)

        log_path = Path(self.base_dir) / "loop-run-log.md"
        try:
            content = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
            content += f"\n{line}"
            log_path.write_text(content, encoding="utf-8")
        except OSError:
            pass  # 写入失败不阻断主流程

        return line

    # --- Loop-engineering 融合 ---

    def _is_paused(self) -> bool:
        """检查 STATE.md 的 LOOPS PAUSED 标记。"""
        state_path = Path(self.base_dir) / "STATE.md"
        if not state_path.exists():
            return False
        try:
            content = state_path.read_text(encoding="utf-8")
            return "⚠️ LOOPS PAUSED" in content
        except OSError:
            return False

    def _check_budget(self) -> bool:
        """检查每日训练预算是否耗尽。"""
        budget_cfg = self.config.get("budget", {})
        max_trains = budget_cfg.get("max_trains_per_day", 3)
        # 简化实现：从 loop-run-log.md 中统计今日训练次数
        log_path = Path(self.base_dir) / "loop-run-log.md"
        if not log_path.exists():
            return True
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        count = 0
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if today in line and '"action": "train"' in line:
                    count += 1
        except OSError:
            pass
        return count < max_trains
