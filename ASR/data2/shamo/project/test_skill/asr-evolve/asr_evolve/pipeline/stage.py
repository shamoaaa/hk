"""Stage 协议定义.

所有流水线阶段须实现此协议，编排器据此做统一的产物检测、执行与 dry-run。
"""

from __future__ import annotations

from typing import Protocol

from asr_evolve.pipeline.models import ArtifactStatus, StageContext, StageResult


class Stage(Protocol):
    """流水线阶段统一接口。"""

    name: str
    destructive: bool  # clean=True，其余=False

    def artifact_exists(self, ctx: StageContext) -> ArtifactStatus:
        """检测本阶段 Stage_Artifact 是否已存在 (R2.1)。"""
        ...

    def plan(self, ctx: StageContext) -> list[str]:
        """dry-run: 返回将执行动作的描述，不触碰磁盘 (R1.4)。"""
        ...

    def execute(self, ctx: StageContext) -> StageResult:
        """执行阶段，产物落盘到 ctx 约定路径 (R2.4)。"""
        ...
