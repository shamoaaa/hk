"""TrainingHealthChecker — 训练健康检查 (R14).

- rc=0 无 .pt → 静默崩溃 (R14.1-14.2)
- 失败阻止进入评估 (R14.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class HealthResult:
    """训练健康检查结果。"""
    passed: bool
    reason: str = ""
    checkpoint_count: int = 0


class TrainingHealthChecker:
    """训练健康检查器 — 捕获 FunASR/hydra 静默崩溃。"""

    def check(self, output_dir: str, return_code: int) -> HealthResult:
        """校验训练产物 (R14.1-14.2).

        Args:
            output_dir: 训练输出目录
            return_code: 训练进程返回码

        Returns:
            HealthResult — passed=False 时阻止进入评估 (R14.3)
        """
        out = Path(output_dir)
        pts = list(out.glob("*.pt")) if out.exists() else []

        # R14.1: 存在至少一个有效 .pt 检查点
        if pts:
            return HealthResult(passed=True, checkpoint_count=len(pts))

        # R14.2: 返回码 0 但无 .pt → 静默崩溃
        if return_code == 0:
            return HealthResult(
                passed=False,
                reason="Silent crash: return_code=0 but no .pt checkpoint found",
            )

        return HealthResult(
            passed=False,
            reason=f"Training failed with return_code={return_code}",
        )
