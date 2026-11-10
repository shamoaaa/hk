"""loop-run-log.md 追加 — 运行日志管理."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json


class RunLogger:
    """追加式运行日志。"""

    def __init__(self, log_path: str = "loop-run-log.md"):
        self.path = Path(log_path)

    def append(
        self,
        loop: str,
        action: str,
        outcome: str,
        cer_before: float | None = None,
        cer_after: float | None = None,
        gpu_hours: float = 0,
        node_id: str | None = None,
        details: str = "",
    ) -> None:
        """追加一条运行记录。"""
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "loop": loop,
            "action": action,
            "outcome": outcome,
            "cer_before": cer_before,
            "cer_after": cer_after,
            "gpu_hours": gpu_hours,
            "node_id": node_id,
            "details": details,
        }
        line = json.dumps(entry, ensure_ascii=False)

        content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        content += f"\n{line}"
        self.path.write_text(content, encoding="utf-8")
