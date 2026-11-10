"""STATE.md 读写 — 环路状态脊柱管理.

- 读取当前状态
- 更新 Last run 时间戳
- 添加/移除高优先级项
- 移动完成项到"已完成"
- 检测 LOOPS PAUSED 标记
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import re


class StateManager:
    """管理 STATE.md 的读写。"""

    def __init__(self, state_path: str = "STATE.md"):
        self.path = Path(state_path)

    def is_paused(self) -> bool:
        """检查是否有 LOOPS PAUSED 标记。"""
        if not self.path.exists():
            return False
        content = self.path.read_text(encoding="utf-8")
        return "⚠️ LOOPS PAUSED" in content

    def read_state(self) -> dict:
        """解析 STATE.md 为结构化数据。"""
        if not self.path.exists():
            return {"last_run": None, "high_priority": [], "watch_list": [], "completed": []}

        content = self.path.read_text(encoding="utf-8")
        state = {
            "last_run": None,
            "high_priority": [],
            "watch_list": [],
            "completed": [],
            "raw": content,
        }

        # 解析 Last run
        m = re.search(r"Last run:\s*(.+)", content)
        if m:
            state["last_run"] = m.group(1).strip()

        # 简单解析各段落的列表项
        sections = re.split(r"^## ", content, flags=re.MULTILINE)
        for section in sections:
            items = re.findall(r"^- (.+)$", section, re.MULTILINE)
            items = [i for i in items if i.strip() != "（暂无）"]
            if "高优先级" in section:
                state["high_priority"] = items
            elif "观察列表" in section:
                state["watch_list"] = items
            elif "已完成" in section:
                state["completed"] = items

        return state

    def update_last_run(self) -> None:
        """更新 Last run 时间戳。"""
        if not self.path.exists():
            return
        content = self.path.read_text(encoding="utf-8")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        content = re.sub(
            r"Last run:.*",
            f"Last run: {now}",
            content,
        )
        self.path.write_text(content, encoding="utf-8")
