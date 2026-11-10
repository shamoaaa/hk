"""DEAD_ENDS.md 硬拦截 — 防止重复尝试已验证无效的方向.

来自 Loop Engineering 的安全设计：
- DEAD_ENDS 不是"建议避免"，而是硬性禁止
- Agent 在 EXECUTE 前必须检查冲突
"""
from __future__ import annotations
from pathlib import Path
import re


class DeadEndGuard:
    """死胡同守卫 — 硬拦截已验证无效的方向。"""

    def __init__(self, dead_ends_path: str = "DEAD_ENDS.md"):
        self.path = Path(dead_ends_path)

    def get_entries(self) -> list[dict]:
        """解析 DEAD_ENDS.md 中的条目。"""
        if not self.path.exists():
            return []

        content = self.path.read_text(encoding="utf-8")
        entries = []
        # 按 ### 分割条目
        sections = re.split(r"^### ", content, flags=re.MULTILINE)
        for section in sections[1:]:  # 跳过开头
            lines = section.strip().split("\n")
            title = lines[0] if lines else ""
            body = "\n".join(lines[1:])
            entries.append({"title": title, "body": body})
        return entries

    def check_conflict(self, proposed_action: str) -> tuple[bool, str]:
        """检查提议的操作是否与死胡同冲突.

        Returns: (is_blocked, reason)
        """
        entries = self.get_entries()
        action_lower = proposed_action.lower()

        for entry in entries:
            # 简单的关键词匹配（可扩展为语义匹配）
            title_lower = entry["title"].lower()
            body_lower = entry["body"].lower()

            # 提取配置关键词
            keywords = re.findall(r"[\w_]+(?:=[\w.]+)?", title_lower + " " + body_lower)
            for kw in keywords:
                if len(kw) > 3 and kw in action_lower:
                    return True, f"与死胡同冲突: '{entry['title']}' (匹配: {kw})"

        return False, ""

    def add_entry(self, title: str, config: str, result: str, reason: str, conclusion: str) -> None:
        """添加新的死胡同条目。"""
        from datetime import date
        entry = f"\n### [{date.today()}] {title}\n"
        entry += f"- 配置: {config}\n"
        entry += f"- 结果: {result}\n"
        entry += f"- 原因分析: {reason}\n"
        entry += f"- 结论: {conclusion}\n"

        content = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
        # 在"## 记录"之后追加
        if "## 记录" in content:
            content = content.replace("## 记录\n", f"## 记录\n{entry}", 1)
        else:
            content += entry
        self.path.write_text(content, encoding="utf-8")
