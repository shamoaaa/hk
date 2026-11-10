"""ExperimentTree — 实验搜索空间的树形管理.

核心设计（灵感来自 AIDE 的树搜索管理实验空间）：
- 每次优化尝试（训练/评估/配置变更）是树上的一个节点
- 节点记录：操作描述、配置快照、评估结果、父子关系
- 支持回溯：从任意节点重新分支
- 支持剪枝：标记失败分支为 dead_end
- 与 VERSION 绑定：每个节点绑定精确的配置/数据集/模型版本

树结构：
```
root (baseline)
├── node-1: augment babble=0.4 → CER 0.92%
│   ├── node-3: augment babble=0.6 → CER 0.88% ✓ (best)
│   │   └── node-5: lr=1e-5 → CER 0.90% ✗ (dead_end)
│   └── node-4: speed_perturb → CER 0.95% ✗ (dead_end)
└── node-2: hotword boost → CER 0.85% ✓✓ (best overall)
```

持久化：experiment_tree.json（append-friendly JSON Lines 备选）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ExperimentNode:
    """实验树节点。"""
    node_id: str                          # 唯一标识（如 "node-1"）
    parent_id: str | None = None          # 父节点 ID（root 为 None）
    action_desc: str = ""                 # 操作描述（如 "augment babble prob=0.6"）
    hypothesis: str = ""                  # 优化假设（如 "提高 babble 概率改善背景噪声场景"）
    config_snapshot: dict = field(default_factory=dict)  # 本次使用的配置快照
    dataset_version: str = ""             # 数据集版本标识
    model_checkpoint: str = ""            # 产出的模型检查点路径
    # 结果
    status: str = "pending"               # pending | running | completed | failed | dead_end
    cer: float | None = None              # 评估 CER（完成后填入）
    stripped_cer: float | None = None     # 剥离唤醒词后的 CER
    instruction_recall: float | None = None  # 指令召回率
    metrics: dict = field(default_factory=dict)  # 其他自定义指标
    # 元数据
    created_at: str = ""                  # ISO 时间戳
    completed_at: str = ""                # 完成时间
    duration_s: float = 0.0               # 耗时（秒）
    gpu_hours: float = 0.0                # GPU 小时
    tags: list[str] = field(default_factory=list)  # 标签（如 ["best", "baseline"]）
    notes: str = ""                       # 备注


class ExperimentTree:
    """实验搜索空间的树形管理。

    功能：
    - add_node: 添加实验节点（指定父节点）
    - record_result: 记录实验结果（CER 等）
    - mark_dead_end: 标记失败分支
    - get_best: 获取当前最优节点
    - get_path: 获取从 root 到指定节点的路径
    - view: 输出 ASCII 树形图
    - save / load: 持久化到 JSON
    """

    def __init__(self, tree_path: str = "experiment_tree.json"):
        self.path = Path(tree_path)
        self.nodes: dict[str, ExperimentNode] = {}
        self._next_id: int = 1
        if self.path.exists():
            self.load()

    # --- 核心操作 ---

    def add_node(
        self,
        action_desc: str,
        parent_id: str | None = None,
        hypothesis: str = "",
        config_snapshot: dict | None = None,
        dataset_version: str = "",
        tags: list[str] | None = None,
    ) -> ExperimentNode:
        """添加实验节点。"""
        node_id = f"node-{self._next_id}"
        self._next_id += 1

        node = ExperimentNode(
            node_id=node_id,
            parent_id=parent_id,
            action_desc=action_desc,
            hypothesis=hypothesis,
            config_snapshot=config_snapshot or {},
            dataset_version=dataset_version,
            status="pending",
            created_at=self._now(),
            tags=tags or [],
        )
        self.nodes[node_id] = node
        self.save()
        return node

    def record_result(
        self,
        node_id: str,
        *,
        status: str = "completed",
        cer: float | None = None,
        stripped_cer: float | None = None,
        instruction_recall: float | None = None,
        model_checkpoint: str = "",
        duration_s: float = 0.0,
        gpu_hours: float = 0.0,
        metrics: dict | None = None,
        notes: str = "",
    ) -> ExperimentNode | None:
        """记录实验结果（R19.5, R21.3）。"""
        node = self.nodes.get(node_id)
        if node is None:
            return None

        node.status = status
        node.cer = cer
        node.stripped_cer = stripped_cer
        node.instruction_recall = instruction_recall
        node.model_checkpoint = model_checkpoint
        node.duration_s = duration_s
        node.gpu_hours = gpu_hours
        node.completed_at = self._now()
        if metrics:
            node.metrics.update(metrics)
        if notes:
            node.notes = notes

        # 自动标记 best
        if status == "completed" and cer is not None:
            best = self.get_best()
            if best is None or (best.cer is not None and cer < best.cer):
                # 移除旧 best 标签
                for n in self.nodes.values():
                    if "best" in n.tags:
                        n.tags.remove("best")
                node.tags.append("best")

        self.save()
        return node

    def mark_dead_end(self, node_id: str, reason: str = "") -> ExperimentNode | None:
        """标记节点为 dead_end（及其所有后代）。"""
        node = self.nodes.get(node_id)
        if node is None:
            return None

        node.status = "dead_end"
        if reason:
            node.notes = reason

        # 递归标记后代
        for child in self._children(node_id):
            child.status = "dead_end"

        self.save()
        return node

    # --- 查询 ---

    def get_best(self) -> ExperimentNode | None:
        """获取当前最优节点（最低 CER）。"""
        completed = [
            n for n in self.nodes.values()
            if n.status == "completed" and n.cer is not None
        ]
        if not completed:
            return None
        return min(completed, key=lambda n: n.cer)  # type: ignore

    def get_path(self, node_id: str) -> list[ExperimentNode]:
        """获取从 root 到指定节点的路径。"""
        path: list[ExperimentNode] = []
        current = self.nodes.get(node_id)
        while current is not None:
            path.append(current)
            current = self.nodes.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def get_leaves(self) -> list[ExperimentNode]:
        """获取所有叶节点（无子节点的节点）。"""
        parent_ids = {n.parent_id for n in self.nodes.values() if n.parent_id}
        return [n for n in self.nodes.values() if n.node_id not in parent_ids]

    def get_active_branches(self) -> list[ExperimentNode]:
        """获取可继续探索的叶节点（非 dead_end）。"""
        return [n for n in self.get_leaves() if n.status != "dead_end"]

    def summary(self) -> dict[str, Any]:
        """实验树摘要。"""
        best = self.get_best()
        return {
            "total_nodes": len(self.nodes),
            "completed": sum(1 for n in self.nodes.values() if n.status == "completed"),
            "dead_ends": sum(1 for n in self.nodes.values() if n.status == "dead_end"),
            "pending": sum(1 for n in self.nodes.values() if n.status == "pending"),
            "best_node": best.node_id if best else None,
            "best_cer": best.cer if best else None,
            "active_branches": len(self.get_active_branches()),
        }

    # --- 可视化 ---

    def view(self) -> str:
        """输出 ASCII 树形图。"""
        lines: list[str] = []
        roots = [n for n in self.nodes.values() if n.parent_id is None]
        for root in roots:
            self._render_node(root, lines, prefix="", is_last=True)
        return "\n".join(lines) if lines else "(empty tree)"

    def _render_node(self, node: ExperimentNode, lines: list[str], prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        status_icon = {
            "completed": "✅", "failed": "❌", "dead_end": "💀",
            "pending": "⏳", "running": "🔄",
        }.get(node.status, "?")

        cer_str = f"CER={node.cer:.4f}" if node.cer is not None else ""
        best_mark = " ⭐" if "best" in node.tags else ""
        line = f"{prefix}{connector}{status_icon} {node.node_id}: {node.action_desc}"
        if cer_str:
            line += f" → {cer_str}{best_mark}"
        lines.append(line)

        children = self._children(node.node_id)
        for i, child in enumerate(children):
            child_prefix = prefix + ("    " if is_last else "│   ")
            self._render_node(child, lines, child_prefix, is_last=(i == len(children) - 1))

    def _children(self, node_id: str) -> list[ExperimentNode]:
        """获取直接子节点。"""
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    # --- 持久化 ---

    def save(self) -> None:
        """保存到 JSON。"""
        data = {
            "next_id": self._next_id,
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
        }
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        """从 JSON 加载。"""
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._next_id = data.get("next_id", 1)
        self.nodes = {}
        for nid, ndata in data.get("nodes", {}).items():
            node = ExperimentNode(**ndata)
            self.nodes[nid] = node

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
