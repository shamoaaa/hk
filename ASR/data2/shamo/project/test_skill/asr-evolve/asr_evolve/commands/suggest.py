# -*- coding: utf-8 -*-
"""suggest 命令 — UCB1 树搜索 + HPO 建议生成.

改进方向 #4: UCB1 树搜索 (AI Scientist v2)
改进方向 #8: 结构化 HPO 历史建议 (AgentHPO)

功能:
  asr-evolve tree suggest-next     基于 UCB1 推荐下一个实验节点
  asr-evolve suggest --budget N    基于 HPO 历史推荐调参方向
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# UCB1 Tree Search (#4)
# ---------------------------------------------------------------------------

def _load_strategy_config() -> dict:
    """加载搜索策略配置。"""
    path = Path("configs/search_strategy.yaml")
    if not path.exists():
        return {"current_phase": "coarse", "phases": {"coarse": {"ucb1_C": 2.0}}}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_tree() -> dict:
    """加载实验树。"""
    path = Path("experiment_tree.json")
    if not path.exists():
        return {"nodes": {}, "next_id": 1}
    return json.loads(path.read_text(encoding="utf-8"))


def _get_node_reward(node: dict, parent_cer: float | None) -> float:
    """计算节点的 reward (CER 改善比例)."""
    if node.get("cer") is None or parent_cer is None or parent_cer == 0:
        return 0.0
    improvement = parent_cer - node["cer"]
    return max(0.0, improvement / parent_cer)


def _get_visit_count(node_id: str, nodes: dict) -> int:
    """计算节点及其子树的总访问次数。"""
    node = nodes.get(node_id, {})
    if node.get("status") in ("pending", None):
        return 0
    count = 1
    for nid, n in nodes.items():
        if n.get("parent_id") == node_id:
            count += _get_visit_count(nid, nodes)
    return count


def _compute_ucb1(
    node_id: str,
    nodes: dict,
    total_visits: int,
    C: float = 1.4,
) -> float:
    """计算节点的 UCB1 分数.

    Score = avg_reward + C * sqrt(ln(N) / n_i)
    - 未探索节点: score = infinity
    - dead_end 节点: score = -infinity
    """
    node = nodes.get(node_id, {})

    # Dead ends get -infinity
    if node.get("status") == "dead_end":
        return float("-inf")

    visit_count = _get_visit_count(node_id, nodes)

    # Never visited = infinity (explore first)
    if visit_count == 0:
        return float("inf")

    # Compute average reward
    parent_id = node.get("parent_id")
    parent_cer = nodes.get(parent_id, {}).get("cer") if parent_id else None
    reward = _get_node_reward(node, parent_cer)

    # UCB1 formula
    if total_visits == 0:
        exploration_term = float("inf")
    else:
        exploration_term = C * math.sqrt(math.log(total_visits) / visit_count)

    return reward + exploration_term


def compute_suggest_next(top_n: int = 5) -> list[dict]:
    """计算 UCB1 推荐节点列表.

    返回可继续探索的节点，按 UCB1 分数排序。
    "可探索"定义：
    - 叶节点且非 dead_end
    - 或已完成但可以从此分支继续
    """
    tree_data = _load_tree()
    nodes = tree_data.get("nodes", {})
    strategy = _load_strategy_config()
    phase = strategy.get("current_phase", "coarse")
    phase_config = strategy.get("phases", {}).get(phase, {})
    C = phase_config.get("ucb1_C", 1.4)

    if not nodes:
        return []

    # Total visits = number of completed experiments
    total_visits = sum(
        1 for n in nodes.values()
        if n.get("status") in ("completed", "dead_end", "failed")
    )

    # Find expandable nodes (completed nodes that could spawn children)
    # + pending nodes that haven't been executed
    candidates = []
    parent_ids = {n.get("parent_id") for n in nodes.values() if n.get("parent_id")}

    for nid, node in nodes.items():
        status = node.get("status", "pending")

        # Skip dead ends
        if status == "dead_end":
            continue

        # Pending nodes are high priority
        if status == "pending":
            ucb1 = float("inf")
            candidates.append({
                "node_id": nid,
                "action_desc": node.get("action_desc", ""),
                "hypothesis": node.get("hypothesis", ""),
                "parent_id": node.get("parent_id"),
                "status": status,
                "ucb1_score": ucb1,
                "visit_count": 0,
                "cer": node.get("cer"),
            })
            continue

        # Completed leaf nodes (can spawn new experiments from here)
        if status == "completed" and nid not in parent_ids:
            # This is a leaf we can explore further from
            ucb1 = _compute_ucb1(nid, nodes, total_visits, C)
            candidates.append({
                "node_id": nid,
                "action_desc": node.get("action_desc", ""),
                "hypothesis": node.get("hypothesis", ""),
                "parent_id": node.get("parent_id"),
                "status": f"completed (expandable)",
                "ucb1_score": ucb1,
                "visit_count": _get_visit_count(nid, nodes),
                "cer": node.get("cer"),
            })

    # Sort by UCB1 score (descending)
    candidates.sort(key=lambda x: x["ucb1_score"], reverse=True)
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# HPO Statistics (#8) — 纯数据输出，不做推荐判断
# ---------------------------------------------------------------------------

def _load_hpo_history() -> dict:
    """加载 HPO 历史。"""
    path = Path("configs/hpo_history.json")
    if not path.exists():
        return {"trials": []}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_hpo_statistics() -> dict:
    """计算 HPO 历史的统计数据（纯数学，Kiro 自行分析）。"""
    history = _load_hpo_history()
    trials = history.get("trials", [])

    if not trials:
        return {"total_trials": 0, "categories": {}, "recent_trend": []}

    # 按 category 聚合
    categories: dict[str, dict] = {}
    all_categories = ["training_recipe", "augmentation", "hotwords", "data_addition"]

    for cat in all_categories:
        cat_trials = [t for t in trials if t.get("category") == cat]
        effective = [t for t in cat_trials if t.get("verdict", "").startswith("effective")]
        categories[cat] = {
            "attempts": len(cat_trials),
            "effective": len(effective),
            "accuracy": round(len(effective) / len(cat_trials), 2) if cat_trials else None,
            "last_verdict": cat_trials[-1].get("verdict") if cat_trials else None,
            "last_params": cat_trials[-1].get("params_changed") if cat_trials else None,
        }

    # 最近趋势
    recent = trials[-5:] if len(trials) >= 5 else trials
    recent_trend = [{"node": t.get("node_id"), "category": t.get("category"),
                     "verdict": t.get("verdict")} for t in recent]

    # 最近有效实验的参数
    effective_trials = [t for t in trials if t.get("verdict", "").startswith("effective")]
    last_effective = effective_trials[-1] if effective_trials else None

    return {
        "total_trials": len(trials),
        "categories": categories,
        "recent_trend": recent_trend,
        "last_effective": {
            "node_id": last_effective.get("node_id"),
            "category": last_effective.get("category"),
            "params_changed": last_effective.get("params_changed"),
            "cer": last_effective.get("result", {}).get("cer"),
        } if last_effective else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_tree_suggest(args) -> None:
    """UCB1 推荐下一个实验方向。"""
    candidates = compute_suggest_next(top_n=getattr(args, "top_n", 5) or 5)

    if not candidates:
        print("实验树为空或所有方向已穷尽。")
        return

    strategy = _load_strategy_config()
    phase = strategy.get("current_phase", "coarse")
    phase_config = strategy.get("phases", {}).get(phase, {})

    print(f"UCB1 推荐节点 | 阶段={phase.upper()} C={phase_config.get('ucb1_C', 1.4)}")
    print("━" * 50)

    for i, c in enumerate(candidates, 1):
        score_str = "∞" if c["ucb1_score"] == float("inf") else f"{c['ucb1_score']:.4f}"
        cer_str = f"CER={c['cer']:.4f}" if c["cer"] is not None else "CER=N/A"

        print(f"\n{i}. [{c['node_id']}] {c['action_desc']}")
        print(f"   UCB1={score_str} | visits={c['visit_count']} | {cer_str}")
        print(f"   状态: {c['status']}")
        if c["hypothesis"]:
            print(f"   假设: {c['hypothesis']}")

    print("\n" + "━" * 50)
    if candidates[0]["ucb1_score"] == float("inf"):
        print("建议: 优先执行未探索节点")
    else:
        print(f"建议: 选择 {candidates[0]['node_id']} (UCB1 最高)")


def cmd_suggest(args) -> None:
    """输出 HPO 历史统计数据（纯数据，Kiro 自行分析）。"""
    stats = compute_hpo_statistics()

    if stats["total_trials"] == 0:
        print("HPO 历史为空，尚无实验数据。")
        return

    print(f"HPO 历史统计 | 总实验: {stats['total_trials']}")
    print("━" * 50)

    # Category stats
    print("\n各方向统计:")
    for cat, data in stats["categories"].items():
        if data["attempts"] > 0:
            acc = f"{data['accuracy']:.0%}" if data["accuracy"] is not None else "N/A"
            print(f"  {cat}: {data['effective']}/{data['attempts']} ({acc})")
            print(f"    最近: {data['last_verdict']} | 参数: {data['last_params']}")
        else:
            print(f"  {cat}: 未尝试")

    # Recent trend
    if stats["recent_trend"]:
        print(f"\n最近趋势:")
        for t in stats["recent_trend"]:
            print(f"  {t['node']}: {t['category']} → {t['verdict']}")

    # Last effective
    if stats["last_effective"]:
        le = stats["last_effective"]
        print(f"\n最近有效实验:")
        print(f"  节点: {le['node_id']} | 方向: {le['category']}")
        print(f"  参数: {le['params_changed']}")
        print(f"  CER: {le['cer']}")

    print("\n" + "━" * 50)
