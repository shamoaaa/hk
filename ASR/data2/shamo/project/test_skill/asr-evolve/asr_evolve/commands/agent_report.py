# -*- coding: utf-8 -*-
"""agent-report 命令 — Agent 效能指标计算（纯数据工具）.

设计原则：代码只做数学计算和数据聚合，不做判断和诊断。
诊断分析由 Kiro 在 THINK 阶段基于 skill-experiment 中的分析指引自行完成。

功能:
  asr-evolve agent-report              输出效能指标（纯数据）
  asr-evolve agent-report --last N     基于最近 N 轮
  asr-evolve agent-report --update     从 HPO 历史重新聚合
  asr-evolve agent-report --json       JSON 格式输出（便于程序消费）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_PERF_FILE = Path("memory/agent_performance.json")
_HPO_FILE = Path("configs/hpo_history.json")


def _load_performance() -> dict:
    """加载 agent performance 数据。"""
    if not _PERF_FILE.exists():
        return {
            "version": 1,
            "total_experiments": 0,
            "effective_experiments": 0,
            "ineffective_experiments": 0,
            "baseline_experiments": 0,
            "dead_ends_hit": 0,
            "backtracks": 0,
            "total_gpu_hours": 0,
            "initial_cer": None,
            "current_best_cer": None,
            "cer_trajectory": [],
            "by_category": {},
        }
    return json.loads(_PERF_FILE.read_text(encoding="utf-8"))


def _save_performance(data: dict) -> None:
    """保存 agent performance 数据。"""
    _PERF_FILE.parent.mkdir(exist_ok=True)
    _PERF_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_performance_from_hpo() -> dict:
    """从 HPO 历史重新计算 agent performance 数据。"""
    if not _HPO_FILE.exists():
        return _load_performance()

    hpo_data = json.loads(_HPO_FILE.read_text(encoding="utf-8"))
    trials = hpo_data.get("trials", [])

    perf = _load_performance()

    # Recompute from trials
    total = len(trials)
    effective = sum(1 for t in trials if t.get("verdict", "").startswith("effective"))
    ineffective = sum(1 for t in trials if "ineffective" in t.get("verdict", ""))
    baselines = sum(1 for t in trials if t.get("verdict") == "baseline")

    perf["total_experiments"] = total
    perf["effective_experiments"] = effective
    perf["ineffective_experiments"] = ineffective
    perf["baseline_experiments"] = baselines

    # CER trajectory
    trajectory = []
    for t in trials:
        cer = t.get("result", {}).get("cer")
        if cer is not None:
            trajectory.append({
                "date": t.get("date", ""),
                "cer": cer,
                "node_id": t.get("node_id", ""),
                "category": t.get("category", ""),
                "verdict": t.get("verdict", ""),
            })
    perf["cer_trajectory"] = trajectory

    if trajectory:
        perf["initial_cer"] = trajectory[0]["cer"]
        perf["current_best_cer"] = min(t["cer"] for t in trajectory)

    # By category
    by_cat: dict[str, dict] = {}
    for t in trials:
        cat = t.get("category", "unknown")
        if cat not in by_cat:
            by_cat[cat] = {"attempts": 0, "effective": 0, "accuracy": None}
        by_cat[cat]["attempts"] += 1
        if t.get("verdict", "").startswith("effective"):
            by_cat[cat]["effective"] += 1

    for cat, stats in by_cat.items():
        if stats["attempts"] > 0:
            stats["accuracy"] = round(stats["effective"] / stats["attempts"], 3)

    perf["by_category"] = by_cat

    _save_performance(perf)
    return perf


def compute_metrics(perf: dict, last_n: int | None = None) -> dict:
    """计算效能指标（纯数学，不含判断）。"""
    trajectory = perf.get("cer_trajectory", [])
    if last_n and len(trajectory) > last_n:
        trajectory = trajectory[-last_n:]

    metrics: dict[str, Any] = {}

    # Decision accuracy
    total = perf.get("total_experiments", 0)
    effective = perf.get("effective_experiments", 0)
    baselines = perf.get("baseline_experiments", 0)
    actual_decisions = total - baselines
    if actual_decisions > 0:
        metrics["decision_accuracy"] = round(effective / actual_decisions, 3)
    else:
        metrics["decision_accuracy"] = None

    # Cumulative efficiency
    initial_cer = perf.get("initial_cer")
    best_cer = perf.get("current_best_cer")
    gpu_hours = perf.get("total_gpu_hours", 0)
    if initial_cer and best_cer and gpu_hours > 0:
        improvement_pct = (initial_cer - best_cer) * 100
        metrics["cumulative_efficiency_pct_per_hour"] = round(improvement_pct / gpu_hours, 4)
    else:
        metrics["cumulative_efficiency_pct_per_hour"] = None

    # Marginal efficiency (last N window)
    window = last_n or 5
    if len(trajectory) >= window:
        recent = trajectory[-window:]
        cer_start = recent[0]["cer"]
        cer_end = min(t["cer"] for t in recent)
        improvement = (cer_start - cer_end) * 100
        window_hours = window * 0.5  # estimate
        metrics["marginal_efficiency_pct_per_hour"] = round(improvement / window_hours, 4) if window_hours > 0 else 0
    else:
        metrics["marginal_efficiency_pct_per_hour"] = None

    # Exploration waste rate
    dead_ends = perf.get("dead_ends_hit", 0)
    if total > 0:
        metrics["exploration_waste_rate"] = round(dead_ends / total, 3)
    else:
        metrics["exploration_waste_rate"] = 0

    # Backtrack cost
    metrics["backtrack_cost"] = perf.get("backtracks", 0)

    # Recent trend (last 3 verdicts)
    if trajectory:
        recent_verdicts = [t["verdict"] for t in trajectory[-3:]]
        metrics["recent_verdicts"] = recent_verdicts

    return metrics


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def cmd_agent_report(args) -> None:
    """输出 Agent 效能指标（纯数据，不含诊断建议）。"""
    do_update = getattr(args, "update", False)
    last_n = getattr(args, "last", None)
    json_output = getattr(args, "json_output", False)

    if do_update:
        perf = update_performance_from_hpo()
        print("✅ 已更新 memory/agent_performance.json\n")
    else:
        perf = _load_performance()

    metrics = compute_metrics(perf, last_n=last_n)

    # JSON mode: output raw data for programmatic consumption
    if json_output:
        output = {"metrics": metrics, "performance": perf}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # Human-readable mode: structured data, no interpretation
    total = perf.get("total_experiments", 0)
    effective = perf.get("effective_experiments", 0)
    ineffective = perf.get("ineffective_experiments", 0)
    gpu_hours = perf.get("total_gpu_hours", 0)

    print("📊 Agent 效能指标")
    print("━" * 50)

    # Core counts
    print(f"总实验: {total} | 有效: {effective} | 无效: {ineffective} | 基线: {perf.get('baseline_experiments', 0)}")

    # Rates
    acc = metrics.get("decision_accuracy")
    print(f"决策准确率: {acc:.0%}" if acc is not None else "决策准确率: N/A (无决策)")
    print(f"GPU 总时间: {gpu_hours:.1f}h")

    cum_eff = metrics.get("cumulative_efficiency_pct_per_hour")
    print(f"累积效率: {cum_eff:.4f} %CER/h" if cum_eff else "累积效率: N/A")

    marg_eff = metrics.get("marginal_efficiency_pct_per_hour")
    window = last_n or 5
    print(f"边际效率 (窗口={window}): {marg_eff:.4f} %CER/h" if marg_eff else f"边际效率 (窗口={window}): N/A (数据不足)")

    waste = metrics.get("exploration_waste_rate", 0)
    print(f"探索浪费率: {waste:.0%}")
    print(f"回退次数: {metrics.get('backtrack_cost', 0)}")

    # CER trajectory
    trajectory = perf.get("cer_trajectory", [])
    if trajectory:
        initial = trajectory[0]["cer"]
        current = perf.get("current_best_cer", trajectory[-1]["cer"])
        print(f"\nCER 轨迹: {initial:.2%} → {current:.2%} (总改善 {(initial-current):.2%})")

    # Recent trend
    recent = metrics.get("recent_verdicts", [])
    if recent:
        print(f"最近趋势: {' → '.join(recent)}")

    # By category (raw numbers)
    by_cat = perf.get("by_category", {})
    if by_cat:
        print("\n分类统计:")
        for cat, stats in by_cat.items():
            acc_cat = stats.get("accuracy")
            if acc_cat is not None:
                print(f"  {cat}: {stats['effective']}/{stats['attempts']} ({acc_cat:.0%})")
            else:
                print(f"  {cat}: 未尝试")

    print("\n" + "━" * 50)
