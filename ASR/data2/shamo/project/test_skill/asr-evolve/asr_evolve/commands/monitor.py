# -*- coding: utf-8 -*-
"""monitor 命令 — 训练 telemetry 数据采集（纯数据工具）.

设计原则：代码只做数据提取和客观检测（NaN），不做判断和建议。
Kiro 自行解读 telemetry 数据并决定是否干预。

功能:
  asr-evolve train monitor     采集训练 telemetry 数据
  asr-evolve train health      本地训练状态摘要
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Telemetry 解析
# ---------------------------------------------------------------------------

def parse_telemetry_line(line: str) -> dict | None:
    """从训练日志行中提取 telemetry 指标.

    FunASR 日志格式示例:
    [2026-07-06] epoch:2 step:40 loss:2.345 lr:0.0001
    """
    metrics = {}
    pairs = re.findall(r'(\w+)[=:]([0-9e\-+.]+)', line)
    for key, val in pairs:
        try:
            metrics[key] = float(val)
        except ValueError:
            metrics[key] = val
    return metrics if metrics else None


def extract_telemetry_summary(telemetry: list[dict]) -> dict:
    """从 telemetry 序列中提取摘要统计（纯数学，无判断）。"""
    summary = {
        "total_entries": len(telemetry),
        "has_nan": False,
        "loss": {},
        "grad_norm": {},
        "lr": {},
    }

    losses = [t.get("loss") for t in telemetry if t.get("loss") is not None]
    grad_norms = [t.get("grad_norm") for t in telemetry if t.get("grad_norm") is not None]
    lrs = [t.get("lr") for t in telemetry if t.get("lr") is not None]

    # NaN 检测（客观事实，不是判断）
    for t in telemetry:
        for key, val in t.items():
            if isinstance(val, float) and (val != val):  # NaN
                summary["has_nan"] = True
                summary["nan_field"] = key
                break

    # Loss 统计
    if losses:
        summary["loss"] = {
            "first": losses[0],
            "last": losses[-1],
            "min": min(losses),
            "max": max(losses),
            "trend": "decreasing" if len(losses) > 1 and losses[-1] < losses[0] else "not_decreasing",
            "values_count": len(losses),
        }
        # 最大单步变化比
        if len(losses) >= 2:
            max_ratio = max(losses[i] / losses[i-1] for i in range(1, len(losses)) if losses[i-1] > 0)
            summary["loss"]["max_step_ratio"] = round(max_ratio, 2)

    # Gradient norm 统计
    if grad_norms:
        summary["grad_norm"] = {
            "avg": round(sum(grad_norms) / len(grad_norms), 4),
            "max": max(grad_norms),
            "min": min(grad_norms),
        }

    # Learning rate
    if lrs:
        summary["lr"] = {
            "current": lrs[-1],
        }

    return summary


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def cmd_train_monitor(args) -> None:
    """采集训练 telemetry 数据（纯数据输出）。"""
    from asr_evolve.commands.train import _load_configs, _load_state, _get_executor

    state = _load_state()
    if not state or state.get("status") != "running":
        print("No running training to monitor.")
        return

    remote_cfg, training_cfg, _ = _load_configs()
    output_dir = state["output_dir"]

    executor = _get_executor(remote_cfg)
    try:
        n_lines = getattr(args, "lines", 50) or 50
        rc, log_output, _ = executor.run_command(
            f"tail -{n_lines} {output_dir}/train_stdout.log 2>/dev/null",
            timeout=15,
        )

        if rc != 0 or not log_output.strip():
            print("Cannot read training log.")
            return

        # Parse telemetry
        telemetry = []
        for line in log_output.strip().split("\n"):
            t = parse_telemetry_line(line)
            if t:
                telemetry.append(t)

        if not telemetry:
            print("No parseable telemetry in recent log lines.")
            print(f"Raw tail ({n_lines} lines):")
            print(log_output[-500:])
            return

        # Extract summary
        summary = extract_telemetry_summary(telemetry)

        # Output (pure data, no judgment)
        print(f"训练 Telemetry | {datetime.now().isoformat()[:16]}")
        print(f"TAG: {state.get('tag', '?')} | Mode: {state.get('mode', 'full')}")
        print("━" * 40)

        # NaN flag
        if summary["has_nan"]:
            print(f"⚠️ NaN 检测到 (字段: {summary.get('nan_field', '?')})")

        # Loss
        loss = summary.get("loss", {})
        if loss:
            print(f"Loss: {loss.get('first', '?'):.4f} → {loss.get('last', '?'):.4f} "
                  f"(min={loss.get('min', '?'):.4f}, max={loss.get('max', '?'):.4f})")
            print(f"  趋势: {loss.get('trend', '?')} | 最大单步比: {loss.get('max_step_ratio', '?')}x")

        # Grad norm
        grad = summary.get("grad_norm", {})
        if grad:
            print(f"Grad Norm: avg={grad.get('avg', '?')}, max={grad.get('max', '?')}")

        # LR
        lr = summary.get("lr", {})
        if lr:
            print(f"LR: {lr.get('current', '?')}")

        print(f"Entries: {summary['total_entries']}")
        print("━" * 40)

    finally:
        executor.disconnect()


def cmd_train_health(args) -> None:
    """本地训练状态摘要（不连接远程）。"""
    state_file = Path("logs/train_state.json")
    if not state_file.exists():
        print("No training state found.")
        return

    state = json.loads(state_file.read_text(encoding="utf-8"))
    print(f"TAG: {state.get('tag', '?')}")
    print(f"Status: {state.get('status', '?')}")
    print(f"Mode: {state.get('mode', 'full')}")
    print(f"Submitted: {state.get('submitted_at', '?')}")

    if state.get("status") == "completed":
        n_pts = state.get("checkpoints", 0)
        print(f"Checkpoints: {n_pts}")
        if n_pts == 0:
            print("⚠️ 0 checkpoints (进程退出但未产出模型)")
    elif state.get("status") == "failed":
        print("Status: failed")
    elif state.get("status") == "running":
        print("Status: running")
        if state.get("finished_at"):
            print(f"Finished: {state['finished_at']}")
