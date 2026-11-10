# -*- coding: utf-8 -*-
"""strategy 命令 — 自适应搜索策略管理.

改进方向 #5: 搜索策略自适应 (OptiMindTune)

功能:
  asr-evolve strategy status           查看当前搜索阶段
  asr-evolve strategy check-transition 检查是否触发阶段切换
  asr-evolve strategy params           查看当前阶段允许的参数范围
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


_STRATEGY_FILE = Path("configs/search_strategy.yaml")


def _load_strategy() -> dict:
    """加载搜索策略配置。"""
    if not _STRATEGY_FILE.exists():
        return {"current_phase": "coarse", "phases": {}, "phase_transitions": {}}
    with open(_STRATEGY_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_strategy(config: dict) -> None:
    """保存搜索策略配置。"""
    with open(_STRATEGY_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _get_current_cer() -> float | None:
    """从 experiment_tree.json 获取当前最优 CER。"""
    tree_path = Path("experiment_tree.json")
    if not tree_path.exists():
        return None
    data = json.loads(tree_path.read_text(encoding="utf-8"))
    nodes = data.get("nodes", {})
    best_cer = None
    for node in nodes.values():
        if node.get("status") == "completed" and node.get("cer") is not None:
            cer = node["cer"]
            if best_cer is None or cer < best_cer:
                best_cer = cer
    return best_cer


def cmd_strategy_status(args) -> None:
    """查看当前搜索阶段。"""
    config = _load_strategy()
    phase = config.get("current_phase", "coarse")
    phases = config.get("phases", {})
    phase_config = phases.get(phase, {})
    transitions = config.get("phase_transitions", {})

    current_cer = _get_current_cer()

    print("搜索策略状态")
    print("━" * 40)
    print(f"当前阶段: {phase.upper()}")
    print(f"描述: {phase_config.get('description', 'N/A')}")
    print(f"当前最优 CER: {current_cer:.4f}" if current_cer else "当前最优 CER: N/A")
    print()
    print(f"UCB1 C值: {phase_config.get('ucb1_C', 'N/A')}")
    print(f"每轮最大参数变更: {phase_config.get('max_params_per_trial', 'N/A')}")
    print(f"强制 Probe: {'是' if phase_config.get('probe_required') else '否'}")
    print(f"死胡同耐心: 连续 {phase_config.get('dead_end_patience', 'N/A')} 次无效")
    print(f"探索/利用比: {phase_config.get('explore_ratio', 'N/A')}")
    print()

    # Show transition thresholds
    print("阶段切换阈值:")
    for name, t in transitions.items():
        trigger = t.get("trigger_cer", "?")
        status = "✅ 已达到" if (current_cer and current_cer < trigger) else "⏳ 未达到"
        print(f"  {name}: CER < {trigger} ({status})")


def cmd_strategy_check_transition(args) -> None:
    """报告当前 CER 是否满足阶段切换条件（纯报告，不自动修改配置）。"""
    config = _load_strategy()
    phase = config.get("current_phase", "coarse")
    transitions = config.get("phase_transitions", {})

    # Get CER from args or current best
    cer = getattr(args, "cer", None)
    if cer is None:
        cer = _get_current_cer()
    if cer is None:
        print("无法获取当前 CER，跳过阶段切换检查。")
        return

    # Check what transition would apply
    next_phase = None
    threshold = None

    if phase == "coarse":
        threshold = transitions.get("coarse_to_fine", {}).get("trigger_cer", 0.10)
        if cer < threshold:
            next_phase = "fine"
    elif phase == "fine":
        threshold = transitions.get("fine_to_exploit", {}).get("trigger_cer", 0.05)
        if cer < threshold:
            next_phase = "exploit"

    # Report (不自动执行切换)
    print(f"阶段切换检查 | CER={cer:.4f}")
    print("━" * 40)
    print(f"当前阶段: {phase.upper()}")

    if next_phase:
        old_config = config.get("phases", {}).get(phase, {})
        new_config = config.get("phases", {}).get(next_phase, {})
        print(f"✅ 满足切换条件: CER={cer:.4f} < {threshold}")
        print(f"   可切换到: {next_phase.upper()}")
        print(f"   变化: UCB1_C {old_config.get('ucb1_C')}→{new_config.get('ucb1_C')}, "
              f"params {old_config.get('max_params_per_trial')}→{new_config.get('max_params_per_trial')}, "
              f"probe {'必须' if new_config.get('probe_required') else '可选'}")
        print(f"\n如需执行切换，请手动修改 configs/search_strategy.yaml 中的 current_phase 字段。")
    else:
        if phase == "coarse":
            target = transitions.get("coarse_to_fine", {}).get("trigger_cer", 0.10)
            print(f"未满足切换条件: CER={cer:.4f} >= {target}")
            print(f"距离下一阶段还差: {cer - target:.4f}")
        elif phase == "fine":
            target = transitions.get("fine_to_exploit", {}).get("trigger_cer", 0.05)
            print(f"未满足切换条件: CER={cer:.4f} >= {target}")
            print(f"距离下一阶段还差: {cer - target:.4f}")
        else:
            print("已在最终阶段 (exploit)")


def cmd_strategy_params(args) -> None:
    """查看当前阶段允许的参数范围。"""
    config = _load_strategy()
    phase = config.get("current_phase", "coarse")
    phase_config = config.get("phases", {}).get(phase, {})
    param_ranges = phase_config.get("param_ranges", {})

    print(f"当前阶段: {phase.upper()} — 允许的参数范围")
    print("━" * 40)
    for param, range_val in param_ranges.items():
        if isinstance(range_val, list):
            print(f"  {param}: [{range_val[0]}, {range_val[1]}]")
        else:
            print(f"  {param}: {range_val}")
    print()
    print(f"探索比例: {phase_config.get('explore_ratio', 'N/A')}")
    print(f"Probe 要求: {'必须' if phase_config.get('probe_required') else '可选'}")


# ---------------------------------------------------------------------------
# CLI 分发
# ---------------------------------------------------------------------------

def cmd_strategy(args) -> None:
    """strategy 命令路由。"""
    action = getattr(args, "strategy_action", None)
    if action == "status":
        cmd_strategy_status(args)
    elif action == "check-transition":
        cmd_strategy_check_transition(args)
    elif action == "params":
        cmd_strategy_params(args)
    else:
        print("用法: asr-evolve strategy {status|check-transition|params}")
