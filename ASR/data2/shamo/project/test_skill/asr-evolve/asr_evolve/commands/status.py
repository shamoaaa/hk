"""status 命令 — 系统概览."""
import json
from pathlib import Path
from asr_evolve.memory.state import StateManager


def cmd_status():
    """读取 STATE.md + experiment_tree.json，输出系统概览。"""
    sm = StateManager("STATE.md")
    state = sm.read_state()

    print("=" * 40)
    print("ASR-Evolve 系统状态")
    print("=" * 40)
    print(f"Last run: {state.get('last_run', '未运行')}")

    # 高优先级
    high_priority = state.get("high_priority", [])
    print(f"高优先级: {len(high_priority)} 项")
    for item in high_priority:
        print(f"  🔴 {item}")

    # 观察列表
    watch_list = state.get("watch_list", [])
    print(f"观察列表: {len(watch_list)} 项")
    for item in watch_list:
        print(f"  👀 {item}")

    # 已完成
    completed = state.get("completed", [])
    print(f"已完成: {len(completed)} 项")

    # 暂停检测
    if sm.is_paused():
        print("\n⚠️  环路已暂停！")

    # 实验树摘要
    tree_path = Path("experiment_tree.json")
    if tree_path.exists():
        try:
            tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
            nodes = tree_data.get("nodes", [])
            total = len(nodes)
            completed_n = sum(1 for n in nodes if n.get("status") == "completed")
            dead_ends = sum(1 for n in nodes if n.get("status") == "dead_end")
            pending = sum(1 for n in nodes if n.get("status") in ("pending", None))
            best_node = min(
                (n for n in nodes if n.get("cer") is not None),
                key=lambda n: n["cer"],
                default=None,
            )
            print(f"\n--- 实验树 ---")
            print(f"总节点: {total}")
            print(f"已完成: {completed_n} | 死胡同: {dead_ends} | 待执行: {pending}")
            if best_node:
                print(f"最优: {best_node.get('node_id', '?')} (CER={best_node['cer']:.4f})")
        except (json.JSONDecodeError, KeyError):
            print("\n--- 实验树: 解析失败 ---")

    print("=" * 40)
