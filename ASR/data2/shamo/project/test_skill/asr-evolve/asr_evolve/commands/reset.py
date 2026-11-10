# -*- coding: utf-8 -*-
"""reset 命令 — 系统重置.

asr-evolve reset remote   清理远程产物（output/脚本/恢复jsonl）
asr-evolve reset local    重置本地记忆文件
asr-evolve reset all      全部重置
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import yaml

from asr_evolve.environment.remote import RemoteExecutor, RemoteConfig


def _get_executor() -> tuple[RemoteExecutor, str]:
    """连接远程并返回 (executor, workspace)。"""
    with open("configs/remote.yaml", encoding="utf-8") as f:
        remote_cfg = yaml.safe_load(f)
    r = remote_cfg["remote"]
    e = RemoteExecutor(RemoteConfig(
        host=r["host"], port=r["port"],
        username=r["username"], password=r.get("password", ""),
        key_file=r.get("key_file", ""),
        workspace=r["workspace"],
    ))
    e.connect()
    return e, r["workspace"]


def reset_remote() -> None:
    """清理远程产物。"""
    executor, ws = _get_executor()
    try:
        cmds = [
            f'rm -rf {ws}/output/*',
            f'rm -f {ws}/run_train_current.sh {ws}/run_minimal.sh {ws}/run_augment.py {ws}/run_merge.sh',
            f'if [ -f {ws}/data/merged/train/train_original.jsonl ]; then '
            f'cp {ws}/data/merged/train/train_original.jsonl {ws}/data/merged/train/train.jsonl; '
            f'rm -f {ws}/data/merged/train/train_augmented.jsonl; '
            f'echo "RESTORED original train.jsonl"; '
            f'else echo "No backup found, train.jsonl unchanged"; fi',
            f'rm -rf {ws}/results/*',
        ]
        for cmd in cmds:
            rc, out, err = executor.run_command(cmd, timeout=30)
            if out.strip():
                print(f"  {out.strip()}")

        # Verify
        rc, out, _ = executor.run_command(f'ls {ws}/output/ 2>/dev/null | wc -l', timeout=10)
        remaining = out.strip()
        print(f"  远程 output/ 剩余: {remaining} 项")
        print("✅ 远程清理完成")
    finally:
        executor.disconnect()


def reset_local() -> None:
    """重置本地记忆文件。"""
    # STATE.md
    Path("STATE.md").write_text(
        "# Loop State — ASR-Evolve\n\n"
        "Last run: （尚未开始）\n\n"
        "## 高优先级（环路正在处理 / 等待人工）\n\n"
        "- 系统已初始化，等待用户指示开始首轮优化\n\n"
        "## 观察列表\n\n—\n\n"
        "## 当前最优\n\n"
        "- 模型: （暂无）\n- CER: （暂无）\n- 测试集: （暂无）\n",
        encoding="utf-8",
    )

    # experiment_tree.json
    Path("experiment_tree.json").write_text(
        json.dumps({"next_id": 1, "nodes": {}}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # DEAD_ENDS.md
    Path("DEAD_ENDS.md").write_text(
        "# Dead Ends — 已验证无效的方向\n\n"
        "> 此文件中的条目是**硬性禁止**——Agent 在 EXECUTE 前必须检查冲突。\n\n"
        "## 记录\n\n（暂无）\n",
        encoding="utf-8",
    )

    # INSIGHTS.md
    Path("INSIGHTS.md").write_text(
        "# Insights — 持久化的优化洞察\n\n"
        "> 跨实验验证的可靠结论。\n\n"
        "## 记录\n\n（暂无）\n",
        encoding="utf-8",
    )

    # loop-run-log.md
    Path("loop-run-log.md").write_text(
        "# Loop Run Log — ASR-Evolve\n\n"
        "每次环路运行追加一条记录。\n\n"
        "## 运行记录\n\n<!-- 环路在下方追加 -->\n",
        encoding="utf-8",
    )

    # configs/decision_rules.json
    Path("configs/decision_rules.json").write_text(
        json.dumps({"version": 1, "rules": [], "candidates": []}, indent=2),
        encoding="utf-8",
    )

    # configs/hpo_history.json
    Path("configs/hpo_history.json").write_text(
        json.dumps({"version": 1, "trials": []}, indent=2),
        encoding="utf-8",
    )

    # configs/search_strategy.yaml — reset current_phase to coarse
    strat_path = Path("configs/search_strategy.yaml")
    if strat_path.exists():
        with open(strat_path, encoding="utf-8") as f:
            strat = yaml.safe_load(f) or {}
        strat["current_phase"] = "coarse"
        with open(strat_path, "w", encoding="utf-8") as f:
            yaml.dump(strat, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # memory/agent_performance.json
    Path("memory").mkdir(exist_ok=True)
    Path("memory/agent_performance.json").write_text(
        json.dumps({
            "version": 1, "total_experiments": 0,
            "effective_experiments": 0, "ineffective_experiments": 0,
            "baseline_experiments": 0, "dead_ends_hit": 0,
            "backtracks": 0, "total_gpu_hours": 0,
            "initial_cer": None, "current_best_cer": None,
            "cer_trajectory": [], "by_category": {},
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # logs/
    logs = Path("logs")
    if logs.exists():
        for f in logs.iterdir():
            f.unlink(missing_ok=True)

    # reports/
    reports = Path("reports")
    if reports.exists():
        for f in reports.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)

    # results/ (eval 产物)
    results = Path("results")
    if results.exists():
        import shutil
        shutil.rmtree(results, ignore_errors=True)
        results.mkdir(exist_ok=True)

    print("✅ 本地重置完成")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_reset(args) -> None:
    """reset 命令路由。"""
    action = getattr(args, "reset_action", None)
    if action == "remote":
        reset_remote()
    elif action == "local":
        reset_local()
    elif action == "all":
        print("--- 远程清理 ---")
        reset_remote()
        print("\n--- 本地重置 ---")
        reset_local()
        print("\n✅ 全部重置完成")
    else:
        print("用法: asr-evolve reset {remote|local|all}")
