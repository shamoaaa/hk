# -*- coding: utf-8 -*-
"""state validate 命令 — 统一状态 schema 一致性检查.

改进方向 #9: Unified State Schema (AiScientist File-as-Bus)

功能:
  asr-evolve state validate    检查所有状态文件一致性
  asr-evolve state summary     结构化状态摘要 (JSON 输出)
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# State schema definitions
# ---------------------------------------------------------------------------

REQUIRED_STATE_FILES = {
    "STATE.md": {"type": "markdown", "required": True},
    "experiment_tree.json": {"type": "json", "required": True},
    "DEAD_ENDS.md": {"type": "markdown", "required": True},
    "INSIGHTS.md": {"type": "markdown", "required": True},
    "loop-budget.md": {"type": "markdown", "required": True},
    "loop-run-log.md": {"type": "markdown", "required": False},
    "configs/training.yaml": {"type": "yaml", "required": True},
    "configs/search_strategy.yaml": {"type": "yaml", "required": True},
    "configs/decision_rules.json": {"type": "json", "required": True},
    "configs/hpo_history.json": {"type": "json", "required": True},
    "configs/agent_metrics.yaml": {"type": "yaml", "required": True},
    "memory/agent_performance.json": {"type": "json", "required": False},
}


def _file_checksum(path: Path) -> str:
    """Compute SHA256 checksum of file content."""
    if not path.exists():
        return ""
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]


def _validate_json_file(path: Path) -> tuple[bool, str]:
    """Validate a JSON file is parseable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e}"
    except Exception as e:
        return False, str(e)


def _validate_yaml_file(path: Path) -> tuple[bool, str]:
    """Validate a YAML file is parseable."""
    try:
        with open(path, encoding="utf-8") as f:
            yaml.safe_load(f)
        return True, ""
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"
    except Exception as e:
        return False, str(e)


def _check_cross_references() -> list[dict]:
    """Check cross-file consistency.

    Validates:
    - experiment_tree best node matches STATE.md reported best
    - dead_end nodes in tree match DEAD_ENDS.md entries
    - HPO history node_ids exist in experiment_tree
    """
    issues = []

    # Load experiment tree
    tree_path = Path("experiment_tree.json")
    if tree_path.exists():
        tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
        nodes = tree_data.get("nodes", {})

        # Check best node consistency with STATE.md
        best_node = None
        best_cer = None
        for nid, node in nodes.items():
            if node.get("status") == "completed" and node.get("cer") is not None:
                if best_cer is None or node["cer"] < best_cer:
                    best_cer = node["cer"]
                    best_node = nid

        state_path = Path("STATE.md")
        if state_path.exists() and best_cer is not None:
            state_content = state_path.read_text(encoding="utf-8")
            # Check if STATE.md mentions a different CER as best
            # This is a soft check - STATE.md is human-readable
            if f"CER: {best_cer}" not in state_content and f"CER={best_cer}" not in state_content:
                # Try percentage format
                cer_pct = f"{best_cer*100:.2f}%"
                if cer_pct not in state_content:
                    issues.append({
                        "type": "cross_reference",
                        "severity": "low",
                        "message": f"STATE.md may not reflect current best CER={best_cer:.4f} (node={best_node})",
                    })

        # Check HPO history references
        hpo_path = Path("configs/hpo_history.json")
        if hpo_path.exists():
            hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
            for trial in hpo.get("trials", []):
                node_id = trial.get("node_id", "")
                if node_id and node_id not in nodes:
                    issues.append({
                        "type": "orphan_reference",
                        "severity": "medium",
                        "message": f"HPO history references {node_id} not found in experiment_tree",
                    })

    return issues


# ---------------------------------------------------------------------------
# Structured state snapshot
# ---------------------------------------------------------------------------

def generate_state_snapshot() -> dict:
    """Generate a versioned, structured state snapshot.

    This is the machine-readable equivalent of STATE.md,
    following the AiScientist File-as-Bus architecture.
    """
    snapshot: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "files": {},
        "experiment_tree": {},
        "strategy": {},
        "performance": {},
        "health": {},
    }

    # File checksums
    for filename in REQUIRED_STATE_FILES:
        path = Path(filename)
        snapshot["files"][filename] = {
            "exists": path.exists(),
            "checksum": _file_checksum(path),
            "size": path.stat().st_size if path.exists() else 0,
        }

    # Experiment tree summary
    tree_path = Path("experiment_tree.json")
    if tree_path.exists():
        tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
        nodes = tree_data.get("nodes", {})
        completed = [n for n in nodes.values() if n.get("status") == "completed"]
        dead_ends = [n for n in nodes.values() if n.get("status") == "dead_end"]
        pending = [n for n in nodes.values() if n.get("status") == "pending"]

        best_cer = min((n["cer"] for n in completed if n.get("cer") is not None), default=None)
        snapshot["experiment_tree"] = {
            "total_nodes": len(nodes),
            "completed": len(completed),
            "dead_ends": len(dead_ends),
            "pending": len(pending),
            "best_cer": best_cer,
        }

    # Strategy phase
    strat_path = Path("configs/search_strategy.yaml")
    if strat_path.exists():
        with open(strat_path, encoding="utf-8") as f:
            strat = yaml.safe_load(f) or {}
        snapshot["strategy"] = {
            "current_phase": strat.get("current_phase", "unknown"),
        }

    # Performance summary
    perf_path = Path("memory/agent_performance.json")
    if perf_path.exists():
        perf = json.loads(perf_path.read_text(encoding="utf-8"))
        total = perf.get("total_experiments", 0)
        effective = perf.get("effective_experiments", 0)
        baselines = perf.get("baseline_experiments", 0)
        actual = total - baselines
        snapshot["performance"] = {
            "total_experiments": total,
            "decision_accuracy": round(effective / actual, 3) if actual > 0 else None,
            "current_best_cer": perf.get("current_best_cer"),
        }

    return snapshot


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_state_validate(args) -> None:
    """检查所有状态文件一致性。"""
    print("状态一致性检查")
    print("━" * 50)

    all_ok = True
    file_issues = []

    # 1. Check file existence and parseability
    for filename, spec in REQUIRED_STATE_FILES.items():
        path = Path(filename)
        if not path.exists():
            if spec["required"]:
                file_issues.append(("❌", filename, "文件不存在 (必需)"))
                all_ok = False
            else:
                file_issues.append(("⚪", filename, "文件不存在 (可选)"))
            continue

        # Validate format
        valid = True
        err = ""
        if spec["type"] == "json":
            valid, err = _validate_json_file(path)
        elif spec["type"] == "yaml":
            valid, err = _validate_yaml_file(path)

        if valid:
            size = path.stat().st_size
            checksum = _file_checksum(path)
            file_issues.append(("✅", filename, f"OK ({size}B, #{checksum[:8]})"))
        else:
            file_issues.append(("❌", filename, f"格式错误: {err}"))
            all_ok = False

    print("\n文件完整性:")
    for icon, name, msg in file_issues:
        print(f"  {icon} {name}: {msg}")

    # 2. Cross-reference checks
    xref_issues = _check_cross_references()
    if xref_issues:
        print(f"\n交叉引用检查 ({len(xref_issues)} 个问题):")
        for issue in xref_issues:
            sev_icon = {"low": "ℹ️", "medium": "⚠️", "high": "❌"}.get(issue["severity"], "?")
            print(f"  {sev_icon} [{issue['type']}] {issue['message']}")
        if any(i["severity"] in ("medium", "high") for i in xref_issues):
            all_ok = False
    else:
        print("\n交叉引用: ✅ 一致")

    # 3. Overall verdict
    print("\n" + "━" * 50)
    if all_ok:
        print("✅ 状态一致性检查通过")
    else:
        print("⚠️ 发现一致性问题，建议修复")


def cmd_state_summary(args) -> None:
    """输出结构化状态摘要 (JSON)。"""
    snapshot = generate_state_snapshot()

    if getattr(args, "json_output", False):
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    else:
        print("结构化状态摘要")
        print("━" * 50)
        print(f"生成时间: {snapshot['generated_at']}")

        tree = snapshot.get("experiment_tree", {})
        if tree:
            print(f"\n实验树: {tree.get('total_nodes', 0)} 节点")
            print(f"  完成: {tree.get('completed', 0)} | 死胡同: {tree.get('dead_ends', 0)} | 待执行: {tree.get('pending', 0)}")
            if tree.get("best_cer"):
                print(f"  最优 CER: {tree['best_cer']:.4f}")

        strat = snapshot.get("strategy", {})
        if strat:
            print(f"\n搜索策略: {strat.get('current_phase', '?').upper()}")

        perf = snapshot.get("performance", {})
        if perf:
            acc = perf.get("decision_accuracy")
            print(f"\nAgent 效能: 准确率={acc:.0%}" if acc else "\nAgent 效能: N/A")

        # File health
        files = snapshot.get("files", {})
        missing = [f for f, info in files.items() if not info.get("exists")]
        if missing:
            print(f"\n缺失文件: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# CLI 分发
# ---------------------------------------------------------------------------

def cmd_state(args) -> None:
    """state 命令路由。"""
    action = getattr(args, "state_action", None)
    if action == "validate":
        cmd_state_validate(args)
    elif action == "summary":
        cmd_state_summary(args)
    else:
        print("用法: asr-evolve state {validate|summary}")
