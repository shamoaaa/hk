"""CLI for managing dynamic optimization strategies.

策略存储在 knowledge_entries 表中（knowledge_type='data_strategy'）。
系统初始化时通过 seed_strategies.py 注入 7 条 seed 策略，
后续迭代中 Kiro 可以通过此 CLI 新增、修改、禁用策略。

Usage:
    # 查看所有活跃策略
    python scripts/manage_strategies.py list

    # 查看策略效果统计
    python scripts/manage_strategies.py stats

    # 新增策略
    python scripts/manage_strategies.py add \
        --name "music_augment_for_high_cer" \
        --trigger "any(sm.scene_type == 'background_music' and sm.cer > 0.02 for sm in rep.scene_metrics)" \
        --commands "python scripts/run_augmentation.py --all --strategy music" \
        --description "music 场景 CER 高时增加 music 增强" \
        --priority 4 \
        --rationale "第3轮迭代发现 music 场景增强不足"

    # 禁用策略
    python scripts/manage_strategies.py disable --name "volume_augment_for_sensitivity"

    # 启用策略
    python scripts/manage_strategies.py enable --name "volume_augment_for_sensitivity"

    # 匹配当前评估报告
    python scripts/manage_strategies.py match --eval-report reports/evaluation/eval_report_iter_001.json
"""

import argparse
import json
import sys
from pathlib import Path

from scripts.core.database import Database
from scripts.core.config_utils import read_json


_DEFAULT_DB_PATH = "metadata.db"


def _get_db(args) -> Database:
    db = Database(getattr(args, "db_path", _DEFAULT_DB_PATH))
    db.initialize()
    return db


def _load_all_strategies(db: Database) -> list[dict]:
    """Load all data_strategy entries from knowledge_entries."""
    rows = db.get_all("knowledge_entries", filters={"knowledge_type": "data_strategy"})
    strategies = []
    for row in rows:
        content = json.loads(row.get("content_json", "{}"))
        content["_entry_id"] = row["entry_id"]
        content["_state"] = row.get("state", "active")
        content["_confidence"] = row.get("confidence", 0.5)
        content["_verified_count"] = row.get("verified_count", 0)
        strategies.append(content)
    return strategies


def cmd_list(args) -> None:
    """List all strategies."""
    db = _get_db(args)
    strategies = _load_all_strategies(db)

    # Also show seed strategies not yet in DB
    from scripts.decision.badcase_strategy_map import DATA_STRATEGIES
    db_names = {s.get("name") for s in strategies}

    output = {"strategies": [], "total": 0}

    for s in strategies:
        output["strategies"].append({
            "name": s.get("name", "unknown"),
            "state": s["_state"],
            "confidence": s["_confidence"],
            "priority": s.get("priority", 0),
            "trigger": s.get("trigger", ""),
            "description": s.get("description", ""),
            "verified_count": s["_verified_count"],
            "source": "database",
        })

    # Show seed strategies not in DB
    for s in DATA_STRATEGIES:
        if s["name"] not in db_names:
            output["strategies"].append({
                "name": s["name"],
                "state": "seed_only",
                "confidence": s.get("confidence", 0.7),
                "priority": s.get("priority", 0),
                "trigger": s.get("trigger", ""),
                "description": s.get("description", ""),
                "verified_count": 0,
                "source": "code",
            })

    output["total"] = len(output["strategies"])
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def cmd_stats(args) -> None:
    """Show strategy effectiveness statistics."""
    db = _get_db(args)
    strategies = _load_all_strategies(db)

    # Get all feedbacks to compute per-strategy stats
    feedbacks = db.get_all("effect_feedbacks")

    output = {"strategies": [], "total_feedbacks": len(feedbacks)}
    for s in strategies:
        name = s.get("name", "unknown")
        # Count feedbacks where this strategy was likely used
        # (simple heuristic: check if strategy name appears in suggestion_content)
        related = [f for f in feedbacks if name in (f.get("suggestion_content") or "")]
        positive = sum(1 for f in related if f.get("is_positive"))
        total = len(related)
        output["strategies"].append({
            "name": name,
            "state": s["_state"],
            "confidence": s["_confidence"],
            "feedback_count": total,
            "positive_count": positive,
            "success_rate": positive / total if total > 0 else None,
        })

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def cmd_add(args) -> None:
    """Add a new strategy."""
    db = _get_db(args)

    # Validate trigger syntax
    try:
        compile(args.trigger, "<trigger>", "eval")
    except SyntaxError as e:
        json.dump({"status": "error", "message": f"Invalid trigger syntax: {e}"}, sys.stdout)
        sys.stdout.write("\n")
        return

    content = {
        "name": args.name,
        "dimension": args.dimension,
        "category": args.category,
        "trigger": args.trigger,
        "description": args.description,
        "commands": [c.strip() for c in args.commands.split(";") if c.strip()],
        "priority": args.priority,
        "rationale": args.rationale or "",
    }

    import uuid
    entry_id = f"strat_{uuid.uuid4().hex[:8]}"
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    db.insert("knowledge_entries", {
        "entry_id": entry_id,
        "knowledge_type": "data_strategy",
        "content_json": json.dumps(content, ensure_ascii=False),
        "confidence": args.confidence,
        "source_iteration_id": args.source_iteration or "",
        "created_at": now,
        "updated_at": now,
        "state": "active",
        "verified_count": 0,
    })

    json.dump({
        "status": "success",
        "entry_id": entry_id,
        "name": args.name,
        "message": f"Strategy '{args.name}' added successfully",
    }, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def cmd_disable(args) -> None:
    """Disable (archive) a strategy."""
    db = _get_db(args)
    strategies = _load_all_strategies(db)

    target = next((s for s in strategies if s.get("name") == args.name), None)
    if not target:
        json.dump({"status": "error", "message": f"Strategy '{args.name}' not found in database"}, sys.stdout)
        sys.stdout.write("\n")
        return

    db.update("knowledge_entries", "entry_id", target["_entry_id"], {"state": "archived"})
    json.dump({"status": "success", "message": f"Strategy '{args.name}' disabled (archived)"}, sys.stdout)
    sys.stdout.write("\n")


def cmd_enable(args) -> None:
    """Re-enable a disabled strategy."""
    db = _get_db(args)
    strategies = _load_all_strategies(db)

    target = next((s for s in strategies if s.get("name") == args.name), None)
    if not target:
        json.dump({"status": "error", "message": f"Strategy '{args.name}' not found in database"}, sys.stdout)
        sys.stdout.write("\n")
        return

    db.update("knowledge_entries", "entry_id", target["_entry_id"], {"state": "active"})
    json.dump({"status": "success", "message": f"Strategy '{args.name}' re-enabled"}, sys.stdout)
    sys.stdout.write("\n")


def cmd_match(args) -> None:
    """Match strategies against an evaluation report."""
    from scripts.decision.badcase_strategy_map import match_strategies_dynamic

    report_data = read_json(args.eval_report)

    # Build a simple object with the fields match_strategies expects
    class _Report:
        def __init__(self, data):
            self.overall_cer = data.get("overall_cer", data.get("cer", 0.0))
            self.overall_wer = data.get("overall_wer", data.get("wer", 0.0))
            self.error_type_distribution = data.get("error_type_distribution", {})
            self.scene_metrics = []
            for sm in data.get("scene_metrics", []):
                self.scene_metrics.append(type("SM", (), sm)())

    report = _Report(report_data)
    matched = match_strategies_dynamic(report, db_path=getattr(args, "db_path", _DEFAULT_DB_PATH))

    output = {
        "status": "success",
        "matched_count": len(matched),
        "strategies": [
            {
                "name": s.get("name"),
                "priority": s.get("priority"),
                "description": s.get("description"),
                "commands": s.get("commands", []),
            }
            for s in matched
        ],
    }
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manage dynamic optimization strategies.")
    parser.add_argument("--db-path", default=_DEFAULT_DB_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all strategies (database + seed).")
    sub.add_parser("stats", help="Show strategy effectiveness statistics.")

    p_add = sub.add_parser("add", help="Add a new strategy.")
    p_add.add_argument("--name", required=True, help="Unique strategy name.")
    p_add.add_argument("--trigger", required=True, help="Python expression (uses 'rep' and 'badcases' vars).")
    p_add.add_argument("--commands", required=True, help="Commands to execute (semicolon-separated).")
    p_add.add_argument("--description", required=True, help="Human-readable description.")
    p_add.add_argument("--dimension", default="data", choices=["data", "post_process", "train_config"])
    p_add.add_argument("--category", default="augmentation")
    p_add.add_argument("--priority", type=int, default=3)
    p_add.add_argument("--confidence", type=float, default=0.5)
    p_add.add_argument("--rationale", default="")
    p_add.add_argument("--source-iteration", default="")

    p_dis = sub.add_parser("disable", help="Disable a strategy.")
    p_dis.add_argument("--name", required=True)

    p_en = sub.add_parser("enable", help="Re-enable a disabled strategy.")
    p_en.add_argument("--name", required=True)

    p_match = sub.add_parser("match", help="Match strategies against an eval report.")
    p_match.add_argument("--eval-report", required=True, help="Path to eval_report JSON.")

    args = parser.parse_args(argv)
    handlers = {
        "list": cmd_list,
        "stats": cmd_stats,
        "add": cmd_add,
        "disable": cmd_disable,
        "enable": cmd_enable,
        "match": cmd_match,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
