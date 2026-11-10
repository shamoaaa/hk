"""CLI script for managing the Skill Knowledge Base.

Usage:
    python scripts/manage_knowledge.py add \\
        --type scene_solution \\
        --content '{"scene":"background_voice","problem":"High CER","solution":"Add SE module"}' \\
        --confidence 0.85 \\
        --source-iteration iter_001

    python scripts/manage_knowledge.py query \\
        --type scene_solution \\
        [--source-iteration iter_001] \\
        [--content-match '{"scene":"background_voice"}']

    python scripts/manage_knowledge.py view

    python scripts/manage_knowledge.py update-confidence \\
        --entry-id ke_abc12345 \\
        --confidence 0.95

    python scripts/manage_knowledge.py trigger-learning \\
        --start-time 2024-01-01T00:00:00 \\
        --end-time 2024-12-31T23:59:59

Manages the Skill Knowledge Base via the SkillKnowledgeBase class and
triggers learning via the SkillEvolutionEngine.  Outputs structured JSON
to stdout.

Requirements: 10.5, 10.6, 10.10
"""

import argparse
import json
import sys
from dataclasses import asdict

from scripts.core.database import Database
from scripts.core.models import KnowledgeEntry
from scripts.decision.skill_knowledge_base import SkillKnowledgeBase
from scripts.decision.skill_evolution import SkillEvolutionEngine


_DEFAULT_DB_PATH = "metadata.db"
_DEFAULT_WORKSPACE = "."


def _get_kb(args: argparse.Namespace) -> SkillKnowledgeBase:
    """Create a SkillKnowledgeBase from CLI arguments."""
    db_path = getattr(args, "db_path", _DEFAULT_DB_PATH)
    db = Database(db_path)
    db.initialize()
    return SkillKnowledgeBase(db=db)


def _get_engine(args: argparse.Namespace) -> SkillEvolutionEngine:
    """Create a SkillEvolutionEngine from CLI arguments."""
    workspace_dir = getattr(args, "workspace", _DEFAULT_WORKSPACE)
    db_path = getattr(args, "db_path", _DEFAULT_DB_PATH)
    db = Database(db_path)
    db.initialize()
    return SkillEvolutionEngine(workspace_dir=workspace_dir, db=db)


def _run_add(args: argparse.Namespace) -> dict:
    """Execute the ``add`` subcommand."""
    kb = _get_kb(args)

    try:
        content = json.loads(args.content)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "message": "Invalid JSON for --content",
        }

    entry = KnowledgeEntry(
        knowledge_type=args.type,
        content=content,
        confidence=args.confidence,
        source_iteration_id=args.source_iteration or "",
    )

    try:
        entry_id = kb.add_knowledge(entry)
    except ValueError as e:
        return {
            "status": "error",
            "message": str(e),
        }

    return {
        "status": "success",
        "entry_id": entry_id,
        "knowledge_type": args.type,
        "confidence": args.confidence,
        "message": f"Knowledge entry added: {entry_id}",
    }


def _run_query(args: argparse.Namespace) -> dict:
    """Execute the ``query`` subcommand."""
    kb = _get_kb(args)

    filters: dict = {}
    if args.source_iteration:
        filters["source_iteration_id"] = args.source_iteration

    if args.content_match:
        try:
            filters["content_match"] = json.loads(args.content_match)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": "Invalid JSON for --content-match",
            }

    # Determine which states to include (default: active only)
    include_stale = getattr(args, "include_stale", False)
    include_archived = getattr(args, "include_archived", False)

    entries = kb.query_knowledge(
        args.type,
        filters=filters or None,
        include_stale=include_stale,
        include_archived=include_archived,
    )

    results = []
    for entry in entries:
        results.append({
            "knowledge_type": entry.knowledge_type,
            "content": entry.content,
            "confidence": entry.confidence,
            "source_iteration_id": entry.source_iteration_id,
            "state": entry.state,
            "verified_count": entry.verified_count,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        })

    return {
        "status": "success",
        "total_entries": len(results),
        "entries": results,
    }


def _run_view(args: argparse.Namespace) -> dict:
    """Execute the ``view`` subcommand."""
    kb = _get_kb(args)
    view = kb.get_structured_view()
    return {
        "status": "success",
        **view,
    }


def _run_update_confidence(args: argparse.Namespace) -> dict:
    """Execute the ``update-confidence`` subcommand."""
    kb = _get_kb(args)

    try:
        kb.update_confidence(args.entry_id, args.confidence)
    except KeyError as e:
        return {
            "status": "error",
            "message": str(e),
        }
    except ValueError as e:
        return {
            "status": "error",
            "message": str(e),
        }

    return {
        "status": "success",
        "entry_id": args.entry_id,
        "new_confidence": args.confidence,
        "message": f"Confidence updated for {args.entry_id}",
    }


def _run_trigger_learning(args: argparse.Namespace) -> dict:
    """Execute the ``trigger-learning`` subcommand."""
    engine = _get_engine(args)

    result = engine.trigger_learning(
        time_range=(args.start_time, args.end_time)
    )

    return {
        "status": "success",
        **result,
        "message": (
            f"Learning triggered: analyzed {result.get('feedbacks_analyzed', 0)} "
            f"feedbacks, updated {result.get('strategies_updated', 0)} strategies"
        ),
    }


def _run_lifecycle(args: argparse.Namespace) -> dict:
    """Execute the ``lifecycle`` subcommand."""
    kb = _get_kb(args)

    action = getattr(args, "lifecycle_action", "status")

    if action == "transition":
        # Manually trigger lifecycle state transitions
        transitions = kb.apply_lifecycle_transitions()
        return {
            "status": "success",
            "action": "transition",
            "transitions": transitions,
            "message": (
                f"Lifecycle transitions applied: "
                f"{transitions['active_to_stale']} active→stale, "
                f"{transitions['stale_to_archived']} stale→archived, "
                f"{transitions['stale_to_active']} stale→active"
            ),
        }
    else:
        # Show status: count entries in each state
        db = kb.db
        all_entries = db.get_all("knowledge_entries")
        counts = {"active": 0, "stale": 0, "archived": 0}
        for row in all_entries:
            state = row.get("state", "active") or "active"
            if state in counts:
                counts[state] += 1

        return {
            "status": "success",
            "action": "status",
            "total_entries": len(all_entries),
            "state_counts": counts,
        }


def main(argv: list[str] | None = None) -> None:
    """Entry point for the manage_knowledge CLI."""
    parser = argparse.ArgumentParser(
        description="Manage the Skill Knowledge Base for the ASR Training Optimizer."
    )
    parser.add_argument(
        "--workspace",
        default=_DEFAULT_WORKSPACE,
        help=f"Workspace root directory (default: {_DEFAULT_WORKSPACE}).",
    )
    parser.add_argument(
        "--db-path",
        default=_DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {_DEFAULT_DB_PATH}).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- add ---
    add_parser = subparsers.add_parser(
        "add", help="Add a knowledge entry."
    )
    add_parser.add_argument(
        "--type", required=True,
        choices=["scene_solution", "param_effect", "data_strategy"],
        help="Knowledge type.",
    )
    add_parser.add_argument(
        "--content", required=True,
        help="Knowledge content as JSON string.",
    )
    add_parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Confidence score (0.0-1.0, default: 0.5).",
    )
    add_parser.add_argument(
        "--source-iteration", default="",
        help="Source iteration ID (optional).",
    )

    # --- query ---
    query_parser = subparsers.add_parser(
        "query", help="Query knowledge entries."
    )
    query_parser.add_argument(
        "--type", required=True,
        choices=["scene_solution", "param_effect", "data_strategy"],
        help="Knowledge type to query.",
    )
    query_parser.add_argument(
        "--source-iteration", default=None,
        help="Filter by source iteration ID (optional).",
    )
    query_parser.add_argument(
        "--content-match", default=None,
        help="Filter by content key-value pairs as JSON string (optional).",
    )
    query_parser.add_argument(
        "--include-stale", action="store_true", default=False,
        help="Include entries with state 'stale' (default: only active).",
    )
    query_parser.add_argument(
        "--include-archived", action="store_true", default=False,
        help="Include entries with state 'archived' (default: only active).",
    )

    # --- view ---
    subparsers.add_parser(
        "view", help="Show structured view of all knowledge entries."
    )

    # --- update-confidence ---
    update_parser = subparsers.add_parser(
        "update-confidence", help="Update confidence of a knowledge entry."
    )
    update_parser.add_argument(
        "--entry-id", required=True, help="Knowledge entry ID."
    )
    update_parser.add_argument(
        "--confidence", type=float, required=True,
        help="New confidence score (0.0-1.0).",
    )

    # --- trigger-learning ---
    learn_parser = subparsers.add_parser(
        "trigger-learning", help="Trigger Skill learning from feedback history."
    )
    learn_parser.add_argument(
        "--start-time", required=True,
        help="Start of time range (ISO format).",
    )
    learn_parser.add_argument(
        "--end-time", required=True,
        help="End of time range (ISO format).",
    )

    # --- lifecycle ---
    lifecycle_parser = subparsers.add_parser(
        "lifecycle", help="Manage knowledge entry lifecycle states."
    )
    lifecycle_parser.add_argument(
        "lifecycle_action",
        nargs="?",
        default="status",
        choices=["status", "transition"],
        help="Action: 'status' to view state counts, 'transition' to trigger state transitions (default: status).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "add": _run_add,
        "query": _run_query,
        "view": _run_view,
        "update-confidence": _run_update_confidence,
        "trigger-learning": _run_trigger_learning,
        "lifecycle": _run_lifecycle,
    }

    result = handlers[args.command](args)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
