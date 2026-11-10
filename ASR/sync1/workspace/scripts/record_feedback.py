"""CLI script for recording optimization effect feedback.

Usage:
    python scripts/record_feedback.py record \\
        --iteration-id iter_001 \\
        --suggestion-id sug_001 \\
        --suggestion-content "Increase attention heads" \\
        --cer-before 0.15 --cer-after 0.12 \\
        --wer-before 0.22 --wer-after 0.18

    python scripts/record_feedback.py label \\
        --iteration-id iter_001 \\
        --cer-before 0.15 --cer-after 0.12

    python scripts/record_feedback.py list [--iteration-id iter_001]

Records optimization effect feedback via the SkillEvolutionEngine,
supporting explicit feedback recording, automatic labeling based on
CER changes, and listing stored feedbacks.  Outputs structured JSON
to stdout.

Requirements: 10.1
"""

import argparse
import json
import sys
from dataclasses import asdict

from scripts.core.database import Database
from scripts.core.models import EffectFeedback
from scripts.decision.skill_evolution import SkillEvolutionEngine


_DEFAULT_DB_PATH = "metadata.db"
_DEFAULT_WORKSPACE = "."


def _get_engine(args: argparse.Namespace) -> SkillEvolutionEngine:
    """Create a SkillEvolutionEngine from CLI arguments."""
    workspace_dir = getattr(args, "workspace", _DEFAULT_WORKSPACE)
    db_path = getattr(args, "db_path", _DEFAULT_DB_PATH)
    db = Database(db_path)
    db.initialize()
    return SkillEvolutionEngine(workspace_dir=workspace_dir, db=db)


def _run_record(args: argparse.Namespace) -> dict:
    """Execute the ``record`` subcommand."""
    engine = _get_engine(args)

    input_analysis = {}
    if args.input_analysis:
        try:
            input_analysis = json.loads(args.input_analysis)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": "Invalid JSON for --input-analysis",
            }

    param_changes = {}
    if args.param_changes:
        try:
            param_changes = json.loads(args.param_changes)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": "Invalid JSON for --param-changes",
            }

    is_positive = args.cer_after < args.cer_before

    feedback = EffectFeedback(
        iteration_id=args.iteration_id,
        suggestion_id=args.suggestion_id or "",
        suggestion_content=args.suggestion_content or "",
        input_analysis=input_analysis,
        param_changes=param_changes,
        cer_before=args.cer_before,
        cer_after=args.cer_after,
        wer_before=args.wer_before,
        wer_after=args.wer_after,
        is_positive=is_positive,
    )

    engine.record_feedback(feedback)

    return {
        "status": "success",
        "iteration_id": args.iteration_id,
        "is_positive": is_positive,
        "cer_change": args.cer_after - args.cer_before,
        "wer_change": args.wer_after - args.wer_before,
        "message": "Feedback recorded successfully",
    }


def _run_label(args: argparse.Namespace) -> dict:
    """Execute the ``label`` subcommand."""
    engine = _get_engine(args)

    feedback = engine.label_feedback(
        iteration_id=args.iteration_id,
        cer_before=args.cer_before,
        cer_after=args.cer_after,
    )

    # Also persist the labeled feedback
    engine.record_feedback(feedback)

    return {
        "status": "success",
        "iteration_id": args.iteration_id,
        "is_positive": feedback.is_positive,
        "cer_before": feedback.cer_before,
        "cer_after": feedback.cer_after,
        "wer_before": feedback.wer_before,
        "wer_after": feedback.wer_after,
        "suggestion_id": feedback.suggestion_id,
        "message": (
            "Feedback labeled as positive (CER improved)"
            if feedback.is_positive
            else "Feedback labeled as negative (CER did not improve)"
        ),
    }


def _run_list(args: argparse.Namespace) -> dict:
    """Execute the ``list`` subcommand."""
    engine = _get_engine(args)

    filters = None
    if args.iteration_id:
        filters = {"iteration_id": args.iteration_id}

    rows = engine.db.get_all(
        "effect_feedbacks",
        filters=filters,
        order_by="created_at DESC",
    )

    feedbacks = []
    for row in rows:
        feedbacks.append({
            "feedback_id": row["feedback_id"],
            "iteration_id": row["iteration_id"],
            "suggestion_id": row.get("suggestion_id", ""),
            "suggestion_content": row.get("suggestion_content", ""),
            "cer_before": row.get("cer_before", 0.0),
            "cer_after": row.get("cer_after", 0.0),
            "wer_before": row.get("wer_before", 0.0),
            "wer_after": row.get("wer_after", 0.0),
            "is_positive": bool(row.get("is_positive", 0)),
            "created_at": row.get("created_at", ""),
        })

    return {
        "status": "success",
        "total_feedbacks": len(feedbacks),
        "feedbacks": feedbacks,
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point for the record_feedback CLI."""
    parser = argparse.ArgumentParser(
        description="Record and manage optimization effect feedback."
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

    # --- record ---
    record_parser = subparsers.add_parser(
        "record", help="Record an optimization effect feedback."
    )
    record_parser.add_argument(
        "--iteration-id", required=True, help="Iteration ID."
    )
    record_parser.add_argument(
        "--suggestion-id", default="", help="Suggestion ID."
    )
    record_parser.add_argument(
        "--suggestion-content", default="", help="Suggestion description."
    )
    record_parser.add_argument(
        "--input-analysis", default=None,
        help="Input analysis data as JSON string.",
    )
    record_parser.add_argument(
        "--param-changes", default=None,
        help="Parameter changes as JSON string.",
    )
    record_parser.add_argument(
        "--cer-before", type=float, required=True, help="CER before optimization."
    )
    record_parser.add_argument(
        "--cer-after", type=float, required=True, help="CER after optimization."
    )
    record_parser.add_argument(
        "--wer-before", type=float, default=0.0, help="WER before optimization."
    )
    record_parser.add_argument(
        "--wer-after", type=float, default=0.0, help="WER after optimization."
    )

    # --- label ---
    label_parser = subparsers.add_parser(
        "label", help="Auto-label feedback for an iteration based on CER change."
    )
    label_parser.add_argument(
        "--iteration-id", required=True, help="Iteration ID to label."
    )
    label_parser.add_argument(
        "--cer-before", type=float, required=True, help="CER before optimization."
    )
    label_parser.add_argument(
        "--cer-after", type=float, required=True, help="CER after optimization."
    )

    # --- list ---
    list_parser = subparsers.add_parser(
        "list", help="List recorded feedbacks."
    )
    list_parser.add_argument(
        "--iteration-id", default=None,
        help="Filter by iteration ID (optional).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "record": _run_record,
        "label": _run_label,
        "list": _run_list,
    }

    result = handlers[args.command](args)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
