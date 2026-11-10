"""CLI script for managing optimization iterations.

Usage:
    python scripts/manage_iterations.py start --description "Optimize background voice CER"
    python scripts/manage_iterations.py history
    python scripts/manage_iterations.py compare --iter-a iter_abc123 --iter-b iter_def456
    python scripts/manage_iterations.py rollback --iteration-id iter_abc123
    python scripts/manage_iterations.py show --iteration-id iter_abc123
    python scripts/manage_iterations.py plateau [--window 3]

Manages optimization iteration lifecycle via the OptimizerSystem, including
starting new iterations, viewing history, comparing iterations, rolling
back, and detecting optimization plateaus.  Outputs structured JSON to
stdout.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import argparse
import json
import sys
from dataclasses import asdict

from scripts.core.database import Database
from scripts.core.models import IterationRecord
from scripts.decision.optimizer_system import OptimizerSystem


_DEFAULT_DB_PATH = "metadata.db"
_DEFAULT_WORKSPACE = "."


def _get_optimizer(args: argparse.Namespace) -> OptimizerSystem:
    """Create an OptimizerSystem from CLI arguments."""
    workspace_dir = getattr(args, "workspace", _DEFAULT_WORKSPACE)
    db_path = getattr(args, "db_path", _DEFAULT_DB_PATH)
    db = Database(db_path)
    db.initialize()
    return OptimizerSystem(workspace_dir=workspace_dir, db=db)


def _iteration_record_to_dict(record: IterationRecord) -> dict:
    """Convert an IterationRecord to a JSON-serializable dict."""
    data = asdict(record)
    # Simplify evaluation reports for display
    if record.evaluation_before is not None:
        data["evaluation_before"] = {
            "overall_cer": record.evaluation_before.overall_cer,
            "overall_wer": record.evaluation_before.overall_wer,
        }
    if record.evaluation_after is not None:
        data["evaluation_after"] = {
            "overall_cer": record.evaluation_after.overall_cer,
            "overall_wer": record.evaluation_after.overall_wer,
        }
    else:
        data["evaluation_after"] = None
    return data


def _run_start(args: argparse.Namespace) -> dict:
    """Execute the ``start`` subcommand."""
    optimizer = _get_optimizer(args)
    description = args.description or ""
    iteration_id = optimizer.start_iteration(description)
    return {
        "status": "success",
        "iteration_id": iteration_id,
        "description": description,
        "message": f"Started new iteration: {iteration_id}",
    }


def _run_history(args: argparse.Namespace) -> dict:
    """Execute the ``history`` subcommand."""
    optimizer = _get_optimizer(args)
    history = optimizer.get_iteration_history()

    entries = [_iteration_record_to_dict(r) for r in history]

    return {
        "status": "success",
        "total_iterations": len(entries),
        "iterations": entries,
    }


def _run_compare(args: argparse.Namespace) -> dict:
    """Execute the ``compare`` subcommand."""
    optimizer = _get_optimizer(args)
    return optimizer.compare_iterations(args.iter_a, args.iter_b)


def _run_rollback(args: argparse.Namespace) -> dict:
    """Execute the ``rollback`` subcommand."""
    optimizer = _get_optimizer(args)
    return optimizer.rollback_to_iteration(args.iteration_id)


def _run_show(args: argparse.Namespace) -> dict:
    """Execute the ``show`` subcommand."""
    optimizer = _get_optimizer(args)
    row = optimizer.db.get_by_id(
        "iterations", "iteration_id", args.iteration_id
    )
    if row is None:
        return {
            "status": "error",
            "message": f"Iteration {args.iteration_id} not found",
        }

    return {
        "status": "success",
        "iteration": dict(row),
    }


def _run_plateau(args: argparse.Namespace) -> dict:
    """Execute the ``plateau`` subcommand."""
    optimizer = _get_optimizer(args)
    window = args.window
    is_plateau = optimizer.check_optimization_plateau(window=window)

    return {
        "status": "success",
        "is_plateau": is_plateau,
        "window": window,
        "message": (
            f"Optimization plateau detected (no CER improvement in last "
            f"{window} iterations)"
            if is_plateau
            else f"No plateau detected (checked last {window} iterations)"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    """Entry point for the manage_iterations CLI."""
    parser = argparse.ArgumentParser(
        description="Manage optimization iterations for the ASR Training Optimizer."
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

    # --- start ---
    start_parser = subparsers.add_parser(
        "start", help="Start a new optimization iteration."
    )
    start_parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of the iteration's purpose.",
    )

    # --- history ---
    subparsers.add_parser(
        "history", help="Show all iteration history."
    )

    # --- compare ---
    compare_parser = subparsers.add_parser(
        "compare", help="Compare two iterations."
    )
    compare_parser.add_argument(
        "--iter-a",
        required=True,
        help="First iteration ID.",
    )
    compare_parser.add_argument(
        "--iter-b",
        required=True,
        help="Second iteration ID.",
    )

    # --- rollback ---
    rollback_parser = subparsers.add_parser(
        "rollback", help="Roll back to a specific iteration."
    )
    rollback_parser.add_argument(
        "--iteration-id",
        required=True,
        help="Iteration ID to roll back to.",
    )

    # --- show ---
    show_parser = subparsers.add_parser(
        "show", help="Show details of a specific iteration."
    )
    show_parser.add_argument(
        "--iteration-id",
        required=True,
        help="Iteration ID to display.",
    )

    # --- plateau ---
    plateau_parser = subparsers.add_parser(
        "plateau", help="Check for optimization plateau."
    )
    plateau_parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Number of recent iterations to check (default: 3).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "start": _run_start,
        "history": _run_history,
        "compare": _run_compare,
        "rollback": _run_rollback,
        "show": _run_show,
        "plateau": _run_plateau,
    }

    result = handlers[args.command](args)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
