"""CLI script for applying post-processing rules and evaluating their effect.

Usage:
    python scripts/apply_postprocess.py apply --text "今天天气很号"
    python scripts/apply_postprocess.py apply --text "今天天气很号" --hotword-file configs/post_process/hotwords.txt
    python scripts/apply_postprocess.py apply --input-file data/texts.txt --reference-file data/refs.txt
    python scripts/apply_postprocess.py apply --config-file configs/post_process/config.yaml --text "测试文本"
    python scripts/apply_postprocess.py evaluate --input-file data/texts.txt --reference-file data/refs.txt
    python scripts/apply_postprocess.py show-config --config-file configs/post_process/config.yaml

Manages post-processing via the PostProcessor, with hotword correction,
regex replacement, and LM reranking.  Outputs structured JSON to stdout.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""

import argparse
import json
import sys
from dataclasses import asdict
from typing import Optional

from scripts.core.config_utils import load_config, read_post_process_config, write_yaml
from scripts.core.models import PostProcessConfig
from scripts.execution.post_processor import PostProcessor

_MAIN_CONFIG = "configs/funasr_config.yaml"


def _sync_postprocess_to_main_config(config: PostProcessConfig) -> None:
    """Write post_process params back to funasr_config.yaml."""
    from dataclasses import asdict as _asdict
    main = load_config(_MAIN_CONFIG)
    pp_data = _asdict(config)
    # Convert HotwordEntry/RegexRule dataclasses to plain dicts
    pp_data["hotword_entries"] = [_asdict(e) for e in config.hotword_entries]
    pp_data["regex_rules"] = [_asdict(r) for r in config.regex_rules]
    main["post_process"] = pp_data
    write_yaml(_MAIN_CONFIG, main)


def _load_config(config_file: Optional[str]) -> PostProcessConfig:
    """Load a PostProcessConfig from *config_file*, main config, or return defaults."""
    if config_file:
        return read_post_process_config(config_file)

    # Try loading from main config
    from scripts.core.models import HotwordEntry, RegexRule
    main = load_config(_MAIN_CONFIG)
    pp_cfg = main.get("post_process", {})
    if pp_cfg:
        hotword_entries = [
            HotwordEntry(**e) for e in pp_cfg.get("hotword_entries", [])
            if isinstance(e, dict)
        ]
        regex_rules = [
            RegexRule(**r) for r in pp_cfg.get("regex_rules", [])
            if isinstance(r, dict)
        ]
        return PostProcessConfig(
            hotword_entries=hotword_entries,
            regex_rules=regex_rules,
            enable_lm_rerank=pp_cfg.get("enable_lm_rerank", False),
            lm_model_path=pp_cfg.get("lm_model_path"),
            pipeline_order=pp_cfg.get("pipeline_order", ["hotword", "regex", "lm_rerank"]),
        )

    return PostProcessConfig()


def _read_lines(path: str) -> list[str]:
    """Read non-empty lines from a text file."""
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _run_apply(args: argparse.Namespace) -> dict:
    """Execute the ``apply`` subcommand."""
    config = _load_config(args.config_file)
    processor = PostProcessor(
        config=config,
        hotword_path=args.hotword_file,
    )

    # Collect input texts
    texts: list[str] = []
    if args.text:
        texts.append(args.text)
    if args.input_file:
        texts.extend(_read_lines(args.input_file))

    if not texts:
        return {
            "status": "error",
            "message": "No input text provided. Use --text or --input-file.",
        }

    # Collect reference texts (optional)
    references: list[str] = []
    if args.reference_file:
        references = _read_lines(args.reference_file)

    results: list[dict] = []
    for i, text in enumerate(texts):
        ref = references[i] if i < len(references) else ""
        result = processor.process(text=text, reference_text=ref)
        results.append(asdict(result))

    return {
        "status": "success",
        "total": len(results),
        "results": results,
    }


def _run_evaluate(args: argparse.Namespace) -> dict:
    """Execute the ``evaluate`` subcommand."""
    config = _load_config(args.config_file)
    processor = PostProcessor(
        config=config,
        hotword_path=args.hotword_file,
    )

    if not args.input_file:
        return {
            "status": "error",
            "message": "Missing --input-file for evaluation.",
        }
    if not args.reference_file:
        return {
            "status": "error",
            "message": "Missing --reference-file for evaluation.",
        }

    texts = _read_lines(args.input_file)
    references = _read_lines(args.reference_file)

    if len(texts) != len(references):
        return {
            "status": "error",
            "message": (
                f"Input lines ({len(texts)}) and reference lines "
                f"({len(references)}) count mismatch."
            ),
        }

    from scripts.core.models import PostProcessResult

    results = []
    for text, ref in zip(texts, references):
        result = processor.process(text=text, reference_text=ref)
        results.append(result)

    effect = processor.evaluate_effect(results)

    return {
        "status": "success",
        "evaluation": effect,
        "sample_results": [asdict(r) for r in results],
    }


def _run_update_config(args: argparse.Namespace) -> dict:
    """Execute the ``update-config`` subcommand.

    Updates the post_process section in funasr_config.yaml and saves
    a versioned snapshot.
    """
    from scripts.core.config_utils import save_versioned_config

    main = load_config(_MAIN_CONFIG)
    pp = main.setdefault("post_process", {})
    changes = []

    if args.hotword_file:
        pp["hotword_file"] = args.hotword_file
        changes.append("hotword_file")
    if args.enable_lm_rerank is not None:
        pp["enable_lm_rerank"] = args.enable_lm_rerank
        changes.append("enable_lm_rerank")
    if args.lm_model_path is not None:
        pp["lm_model_path"] = args.lm_model_path
        changes.append("lm_model_path")
    if args.add_regex:
        import re as _re
        parts = args.add_regex.split("->", 1)
        if len(parts) == 2:
            rule = {"pattern": parts[0].strip(), "replacement": parts[1].strip(),
                    "description": "added via CLI", "priority": 0}
            pp.setdefault("regex_rules", []).append(rule)
            changes.append("regex_rules")

    if not changes:
        return {"status": "noop", "message": "No changes specified."}

    write_yaml(_MAIN_CONFIG, main)
    save_versioned_config("configs/post_process", "postprocess", pp)

    return {
        "status": "success",
        "message": f"Updated: {', '.join(changes)}",
        "post_process": pp,
    }


def _run_show_config(args: argparse.Namespace) -> dict:
    """Execute the ``show-config`` subcommand."""
    if args.config_file:
        try:
            config = read_post_process_config(args.config_file)
        except FileNotFoundError:
            return {"status": "error", "message": f"Config file not found: {args.config_file}"}
        except Exception as exc:
            return {"status": "error", "message": f"Failed to read config: {exc}"}
        return {"status": "success", "config": asdict(config)}

    # Fall back to main config
    main = load_config(_MAIN_CONFIG)
    pp = main.get("post_process", {})
    return {"status": "success", "config": pp if pp else asdict(PostProcessConfig())}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the apply_postprocess CLI."""
    parser = argparse.ArgumentParser(
        description="Apply post-processing rules and evaluate their effect."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- apply ---
    apply_parser = subparsers.add_parser(
        "apply", help="Apply post-processing to input text(s)."
    )
    apply_parser.add_argument(
        "--config-file",
        default=None,
        help="Path to a PostProcessConfig YAML file.",
    )
    apply_parser.add_argument(
        "--hotword-file",
        default=None,
        help="Path to a hotword text file (tab-separated).",
    )
    apply_parser.add_argument(
        "--text",
        default=None,
        help="A single input text to process.",
    )
    apply_parser.add_argument(
        "--input-file",
        default=None,
        help="Path to a file with one input text per line.",
    )
    apply_parser.add_argument(
        "--reference-file",
        default=None,
        help="Path to a file with one reference text per line.",
    )

    # --- evaluate ---
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate post-processing effect on a set of texts."
    )
    eval_parser.add_argument(
        "--config-file",
        default=None,
        help="Path to a PostProcessConfig YAML file.",
    )
    eval_parser.add_argument(
        "--hotword-file",
        default=None,
        help="Path to a hotword text file (tab-separated).",
    )
    eval_parser.add_argument(
        "--input-file",
        default=None,
        help="Path to a file with one input text per line.",
    )
    eval_parser.add_argument(
        "--reference-file",
        default=None,
        help="Path to a file with one reference text per line.",
    )

    # --- show-config ---
    show_parser = subparsers.add_parser(
        "show-config", help="Show current post-process configuration."
    )
    show_parser.add_argument(
        "--config-file",
        default=None,
        help="Path to a PostProcessConfig YAML file.",
    )

    # --- update-config ---
    update_parser = subparsers.add_parser(
        "update-config", help="Update post-process config in funasr_config.yaml."
    )
    update_parser.add_argument(
        "--hotword-file", default=None,
        help="Set the hotword file path.",
    )
    update_parser.add_argument(
        "--enable-lm-rerank", type=bool, default=None,
        help="Enable/disable LM reranking.",
    )
    update_parser.add_argument(
        "--lm-model-path", default=None,
        help="Set the LM model path.",
    )
    update_parser.add_argument(
        "--add-regex", default=None,
        help="Add a regex rule: 'pattern->replacement'.",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "apply": _run_apply,
        "evaluate": _run_evaluate,
        "show-config": _run_show_config,
        "update-config": _run_update_config,
    }

    result = handlers[args.command](args)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
