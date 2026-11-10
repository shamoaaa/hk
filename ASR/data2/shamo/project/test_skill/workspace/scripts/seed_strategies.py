"""CLI for seeding the knowledge base with badcase→strategy mappings.

一次性注入默认的"前置数据优化手段"映射表到知识库。这些条目作为 seed，
后续随着迭代的 effect_feedback 自动调整置信度（参见 Skill Evolution）。

Usage:
    python scripts/seed_strategies.py
    python scripts/seed_strategies.py --list    # 只列出不写入
"""

import argparse
import json
import sys

from scripts.decision.badcase_strategy_map import (
    DATA_STRATEGIES,
    list_all_strategies,
    seed_knowledge_base,
)
from scripts.core.database import Database
from scripts.decision.skill_knowledge_base import SkillKnowledgeBase


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Seed the knowledge base with badcase→strategy mappings."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Only list strategies; don't write to knowledge base.",
    )
    parser.add_argument(
        "--db-path", default="metadata.db",
        help="SQLite database path (default: metadata.db).",
    )
    args = parser.parse_args(argv)

    if args.list:
        json.dump({
            "total_strategies": len(DATA_STRATEGIES),
            "strategies": list_all_strategies(),
        }, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return

    db = Database(args.db_path)
    db.initialize()
    kb = SkillKnowledgeBase(db=db)
    n = seed_knowledge_base(kb)

    json.dump({
        "status": "success",
        "seeded_count": n,
        "total_strategies": len(DATA_STRATEGIES),
    }, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
