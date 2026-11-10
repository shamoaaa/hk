"""Skill Evolution Engine for the ASR Training Optimizer.

Provides the ``SkillEvolutionEngine`` class which manages the self-learning
and continuous improvement of LLM Skill capabilities.  This module implements
effect feedback recording, feedback labeling, pattern analysis, strategy
weight updates, Skill file management, and version control.

Requirements: 10.1, 10.2, 10.3, 10.4
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

import shutil

from scripts.core.database import Database
from scripts.core.models import EffectFeedback, KnowledgeEntry, SkillVersion, StrategyWeight


class SkillEvolutionEngine:
    """Skill self-evolution engine managing learning and improvement.

    Records optimization effect feedback, labels feedback as positive or
    negative based on CER changes, and provides the foundation for
    pattern analysis and strategy evolution.

    Parameters
    ----------
    workspace_dir : str
        Root workspace directory containing ``configs/``, ``knowledge/``,
        ``scripts/``, and other project directories.
    db : Database
        Initialized database instance for persisting feedback records,
        knowledge entries, skill versions, and strategy weights.
    """

    def __init__(self, workspace_dir: str, db: Database) -> None:
        self.workspace_dir = workspace_dir
        self.db = db

        # Derived paths
        self.skills_dir = os.path.join(workspace_dir, ".kiro", "skills")
        self.knowledge_dir = os.path.join(workspace_dir, "knowledge")
        self.skill_versions_dir = os.path.join(
            self.knowledge_dir, "skill_versions"
        )
        self.configs_dir = os.path.join(workspace_dir, "configs")

    # ------------------------------------------------------------------
    # Feedback recording
    # ------------------------------------------------------------------

    def record_feedback(self, feedback: EffectFeedback) -> None:
        """Record an optimization effect feedback to the database.

        Generates a unique ``feedback_id``, serializes the
        ``input_analysis`` and ``param_changes`` dicts as JSON, and
        stores the complete feedback record in the ``effect_feedbacks``
        table.

        Parameters
        ----------
        feedback : EffectFeedback
            The feedback record to persist.  All fields should be
            populated by the caller except that a ``feedback_id`` will
            be generated automatically.
        """
        feedback_id = f"fb_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        self.db.insert("effect_feedbacks", {
            "feedback_id": feedback_id,
            "iteration_id": feedback.iteration_id,
            "suggestion_id": feedback.suggestion_id,
            "suggestion_content": feedback.suggestion_content,
            "input_analysis_json": json.dumps(
                feedback.input_analysis, ensure_ascii=False
            ),
            "param_changes_json": json.dumps(
                feedback.param_changes, ensure_ascii=False
            ),
            "cer_before": feedback.cer_before,
            "cer_after": feedback.cer_after,
            "wer_before": feedback.wer_before,
            "wer_after": feedback.wer_after,
            "is_positive": 1 if feedback.is_positive else 0,
            "created_at": now,
        })

    # ------------------------------------------------------------------
    # Feedback labeling
    # ------------------------------------------------------------------

    def label_feedback(
        self,
        iteration_id: str,
        cer_before: float,
        cer_after: float,
    ) -> EffectFeedback:
        """Create a labeled ``EffectFeedback`` for an iteration.

        Looks up the iteration in the database to retrieve context
        (description, param changes), then creates an ``EffectFeedback``
        with ``is_positive`` set to ``True`` when ``cer_after < cer_before``
        (i.e. CER improved) and ``False`` otherwise.

        Parameters
        ----------
        iteration_id : str
            The iteration to label feedback for.
        cer_before : float
            CER value before the optimization.
        cer_after : float
            CER value after the optimization.

        Returns
        -------
        EffectFeedback
            A fully populated feedback record with the ``is_positive``
            flag set based on CER comparison.
        """
        is_positive = cer_after < cer_before

        # Look up iteration context from the database
        row = self.db.get_by_id("iterations", "iteration_id", iteration_id)

        suggestion_content = ""
        input_analysis: dict = {}
        param_changes: dict = {}
        wer_before = 0.0
        wer_after = 0.0

        if row is not None:
            suggestion_content = row.get("description", "")
            param_changes = _safe_json_loads(
                row.get("param_changes_json", "{}")
            )
            input_analysis = _safe_json_loads(
                row.get("model_changes_json", "{}")
            )
            wer_before = row.get("wer_before") or 0.0
            wer_after = row.get("wer_after") or 0.0

        now = datetime.now(timezone.utc).isoformat()

        feedback = EffectFeedback(
            iteration_id=iteration_id,
            suggestion_id=f"sug_{uuid.uuid4().hex[:8]}",
            suggestion_content=suggestion_content,
            input_analysis=input_analysis,
            param_changes=param_changes,
            cer_before=cer_before,
            cer_after=cer_after,
            wer_before=wer_before,
            wer_after=wer_after,
            is_positive=is_positive,
            timestamp=now,
        )

        return feedback

    # ------------------------------------------------------------------
    # Pattern analysis
    # ------------------------------------------------------------------

    def analyze_patterns(self, min_feedback_count: int) -> dict:
        """Analyze historical feedback patterns to extract strategy insights.

        Queries all ``effect_feedbacks`` from the database, groups them
        by ``suggestion_content`` (the strategy dimension), and computes
        success metrics for each strategy.

        Parameters
        ----------
        min_feedback_count : int
            Minimum number of total feedbacks required before analysis
            is performed.  If the database contains fewer feedbacks,
            an empty analysis dict is returned.

        Returns
        -------
        dict
            A dict with keys:

            - ``total_feedbacks`` – total number of feedback records
            - ``strategies`` – dict mapping strategy name to its stats
            - ``high_success_strategies`` – list of strategy names with
              success_rate above the average
            - ``low_success_strategies`` – list of strategy names with
              success_rate at or below the average
            - ``average_success_rate`` – the overall average success rate
        """
        rows = self.db.get_all("effect_feedbacks")
        total = len(rows)

        if total < min_feedback_count:
            return {}

        # Group feedbacks by suggestion_content (strategy name)
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            strategy_name = row.get("suggestion_content", "") or ""
            groups[strategy_name].append(row)

        strategies: dict[str, dict] = {}
        for name, feedbacks in groups.items():
            total_count = len(feedbacks)
            positive_count = sum(
                1 for fb in feedbacks if fb.get("is_positive")
            )
            success_rate = positive_count / total_count if total_count > 0 else 0.0

            # Average CER improvement for positive feedbacks
            positive_improvements = [
                (fb.get("cer_before", 0.0) or 0.0) - (fb.get("cer_after", 0.0) or 0.0)
                for fb in feedbacks
                if fb.get("is_positive")
            ]
            avg_cer_improvement = (
                sum(positive_improvements) / len(positive_improvements)
                if positive_improvements
                else 0.0
            )

            strategies[name] = {
                "total_count": total_count,
                "positive_count": positive_count,
                "success_rate": success_rate,
                "avg_cer_improvement": avg_cer_improvement,
            }

        # Compute average success rate across all strategies
        if strategies:
            average_success_rate = (
                sum(s["success_rate"] for s in strategies.values())
                / len(strategies)
            )
        else:
            average_success_rate = 0.0

        high_success: list[str] = []
        low_success: list[str] = []
        for name, stats in strategies.items():
            if stats["success_rate"] > average_success_rate:
                high_success.append(name)
            else:
                low_success.append(name)

        return {
            "total_feedbacks": total,
            "strategies": strategies,
            "high_success_strategies": high_success,
            "low_success_strategies": low_success,
            "average_success_rate": average_success_rate,
        }

    # ------------------------------------------------------------------
    # Strategy weight updates
    # ------------------------------------------------------------------

    def update_strategy_weights(
        self, pattern_analysis: dict
    ) -> list[StrategyWeight]:
        """Update strategy recommendation weights based on pattern analysis.

        For each strategy in *pattern_analysis*, computes a weight
        proportional to its success rate.  High-success strategies
        receive higher weights; low-success strategies receive lower
        weights.  Weights are persisted to the ``strategy_weights``
        table using an upsert pattern.

        Parameters
        ----------
        pattern_analysis : dict
            The output of :meth:`analyze_patterns`.  If empty or missing
            the ``strategies`` key, an empty list is returned.

        Returns
        -------
        list[StrategyWeight]
            The list of persisted ``StrategyWeight`` objects.
        """
        strategies = pattern_analysis.get("strategies")
        if not strategies:
            return []

        average_success_rate = pattern_analysis.get("average_success_rate", 0.0)
        now = datetime.now(timezone.utc).isoformat()

        result: list[StrategyWeight] = []
        for name, stats in strategies.items():
            success_rate = stats["success_rate"]
            # Weight formula: base weight scaled by success rate relative
            # to the average.  Ensures high-success >= low-success.
            if average_success_rate > 0:
                weight = success_rate / average_success_rate
            else:
                weight = 1.0
            # Clamp weight to [0.1, 2.0] for practical use
            weight = max(0.1, min(2.0, weight))

            sw = StrategyWeight(
                strategy_name=name,
                weight=weight,
                success_rate=success_rate,
                sample_count=stats["total_count"],
            )
            result.append(sw)

            # Upsert into strategy_weights table
            existing = self.db.get_by_id(
                "strategy_weights", "strategy_name", name
            )
            if existing:
                self.db.update(
                    "strategy_weights",
                    "strategy_name",
                    name,
                    {
                        "weight": weight,
                        "success_rate": success_rate,
                        "sample_count": stats["total_count"],
                        "updated_at": now,
                    },
                )
            else:
                self.db.insert("strategy_weights", {
                    "strategy_name": name,
                    "weight": weight,
                    "success_rate": success_rate,
                    "sample_count": stats["total_count"],
                    "version_id": None,
                    "updated_at": now,
                })

        return result

    # ------------------------------------------------------------------
    # Skill file updates and version management
    # ------------------------------------------------------------------

    def update_skill_files(
        self, strategy_weights: list[StrategyWeight]
    ) -> dict:
        """Update Skill files with current strategy weights.

        Reads all ``.md`` files from :attr:`skills_dir` and appends or
        updates a "Strategy Weights" section at the bottom of each file
        with the current strategy weights sorted by weight descending.

        Parameters
        ----------
        strategy_weights : list[StrategyWeight]
            The strategy weights to write into each Skill file.

        Returns
        -------
        dict
            A summary with keys ``updated_files`` (list of updated file
            paths) and ``strategy_count`` (number of strategies written).
        """
        if not os.path.isdir(self.skills_dir):
            return {"updated_files": [], "strategy_count": 0}

        # Sort strategies by weight descending
        sorted_weights = sorted(
            strategy_weights, key=lambda sw: sw.weight, reverse=True
        )

        # Build the strategy weights section content
        lines = ["\n## Strategy Weights\n"]
        for sw in sorted_weights:
            lines.append(
                f"- **{sw.strategy_name}**: weight={sw.weight:.4f}, "
                f"success_rate={sw.success_rate:.4f}"
            )
        section_content = "\n".join(lines) + "\n"

        updated_files: list[str] = []
        for filename in sorted(os.listdir(self.skills_dir)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(self.skills_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove existing Strategy Weights section if present
            marker = "## Strategy Weights"
            idx = content.find(marker)
            if idx != -1:
                content = content[:idx].rstrip("\n")

            # Append the new section
            content = content.rstrip("\n") + "\n" + section_content

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            updated_files.append(filepath)

        return {
            "updated_files": updated_files,
            "strategy_count": len(sorted_weights),
        }

    def create_version_snapshot(
        self, change_description: str
    ) -> SkillVersion:
        """Create a Skill version snapshot.

        Captures the current strategy rules, knowledge base state, and
        copies all Skill files to a versioned directory under
        ``knowledge/skill_versions/{version_id}/``.

        Parameters
        ----------
        change_description : str
            Human-readable description of what changed in this version.

        Returns
        -------
        SkillVersion
            The persisted version record.
        """
        version_id = f"sv_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        # Capture current strategy rules from strategy_weights table
        strategy_rows = self.db.get_all("strategy_weights")
        strategy_rules: dict[str, dict] = {}
        for row in strategy_rows:
            strategy_rules[row["strategy_name"]] = {
                "weight": row["weight"],
                "success_rate": row["success_rate"],
                "sample_count": row["sample_count"],
            }

        # Capture knowledge base state from knowledge_entries table
        knowledge_rows = self.db.get_all("knowledge_entries")
        knowledge_snapshot: dict[str, list[dict]] = {"entries": []}
        for row in knowledge_rows:
            entry_data: dict = {
                "entry_id": row["entry_id"],
                "knowledge_type": row["knowledge_type"],
                "confidence": row["confidence"],
                "source_iteration_id": row.get("source_iteration_id", ""),
            }
            # Parse content_json if present
            content_json = row.get("content_json", "{}")
            entry_data["content"] = _safe_json_loads(content_json)
            knowledge_snapshot["entries"].append(entry_data)

        # Copy skill files to version directory
        version_dir = os.path.join(self.skill_versions_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)

        if os.path.isdir(self.skills_dir):
            for filename in os.listdir(self.skills_dir):
                if filename.endswith(".md"):
                    src = os.path.join(self.skills_dir, filename)
                    dst = os.path.join(version_dir, filename)
                    shutil.copy2(src, dst)

        # Save knowledge snapshot as JSON in the version directory
        snapshot_path = os.path.join(version_dir, "knowledge_snapshot.json")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(knowledge_snapshot, f, indent=2, ensure_ascii=False)

        # Compute feedback metrics
        feedback_rows = self.db.get_all("effect_feedbacks")
        total_feedbacks = len(feedback_rows)
        positive_count = sum(
            1 for fb in feedback_rows if fb.get("is_positive")
        )
        positive_feedback_rate = (
            positive_count / total_feedbacks if total_feedbacks > 0 else 0.0
        )

        cer_improvements = [
            (fb.get("cer_before", 0.0) or 0.0)
            - (fb.get("cer_after", 0.0) or 0.0)
            for fb in feedback_rows
            if fb.get("is_positive")
        ]
        avg_cer_improvement = (
            sum(cer_improvements) / len(cer_improvements)
            if cer_improvements
            else 0.0
        )

        # Persist version record to skill_versions table
        self.db.insert("skill_versions", {
            "version_id": version_id,
            "strategy_rules_json": json.dumps(
                strategy_rules, ensure_ascii=False
            ),
            "knowledge_snapshot_path": snapshot_path,
            "change_description": change_description,
            "positive_feedback_rate": positive_feedback_rate,
            "avg_cer_improvement": avg_cer_improvement,
            "created_at": now,
        })

        return SkillVersion(
            version_id=version_id,
            strategy_rules=strategy_rules,
            knowledge_base_snapshot=knowledge_snapshot,
            change_description=change_description,
            positive_feedback_rate=positive_feedback_rate,
            avg_cer_improvement=avg_cer_improvement,
            created_at=now,
        )

    def rollback_version(self, version_id: str) -> bool:
        """Roll back Skill files to a previous version snapshot.

        Looks up *version_id* in the ``skill_versions`` table.  If found,
        copies all ``.md`` files from the version's snapshot directory
        (``knowledge/skill_versions/{version_id}/``) back to
        ``.kiro/skills/``.

        Parameters
        ----------
        version_id : str
            The version to roll back to.

        Returns
        -------
        bool
            ``True`` if the rollback succeeded, ``False`` if the version
            was not found or the snapshot directory is missing.
        """
        row = self.db.get_by_id(
            "skill_versions", "version_id", version_id
        )
        if row is None:
            return False

        version_dir = os.path.join(self.skill_versions_dir, version_id)
        if not os.path.isdir(version_dir):
            return False

        os.makedirs(self.skills_dir, exist_ok=True)

        md_files = [
            f for f in os.listdir(version_dir) if f.endswith(".md")
        ]
        for filename in md_files:
            src = os.path.join(version_dir, filename)
            dst = os.path.join(self.skills_dir, filename)
            shutil.copy2(src, dst)

        return True

    def check_auto_rollback(self) -> Optional[str]:
        """Check whether the latest Skill version should be auto-rolled back.

        Retrieves all skill versions ordered by ``created_at``.  If the
        latest version's ``positive_feedback_rate`` is lower than the
        previous version's rate, the system automatically rolls back to
        the previous version.

        Returns
        -------
        Optional[str]
            The ``version_id`` that was rolled back to, or ``None`` if
            no rollback was needed (fewer than 2 versions, or the latest
            version is performing at least as well as the previous one).
        """
        versions = self.db.get_all(
            "skill_versions", order_by="created_at ASC"
        )
        if len(versions) < 2:
            return None

        previous = versions[-2]
        latest = versions[-1]

        prev_rate = previous.get("positive_feedback_rate", 0.0) or 0.0
        latest_rate = latest.get("positive_feedback_rate", 0.0) or 0.0

        if latest_rate < prev_rate:
            prev_id = previous["version_id"]
            self.rollback_version(prev_id)
            return prev_id

        return None

    def trigger_learning(
        self, time_range: tuple[str, str]
    ) -> dict:
        """Manually trigger Skill learning from feedbacks in a time range.

        Queries all ``effect_feedbacks``, filters by ``created_at``
        within *time_range*, then runs pattern analysis and strategy
        weight updates on the filtered data.

        Parameters
        ----------
        time_range : tuple[str, str]
            A ``(start_time, end_time)`` pair of ISO-format strings.

        Returns
        -------
        dict
            A summary with keys ``feedbacks_analyzed``,
            ``strategies_updated``, and ``version_id``.  Returns an
            empty dict with zero counts if no feedbacks fall within the
            range.
        """
        start_time, end_time = time_range

        all_feedbacks = self.db.get_all("effect_feedbacks")
        filtered = [
            fb for fb in all_feedbacks
            if start_time <= (fb.get("created_at") or "") <= end_time
        ]

        if not filtered:
            return {
                "feedbacks_analyzed": 0,
                "strategies_updated": 0,
                "version_id": None,
            }

        # Run analysis on filtered feedbacks by temporarily considering
        # only the filtered set.  We use analyze_patterns with
        # min_feedback_count=1 since we already confirmed we have data.
        # However, analyze_patterns reads from the DB directly, so we
        # perform the analysis inline on the filtered set.
        groups: dict[str, list[dict]] = defaultdict(list)
        for fb in filtered:
            strategy_name = fb.get("suggestion_content", "") or ""
            groups[strategy_name].append(fb)

        strategies: dict[str, dict] = {}
        for name, feedbacks in groups.items():
            total_count = len(feedbacks)
            positive_count = sum(
                1 for fb in feedbacks if fb.get("is_positive")
            )
            success_rate = (
                positive_count / total_count if total_count > 0 else 0.0
            )
            positive_improvements = [
                (fb.get("cer_before", 0.0) or 0.0)
                - (fb.get("cer_after", 0.0) or 0.0)
                for fb in feedbacks
                if fb.get("is_positive")
            ]
            avg_cer_improvement = (
                sum(positive_improvements) / len(positive_improvements)
                if positive_improvements
                else 0.0
            )
            strategies[name] = {
                "total_count": total_count,
                "positive_count": positive_count,
                "success_rate": success_rate,
                "avg_cer_improvement": avg_cer_improvement,
            }

        if strategies:
            average_success_rate = (
                sum(s["success_rate"] for s in strategies.values())
                / len(strategies)
            )
        else:
            average_success_rate = 0.0

        high_success: list[str] = []
        low_success: list[str] = []
        for name, stats in strategies.items():
            if stats["success_rate"] > average_success_rate:
                high_success.append(name)
            else:
                low_success.append(name)

        pattern_analysis = {
            "total_feedbacks": len(filtered),
            "strategies": strategies,
            "high_success_strategies": high_success,
            "low_success_strategies": low_success,
            "average_success_rate": average_success_rate,
        }

        weights = self.update_strategy_weights(pattern_analysis)
        self.update_skill_files(weights)
        version = self.create_version_snapshot(
            f"Learning triggered for range {start_time} to {end_time}"
        )

        return {
            "feedbacks_analyzed": len(filtered),
            "strategies_updated": len(weights),
            "version_id": version.version_id,
        }

    # ------------------------------------------------------------------
    # Skill quick reference table (Task 20.5)
    # ------------------------------------------------------------------

    def generate_skill_summary_table(self, skill_name: str) -> str:
        """Generate a quick-reference summary table for a Skill file.

        Extracts the top strategies (by verified count and success rate)
        from ``strategy_weights`` and ``knowledge_entries`` tables that
        are relevant to the given skill, and formats them as a markdown
        table (max 10 entries).

        Parameters
        ----------
        skill_name : str
            The skill name (e.g. ``"asr-training-diagnostics"``).

        Returns
        -------
        str
            A markdown-formatted quick-reference section.
        """
        # Gather strategy weights sorted by success_rate desc, then sample_count desc
        all_weights = self.db.get_all("strategy_weights")
        sorted_weights = sorted(
            all_weights,
            key=lambda r: (r.get("success_rate", 0.0) or 0.0, r.get("sample_count", 0) or 0),
            reverse=True,
        )

        # Gather high-confidence active knowledge entries
        all_knowledge = self.db.get_all(
            "knowledge_entries",
            filters={"state": "active"},
        )
        high_conf_knowledge = sorted(
            [k for k in all_knowledge if (k.get("confidence", 0.0) or 0.0) >= 0.6],
            key=lambda r: (r.get("verified_count", 0) or 0, r.get("confidence", 0.0) or 0.0),
            reverse=True,
        )

        # Build table rows (max 10)
        rows: list[tuple[str, str, str, str]] = []

        for sw in sorted_weights:
            if len(rows) >= 10:
                break
            rows.append((
                sw.get("strategy_name", ""),
                "strategy",
                f"{(sw.get('success_rate', 0.0) or 0.0):.0%}",
                str(sw.get("sample_count", 0) or 0),
            ))

        for ke in high_conf_knowledge:
            if len(rows) >= 10:
                break
            content = _safe_json_loads(ke.get("content_json", "{}"))
            summary = content.get("summary", "") or content.get("description", "") or ke.get("knowledge_type", "")
            # Truncate long summaries
            if len(summary) > 50:
                summary = summary[:47] + "..."
            rows.append((
                summary,
                ke.get("knowledge_type", ""),
                f"{(ke.get('confidence', 0.0) or 0.0):.0%}",
                str(ke.get("verified_count", 0) or 0),
            ))

        if not rows:
            return ""

        # Format as markdown table
        lines = [
            "## 经验速查",
            "",
            "| 策略/知识 | 类型 | 成功率/置信度 | 验证次数 |",
            "|-----------|------|---------------|----------|",
        ]
        for name, ktype, rate, count in rows[:10]:
            lines.append(f"| {name} | {ktype} | {rate} | {count} |")
        lines.append("")

        return "\n".join(lines)

    def update_skill_quick_reference(self, skill_name: str) -> dict:
        """Write the quick-reference table into the corresponding Skill file.

        Inserts or replaces the "经验速查" section at the top of the
        Skill file, before the "能力描述" section.

        Parameters
        ----------
        skill_name : str
            The skill name (e.g. ``"asr-training-diagnostics"``).

        Returns
        -------
        dict
            A dict with keys ``success`` (bool), ``file_path`` (str),
            and ``table_rows`` (int).
        """
        filename = f"{skill_name}.md"
        filepath = os.path.join(self.skills_dir, filename)

        if not os.path.isfile(filepath):
            return {"success": False, "file_path": filepath, "table_rows": 0}

        table_content = self.generate_skill_summary_table(skill_name)
        if not table_content:
            return {"success": True, "file_path": filepath, "table_rows": 0}

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove existing 经验速查 section if present
        marker_start = "## 经验速查"
        marker_end_candidates = ["## 能力描述", "## 数据来源", "## 分析规则"]

        idx_start = content.find(marker_start)
        if idx_start != -1:
            # Find the next section after 经验速查
            idx_end = len(content)
            for candidate in marker_end_candidates:
                pos = content.find(candidate, idx_start + len(marker_start))
                if pos != -1 and pos < idx_end:
                    idx_end = pos
            # Also check for any ## heading after the start
            search_pos = idx_start + len(marker_start)
            next_heading = content.find("\n## ", search_pos)
            if next_heading != -1 and next_heading + 1 < idx_end:
                idx_end = next_heading + 1
            content = content[:idx_start] + content[idx_end:]

        # Insert the table before "## 能力描述"
        insert_marker = "## 能力描述"
        insert_pos = content.find(insert_marker)
        if insert_pos != -1:
            content = (
                content[:insert_pos]
                + table_content + "\n"
                + content[insert_pos:]
            )
        else:
            # If no "能力描述" section, insert after the title (first line)
            first_newline = content.find("\n")
            if first_newline != -1:
                content = (
                    content[:first_newline + 1]
                    + "\n" + table_content + "\n"
                    + content[first_newline + 1:]
                )
            else:
                content = content + "\n\n" + table_content

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Count table rows (lines starting with |, excluding header)
        table_rows = sum(
            1 for line in table_content.split("\n")
            if line.startswith("| ") and not line.startswith("| 策略") and not line.startswith("|---")
        )

        return {"success": True, "file_path": filepath, "table_rows": table_rows}

    # ------------------------------------------------------------------
    # Skill file rule auto-update (Task 20.6)
    # ------------------------------------------------------------------

    def update_skill_rules_from_patterns(
        self, patterns: dict, weights: list[StrategyWeight]
    ) -> dict:
        """Auto-update Skill file rule parameters based on pattern analysis.

        For strategies with ``sample_count >= 3`` whose actual success
        rate deviates from the annotated confidence in the Skill file by
        more than 0.1, this method patches the Skill file to reflect the
        real data.

        Parameters
        ----------
        patterns : dict
            Output of :meth:`analyze_patterns`.
        weights : list[StrategyWeight]
            Current strategy weights.

        Returns
        -------
        dict
            A summary with keys ``updated_rules`` (int),
            ``deprecated_rules`` (list[str]), ``appended_rules``
            (list[str]).
        """
        if not os.path.isdir(self.skills_dir):
            return {"updated_rules": 0, "deprecated_rules": [], "appended_rules": []}

        strategies = patterns.get("strategies", {})
        updated_rules = 0

        # Build a weight lookup
        weight_map = {sw.strategy_name: sw for sw in weights}

        for filename in sorted(os.listdir(self.skills_dir)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(self.skills_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            modified = False
            for strategy_name, stats in strategies.items():
                sample_count = stats.get("total_count", 0)
                if sample_count < 3:
                    continue
                actual_rate = stats.get("success_rate", 0.0)

                # Look for the strategy name in the file and update confidence
                # Pattern: "置信度：X.X" or "置信度: X.X" or "confidence: X.X"
                # Search for lines mentioning this strategy
                if strategy_name not in content:
                    continue

                # Update confidence values near the strategy mention
                for conf_pattern in [
                    r"(置信度[：:]\s*)([\d.]+)",
                    r"(confidence[：:]\s*)([\d.]+)",
                ]:
                    matches = list(re.finditer(conf_pattern, content))
                    for match in matches:
                        # Check if this confidence annotation is near the strategy
                        start = max(0, match.start() - 200)
                        context_before = content[start:match.start()]
                        if strategy_name in context_before or match.start() < content.find(strategy_name) + 300:
                            old_conf = float(match.group(2))
                            if abs(old_conf - actual_rate) > 0.1:
                                new_conf_str = f"{match.group(1)}{actual_rate:.2f}"
                                content = content[:match.start()] + new_conf_str + content[match.end():]
                                modified = True
                                updated_rules += 1
                                break

            if modified:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

        # Run deprecation and appending
        deprecated = self.deprecate_failed_rules()
        appended = self.append_discovered_rules()

        return {
            "updated_rules": updated_rules,
            "deprecated_rules": deprecated,
            "appended_rules": appended,
        }

    def deprecate_failed_rules(self, min_failures: int = 3) -> list[str]:
        """Mark rules that have consecutively failed validation as deprecated.

        Looks at ``strategy_weights`` for strategies with
        ``sample_count >= min_failures`` and ``success_rate < 0.3``.
        For each such strategy found in a Skill file, adds a
        ``[DEPRECATED]`` marker.

        Parameters
        ----------
        min_failures : int
            Minimum number of verification attempts required before
            deprecation (default 3).

        Returns
        -------
        list[str]
            Names of strategies that were marked as deprecated.
        """
        if not os.path.isdir(self.skills_dir):
            return []

        all_weights = self.db.get_all("strategy_weights")
        failed_strategies = [
            row["strategy_name"]
            for row in all_weights
            if (row.get("sample_count", 0) or 0) >= min_failures
            and (row.get("success_rate", 1.0) or 1.0) < 0.3
        ]

        if not failed_strategies:
            return []

        deprecated: list[str] = []
        for filename in sorted(os.listdir(self.skills_dir)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(self.skills_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            modified = False
            for strategy_name in failed_strategies:
                if strategy_name not in content:
                    continue
                # Don't double-deprecate
                deprecated_marker = f"[DEPRECATED] {strategy_name}"
                if deprecated_marker in content:
                    continue
                # Add deprecated marker before the strategy mention
                content = content.replace(
                    strategy_name,
                    f"[DEPRECATED] {strategy_name}",
                    1,  # Only first occurrence
                )
                modified = True
                deprecated.append(strategy_name)

            if modified:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

        return deprecated

    def append_discovered_rules(
        self, min_confidence: float = 0.8, min_verified: int = 3
    ) -> list[str]:
        """Append high-confidence new patterns as rules to Skill files.

        Queries ``knowledge_entries`` for active entries with
        ``confidence >= min_confidence`` and
        ``verified_count >= min_verified``, then appends them to the
        appropriate Skill file under an "自动发现的规则" section.

        Parameters
        ----------
        min_confidence : float
            Minimum confidence threshold (default 0.8).
        min_verified : int
            Minimum verification count (default 3).

        Returns
        -------
        list[str]
            Descriptions of rules that were appended.
        """
        if not os.path.isdir(self.skills_dir):
            return []

        all_entries = self.db.get_all(
            "knowledge_entries",
            filters={"state": "active"},
        )

        # Filter high-confidence, well-verified entries
        candidates = [
            row for row in all_entries
            if (row.get("confidence", 0.0) or 0.0) >= min_confidence
            and (row.get("verified_count", 0) or 0) >= min_verified
        ]

        if not candidates:
            return []

        # Map knowledge_type to skill file
        type_to_skill = {
            "scene_solution": "asr-training-diagnostics.md",
            "param_effect": "asr-model-tuning.md",
            "data_strategy": "asr-data-optimization.md",
        }

        appended: list[str] = []
        # Group candidates by target skill file
        by_file: dict[str, list[dict]] = defaultdict(list)
        for entry in candidates:
            ktype = entry.get("knowledge_type", "")
            target_file = type_to_skill.get(ktype, "asr-training-diagnostics.md")
            by_file[target_file].append(entry)

        for filename, entries in by_file.items():
            filepath = os.path.join(self.skills_dir, filename)
            if not os.path.isfile(filepath):
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Find or create "自动发现的规则" section
            section_marker = "## 自动发现的规则"
            if section_marker not in content:
                content = content.rstrip("\n") + f"\n\n{section_marker}\n\n"

            new_rules: list[str] = []
            for entry in entries:
                entry_content = _safe_json_loads(entry.get("content_json", "{}"))
                description = (
                    entry_content.get("summary", "")
                    or entry_content.get("description", "")
                    or entry_content.get("solution", "")
                    or str(entry_content)
                )
                confidence = entry.get("confidence", 0.0) or 0.0
                verified = entry.get("verified_count", 0) or 0

                # Skip if already present in the file
                if description and description in content:
                    continue

                rule_text = (
                    f"- **{description}**\n"
                    f"  - 置信度: {confidence:.2f}，验证次数: {verified}\n"
                    f"  - 来源迭代: {entry.get('source_iteration_id', 'N/A')}\n"
                )
                new_rules.append(rule_text)
                appended.append(description)

            if new_rules:
                # Append rules after the section marker
                idx = content.find(section_marker)
                insert_pos = idx + len(section_marker)
                # Skip past any existing newlines after the marker
                while insert_pos < len(content) and content[insert_pos] == "\n":
                    insert_pos += 1
                rules_text = "\n".join(new_rules) + "\n"
                content = content[:insert_pos] + "\n" + rules_text + content[insert_pos:]

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

        return appended

    def compare_versions(
        self, version_a: str, version_b: str
    ) -> dict:
        """Compare quality metrics between two Skill versions.

        Parameters
        ----------
        version_a : str
            Version ID of the first version.
        version_b : str
            Version ID of the second version.

        Returns
        -------
        dict
            A dict with keys ``version_a``, ``version_b`` (each
            containing ``positive_feedback_rate`` and
            ``avg_cer_improvement``), and ``deltas`` with the
            differences (B − A).  Returns an empty dict if either
            version is not found.
        """
        row_a = self.db.get_by_id(
            "skill_versions", "version_id", version_a
        )
        row_b = self.db.get_by_id(
            "skill_versions", "version_id", version_b
        )

        if row_a is None or row_b is None:
            return {}

        rate_a = row_a.get("positive_feedback_rate", 0.0) or 0.0
        rate_b = row_b.get("positive_feedback_rate", 0.0) or 0.0
        cer_a = row_a.get("avg_cer_improvement", 0.0) or 0.0
        cer_b = row_b.get("avg_cer_improvement", 0.0) or 0.0

        return {
            "version_a": {
                "version_id": version_a,
                "positive_feedback_rate": rate_a,
                "avg_cer_improvement": cer_a,
            },
            "version_b": {
                "version_id": version_b,
                "positive_feedback_rate": rate_b,
                "avg_cer_improvement": cer_b,
            },
            "deltas": {
                "positive_feedback_rate": rate_b - rate_a,
                "avg_cer_improvement": cer_b - cer_a,
            },
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_json_loads(value: object) -> dict:
    """Parse a JSON string, returning an empty dict on failure."""
    if not value:
        return {}
    try:
        result = json.loads(str(value))
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
