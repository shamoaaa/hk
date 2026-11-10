"""Skill Knowledge Base for the ASR Training Optimizer.

Provides the ``SkillKnowledgeBase`` class which manages domain expert
knowledge accumulated during optimization iterations.  Knowledge entries
capture scene-problem-solution mappings, parameter adjustment effects,
and data feature-augmentation strategy correspondences.

Requirements: 10.5, 10.6
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from scripts.core.database import Database
from scripts.core.models import KnowledgeEntry


# Valid knowledge types matching the DB CHECK constraint
_VALID_KNOWLEDGE_TYPES = frozenset({
    "scene_solution",
    "param_effect",
    "data_strategy",
})


class SkillKnowledgeBase:
    """Skill knowledge base managing accumulated domain expert knowledge.

    Stores and retrieves knowledge entries that capture relationships
    discovered during ASR optimization iterations, including:

    - **scene_solution**: scene → problem → solution mappings
    - **param_effect**: parameter adjustment → effect patterns
    - **data_strategy**: data feature → augmentation strategy correspondences

    Parameters
    ----------
    db : Database
        Initialized database instance for persisting knowledge entries
        in the ``knowledge_entries`` table.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Add knowledge
    # ------------------------------------------------------------------

    def add_knowledge(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry to the knowledge base.

        Generates a unique ``entry_id``, validates the knowledge type,
        serializes the ``content`` dict as JSON, and persists the entry
        to the ``knowledge_entries`` table.

        Parameters
        ----------
        entry : KnowledgeEntry
            The knowledge entry to add.  The ``knowledge_type`` must be
            one of ``"scene_solution"``, ``"param_effect"``, or
            ``"data_strategy"``.

        Returns
        -------
        str
            The generated ``entry_id`` for the new knowledge entry.

        Raises
        ------
        ValueError
            If ``entry.knowledge_type`` is not a valid type.
        """
        if entry.knowledge_type not in _VALID_KNOWLEDGE_TYPES:
            raise ValueError(
                f"Invalid knowledge_type '{entry.knowledge_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_KNOWLEDGE_TYPES))}"
            )

        entry_id = f"ke_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        self.db.insert("knowledge_entries", {
            "entry_id": entry_id,
            "knowledge_type": entry.knowledge_type,
            "content_json": json.dumps(
                entry.content, ensure_ascii=False
            ),
            "confidence": entry.confidence,
            "source_iteration_id": entry.source_iteration_id,
            "created_at": entry.created_at or now,
            "updated_at": entry.updated_at or now,
            "state": entry.state or "active",
            "verified_count": entry.verified_count,
            "last_verified_at": entry.last_verified_at or None,
            "auto_confidence": entry.auto_confidence,
        })

        return entry_id

    # ------------------------------------------------------------------
    # Add knowledge (strict / verified mode)
    # ------------------------------------------------------------------

    def add_knowledge_verified(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry with strict action-verification checks.

        Enforces the Action Verification Principle:
        1. Must originate from a complete iteration loop (iteration_id non-empty)
        2. Associated effect_feedback must have is_positive=True
        3. Same-type knowledge must have been verified in at least 2
           different iterations

        Parameters
        ----------
        entry : KnowledgeEntry
            The knowledge entry to add.

        Returns
        -------
        str
            The generated ``entry_id`` for the new knowledge entry.

        Raises
        ------
        ValueError
            If the entry fails any verification check.
        """
        if entry.knowledge_type not in _VALID_KNOWLEDGE_TYPES:
            raise ValueError(
                f"Invalid knowledge_type '{entry.knowledge_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_KNOWLEDGE_TYPES))}"
            )

        # Check 1: Must come from a complete iteration loop
        if not entry.source_iteration_id:
            raise ValueError(
                "Verified knowledge must have a non-empty source_iteration_id "
                "(must originate from a complete iteration loop)."
            )

        # Check 2: Associated effect_feedback must be positive
        feedbacks = self.db.get_all(
            "effect_feedbacks",
            filters={"iteration_id": entry.source_iteration_id},
        )
        has_positive = any(fb.get("is_positive") for fb in feedbacks)
        if not has_positive:
            raise ValueError(
                f"No positive effect_feedback found for iteration "
                f"'{entry.source_iteration_id}'. Verified knowledge requires "
                f"at least one positive feedback from the source iteration."
            )

        # Check 3: Same-type knowledge verified in at least 2 different iterations
        same_type_entries = self.db.get_all(
            "knowledge_entries",
            filters={"knowledge_type": entry.knowledge_type},
        )
        verified_iterations = set()
        for row in same_type_entries:
            src_iter = row.get("source_iteration_id", "")
            if src_iter:
                verified_iterations.add(src_iter)
        # Include the current iteration
        verified_iterations.add(entry.source_iteration_id)

        if len(verified_iterations) < 2:
            raise ValueError(
                f"Same-type knowledge ('{entry.knowledge_type}') must be "
                f"verified in at least 2 different iterations. "
                f"Currently verified in {len(verified_iterations)} iteration(s)."
            )

        # All checks passed, add the entry
        return self.add_knowledge(entry)

    # ------------------------------------------------------------------
    # Query knowledge
    # ------------------------------------------------------------------

    def query_knowledge(
        self,
        knowledge_type: str,
        filters: Optional[dict] = None,
        include_stale: bool = False,
        include_archived: bool = False,
    ) -> list[KnowledgeEntry]:
        """Query knowledge entries by type and optional filters.

        By default, only returns entries with state ``"active"``.
        Use ``include_stale`` and ``include_archived`` to include
        entries in those states.

        Parameters
        ----------
        knowledge_type : str
            The type of knowledge to query.  Must be one of
            ``"scene_solution"``, ``"param_effect"``, or
            ``"data_strategy"``.
        filters : dict, optional
            Additional column-level filters passed to the database
            query.  Supported filter keys include
            ``"source_iteration_id"`` and ``"confidence"`` (exact match).
            Content-level filtering (matching keys inside the JSON
            ``content`` field) is supported via the special key
            ``"content_match"`` — a dict whose key-value pairs must all
            appear in the entry's ``content`` for it to be included.
        include_stale : bool
            If ``True``, include entries with state ``"stale"``
            (default ``False``).
        include_archived : bool
            If ``True``, include entries with state ``"archived"``
            (default ``False``).

        Returns
        -------
        list[KnowledgeEntry]
            Matching knowledge entries, ordered by ``created_at``
            descending (newest first).
        """
        db_filters: dict = {"knowledge_type": knowledge_type}
        content_match: Optional[dict] = None

        if filters:
            for key, value in filters.items():
                if key == "content_match":
                    content_match = value
                else:
                    db_filters[key] = value

        # Determine allowed states
        allowed_states = {"active"}
        if include_stale:
            allowed_states.add("stale")
        if include_archived:
            allowed_states.add("archived")

        rows = self.db.get_all(
            "knowledge_entries",
            filters=db_filters,
            order_by="created_at DESC",
        )

        entries: list[KnowledgeEntry] = []
        for row in rows:
            # Filter by state
            row_state = row.get("state", "active") or "active"
            if row_state not in allowed_states:
                continue

            content = _safe_json_loads(row.get("content_json", "{}"))

            # Apply content-level filtering if requested
            if content_match is not None:
                if not _content_matches(content, content_match):
                    continue

            entries.append(KnowledgeEntry(
                knowledge_type=row["knowledge_type"],
                content=content,
                confidence=row.get("confidence", 0.0) or 0.0,
                source_iteration_id=row.get("source_iteration_id", ""),
                created_at=row.get("created_at", ""),
                updated_at=row.get("updated_at", ""),
                state=row_state,
                verified_count=row.get("verified_count", 0) or 0,
                last_verified_at=row.get("last_verified_at", "") or "",
                auto_confidence=row.get("auto_confidence"),
            ))

        return entries

    # ------------------------------------------------------------------
    # Structured view
    # ------------------------------------------------------------------

    def get_structured_view(self) -> dict:
        """Return a structured view of all knowledge entries.

        Groups entries by ``knowledge_type`` and includes summary
        statistics for each group.

        Returns
        -------
        dict
            A dict with keys:

            - ``total_entries`` – total number of knowledge entries
            - ``by_type`` – dict mapping each knowledge type to a dict
              containing ``count``, ``avg_confidence``, and ``entries``
              (list of entry dicts with ``entry_id``, ``content``,
              ``confidence``, ``source_iteration_id``, ``created_at``)
            - ``types_summary`` – list of dicts with ``type``, ``count``,
              and ``avg_confidence`` for quick overview
        """
        all_rows = self.db.get_all(
            "knowledge_entries",
            order_by="created_at DESC",
        )

        groups: dict[str, list[dict]] = defaultdict(list)
        for row in all_rows:
            groups[row["knowledge_type"]].append(row)

        by_type: dict[str, dict] = {}
        types_summary: list[dict] = []

        for ktype in sorted(_VALID_KNOWLEDGE_TYPES):
            rows = groups.get(ktype, [])
            count = len(rows)

            confidences = [
                (r.get("confidence", 0.0) or 0.0) for r in rows
            ]
            avg_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            entries_list: list[dict] = []
            for row in rows:
                entries_list.append({
                    "entry_id": row["entry_id"],
                    "content": _safe_json_loads(
                        row.get("content_json", "{}")
                    ),
                    "confidence": row.get("confidence", 0.0) or 0.0,
                    "source_iteration_id": row.get(
                        "source_iteration_id", ""
                    ),
                    "created_at": row.get("created_at", ""),
                })

            by_type[ktype] = {
                "count": count,
                "avg_confidence": avg_confidence,
                "entries": entries_list,
            }

            types_summary.append({
                "type": ktype,
                "count": count,
                "avg_confidence": avg_confidence,
            })

        return {
            "total_entries": len(all_rows),
            "by_type": by_type,
            "types_summary": types_summary,
        }

    # ------------------------------------------------------------------
    # Update confidence
    # ------------------------------------------------------------------

    def update_confidence(
        self, entry_id: str, new_confidence: float
    ) -> None:
        """Update the confidence score of a knowledge entry.

        Parameters
        ----------
        entry_id : str
            The ID of the knowledge entry to update.
        new_confidence : float
            The new confidence value (should be between 0.0 and 1.0).

        Raises
        ------
        ValueError
            If ``new_confidence`` is not in the range [0.0, 1.0].
        KeyError
            If no entry with the given ``entry_id`` exists.
        """
        if not (0.0 <= new_confidence <= 1.0):
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {new_confidence}"
            )

        existing = self.db.get_by_id(
            "knowledge_entries", "entry_id", entry_id
        )
        if existing is None:
            raise KeyError(f"Knowledge entry '{entry_id}' not found")

        now = datetime.now(timezone.utc).isoformat()
        self.db.update(
            "knowledge_entries",
            "entry_id",
            entry_id,
            {
                "confidence": new_confidence,
                "updated_at": now,
            },
        )

    # ------------------------------------------------------------------
    # Auto-confidence computation
    # ------------------------------------------------------------------

    def compute_confidence(self, entry_id: str) -> float:
        """Compute confidence for a knowledge entry based on feedback data.

        Uses the formula:
            confidence = (success_count / total_count) * 0.95^(days_since_last_verification / 30)

        The method queries ``effect_feedbacks`` for feedbacks related to
        the same knowledge type and source iteration, computes the
        success rate, applies a time decay factor based on
        ``last_verified_at``, and updates the entry's
        ``auto_confidence`` field.

        Parameters
        ----------
        entry_id : str
            The ID of the knowledge entry to compute confidence for.

        Returns
        -------
        float
            The computed confidence value (0.0-1.0).

        Raises
        ------
        KeyError
            If no entry with the given ``entry_id`` exists.
        """
        existing = self.db.get_by_id(
            "knowledge_entries", "entry_id", entry_id
        )
        if existing is None:
            raise KeyError(f"Knowledge entry '{entry_id}' not found")

        source_iteration_id = existing.get("source_iteration_id", "")

        # Query all feedbacks related to this entry's source iteration
        # and knowledge type context
        all_feedbacks = self.db.get_all("effect_feedbacks")

        # Filter feedbacks relevant to this knowledge entry:
        # - same iteration_id as source, OR
        # - feedbacks from iterations that reference similar content
        relevant_feedbacks = [
            fb for fb in all_feedbacks
            if fb.get("iteration_id") == source_iteration_id
        ]

        # If no direct match, use all feedbacks for the same suggestion type
        if not relevant_feedbacks:
            knowledge_type = existing.get("knowledge_type", "")
            content = _safe_json_loads(existing.get("content_json", "{}"))
            # Use all feedbacks as fallback
            relevant_feedbacks = all_feedbacks

        total_count = len(relevant_feedbacks)
        if total_count == 0:
            # No feedback data, return current confidence unchanged
            return existing.get("confidence", 0.0) or 0.0

        success_count = sum(
            1 for fb in relevant_feedbacks if fb.get("is_positive")
        )
        success_rate = success_count / total_count

        # Compute time decay factor based on last_verified_at
        last_verified = existing.get("last_verified_at") or existing.get("updated_at") or existing.get("created_at") or ""
        days_since = 0.0
        if last_verified:
            try:
                last_dt = datetime.fromisoformat(last_verified.replace("Z", "+00:00"))
                now_dt = datetime.now(timezone.utc)
                days_since = (now_dt - last_dt).total_seconds() / 86400.0
            except (ValueError, TypeError):
                days_since = 0.0

        time_decay_factor = 0.95 ** (days_since / 30.0)
        confidence = success_rate * time_decay_factor

        # Clamp to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        # Update the entry's auto_confidence field
        now = datetime.now(timezone.utc).isoformat()
        self.db.update(
            "knowledge_entries",
            "entry_id",
            entry_id,
            {
                "auto_confidence": confidence,
                "updated_at": now,
            },
        )

        return confidence

    # ------------------------------------------------------------------
    # Lifecycle state transitions
    # ------------------------------------------------------------------

    def apply_lifecycle_transitions(self) -> dict:
        """Apply lifecycle state transitions to all knowledge entries.

        Transitions:
        - active → stale: entry has not been verified in over 60 days
        - stale → archived: entry has not been verified in over 120 days
          OR auto_confidence < 0.3
        - stale → active: entry has been re-verified (last_verified_at
          updated within the last 60 days)

        Returns
        -------
        dict
            A summary with keys ``active_to_stale``, ``stale_to_archived``,
            ``stale_to_active`` indicating the count of transitions made.
        """
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()

        all_entries = self.db.get_all("knowledge_entries")

        active_to_stale = 0
        stale_to_archived = 0
        stale_to_active = 0

        for row in all_entries:
            entry_id = row["entry_id"]
            state = row.get("state", "active") or "active"
            last_verified = row.get("last_verified_at") or row.get("updated_at") or row.get("created_at") or ""
            auto_confidence = row.get("auto_confidence")

            # Compute days since last verification
            days_since = self._days_since(last_verified, now_dt)

            if state == "active":
                # active → stale: over 60 days without verification
                if days_since > 60:
                    self.db.update(
                        "knowledge_entries", "entry_id", entry_id,
                        {"state": "stale", "updated_at": now_iso},
                    )
                    active_to_stale += 1

            elif state == "stale":
                # stale → active: re-verified within 60 days
                if days_since <= 60 and last_verified:
                    self.db.update(
                        "knowledge_entries", "entry_id", entry_id,
                        {"state": "active", "updated_at": now_iso},
                    )
                    stale_to_active += 1
                # stale → archived: over 120 days OR confidence < 0.3
                elif days_since > 120 or (
                    auto_confidence is not None and auto_confidence < 0.3
                ):
                    self.db.update(
                        "knowledge_entries", "entry_id", entry_id,
                        {"state": "archived", "updated_at": now_iso},
                    )
                    stale_to_archived += 1

        return {
            "active_to_stale": active_to_stale,
            "stale_to_archived": stale_to_archived,
            "stale_to_active": stale_to_active,
        }

    @staticmethod
    def _days_since(iso_timestamp: str, now_dt: datetime) -> float:
        """Compute days elapsed since an ISO timestamp."""
        if not iso_timestamp:
            return 999.0  # Treat missing timestamps as very old
        try:
            dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
            return (now_dt - dt).total_seconds() / 86400.0
        except (ValueError, TypeError):
            return 999.0


def _safe_json_loads(value: object) -> dict:
    """Parse a JSON string, returning an empty dict on failure."""
    if not value:
        return {}
    try:
        result = json.loads(str(value))
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _content_matches(content: dict, match_criteria: dict) -> bool:
    """Check whether *content* contains all key-value pairs in *match_criteria*.

    Parameters
    ----------
    content : dict
        The content dict from a knowledge entry.
    match_criteria : dict
        Key-value pairs that must all be present in *content*.

    Returns
    -------
    bool
        ``True`` if every key in *match_criteria* exists in *content*
        and the corresponding values are equal.
    """
    for key, value in match_criteria.items():
        if key not in content:
            return False
        if content[key] != value:
            return False
    return True
