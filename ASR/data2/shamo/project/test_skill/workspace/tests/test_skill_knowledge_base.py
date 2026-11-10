"""Tests for the SkillKnowledgeBase module.

Covers knowledge entry CRUD operations, querying by type and filters,
structured view generation, and confidence updates.

Requirements: 10.5, 10.6
"""

import json

import pytest

from scripts.core.database import Database
from scripts.core.models import KnowledgeEntry
from scripts.decision.skill_knowledge_base import SkillKnowledgeBase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """Return an initialized Database instance."""
    path = str(tmp_path / "test_metadata.db")
    database = Database(path)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def kb(db):
    """Return a SkillKnowledgeBase instance."""
    return SkillKnowledgeBase(db=db)


@pytest.fixture
def sample_scene_solution():
    """Return a sample scene_solution KnowledgeEntry."""
    return KnowledgeEntry(
        knowledge_type="scene_solution",
        content={
            "scene": "background_voice",
            "problem": "High CER in background voice scenario",
            "solution": "Add speech enhancement frontend module",
        },
        confidence=0.85,
        source_iteration_id="iter_001",
    )


@pytest.fixture
def sample_param_effect():
    """Return a sample param_effect KnowledgeEntry."""
    return KnowledgeEntry(
        knowledge_type="param_effect",
        content={
            "parameter": "learning_rate",
            "change": "increased from 1e-5 to 5e-5",
            "effect": "CER improved by 2% on background voice",
        },
        confidence=0.72,
        source_iteration_id="iter_002",
    )


@pytest.fixture
def sample_data_strategy():
    """Return a sample data_strategy KnowledgeEntry."""
    return KnowledgeEntry(
        knowledge_type="data_strategy",
        content={
            "data_feature": "low SNR samples",
            "strategy": "Apply background voice augmentation",
            "expected_improvement": "5-10% CER reduction",
        },
        confidence=0.60,
        source_iteration_id="iter_003",
    )


# ---------------------------------------------------------------------------
# add_knowledge tests
# ---------------------------------------------------------------------------


class TestAddKnowledge:
    """Tests for SkillKnowledgeBase.add_knowledge."""

    def test_returns_entry_id(self, kb, sample_scene_solution):
        """add_knowledge should return a generated entry_id."""
        entry_id = kb.add_knowledge(sample_scene_solution)
        assert entry_id.startswith("ke_")
        assert len(entry_id) > 3

    def test_persists_to_database(self, kb, db, sample_scene_solution):
        """add_knowledge should persist the entry to knowledge_entries table."""
        kb.add_knowledge(sample_scene_solution)
        rows = db.get_all("knowledge_entries")
        assert len(rows) == 1

    def test_stores_correct_knowledge_type(self, kb, db, sample_scene_solution):
        """The stored knowledge_type should match the input."""
        kb.add_knowledge(sample_scene_solution)
        rows = db.get_all("knowledge_entries")
        assert rows[0]["knowledge_type"] == "scene_solution"

    def test_serializes_content_as_json(self, kb, db, sample_scene_solution):
        """The content dict should be serialized as a JSON string."""
        kb.add_knowledge(sample_scene_solution)
        rows = db.get_all("knowledge_entries")
        content = json.loads(rows[0]["content_json"])
        assert content["scene"] == "background_voice"
        assert content["problem"] == "High CER in background voice scenario"

    def test_stores_confidence(self, kb, db, sample_scene_solution):
        """The confidence value should be stored correctly."""
        kb.add_knowledge(sample_scene_solution)
        rows = db.get_all("knowledge_entries")
        assert rows[0]["confidence"] == pytest.approx(0.85)

    def test_stores_source_iteration_id(self, kb, db, sample_scene_solution):
        """The source_iteration_id should be stored correctly."""
        kb.add_knowledge(sample_scene_solution)
        rows = db.get_all("knowledge_entries")
        assert rows[0]["source_iteration_id"] == "iter_001"

    def test_generates_unique_ids(self, kb, sample_scene_solution):
        """Each call should generate a unique entry_id."""
        id1 = kb.add_knowledge(sample_scene_solution)
        id2 = kb.add_knowledge(sample_scene_solution)
        assert id1 != id2

    def test_stores_timestamps(self, kb, db, sample_scene_solution):
        """add_knowledge should set created_at and updated_at timestamps."""
        kb.add_knowledge(sample_scene_solution)
        rows = db.get_all("knowledge_entries")
        assert rows[0]["created_at"] is not None
        assert len(rows[0]["created_at"]) > 0
        assert rows[0]["updated_at"] is not None
        assert len(rows[0]["updated_at"]) > 0

    def test_rejects_invalid_knowledge_type(self, kb):
        """add_knowledge should raise ValueError for invalid knowledge_type."""
        entry = KnowledgeEntry(
            knowledge_type="invalid_type",
            content={"key": "value"},
            confidence=0.5,
        )
        with pytest.raises(ValueError, match="Invalid knowledge_type"):
            kb.add_knowledge(entry)

    def test_all_three_types_accepted(self, kb, sample_scene_solution,
                                      sample_param_effect, sample_data_strategy):
        """All three valid knowledge types should be accepted."""
        id1 = kb.add_knowledge(sample_scene_solution)
        id2 = kb.add_knowledge(sample_param_effect)
        id3 = kb.add_knowledge(sample_data_strategy)
        assert all(eid.startswith("ke_") for eid in [id1, id2, id3])

    def test_empty_content_dict(self, kb, db):
        """add_knowledge should handle empty content dict."""
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={},
            confidence=0.5,
            source_iteration_id="iter_empty",
        )
        entry_id = kb.add_knowledge(entry)
        assert entry_id.startswith("ke_")

        rows = db.get_all("knowledge_entries")
        content = json.loads(rows[0]["content_json"])
        assert content == {}

    def test_preserves_provided_timestamps(self, kb, db):
        """If created_at/updated_at are provided, they should be used."""
        entry = KnowledgeEntry(
            knowledge_type="param_effect",
            content={"test": True},
            confidence=0.5,
            source_iteration_id="iter_ts",
            created_at="2024-06-01T00:00:00",
            updated_at="2024-06-01T00:00:00",
        )
        kb.add_knowledge(entry)
        rows = db.get_all("knowledge_entries")
        assert rows[0]["created_at"] == "2024-06-01T00:00:00"
        assert rows[0]["updated_at"] == "2024-06-01T00:00:00"


# ---------------------------------------------------------------------------
# query_knowledge tests
# ---------------------------------------------------------------------------


class TestQueryKnowledge:
    """Tests for SkillKnowledgeBase.query_knowledge."""

    def test_returns_entries_of_specified_type(self, kb, sample_scene_solution,
                                                sample_param_effect):
        """query_knowledge should only return entries of the requested type."""
        kb.add_knowledge(sample_scene_solution)
        kb.add_knowledge(sample_param_effect)

        results = kb.query_knowledge("scene_solution")
        assert len(results) == 1
        assert results[0].knowledge_type == "scene_solution"

    def test_returns_empty_for_no_matches(self, kb, sample_scene_solution):
        """query_knowledge should return [] when no entries match."""
        kb.add_knowledge(sample_scene_solution)
        results = kb.query_knowledge("param_effect")
        assert results == []

    def test_returns_knowledge_entry_instances(self, kb, sample_scene_solution):
        """query_knowledge should return KnowledgeEntry dataclass instances."""
        kb.add_knowledge(sample_scene_solution)
        results = kb.query_knowledge("scene_solution")
        assert len(results) == 1
        assert isinstance(results[0], KnowledgeEntry)

    def test_content_is_deserialized(self, kb, sample_scene_solution):
        """The content field should be deserialized from JSON."""
        kb.add_knowledge(sample_scene_solution)
        results = kb.query_knowledge("scene_solution")
        assert results[0].content["scene"] == "background_voice"

    def test_filter_by_source_iteration_id(self, kb):
        """query_knowledge should support filtering by source_iteration_id."""
        entry1 = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"scene": "clean"},
            confidence=0.8,
            source_iteration_id="iter_A",
        )
        entry2 = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"scene": "noisy"},
            confidence=0.7,
            source_iteration_id="iter_B",
        )
        kb.add_knowledge(entry1)
        kb.add_knowledge(entry2)

        results = kb.query_knowledge(
            "scene_solution",
            filters={"source_iteration_id": "iter_A"},
        )
        assert len(results) == 1
        assert results[0].content["scene"] == "clean"

    def test_content_match_filter(self, kb):
        """query_knowledge should support content_match filtering."""
        entry1 = KnowledgeEntry(
            knowledge_type="param_effect",
            content={"parameter": "learning_rate", "direction": "increase"},
            confidence=0.8,
            source_iteration_id="iter_1",
        )
        entry2 = KnowledgeEntry(
            knowledge_type="param_effect",
            content={"parameter": "batch_size", "direction": "decrease"},
            confidence=0.7,
            source_iteration_id="iter_2",
        )
        kb.add_knowledge(entry1)
        kb.add_knowledge(entry2)

        results = kb.query_knowledge(
            "param_effect",
            filters={"content_match": {"parameter": "learning_rate"}},
        )
        assert len(results) == 1
        assert results[0].content["parameter"] == "learning_rate"

    def test_content_match_multiple_criteria(self, kb):
        """content_match should require all criteria to match."""
        entry = KnowledgeEntry(
            knowledge_type="data_strategy",
            content={"feature": "low_snr", "strategy": "augment", "scene": "noisy"},
            confidence=0.6,
            source_iteration_id="iter_1",
        )
        kb.add_knowledge(entry)

        # Both criteria match
        results = kb.query_knowledge(
            "data_strategy",
            filters={"content_match": {"feature": "low_snr", "strategy": "augment"}},
        )
        assert len(results) == 1

        # One criterion doesn't match
        results = kb.query_knowledge(
            "data_strategy",
            filters={"content_match": {"feature": "low_snr", "strategy": "filter"}},
        )
        assert len(results) == 0

    def test_returns_multiple_entries(self, kb):
        """query_knowledge should return all matching entries."""
        for i in range(3):
            entry = KnowledgeEntry(
                knowledge_type="scene_solution",
                content={"index": i},
                confidence=0.5 + i * 0.1,
                source_iteration_id=f"iter_{i}",
            )
            kb.add_knowledge(entry)

        results = kb.query_knowledge("scene_solution")
        assert len(results) == 3

    def test_ordered_by_created_at_desc(self, kb):
        """Results should be ordered by created_at descending (newest first)."""
        entry_old = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"order": "first"},
            confidence=0.5,
            source_iteration_id="iter_old",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        entry_new = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"order": "second"},
            confidence=0.5,
            source_iteration_id="iter_new",
            created_at="2024-06-01T00:00:00",
            updated_at="2024-06-01T00:00:00",
        )
        kb.add_knowledge(entry_old)
        kb.add_knowledge(entry_new)

        results = kb.query_knowledge("scene_solution")
        assert len(results) == 2
        assert results[0].content["order"] == "second"
        assert results[1].content["order"] == "first"

    def test_no_filters(self, kb, sample_scene_solution):
        """query_knowledge should work with filters=None."""
        kb.add_knowledge(sample_scene_solution)
        results = kb.query_knowledge("scene_solution", filters=None)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# get_structured_view tests
# ---------------------------------------------------------------------------


class TestGetStructuredView:
    """Tests for SkillKnowledgeBase.get_structured_view."""

    def test_empty_knowledge_base(self, kb):
        """get_structured_view should handle an empty knowledge base."""
        view = kb.get_structured_view()
        assert view["total_entries"] == 0
        assert len(view["by_type"]) == 3  # All three types present
        assert len(view["types_summary"]) == 3
        for summary in view["types_summary"]:
            assert summary["count"] == 0
            assert summary["avg_confidence"] == 0.0

    def test_groups_by_type(self, kb, sample_scene_solution,
                            sample_param_effect, sample_data_strategy):
        """get_structured_view should group entries by knowledge_type."""
        kb.add_knowledge(sample_scene_solution)
        kb.add_knowledge(sample_param_effect)
        kb.add_knowledge(sample_data_strategy)

        view = kb.get_structured_view()
        assert view["total_entries"] == 3
        assert view["by_type"]["scene_solution"]["count"] == 1
        assert view["by_type"]["param_effect"]["count"] == 1
        assert view["by_type"]["data_strategy"]["count"] == 1

    def test_computes_avg_confidence(self, kb):
        """get_structured_view should compute average confidence per type."""
        for conf in [0.6, 0.8, 1.0]:
            entry = KnowledgeEntry(
                knowledge_type="scene_solution",
                content={"conf": conf},
                confidence=conf,
                source_iteration_id="iter_avg",
            )
            kb.add_knowledge(entry)

        view = kb.get_structured_view()
        avg = view["by_type"]["scene_solution"]["avg_confidence"]
        assert avg == pytest.approx(0.8)

    def test_entries_contain_expected_fields(self, kb, sample_scene_solution):
        """Each entry in the structured view should have expected fields."""
        kb.add_knowledge(sample_scene_solution)
        view = kb.get_structured_view()

        entries = view["by_type"]["scene_solution"]["entries"]
        assert len(entries) == 1
        entry = entries[0]
        assert "entry_id" in entry
        assert "content" in entry
        assert "confidence" in entry
        assert "source_iteration_id" in entry
        assert "created_at" in entry

    def test_types_summary_matches_by_type(self, kb, sample_scene_solution,
                                            sample_param_effect):
        """types_summary should be consistent with by_type data."""
        kb.add_knowledge(sample_scene_solution)
        kb.add_knowledge(sample_param_effect)

        view = kb.get_structured_view()
        summary_map = {s["type"]: s for s in view["types_summary"]}

        for ktype, data in view["by_type"].items():
            assert summary_map[ktype]["count"] == data["count"]
            assert summary_map[ktype]["avg_confidence"] == pytest.approx(
                data["avg_confidence"]
            )

    def test_all_three_types_always_present(self, kb, sample_scene_solution):
        """All three knowledge types should appear even if some have no entries."""
        kb.add_knowledge(sample_scene_solution)
        view = kb.get_structured_view()

        assert "scene_solution" in view["by_type"]
        assert "param_effect" in view["by_type"]
        assert "data_strategy" in view["by_type"]

    def test_multiple_entries_per_type(self, kb):
        """get_structured_view should handle multiple entries per type."""
        for i in range(5):
            entry = KnowledgeEntry(
                knowledge_type="param_effect",
                content={"index": i},
                confidence=0.5,
                source_iteration_id=f"iter_{i}",
            )
            kb.add_knowledge(entry)

        view = kb.get_structured_view()
        assert view["by_type"]["param_effect"]["count"] == 5
        assert len(view["by_type"]["param_effect"]["entries"]) == 5


# ---------------------------------------------------------------------------
# update_confidence tests
# ---------------------------------------------------------------------------


class TestUpdateConfidence:
    """Tests for SkillKnowledgeBase.update_confidence."""

    def test_updates_confidence_value(self, kb, db, sample_scene_solution):
        """update_confidence should change the confidence in the database."""
        entry_id = kb.add_knowledge(sample_scene_solution)
        kb.update_confidence(entry_id, 0.95)

        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["confidence"] == pytest.approx(0.95)

    def test_updates_updated_at_timestamp(self, kb, db):
        """update_confidence should update the updated_at timestamp."""
        entry = KnowledgeEntry(
            knowledge_type="scene_solution",
            content={"test": True},
            confidence=0.5,
            source_iteration_id="iter_ts",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00",
        )
        entry_id = kb.add_knowledge(entry)
        kb.update_confidence(entry_id, 0.9)

        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["updated_at"] != "2024-01-01T00:00:00"

    def test_raises_for_nonexistent_entry(self, kb):
        """update_confidence should raise KeyError for unknown entry_id."""
        with pytest.raises(KeyError, match="not found"):
            kb.update_confidence("ke_nonexistent", 0.5)

    def test_raises_for_confidence_above_one(self, kb, sample_scene_solution):
        """update_confidence should raise ValueError for confidence > 1.0."""
        entry_id = kb.add_knowledge(sample_scene_solution)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            kb.update_confidence(entry_id, 1.5)

    def test_raises_for_negative_confidence(self, kb, sample_scene_solution):
        """update_confidence should raise ValueError for confidence < 0.0."""
        entry_id = kb.add_knowledge(sample_scene_solution)
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            kb.update_confidence(entry_id, -0.1)

    def test_boundary_confidence_zero(self, kb, db, sample_scene_solution):
        """update_confidence should accept 0.0 as a valid confidence."""
        entry_id = kb.add_knowledge(sample_scene_solution)
        kb.update_confidence(entry_id, 0.0)

        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["confidence"] == pytest.approx(0.0)

    def test_boundary_confidence_one(self, kb, db, sample_scene_solution):
        """update_confidence should accept 1.0 as a valid confidence."""
        entry_id = kb.add_knowledge(sample_scene_solution)
        kb.update_confidence(entry_id, 1.0)

        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["confidence"] == pytest.approx(1.0)

    def test_multiple_updates(self, kb, db, sample_scene_solution):
        """Multiple confidence updates should all take effect."""
        entry_id = kb.add_knowledge(sample_scene_solution)

        kb.update_confidence(entry_id, 0.3)
        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["confidence"] == pytest.approx(0.3)

        kb.update_confidence(entry_id, 0.9)
        row = db.get_by_id("knowledge_entries", "entry_id", entry_id)
        assert row["confidence"] == pytest.approx(0.9)
