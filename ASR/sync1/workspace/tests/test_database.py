"""Tests for the SQLite database module.

Covers initialization, CRUD operations, connection management,
and table validation for all six core tables.
"""

import os
import sqlite3
import tempfile

import pytest

from scripts.core.database import Database, _TABLE_PRIMARY_KEYS


@pytest.fixture
def db_path(tmp_path):
    """Return a temporary database file path."""
    return str(tmp_path / "test.db")


@pytest.fixture
def db(db_path):
    """Return an initialized Database instance (auto-closed after test)."""
    database = Database(db_path)
    database.initialize()
    yield database
    database.close()


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestInitialization:
    def test_creates_all_six_tables(self, db):
        """initialize() should create all six expected tables."""
        conn = db.get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = sorted(row["name"] for row in cursor.fetchall())
        expected = sorted(_TABLE_PRIMARY_KEYS.keys())
        assert tables == expected

    def test_initialize_is_idempotent(self, db):
        """Calling initialize() twice should not raise."""
        db.initialize()  # second call
        conn = db.get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = sorted(row["name"] for row in cursor.fetchall())
        assert len(tables) == 6

    def test_wal_mode_enabled(self, db):
        """Connection should use WAL journal mode."""
        conn = db.get_connection()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, db):
        """Foreign key enforcement should be on."""
        conn = db.get_connection()
        cursor = conn.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1


# ------------------------------------------------------------------
# Connection management
# ------------------------------------------------------------------


class TestConnectionManagement:
    def test_context_manager(self, db_path):
        """Database should work as a context manager."""
        with Database(db_path) as database:
            database.initialize()
            assert database._conn is not None
        # After exiting, connection should be closed
        assert database._conn is None

    def test_close_when_not_connected(self, db_path):
        """close() on an unopened database should not raise."""
        database = Database(db_path)
        database.close()  # should be a no-op

    def test_get_connection_returns_same_object(self, db):
        """Repeated get_connection() calls return the same connection."""
        conn1 = db.get_connection()
        conn2 = db.get_connection()
        assert conn1 is conn2

    def test_row_factory_is_sqlite_row(self, db):
        """Connection row_factory should be sqlite3.Row."""
        conn = db.get_connection()
        assert conn.row_factory is sqlite3.Row


# ------------------------------------------------------------------
# CRUD: insert / get_by_id
# ------------------------------------------------------------------


class TestInsertAndGetById:
    def test_insert_and_retrieve_iteration(self, db):
        row = {
            "iteration_id": "iter_001",
            "description": "First iteration",
            "status": "pending",
            "cer_before": 0.15,
            "cer_after": None,
            "wer_before": 0.25,
            "wer_after": None,
            "param_changes_json": "{}",
            "model_changes_json": "{}",
            "data_changes_json": "{}",
            "created_at": "2024-01-01T00:00:00",
            "completed_at": None,
        }
        pk = db.insert("iterations", row)
        assert pk == "iter_001"

        result = db.get_by_id("iterations", "iteration_id", "iter_001")
        assert result is not None
        assert result["description"] == "First iteration"
        assert result["status"] == "pending"
        assert result["cer_before"] == 0.15

    def test_insert_and_retrieve_effect_feedback(self, db):
        row = {
            "feedback_id": "fb_001",
            "iteration_id": None,
            "suggestion_id": "sug_001",
            "suggestion_content": "Increase LR",
            "input_analysis_json": "{}",
            "param_changes_json": "{}",
            "cer_before": 0.12,
            "cer_after": 0.10,
            "wer_before": 0.20,
            "wer_after": 0.18,
            "is_positive": 1,
            "created_at": "2024-01-01",
        }
        pk = db.insert("effect_feedbacks", row)
        assert pk == "fb_001"

        result = db.get_by_id("effect_feedbacks", "feedback_id", "fb_001")
        assert result["is_positive"] == 1

    def test_insert_knowledge_entry(self, db):
        row = {
            "entry_id": "ke_001",
            "knowledge_type": "scene_solution",
            "content_json": '{"scene": "background_voice"}',
            "confidence": 0.85,
            "source_iteration_id": None,
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        }
        pk = db.insert("knowledge_entries", row)
        assert pk == "ke_001"

    def test_insert_skill_version(self, db):
        row = {
            "version_id": "v001",
            "strategy_rules_json": "{}",
            "knowledge_snapshot_path": "/path/to/snapshot",
            "change_description": "Initial version",
            "positive_feedback_rate": 0.0,
            "avg_cer_improvement": 0.0,
            "created_at": "2024-01-01",
        }
        pk = db.insert("skill_versions", row)
        assert pk == "v001"

    def test_insert_strategy_weight(self, db):
        # Insert a skill_version first (foreign key)
        db.insert("skill_versions", {
            "version_id": "v001",
            "strategy_rules_json": "{}",
            "knowledge_snapshot_path": "",
            "change_description": "",
            "positive_feedback_rate": 0.0,
            "avg_cer_improvement": 0.0,
            "created_at": "2024-01-01",
        })
        row = {
            "strategy_name": "increase_lr",
            "weight": 0.8,
            "success_rate": 0.75,
            "sample_count": 10,
            "version_id": "v001",
            "updated_at": "2024-01-01",
        }
        pk = db.insert("strategy_weights", row)
        assert pk == "increase_lr"

    def test_insert_param_history(self, db):
        row = {
            "history_id": "ph_001",
            "iteration_id": None,
            "config_json": '{"learning_rate": 0.001}',
            "cer_result": 0.10,
            "wer_result": 0.18,
            "created_at": "2024-01-01",
        }
        pk = db.insert("param_history", row)
        assert pk == "ph_001"

    def test_get_by_id_returns_none_for_missing(self, db):
        result = db.get_by_id("iterations", "iteration_id", "nonexistent")
        assert result is None


# ------------------------------------------------------------------
# CRUD: get_all
# ------------------------------------------------------------------


class TestGetAll:
    def test_get_all_empty_table(self, db):
        rows = db.get_all("iterations")
        assert rows == []

    def test_get_all_with_data(self, db):
        for i in range(3):
            db.insert("iterations", {
                "iteration_id": f"iter_{i:03d}",
                "status": "pending" if i < 2 else "completed",
                "created_at": f"2024-01-0{i + 1}",
            })
        rows = db.get_all("iterations")
        assert len(rows) == 3

    def test_get_all_with_filter(self, db):
        for i in range(3):
            db.insert("iterations", {
                "iteration_id": f"iter_{i:03d}",
                "status": "pending" if i < 2 else "completed",
                "created_at": f"2024-01-0{i + 1}",
            })
        rows = db.get_all("iterations", filters={"status": "pending"})
        assert len(rows) == 2

    def test_get_all_with_order_by(self, db):
        db.insert("iterations", {
            "iteration_id": "iter_a",
            "status": "pending",
            "created_at": "2024-01-02",
        })
        db.insert("iterations", {
            "iteration_id": "iter_b",
            "status": "pending",
            "created_at": "2024-01-01",
        })
        rows = db.get_all("iterations", order_by="created_at ASC")
        assert rows[0]["iteration_id"] == "iter_b"
        assert rows[1]["iteration_id"] == "iter_a"


# ------------------------------------------------------------------
# CRUD: update
# ------------------------------------------------------------------


class TestUpdate:
    def test_update_existing_row(self, db):
        db.insert("iterations", {
            "iteration_id": "iter_001",
            "status": "pending",
            "created_at": "2024-01-01",
        })
        updated = db.update(
            "iterations", "iteration_id", "iter_001",
            {"status": "completed", "cer_after": 0.08},
        )
        assert updated is True

        row = db.get_by_id("iterations", "iteration_id", "iter_001")
        assert row["status"] == "completed"
        assert row["cer_after"] == 0.08

    def test_update_nonexistent_row(self, db):
        updated = db.update(
            "iterations", "iteration_id", "nonexistent",
            {"status": "completed"},
        )
        assert updated is False


# ------------------------------------------------------------------
# CRUD: delete
# ------------------------------------------------------------------


class TestDelete:
    def test_delete_existing_row(self, db):
        db.insert("iterations", {
            "iteration_id": "iter_001",
            "status": "pending",
            "created_at": "2024-01-01",
        })
        deleted = db.delete("iterations", "iteration_id", "iter_001")
        assert deleted is True
        assert db.get_by_id("iterations", "iteration_id", "iter_001") is None

    def test_delete_nonexistent_row(self, db):
        deleted = db.delete("iterations", "iteration_id", "nonexistent")
        assert deleted is False


# ------------------------------------------------------------------
# CRUD: count
# ------------------------------------------------------------------


class TestCount:
    def test_count_empty(self, db):
        assert db.count("iterations") == 0

    def test_count_with_rows(self, db):
        for i in range(5):
            db.insert("iterations", {
                "iteration_id": f"iter_{i:03d}",
                "status": "pending" if i < 3 else "completed",
                "created_at": "2024-01-01",
            })
        assert db.count("iterations") == 5

    def test_count_with_filter(self, db):
        for i in range(5):
            db.insert("iterations", {
                "iteration_id": f"iter_{i:03d}",
                "status": "pending" if i < 3 else "completed",
                "created_at": "2024-01-01",
            })
        assert db.count("iterations", filters={"status": "pending"}) == 3
        assert db.count("iterations", filters={"status": "completed"}) == 2


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


class TestValidation:
    def test_invalid_table_raises_value_error(self, db):
        with pytest.raises(ValueError, match="Unknown table"):
            db.insert("nonexistent_table", {"id": "1"})

    def test_invalid_table_on_get_all(self, db):
        with pytest.raises(ValueError, match="Unknown table"):
            db.get_all("bad_table")

    def test_invalid_table_on_count(self, db):
        with pytest.raises(ValueError, match="Unknown table"):
            db.count("bad_table")

    def test_status_check_constraint(self, db):
        """Inserting an invalid status should raise IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            db.insert("iterations", {
                "iteration_id": "iter_bad",
                "status": "invalid_status",
                "created_at": "2024-01-01",
            })

    def test_knowledge_type_check_constraint(self, db):
        """Inserting an invalid knowledge_type should raise IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            db.insert("knowledge_entries", {
                "entry_id": "ke_bad",
                "knowledge_type": "invalid_type",
                "content_json": "{}",
                "confidence": 0.5,
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
            })
