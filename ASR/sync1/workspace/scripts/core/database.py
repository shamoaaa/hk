"""SQLite database initialization and CRUD operations for the ASR Training Optimizer.

Provides the Database class for managing SQLite connections and performing
CRUD operations on the six core tables: iterations, effect_feedbacks,
knowledge_entries, skill_versions, strategy_weights, and param_history.

Requirements: 9.1, 10.1, 10.5, 10.7
"""

import json
import sqlite3
from contextlib import contextmanager
from typing import Optional


# SQL statements for creating the six core tables
_CREATE_TABLES_SQL = [
    """
    CREATE TABLE IF NOT EXISTS iterations (
        iteration_id TEXT PRIMARY KEY,
        description TEXT,
        status TEXT CHECK(status IN ('pending', 'running', 'completed', 'failed')),
        cer_before REAL,
        cer_after REAL,
        wer_before REAL,
        wer_after REAL,
        param_changes_json TEXT,
        model_changes_json TEXT,
        data_changes_json TEXT,
        created_at TEXT,
        completed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS effect_feedbacks (
        feedback_id TEXT PRIMARY KEY,
        iteration_id TEXT REFERENCES iterations(iteration_id),
        suggestion_id TEXT,
        suggestion_content TEXT,
        input_analysis_json TEXT,
        param_changes_json TEXT,
        cer_before REAL,
        cer_after REAL,
        wer_before REAL,
        wer_after REAL,
        is_positive INTEGER,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS knowledge_entries (
        entry_id TEXT PRIMARY KEY,
        knowledge_type TEXT CHECK(knowledge_type IN ('scene_solution', 'param_effect', 'data_strategy')),
        content_json TEXT,
        confidence REAL,
        source_iteration_id TEXT,
        created_at TEXT,
        updated_at TEXT,
        state TEXT DEFAULT 'active' CHECK(state IN ('active', 'stale', 'archived')),
        verified_count INTEGER DEFAULT 0,
        last_verified_at TEXT,
        auto_confidence REAL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS skill_versions (
        version_id TEXT PRIMARY KEY,
        strategy_rules_json TEXT,
        knowledge_snapshot_path TEXT,
        change_description TEXT,
        positive_feedback_rate REAL,
        avg_cer_improvement REAL,
        created_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS strategy_weights (
        strategy_name TEXT PRIMARY KEY,
        weight REAL,
        success_rate REAL,
        sample_count INTEGER,
        version_id TEXT REFERENCES skill_versions(version_id),
        updated_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS param_history (
        history_id TEXT PRIMARY KEY,
        iteration_id TEXT REFERENCES iterations(iteration_id),
        config_json TEXT,
        cer_result REAL,
        wer_result REAL,
        created_at TEXT
    )
    """,
]

# Mapping from table name to its primary key column
_TABLE_PRIMARY_KEYS = {
    "iterations": "iteration_id",
    "effect_feedbacks": "feedback_id",
    "knowledge_entries": "entry_id",
    "skill_versions": "version_id",
    "strategy_weights": "strategy_name",
    "param_history": "history_id",
}


class Database:
    """SQLite database manager for the ASR Training Optimizer.

    Handles connection lifecycle, table initialization, and generic CRUD
    operations for all six core tables.

    Usage::

        db = Database("workspace/metadata.db")
        db.initialize()

        # Insert a row
        db.insert("iterations", {
            "iteration_id": "iter_001",
            "status": "pending",
            "created_at": "2024-01-01T00:00:00",
        })

        # Query rows
        rows = db.get_all("iterations", filters={"status": "pending"})

        # Context manager
        with Database("workspace/metadata.db") as db:
            db.initialize()
            db.insert(...)
    """

    def __init__(self, db_path: str = "metadata.db") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Return the current connection, creating one if needed.

        The connection is configured with:
        - ``sqlite3.Row`` row factory for dict-like access
        - WAL journal mode for better concurrent read performance
        - Foreign key enforcement enabled
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        """Close the database connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        self.get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all six tables if they don't already exist,
        and run lightweight ALTER TABLE migrations for any newly-added columns."""
        conn = self.get_connection()
        for sql in _CREATE_TABLES_SQL:
            conn.execute(sql)

        # ── Column migrations (idempotent) ──
        # Each migration: (table, column_name, column_def)
        # ALTER TABLE ADD COLUMN only runs if the column doesn't already exist.
        _MIGRATIONS = [
            ("knowledge_entries", "state", "TEXT DEFAULT 'active'"),
            ("knowledge_entries", "verified_count", "INTEGER DEFAULT 0"),
            ("knowledge_entries", "last_verified_at", "TEXT"),
            ("knowledge_entries", "auto_confidence", "REAL"),
        ]
        for table, col, col_def in _MIGRATIONS:
            # PRAGMA table_info returns rows: (cid, name, type, notnull, dflt_value, pk)
            cur = conn.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cur.fetchall()}
            if col not in existing:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                except Exception as e:
                    # Best-effort: log but don't block initialization
                    import logging
                    logging.getLogger(__name__).warning(
                        "Migration %s.%s failed: %s", table, col, e
                    )

        conn.commit()

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def insert(self, table: str, data: dict) -> str:
        """Insert a row into *table* and return its primary key value.

        Parameters
        ----------
        table:
            Target table name (must be one of the six known tables).
        data:
            Column-name → value mapping.  The primary key column **must**
            be present in *data*.

        Returns
        -------
        str
            The primary key value of the inserted row.
        """
        self._validate_table(table)
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        values = list(data.values())

        conn = self.get_connection()
        conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            values,
        )
        conn.commit()

        pk_col = _TABLE_PRIMARY_KEYS[table]
        return str(data[pk_col])

    def get_by_id(
        self, table: str, id_column: str, id_value: str
    ) -> Optional[dict]:
        """Return a single row as a dict, or ``None`` if not found.

        Parameters
        ----------
        table:
            Target table name.
        id_column:
            Column to match against (typically the primary key).
        id_value:
            Value to look up.
        """
        self._validate_table(table)
        conn = self.get_connection()
        cursor = conn.execute(
            f"SELECT * FROM {table} WHERE {id_column} = ?", (id_value,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_all(
        self,
        table: str,
        filters: Optional[dict] = None,
        order_by: Optional[str] = None,
    ) -> list[dict]:
        """Return all rows matching optional *filters*.

        Parameters
        ----------
        table:
            Target table name.
        filters:
            Column-name → value pairs used as ``WHERE col = ?`` clauses
            joined with ``AND``.  Pass ``None`` or ``{}`` for no filtering.
        order_by:
            Optional SQL ``ORDER BY`` clause content, e.g.
            ``"created_at DESC"``.
        """
        self._validate_table(table)
        conn = self.get_connection()

        query = f"SELECT * FROM {table}"
        params: list = []

        if filters:
            conditions = " AND ".join(f"{col} = ?" for col in filters)
            params = list(filters.values())
            query += f" WHERE {conditions}"

        if order_by:
            query += f" ORDER BY {order_by}"

        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def update(
        self, table: str, id_column: str, id_value: str, data: dict
    ) -> bool:
        """Update a row identified by *id_column*/*id_value*.

        Parameters
        ----------
        table:
            Target table name.
        id_column:
            Column used to locate the row.
        id_value:
            Value of the identifying column.
        data:
            Column-name → new-value mapping for the columns to update.

        Returns
        -------
        bool
            ``True`` if exactly one row was updated, ``False`` otherwise.
        """
        self._validate_table(table)
        set_clause = ", ".join(f"{col} = ?" for col in data)
        values = list(data.values()) + [id_value]

        conn = self.get_connection()
        cursor = conn.execute(
            f"UPDATE {table} SET {set_clause} WHERE {id_column} = ?",
            values,
        )
        conn.commit()
        return cursor.rowcount == 1

    def delete(self, table: str, id_column: str, id_value: str) -> bool:
        """Delete a row identified by *id_column*/*id_value*.

        Returns
        -------
        bool
            ``True`` if exactly one row was deleted, ``False`` otherwise.
        """
        self._validate_table(table)
        conn = self.get_connection()
        cursor = conn.execute(
            f"DELETE FROM {table} WHERE {id_column} = ?", (id_value,)
        )
        conn.commit()
        return cursor.rowcount == 1

    def count(self, table: str, filters: Optional[dict] = None) -> int:
        """Return the number of rows in *table*, optionally filtered.

        Parameters
        ----------
        table:
            Target table name.
        filters:
            Optional column-name → value pairs for ``WHERE`` filtering.
        """
        self._validate_table(table)
        conn = self.get_connection()

        query = f"SELECT COUNT(*) FROM {table}"
        params: list = []

        if filters:
            conditions = " AND ".join(f"{col} = ?" for col in filters)
            params = list(filters.values())
            query += f" WHERE {conditions}"

        cursor = conn.execute(query, params)
        return cursor.fetchone()[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_table(table: str) -> None:
        """Raise ``ValueError`` if *table* is not a known table name."""
        if table not in _TABLE_PRIMARY_KEYS:
            raise ValueError(
                f"Unknown table '{table}'. "
                f"Valid tables: {', '.join(sorted(_TABLE_PRIMARY_KEYS))}"
            )
