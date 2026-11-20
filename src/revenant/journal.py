"""Append-only, SQLite-backed event journal.

The journal is the single source of truth. A run *is* its ordered list of
events; everything else (status, usage, wake-up time) is a cache that makes
listing and scheduling cheap.

Two properties matter for correctness:

* **Fencing.** Every append checks, inside the same transaction, that the
  writer still owns the run's lease. A worker that was presumed dead and
  replaced can never corrupt the history, however late it wakes up.
* **Gap-free ordering.** ``(run_id, seq)`` is the primary key and writers
  always append at ``seq == len(history)``, so two writers can never both
  succeed at the same position.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .errors import LeaseLost, RunNotFound

TERMINAL_STATUSES = ("completed", "failed")
TERMINAL_EVENTS = ("run_completed", "run_failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    workflow       TEXT NOT NULL,
    status         TEXT NOT NULL,
    input          TEXT NOT NULL,
    options        TEXT NOT NULL DEFAULT '{}',
    result         TEXT,
    error          TEXT,
    usage          TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    parent_run_id  TEXT,
    forked_at_seq  INTEGER,
    lease_owner    TEXT,
    lease_expires  REAL,
    wake_at        REAL,
    waiting_on     TEXT
);
CREATE TABLE IF NOT EXISTS events (
    run_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    type     TEXT NOT NULL,
    name     TEXT NOT NULL,
    payload  TEXT NOT NULL,
    ts       REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS signals (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    name      TEXT NOT NULL,
    payload   TEXT NOT NULL,
    ts        REAL NOT NULL,
    consumed  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
CREATE INDEX IF NOT EXISTS idx_signals_pending ON signals (run_id, name, consumed);
"""

_JSON_COLUMNS = ("input", "options", "result", "error", "usage")
_UPDATABLE = {"status", "result", "error", "usage", "wake_at", "waiting_on", "workflow"}


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int
    type: str
    name: str
    payload: dict[str, Any]
    ts: float


@dataclass(frozen=True)
class Signal:
    id: int
    run_id: str
    name: str
    payload: Any
    ts: float


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    workflow: str
    status: str
    input: Any
    options: dict[str, Any]
    result: Any
    error: dict[str, Any] | None
    usage: dict[str, Any] | None
    created_at: float
    updated_at: float
    parent_run_id: str | None
    forked_at_seq: int | None
    lease_owner: str | None
    lease_expires: float | None
    wake_at: float | None
    waiting_on: str | None


def _dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


class Journal:
    """Thread-safe handle on one journal database."""

    def __init__(
        self,
        path: str = "revenant.db",
        *,
        clock: Callable[[], float] = time.time,
        durability: str = "normal",
    ) -> None:
        if durability not in ("normal", "full"):
            raise ValueError("durability must be 'normal' or 'full'")
        self.path = str(path)
        self.clock = clock
        #: Test hook, called as ``before_append(event_type, seq)`` ahead of every
        #: write. Raising from it simulates a crash at that exact point.
        self.before_append: Callable[[str, int], None] | None = None
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA synchronous={durability.upper()}")
        with self._lock:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # ------------------------------------------------------------------ runs

    def create_run(
        self,
        run_id: str,
        workflow: str,
        input: Any,
        options: dict[str, Any] | None = None,
    ) -> RunRecord:
        now = self.clock()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, workflow, status, input, options, created_at, updated_at)"
                " VALUES (?, ?, 'pending', ?, ?, ?, ?)",
                (run_id, workflow, _dumps(input), _dumps(options or {}), now, now),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFound(f"no run with id {run_id!r}")
        return self._to_run(row)

    def find_run(self, prefix: str) -> RunRecord:
        """Resolve a run by id or by unique id prefix."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE run_id = ? OR run_id LIKE ? ORDER BY created_at LIMIT 3",
                (prefix, prefix.replace("%", "").replace("_", "") + "%"),
            ).fetchall()
        exact = [r for r in rows if r["run_id"] == prefix]
        if exact:
            return self._to_run(exact[0])
        if not rows:
            raise RunNotFound(f"no run matching {prefix!r}")
        if len(rows) > 1:
            raise RunNotFound(f"run id prefix {prefix!r} is ambiguous")
        return self._to_run(rows[0])

    def list_runs(self, status: str | None = None, limit: int = 100) -> list[RunRecord]:
        query = "SELECT * FROM runs"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC, run_id LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._to_run(r) for r in rows]

    def update_run(self, run_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"cannot update run fields: {sorted(unknown)}")
        columns = []
        params: list[Any] = []
        for key, value in fields.items():
            columns.append(f"{key} = ?")
            if key in _JSON_COLUMNS and value is not None:
                value = _dumps(value)
            params.append(value)
        columns.append("updated_at = ?")
        params.extend([self.clock(), run_id])
        with self._tx() as conn:
            conn.execute(f"UPDATE runs SET {', '.join(columns)} WHERE run_id = ?", params)

    def due_runs(self, now: float) -> list[str]:
        """Runs a worker should pick up right now.

        That is: never-started runs, timers that have fired, signal waits whose
        signal has arrived, and runs whose worker died (expired lease).
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT run_id FROM runs
                WHERE (lease_owner IS NULL OR lease_expires < :now)
                  AND (
                        status = 'pending'
                     OR status = 'running'
                     OR (status = 'suspended' AND wake_at IS NOT NULL AND wake_at <= :now)
                     OR (status = 'suspended' AND waiting_on IS NOT NULL AND EXISTS (
                            SELECT 1 FROM signals s
                            WHERE s.run_id = runs.run_id
                              AND s.name = runs.waiting_on
                              AND s.consumed = 0))
                  )
                ORDER BY created_at, run_id
                """,
                {"now": now},
            ).fetchall()
        return [r["run_id"] for r in rows]

    # ---------------------------------------------------------------- leases

    def acquire_lease(self, run_id: str, owner: str, ttl: float, *, force: bool = False) -> bool:
        now = self.clock()
        with self._tx() as conn:
            if force:
                cur = conn.execute(
                    "UPDATE runs SET lease_owner = ?, lease_expires = ? WHERE run_id = ?",
                    (owner, now + ttl, run_id),
                )
            else:
                cur = conn.execute(
                    "UPDATE runs SET lease_owner = ?, lease_expires = ? WHERE run_id = ?"
                    " AND (lease_owner IS NULL OR lease_owner = ? OR lease_expires < ?)",
                    (owner, now + ttl, run_id, owner, now),
                )
            return cur.rowcount == 1

    def renew_lease(self, run_id: str, owner: str, ttl: float) -> bool:
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE runs SET lease_expires = ? WHERE run_id = ? AND lease_owner = ?",
                (self.clock() + ttl, run_id, owner),
            )
            return cur.rowcount == 1

    def release_lease(self, run_id: str, owner: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE runs SET lease_owner = NULL, lease_expires = NULL"
                " WHERE run_id = ? AND lease_owner = ?",
                (run_id, owner),
            )

    # ---------------------------------------------------------------- events

    def events(self, run_id: str) -> list[Event]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY seq", (run_id,)
            ).fetchall()
        return [
            Event(r["run_id"], r["seq"], r["type"], r["name"], json.loads(r["payload"]), r["ts"])
            for r in rows
        ]

    def append(
        self,
        run_id: str,
        seq: int,
        type: str,
        name: str,
        payload: dict[str, Any],
        *,
        owner: str,
        lease_ttl: float,
        consume_signal: int | None = None,
    ) -> Event:
        """Append one event, fenced by the run's lease."""
        if self.before_append is not None:
            self.before_append(type, seq)
        data = _dumps(payload)
        ts = self.clock()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT lease_owner FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise RunNotFound(f"no run with id {run_id!r}")
            if row["lease_owner"] != owner:
                raise LeaseLost(
                    f"worker {owner!r} no longer owns run {run_id} "
                    f"(current owner: {row['lease_owner']!r})"
                )
            try:
                conn.execute(
                    "INSERT INTO events (run_id, seq, type, name, payload, ts)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, seq, type, name, data, ts),
                )
            except sqlite3.IntegrityError as exc:
                raise LeaseLost(
                    f"event {seq} of run {run_id} was already written by another worker"
                ) from exc
            if consume_signal is not None:
                conn.execute("UPDATE signals SET consumed = 1 WHERE id = ?", (consume_signal,))
            conn.execute(
                "UPDATE runs SET lease_expires = ?, updated_at = ? WHERE run_id = ?",
                (ts + lease_ttl, ts, run_id),
            )
        return Event(run_id, seq, type, name, json.loads(data), ts)

    # --------------------------------------------------------------- signals

    def add_signal(self, run_id: str, name: str, payload: Any) -> int:
        self.get_run(run_id)
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO signals (run_id, name, payload, ts) VALUES (?, ?, ?, ?)",
                (run_id, name, _dumps(payload), self.clock()),
            )
            return int(cur.lastrowid or 0)

    def next_signal(self, run_id: str, name: str) -> Signal | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM signals WHERE run_id = ? AND name = ? AND consumed = 0"
                " ORDER BY id LIMIT 1",
                (run_id, name),
            ).fetchone()
        if row is None:
            return None
        return Signal(row["id"], row["run_id"], row["name"], json.loads(row["payload"]), row["ts"])

    # ------------------------------------------------------------------ fork

    def fork(
        self,
        run_id: str,
        new_run_id: str,
        at_seq: int,
        *,
        workflow: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Create a new run sharing the first ``at_seq`` events of ``run_id``."""
        source = self.get_run(run_id)
        now = self.clock()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, workflow, status, input, options, created_at,"
                " updated_at, parent_run_id, forked_at_seq)"
                " VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)",
                (
                    new_run_id,
                    workflow or source.workflow,
                    _dumps(source.input),
                    _dumps(source.options if options is None else options),
                    now,
                    now,
                    run_id,
                    at_seq,
                ),
            )
            conn.execute(
                "INSERT INTO events (run_id, seq, type, name, payload, ts)"
                " SELECT ?, seq, type, name, payload, ts FROM events"
                " WHERE run_id = ? AND seq < ?",
                (new_run_id, run_id, at_seq),
            )
        return self.get_run(new_run_id)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _to_run(row: sqlite3.Row) -> RunRecord:
        def load(column: str) -> Any:
            raw = row[column]
            return None if raw is None else json.loads(raw)

        return RunRecord(
            run_id=row["run_id"],
            workflow=row["workflow"],
            status=row["status"],
            input=load("input"),
            options=load("options") or {},
            result=load("result"),
            error=load("error"),
            usage=load("usage"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            parent_run_id=row["parent_run_id"],
            forked_at_seq=row["forked_at_seq"],
            lease_owner=row["lease_owner"],
            lease_expires=row["lease_expires"],
            wake_at=row["wake_at"],
            waiting_on=row["waiting_on"],
        )
