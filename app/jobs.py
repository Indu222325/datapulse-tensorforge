"""Asynchronous batch jobs: SQLite-backed store + one background worker thread.

Design choices (each maps to a rule in the API contract):
  * State lives in a SQLite file, not in process memory -> survives restarts, shareable between workers.
  * A job found in `running` state at start-up was interrupted by a restart -> marked `failed`
    with error.code = "interrupted". Jobs still `queued` are simply picked up again.
  * One worker thread processes one job at a time in small chunks, so /health and /predict stay
    responsive and `processed` moves steadily.
  * Capacity: 1 running + 3 queued. A 5th submission gets 429.
  * Finished jobs are kept for JOB_RETENTION_SECONDS (default 24 h, contract minimum is 6 h);
    afterwards the id answers 410. Deleted ids answer 404.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger("datapulse.jobs")

MAX_ACTIVE_JOBS = 4          # 1 running + 3 queued
CHUNK = 100
ACTIVE = ("queued", "running")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY, status TEXT NOT NULL, total INTEGER NOT NULL, processed INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, started_at REAL, finished_at REAL, expires_at REAL, model_version TEXT,
    error_code TEXT, error_message TEXT, idem_key TEXT, tickets TEXT, results TEXT
);
CREATE INDEX IF NOT EXISTS jobs_idem ON jobs(idem_key);
CREATE TABLE IF NOT EXISTS expired (job_id TEXT PRIMARY KEY, at REAL NOT NULL);
"""


def iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class QueueFull(Exception):
    pass


class JobStore:
    def __init__(self, db_path: str | Path, retention_seconds: int):
        self.retention = retention_seconds
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)

    # ---- lifecycle ------------------------------------------------------
    def recover_after_restart(self) -> int:
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status='failed', error_code='interrupted', "
                "error_message='Service restarted while the job was running.', finished_at=?, expires_at=?, "
                "tickets=NULL WHERE status='running'", (now, now + self.retention))
            return cur.rowcount

    def purge_expired(self) -> None:
        now = time.time()
        with self._lock:
            rows = self._db.execute("SELECT job_id FROM jobs WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)).fetchall()
            for row in rows:
                self._db.execute("INSERT OR REPLACE INTO expired(job_id, at) VALUES (?, ?)", (row["job_id"], now))
                self._db.execute("DELETE FROM jobs WHERE job_id=?", (row["job_id"],))
            self._db.execute("DELETE FROM expired WHERE at < ?", (now - 7 * 86400,))

    # ---- API operations -------------------------------------------------
    def submit(self, tickets: list[dict], model_version: str, idem_key: str | None) -> tuple[dict, bool]:
        """Returns (status dict, created?). Raises QueueFull when capacity is reached."""
        self.purge_expired()
        with self._lock:
            if idem_key:
                row = self._db.execute("SELECT * FROM jobs WHERE idem_key=? ORDER BY created_at DESC LIMIT 1", (idem_key,)).fetchone()
                if row is not None:
                    return self._status(row), False
            active = self._db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if active >= MAX_ACTIVE_JOBS:
                raise QueueFull()
            job_id = str(uuid.uuid4())
            self._db.execute(
                "INSERT INTO jobs(job_id, status, total, processed, created_at, model_version, idem_key, tickets) "
                "VALUES (?, 'queued', ?, 0, ?, ?, ?, ?)",
                (job_id, len(tickets), time.time(), model_version, idem_key, json.dumps(tickets)))
            row = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return self._status(row), True

    def lookup(self, job_id: str) -> tuple[str, dict | None]:
        """-> ("ok", status) | ("expired", None) | ("missing", None)"""
        self.purge_expired()
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is not None:
                return "ok", self._status(row)
            gone = self._db.execute("SELECT 1 FROM expired WHERE job_id=?", (job_id,)).fetchone()
            return ("expired" if gone else "missing"), None

    def results(self, job_id: str) -> list[dict] | None:
        with self._lock:
            row = self._db.execute("SELECT results FROM jobs WHERE job_id=? AND status='succeeded'", (job_id,)).fetchone()
        return json.loads(row["results"]) if row and row["results"] else None

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM jobs WHERE job_id=?", (job_id,)).rowcount > 0

    # ---- worker side ----------------------------------------------------
    def claim_next(self) -> tuple[str, list[dict]] | None:
        with self._lock:
            row = self._db.execute("SELECT job_id, tickets FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE jobs SET status='running', started_at=? WHERE job_id=?", (time.time(), row["job_id"]))
            return row["job_id"], json.loads(row["tickets"])

    def progress(self, job_id: str, processed: int) -> bool:
        """Returns False when the job was deleted (= cancelled) meanwhile."""
        with self._lock:
            return self._db.execute("UPDATE jobs SET processed=? WHERE job_id=? AND status='running'", (processed, job_id)).rowcount > 0

    def finish(self, job_id: str, results: list[dict]) -> None:
        now = time.time()
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status='succeeded', processed=total, finished_at=?, expires_at=?, results=?, tickets=NULL "
                "WHERE job_id=? AND status='running'", (now, now + self.retention, json.dumps(results), job_id))

    def fail(self, job_id: str, code: str, message: str) -> None:
        now = time.time()
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status='failed', error_code=?, error_message=?, finished_at=?, expires_at=?, tickets=NULL "
                "WHERE job_id=? AND status='running'", (code, message, now, now + self.retention, job_id))

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _status(row: sqlite3.Row) -> dict:
        error = None
        if row["status"] == "failed":
            error = {"code": row["error_code"] or "internal_error", "message": row["error_message"] or "Job failed."}
        return {
            "job_id": row["job_id"], "status": row["status"], "total": row["total"], "processed": row["processed"],
            "created_at": iso(row["created_at"]), "started_at": iso(row["started_at"]),
            "finished_at": iso(row["finished_at"]), "expires_at": iso(row["expires_at"]),
            "model_version": row["model_version"], "error": error,
        }


class JobWorker(threading.Thread):
    """Single background thread: claims the oldest queued job and classifies it chunk by chunk."""

    def __init__(self, store: JobStore, predict_fn):
        super().__init__(name="job-worker", daemon=True)
        self.store = store
        self.predict_fn = predict_fn
        self.wake = threading.Event()
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()
        self.wake.set()

    def run(self) -> None:
        last_purge = 0.0
        while not self._halt.is_set():
            try:
                if time.time() - last_purge > 60:
                    self.store.purge_expired()
                    last_purge = time.time()
                claimed = self.store.claim_next()
                if claimed is None:
                    self.wake.wait(timeout=1.0)
                    self.wake.clear()
                    continue
                self._process(*claimed)
            except Exception:  # never let the worker die
                log.exception("job worker loop error")
                time.sleep(1.0)

    def _process(self, job_id: str, tickets: list[dict]) -> None:
        results: list[dict] = []
        try:
            for start in range(0, len(tickets), CHUNK):
                if self._halt.is_set():
                    return  # shutting down: job stays 'running' and is marked 'interrupted' on next start
                results.extend(self.predict_fn(tickets[start:start + CHUNK]))
                if not self.store.progress(job_id, len(results)):
                    return  # deleted / cancelled by the client
                time.sleep(0)  # give the API threads a turn
            self.store.finish(job_id, results)
        except Exception:
            log.exception("job %s failed", job_id)
            self.store.fail(job_id, "internal_error", "The job could not be completed.")
