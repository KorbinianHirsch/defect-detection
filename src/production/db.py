"""SQLite event log for the production simulation.

One row per part processed by the worker: score, decision, latency, and
(only because this replays labeled benchmark data) the true defect label,
so the dashboard can show a live accuracy panel.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "production" / "events.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    category TEXT NOT NULL,
    filename TEXT NOT NULL,
    true_label TEXT,
    score REAL NOT NULL,
    threshold REAL NOT NULL,
    is_anomaly INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    sorted_path TEXT NOT NULL
);
"""


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)


@contextmanager
def connect(db_path: Path = DB_PATH):
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_event(
    conn: sqlite3.Connection,
    *,
    timestamp: str,
    category: str,
    filename: str,
    true_label: str | None,
    score: float,
    threshold: float,
    is_anomaly: bool,
    latency_ms: float,
    sorted_path: str,
) -> None:
    conn.execute(
        "INSERT INTO events "
        "(timestamp, category, filename, true_label, score, threshold, is_anomaly, latency_ms, sorted_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, category, filename, true_label, score, threshold, int(is_anomaly), latency_ms, sorted_path),
    )
    conn.commit()
