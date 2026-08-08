"""Persistent scan history backed by SQLite.

Stores one row per completed detection (image or video): the media name,
the verdict + confidence, the per-model scores, the paths of the artifacts
(heatmap / face crop / source media), and a UTC timestamp.

Thread-safety: sqlite3 connections are created per operation so the Flask
workers and the session sweeper never share a connection concurrently.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import resolve

_lock = threading.RLock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    kind        TEXT NOT NULL,            -- 'image' | 'video'
    original_name TEXT NOT NULL,
    verdict     TEXT NOT NULL,            -- 'REAL' | 'FAKE'
    confidence  REAL NOT NULL,
    p_fake      REAL NOT NULL,
    threshold   REAL NOT NULL,
    scores      TEXT NOT NULL,            -- JSON dict of per-model scores
    frames_analyzed INTEGER,
    media_url   TEXT NOT NULL,
    face_url    TEXT,
    heatmap_url TEXT,
    created_at  TEXT NOT NULL             -- ISO-8601 UTC
);
CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at DESC);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_history(db_path=None) -> Path:
    """Create the schema if missing and return the resolved DB path."""
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        conn.executescript(_SCHEMA)
    return path


def add_scan(record: dict, db_path=None) -> int:
    """Insert one detection result. `record` is the API result dict."""
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO scans
                (session_id, kind, original_name, verdict, confidence,
                 p_fake, threshold, scores, frames_analyzed,
                 media_url, face_url, heatmap_url, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("session_id", ""),
                record.get("kind", "image"),
                record.get("original_name", ""),
                record.get("verdict", "REAL"),
                float(record.get("confidence", 0.0)),
                float(record.get("p_fake", 0.0)),
                float(record.get("threshold", 0.5)),
                json.dumps(record.get("scores") or {}),
                record.get("faces_analyzed"),
                record.get("media_url", ""),
                record.get("face_url"),
                record.get("heatmap_url"),
                record.get("created_at") or datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_scans(limit: int = 100, offset: int = 0,
               kind: str | None = None, verdict: str | None = None,
               db_path=None) -> list[dict]:
    """Return scan history rows newest-first, with the SQLite integer id."""
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    sql = "SELECT * FROM scans WHERE 1=1"
    params: list = []
    if kind in ("image", "video"):
        sql += " AND kind = ?"
        params.append(kind)
    if verdict in ("REAL", "FAKE"):
        sql += " AND verdict = ?"
        params.append(verdict)
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params += [int(limit), int(offset)]

    with _lock, _connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_scan(scan_id: int, db_path=None) -> dict | None:
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?",
                           (int(scan_id),)).fetchone()
    return _row_to_dict(row) if row else None


def count_scans(db_path=None) -> int:
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0])


def delete_scan(scan_id: int, db_path=None) -> bool:
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        cur = conn.execute("DELETE FROM scans WHERE id = ?", (int(scan_id),))
        conn.commit()
    return cur.rowcount > 0


def clear_all(db_path=None) -> int:
    path = Path(db_path) if db_path else resolve("data") / "history.db"
    with _lock, _connect(path) as conn:
        cur = conn.execute("DELETE FROM scans")
        conn.commit()
    return int(cur.rowcount)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = {k: row[k] for k in row.keys()}
    try:
        d["scores"] = json.loads(d["scores"]) if d.get("scores") else {}
    except (ValueError, TypeError):
        d["scores"] = {}
    return d