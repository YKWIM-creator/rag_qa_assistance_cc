import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional


class ScraperDB:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS queue (
                url      TEXT PRIMARY KEY,
                school   TEXT NOT NULL,
                status   TEXT NOT NULL DEFAULT 'pending',
                added_at TEXT
            );
            CREATE TABLE IF NOT EXISTS pages (
                url          TEXT PRIMARY KEY,
                school       TEXT NOT NULL,
                title        TEXT,
                markdown     TEXT,
                content_hash TEXT,
                page_type    TEXT NOT NULL DEFAULT 'general',
                scraped_at   TEXT
            );
        """)
        self.conn.commit()

    def enqueue_new(self, urls: list[str], school: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.executemany(
            "INSERT OR IGNORE INTO queue (url, school, status, added_at) VALUES (?, ?, 'pending', ?)",
            [(url, school, now) for url in urls],
        )
        self.conn.commit()

    def next_pending(self) -> Optional[str]:
        row = self.conn.execute(
            "SELECT url FROM queue WHERE status = 'pending' LIMIT 1"
        ).fetchone()
        return row["url"] if row else None

    def mark(self, url: str, status: str):
        self.conn.execute("UPDATE queue SET status = ? WHERE url = ?", (status, url))
        self.conn.commit()

    def hash_exists(self, content_hash: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM pages WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row is not None

    def save_page(self, url: str, school: str, title: str, markdown: str,
                  content_hash: str, page_type: str = "general"):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO pages
               (url, school, title, markdown, content_hash, page_type, scraped_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (url, school, title, markdown, content_hash, page_type, now),
        )
        self.conn.commit()

    def get_pages_for_school(self, school: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM pages WHERE school = ?", (school,)
        ).fetchall()
        return [dict(row) for row in rows]

    def clear_school(self, school: str):
        self.conn.execute("DELETE FROM queue WHERE school = ?", (school,))
        self.conn.execute("DELETE FROM pages WHERE school = ?", (school,))
        self.conn.commit()

    def close(self):
        self.conn.close()
