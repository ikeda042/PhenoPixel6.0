from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

DB_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DB_DIR / "activity_tracker.db"

_INITIALIZED = False

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    action_name TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activity_log_created_at
    ON activity_log (created_at);
CREATE INDEX IF NOT EXISTS idx_activity_log_action_name
    ON activity_log (action_name);
"""


async def init_db() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    DB_DIR.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_SCHEMA)
        await db.commit()
    _INITIALIZED = True


@asynccontextmanager
async def get_db() -> aiosqlite.Connection:
    await init_db()
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()
