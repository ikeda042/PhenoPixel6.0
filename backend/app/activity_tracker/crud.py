from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from app.activity_tracker.async_databases import get_db

ACTION_TOP_PAGE = "top_page"
ACTION_CELL_EXTRACTION = "cell_extraction"
ACTION_BULK_ENGINE = "bulk_engine"

ALLOWED_ACTIONS = {
    ACTION_TOP_PAGE,
    ACTION_CELL_EXTRACTION,
    ACTION_BULK_ENGINE,
}

_MAX_ACTION_LENGTH = 64


def _normalize_action_name(action_name: str) -> str:
    cleaned = (action_name or "").strip()
    if not cleaned:
        raise ValueError("Action name is required")
    if len(cleaned) > _MAX_ACTION_LENGTH:
        raise ValueError("Action name is too long")
    return cleaned


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


async def record_activity(action_name: str, created_at: datetime | None = None) -> None:
    normalized = _normalize_action_name(action_name)
    timestamp = created_at or datetime.utcnow()
    async with get_db() as db:
        await db.execute(
            "INSERT INTO activity_log (created_at, action_name) VALUES (?, ?)",
            (_format_timestamp(timestamp), normalized),
        )
        await db.commit()


def record_activity_sync(action_name: str, created_at: datetime | None = None) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(record_activity(action_name, created_at=created_at))
        return
    loop.create_task(record_activity(action_name, created_at=created_at))


async def get_daily_activity_summary(
    days: int = 7,
    action_name: str | None = None,
) -> dict[str, Any]:
    if days < 1:
        raise ValueError("Days must be at least 1")
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days - 1)

    params: list[Any] = [start_date.isoformat(), end_date.isoformat()]
    filter_clause = ""
    if action_name:
        normalized = _normalize_action_name(action_name)
        filter_clause = "AND action_name = ?"
        params.append(normalized)

    query = f"""
        SELECT date(created_at) AS day, COUNT(*) AS count
        FROM activity_log
        WHERE date(created_at) BETWEEN ? AND ?
        {filter_clause}
        GROUP BY date(created_at)
        ORDER BY date(created_at)
    """

    async with get_db() as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

    counts_by_day = {row["day"]: row["count"] for row in rows}
    points: list[dict[str, Any]] = []
    total = 0
    for index in range(days):
        day = start_date + timedelta(days=index)
        day_str = day.isoformat()
        count = int(counts_by_day.get(day_str, 0))
        total += count
        points.append({"date": day_str, "count": count})

    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "points": points,
        "total": total,
    }
