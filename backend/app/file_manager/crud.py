from pathlib import Path
from typing import AsyncIterator

import aiofiles
from aiofiles import os as aioos

FILES_DIR = Path(__file__).resolve().parent / "storage"
UPLOAD_CHUNK_SIZE = 1024 * 1024 * 100


def ensure_files_dir() -> Path:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    return FILES_DIR


def sanitize_filename(filename: str) -> str:
    cleaned = (filename or "").strip()
    if not cleaned:
        raise ValueError("Filename is required")
    if Path(cleaned).name != cleaned:
        raise ValueError("Invalid filename")
    if cleaned.startswith(".") or cleaned in {".", ".."}:
        raise ValueError("Invalid filename")
    return cleaned


def resolve_file_path(filename: str) -> Path:
    sanitized = sanitize_filename(filename)
    return ensure_files_dir() / sanitized


async def list_files() -> list[dict[str, object]]:
    files_dir = ensure_files_dir()
    if not files_dir.is_dir():
        return []
    entries = await aioos.listdir(files_dir)
    files: list[dict[str, object]] = []
    for name in entries:
        if name.startswith("."):
            continue
        path = files_dir / name
        if not path.is_file():
            continue
        stat = await aioos.stat(path)
        files.append({"name": name, "size": stat.st_size, "modified": stat.st_mtime})
    return sorted(files, key=lambda item: str(item["name"]).lower())


async def save_upload(filename: str, reader: AsyncIterator[bytes]) -> dict[str, object]:
    sanitized = sanitize_filename(filename)
    file_path = ensure_files_dir() / sanitized
    total_size = 0
    async with aiofiles.open(file_path, "wb") as out_file:
        async for chunk in reader:
            if not chunk:
                continue
            total_size += len(chunk)
            await out_file.write(chunk)
    return {"filename": sanitized, "size": total_size}


async def read_file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    async with aiofiles.open(path, "rb") as file_handle:
        while True:
            chunk = await file_handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


async def delete_file(filename: str) -> str:
    file_path = resolve_file_path(filename)
    if not file_path.is_file():
        raise FileNotFoundError("File not found")
    await aioos.remove(file_path)
    return file_path.name
