from pathlib import Path
from typing import Annotated

import aiofiles
from aiofiles import os as aioos
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse


router_extracted_data = APIRouter(tags=["extracted_data"])
EXTRACTED_DATA_DIR = Path(__file__).resolve().parent


async def _list_extracted_folders() -> list[str]:
    if not EXTRACTED_DATA_DIR.exists():
        return []
    entries = await aioos.listdir(EXTRACTED_DATA_DIR)
    folders = [
        name
        for name in entries
        if (EXTRACTED_DATA_DIR / name).is_dir() and not name.startswith((".", "__"))
    ]
    return sorted(folders)


def _sanitize_folder(folder: str) -> str:
    cleaned = (folder or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Folder is required")
    if Path(cleaned).name != cleaned:
        raise HTTPException(status_code=400, detail="Invalid folder name")
    if cleaned.startswith(".") or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid folder name")
    return cleaned


@router_extracted_data.get("/get-folder-names", response_model=list[str])
async def get_folder_names() -> list[str]:
    return await _list_extracted_folders()


async def _read_file_chunks(path: Path, chunk_size: int = 1024 * 1024):
    async with aiofiles.open(path, "rb") as file_handle:
        while True:
            chunk = await file_handle.read(chunk_size)
            if not chunk:
                break
            yield chunk


@router_extracted_data.get("/get-extracted-image")
async def get_extracted_image(
    folder: Annotated[str, Query()] = ...,
    n: Annotated[int, Query(ge=0)] = ...,
):
    folder_name = _sanitize_folder(folder)
    folder_path = EXTRACTED_DATA_DIR / folder_name
    if not folder_path.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    file_path = folder_path / f"{n}.png"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return StreamingResponse(_read_file_chunks(file_path), media_type="image/png")


@router_extracted_data.get("/get-extracted-image-count")
async def get_extracted_image_count(
    folder: Annotated[str, Query()] = ...,
):
    folder_name = _sanitize_folder(folder)
    folder_path = EXTRACTED_DATA_DIR / folder_name
    if not folder_path.is_dir():
        raise HTTPException(status_code=404, detail="Folder not found")
    entries = await aioos.listdir(folder_path)
    count = sum(
        1
        for name in entries
        if (folder_path / name).is_file() and name.lower().endswith(".png")
    )
    return {"count": count}
