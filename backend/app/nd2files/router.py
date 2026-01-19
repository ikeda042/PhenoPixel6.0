import os
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router_nd2 = APIRouter(tags=["nd2files"])
UPLOAD_DIR = Path(__file__).resolve().parent
UPLOAD_CHUNK_SIZE = 1024 * 1024 * 100


class Nd2BulkDeleteRequest(BaseModel):
    filenames: list[str]


def _ensure_upload_dir() -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOAD_DIR


def _list_nd2_files(upload_dir: Path) -> list[str]:
    if not upload_dir.is_dir():
        return []
    return sorted(
        [
            entry.name
            for entry in upload_dir.iterdir()
            if entry.is_file()
            and entry.name.endswith(".nd2")
            and not entry.name.endswith("timelapse.nd2")
        ]
    )


def _sanitize_nd2_filename(filename: str) -> str:
    cleaned = os.path.basename(filename or "")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Filename is required")
    base, ext = os.path.splitext(cleaned)
    if not base:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if ext.lower() != ".nd2":
        raise HTTPException(status_code=400, detail="Only .nd2 files are supported")
    base = base.replace(".", "p")
    return f"{base}.nd2"


@router_nd2.post("/nd2_files")
async def upload_nd2_file(file: UploadFile = File(...)):
    sanitized = _sanitize_nd2_filename(file.filename or "")
    file_path = _ensure_upload_dir() / sanitized
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await file.read(UPLOAD_CHUNK_SIZE):
                await out_file.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"filename": sanitized})


@router_nd2.get("/nd2_files")
async def get_nd2_files():
    upload_dir = _ensure_upload_dir()
    return JSONResponse(content={"files": _list_nd2_files(upload_dir)})


@router_nd2.delete("/nd2_files/{filename}")
def delete_nd2_file(filename: str):
    sanitized = _sanitize_nd2_filename(filename)
    file_path = _ensure_upload_dir() / sanitized
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        file_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content={"deleted": True, "filename": sanitized})


@router_nd2.post("/nd2_files/bulk-delete")
def bulk_delete_nd2_files(payload: Nd2BulkDeleteRequest):
    if not payload.filenames:
        raise HTTPException(status_code=400, detail="No filenames provided")

    upload_dir = _ensure_upload_dir()
    deleted: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    seen: set[str] = set()

    for filename in payload.filenames:
        if filename in seen:
            continue
        seen.add(filename)
        try:
            sanitized = _sanitize_nd2_filename(filename)
        except HTTPException:
            invalid.append(filename)
            continue
        file_path = upload_dir / sanitized
        if not file_path.exists():
            missing.append(sanitized)
            continue
        try:
            file_path.unlink()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        deleted.append(sanitized)

    return JSONResponse(
        content={"deleted": deleted, "missing": missing, "invalid": invalid}
    )
