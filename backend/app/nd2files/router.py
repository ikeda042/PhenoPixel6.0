import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import aiofiles
import nd2reader
import numpy as np
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


def _to_jsonable(value: Any, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val, depth + 1, max_depth) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item, depth + 1, max_depth) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist(), depth + 1, max_depth)
    if isinstance(value, np.generic):
        return _to_jsonable(value.item(), depth + 1, max_depth)
    return str(value)


def _extract_nd2_metadata(file_path: Path) -> dict[str, Any]:
    stats = file_path.stat()
    with nd2reader.ND2Reader(str(file_path)) as images:
        reader_info: dict[str, Any] = {
            "axes": getattr(images, "axes", None),
            "sizes": dict(getattr(images, "sizes", {}) or {}),
            "bundle_axes": getattr(images, "bundle_axes", None),
            "iter_axes": getattr(images, "iter_axes", None),
            "default_coords": getattr(images, "default_coords", None),
            "ndim": getattr(images, "ndim", None),
            "shape": getattr(images, "shape", None),
            "dtype": str(images.pixel_type) if getattr(images, "pixel_type", None) else None,
        }
        try:
            reader_info["num_frames"] = len(images)
        except Exception:
            reader_info["num_frames"] = None
        try:
            reader_info["frame_rate_fps"] = float(images.frame_rate)
        except Exception:
            reader_info["frame_rate_fps"] = None
        try:
            reader_info["timesteps_ms"] = images.timesteps
        except Exception:
            reader_info["timesteps_ms"] = None
        try:
            reader_info["parser_supported"] = bool(images.parser.supported)
        except Exception:
            reader_info["parser_supported"] = None
        payload: dict[str, Any] = {
            "file": {
                "name": file_path.name,
                "size_bytes": int(stats.st_size),
                "modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            },
            "reader": reader_info,
            "metadata": getattr(images, "metadata", None),
        }
    return _to_jsonable(payload)


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


@router_nd2.get("/nd2_files/{filename}/metadata")
def get_nd2_file_metadata(filename: str):
    sanitized = _sanitize_nd2_filename(filename)
    file_path = _ensure_upload_dir() / sanitized
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        metadata = _extract_nd2_metadata(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content=metadata)
