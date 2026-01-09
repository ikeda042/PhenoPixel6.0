from typing import AsyncIterator

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.file_manager.crud import (
    UPLOAD_CHUNK_SIZE,
    delete_file,
    list_files,
    read_file_chunks,
    resolve_file_path,
    save_upload,
)

router_file_manager = APIRouter(tags=["file_manager"])


async def _iter_upload(file: UploadFile) -> AsyncIterator[bytes]:
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


@router_file_manager.get("/filemanager/files")
async def list_files_endpoint():
    try:
        files = await list_files()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(content={"files": files})


@router_file_manager.post("/filemanager/files")
async def upload_file_endpoint(file: UploadFile = File(...)):
    try:
        payload = await save_upload(file.filename or "", _iter_upload(file))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await file.close()
    return JSONResponse(content=payload)


@router_file_manager.get("/filemanager/files/{filename}")
async def download_file_endpoint(filename: str):
    try:
        file_path = resolve_file_path(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    headers = {"Content-Disposition": f'attachment; filename="{file_path.name}"'}
    return StreamingResponse(
        read_file_chunks(file_path),
        media_type="application/octet-stream",
        headers=headers,
    )


@router_file_manager.delete("/filemanager/files/{filename}")
async def delete_file_endpoint(filename: str):
    try:
        deleted = await delete_file(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(content={"deleted": True, "filename": deleted})
