import asyncio
import io
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import aiofiles
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.database_manager.crud import DatabaseManagerCrud


router_database_manager = APIRouter(tags=["database_manager"])
annotation_executor = ProcessPoolExecutor()
heatmap_executor = ProcessPoolExecutor(max_workers=1)
UPLOAD_CHUNK_SIZE = 1024 * 1024 * 100


@router_database_manager.get("/get-databases", response_model=list[str])
async def get_databases() -> list[str]:
    return await DatabaseManagerCrud.list_databases()


@router_database_manager.post("/database_files")
async def upload_database(file: UploadFile = File(...)) -> dict:
    try:
        sanitized = DatabaseManagerCrud.sanitize_db_name(file.filename or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    DatabaseManagerCrud.DATABASES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DatabaseManagerCrud.DATABASES_DIR / sanitized
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await file.read(UPLOAD_CHUNK_SIZE):
                await out_file.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    try:
        DatabaseManagerCrud.migrate_database(sanitized)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"filename": sanitized}


@router_database_manager.get("/database_files/{dbname}")
async def download_database_endpoint(dbname: str) -> StreamingResponse:
    try:
        db_path = DatabaseManagerCrud.resolve_database_path(dbname)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not db_path.is_file():
        raise HTTPException(status_code=404, detail="Database not found")
    headers = {"Content-Disposition": f'attachment; filename="{db_path.name}"'}
    return StreamingResponse(
        DatabaseManagerCrud.read_database_chunks(db_path),
        media_type="application/octet-stream",
        headers=headers,
    )


@router_database_manager.delete("/database_files/{dbname}")
async def delete_database_endpoint(dbname: str) -> dict:
    try:
        deleted = await DatabaseManagerCrud.delete_database(dbname)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"deleted": True, "filename": deleted}


@router_database_manager.get("/get-cell-ids", response_model=list[str])
def get_cell_ids_endpoint(dbname: str = Query(...)) -> list[str]:
    try:
        return DatabaseManagerCrud.get_cell_ids(dbname)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router_database_manager.get("/get-cell-ids-by-label", response_model=list[str])
def get_cell_ids_by_label_endpoint(
    dbname: str = Query(...),
    label: str = Query(...),
) -> list[str]:
    try:
        return DatabaseManagerCrud.get_cell_ids_by_label(dbname, label)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router_database_manager.get("/get-manual-labels", response_model=list[str])
def get_manual_labels_endpoint(
    dbname: str = Query(...),
) -> list[str]:
    try:
        return DatabaseManagerCrud.get_manual_labels(dbname)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router_database_manager.get("/get-cell-contour")
def get_cell_contour_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
) -> dict:
    try:
        contour = DatabaseManagerCrud.get_cell_contour(dbname, cell_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell contour not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"contour": contour}


@router_database_manager.get("/get-cell-label", response_model=str)
def get_cell_label_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
) -> str:
    try:
        return DatabaseManagerCrud.get_cell_label(dbname, cell_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell label not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router_database_manager.get("/get-cell-image")
def get_cell_image_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    image_type: str = Query(..., description="ph | fluo1 | fluo2"),
    draw_contour: bool = Query(False),
    draw_scale_bar: bool = Query(False),
) -> StreamingResponse:
    try:
        image_bytes = DatabaseManagerCrud.get_cell_image(
            dbname,
            cell_id,
            image_type,
            draw_contour=draw_contour,
            draw_scale_bar=draw_scale_bar,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell image not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router_database_manager.get("/get-cell-image-optical-boost")
def get_cell_image_optical_boost_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    image_type: str = Query(..., description="fluo1 | fluo2"),
    draw_contour: bool = Query(False),
    draw_scale_bar: bool = Query(False),
) -> StreamingResponse:
    try:
        image_bytes = DatabaseManagerCrud.get_cell_image_optical_boost(
            dbname,
            cell_id,
            image_type,
            draw_contour=draw_contour,
            draw_scale_bar=draw_scale_bar,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell image not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router_database_manager.get("/get-cell-overlay")
def get_cell_overlay_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
) -> StreamingResponse:
    try:
        image_bytes = DatabaseManagerCrud.get_cell_overlay(dbname, cell_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell overlay not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router_database_manager.patch("/update-cell-label")
def update_cell_label_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    label: str = Query(...),
) -> dict:
    try:
        updated = DatabaseManagerCrud.update_cell_label(dbname, cell_id, label)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell label not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"cell_id": cell_id, "label": updated}


@router_database_manager.patch("/elastic-contour")
def elastic_contour_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    delta: int = Query(0),
) -> dict:
    try:
        contour = DatabaseManagerCrud.apply_elastic_contour(dbname, cell_id, delta)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell contour not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"cell_id": cell_id, "contour": contour}


@router_database_manager.get("/get-cell-replot")
def get_cell_replot_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    image_type: str = Query("fluo1", description="ph | fluo1 | fluo2 | overlay"),
    degree: int = Query(4, ge=1),
    dark_mode: bool = Query(False),
) -> StreamingResponse:
    try:
        image_bytes = DatabaseManagerCrud.get_cell_replot(
            dbname,
            cell_id,
            image_type=image_type,
            degree=degree,
            dark_mode=dark_mode,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell image not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router_database_manager.get("/get-cell-heatmap")
async def get_cell_heatmap_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    image_type: str = Query("fluo1", description="fluo1 | fluo2"),
    degree: int = Query(4, ge=1),
) -> StreamingResponse:
    try:
        loop = asyncio.get_running_loop()
        image_bytes = await loop.run_in_executor(
            heatmap_executor,
            DatabaseManagerCrud.get_cell_heatmap,
            dbname,
            cell_id,
            image_type,
            degree,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell image not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router_database_manager.get("/get-cell-distribution")
def get_cell_distribution_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    image_type: str = Query("fluo1", description="ph | fluo1 | fluo2"),
) -> StreamingResponse:
    try:
        image_bytes = DatabaseManagerCrud.get_cell_intensity_distribution(
            dbname, cell_id, image_type
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell image not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router_database_manager.get("/get-cell-map256")
async def get_cell_map256_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    image_type: str = Query("fluo1", description="fluo1 | fluo2"),
    degree: int = Query(4, ge=1),
) -> StreamingResponse:
    try:
        loop = asyncio.get_running_loop()
        image_bytes = await loop.run_in_executor(
            heatmap_executor,
            DatabaseManagerCrud.get_cell_map256,
            dbname,
            cell_id,
            image_type,
            degree,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell image not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router_database_manager.get("/get-cell-map256-jet")
async def get_cell_map256_jet_endpoint(
    dbname: str = Query(...),
    cell_id: str = Query(...),
    image_type: str = Query("fluo1", description="fluo1 | fluo2"),
    degree: int = Query(4, ge=1),
) -> StreamingResponse:
    try:
        loop = asyncio.get_running_loop()
        image_bytes = await loop.run_in_executor(
            heatmap_executor,
            DatabaseManagerCrud.get_cell_map256_jet,
            dbname,
            cell_id,
            image_type,
            degree,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except LookupError:
        raise HTTPException(status_code=404, detail="Cell image not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@router_database_manager.get("/get-annotation-zip")
async def get_annotation_zip_endpoint(
    dbname: str = Query(...),
    image_type: str = Query("ph", description="ph | fluo1 | fluo2"),
    raw: bool = Query(False),
    downscale: Optional[float] = Query(None, ge=0.05, le=1.0),
) -> StreamingResponse:
    try:
        loop = asyncio.get_running_loop()
        zip_bytes = await loop.run_in_executor(
            annotation_executor,
            DatabaseManagerCrud.get_annotation_zip,
            dbname,
            image_type,
            raw,
            downscale,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Database not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return StreamingResponse(io.BytesIO(zip_bytes), media_type="application/zip")
