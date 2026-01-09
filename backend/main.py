import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.bulk_engine.router import router_bulk_engine
from app.cellextraction.router import router_cellextraction
from app.database_manager.router import router_database_manager
from app.extracted_data.router import router_extracted_data
from app.file_manager.router import router_file_manager
from app.nd2files.router import router_nd2
from app.nd2parser.router import router_nd2parser
from app.system.router import router_system

API_PREFIX = "/api/v1"
app = FastAPI(docs_url=f"{API_PREFIX}/docs", openapi_url=f"{API_PREFIX}/openapi.json")
logger = logging.getLogger("uvicorn.error")
FRONTEND_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST_DIR / "index.html"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router_nd2, prefix=API_PREFIX)
app.include_router(router_nd2parser, prefix=API_PREFIX)
app.include_router(router_extracted_data, prefix=API_PREFIX)
app.include_router(router_cellextraction, prefix=API_PREFIX)
app.include_router(router_database_manager, prefix=API_PREFIX)
app.include_router(router_bulk_engine, prefix=API_PREFIX)
app.include_router(router_file_manager, prefix=API_PREFIX)
app.include_router(router_system, prefix=API_PREFIX)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 500:
        logger.error(
            "HTTP %s %s %s: %s",
            exc.status_code,
            request.method,
            request.url.path,
            exc.detail,
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.get(f"{API_PREFIX}/")
def read_root():
    return {"message": "Hello from PhenoPixel"}


@app.get(f"{API_PREFIX}/health")
def read_health():
    return {"status": "ok"}


if FRONTEND_INDEX.is_file():
    assets_dir = FRONTEND_DIST_DIR / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(FRONTEND_INDEX)

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        if full_path.startswith(API_PREFIX.lstrip("/")):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = FRONTEND_DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_INDEX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=True)
