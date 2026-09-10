"""
App entrypoint. Wires up logging, DB init, routes, exception handlers,
and (optionally) serves the frontend templates/static files so a single
deployment covers both frontend + API -- simplest path for a free-tier
deploy target like Render/Railway/Koyeb.
"""
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.database import init_db
from app.api.routes import documents

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)

# --- DB init ---
init_db()

# --- API routes ---
app.include_router(documents.router)

# --- Frontend (served by the same app; adjust paths if you deploy separately) ---
templates = Jinja2Templates(directory="../frontend/templates")
app.mount("/static", StaticFiles(directory="../frontend/static"), name="static")


@app.get("/")
def dashboard_page(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


@app.get("/documents/{document_name}/view")
def document_detail_page(request: Request, document_name: str):
    return templates.TemplateResponse(
        request=request, name="document_result.html", context={"document_name": document_name}
    )



# --- HTTPException handler: routes raise HTTPException(detail={"error": {...}});
# FastAPI's default wraps that under a "detail" key, which breaks the mandated
# {"error": {"code", "message"}} contract (spec 5.3). Unwrap it here so every
# error response -- validation failures, 404s, 500s -- has the exact same shape.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        content = detail
    else:
        content = {"error": {"code": "HTTP_ERROR", "message": str(detail)}}
    return JSONResponse(status_code=exc.status_code, content=content)


# --- Global exception handler: never leak stack traces to the client ---
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}},
    )
