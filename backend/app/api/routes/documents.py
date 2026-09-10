"""
REST API routes (spec section 5). Kept thin -- all real work happens
in app.services.document_service.
"""
import logging
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.document import DocumentType
from app.services.document_service import process_document
from app.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["documents"])

SUPPORTED_CONTENT_TYPES = {"application/pdf", "image/jpeg", "image/jpg", "image/png"}


@router.post("/documents/process")
async def process_document_endpoint(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(...),
    db: Session = Depends(get_db),
):
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail={
            "error": {"code": "EMPTY_FILE", "message": "The uploaded file is empty."}
        })

    try:
        result = process_document(
            db=db,
            file_bytes=file_bytes,
            filename=file.filename,
            content_type=file.content_type or "",
            document_type=document_type.value,
        )
    except Exception:
        # Anything unexpected: log full detail server-side, never leak internals to the client
        logger.exception("Unexpected error processing '%s'", file.filename)
        raise HTTPException(status_code=500, detail={
            "error": {"code": "INTERNAL_ERROR", "message": "Document processing failed unexpectedly."}
        })

    return result


@router.get("/documents/{document_name}")
def get_document_by_name(document_name: str, db: Session = Depends(get_db)):
    repo = DocumentRepository(db)
    record = repo.get_latest_by_name(document_name)
    if not record:
        raise HTTPException(status_code=404, detail={
            "error": {"code": "DOCUMENT_NOT_FOUND", "message": f"No processed result found for '{document_name}'."}
        })
    return record.result_json


@router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    repo = DocumentRepository(db)
    records = repo.list_latest_per_document()
    return {
        "documents": [
            {
                "document_name": r.document_name,
                "document_type": r.document_type,
                "processing_status": r.processing_status,
                "processed_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
        "count": len(records),
    }


@router.get("/health")
def health_check():
    return {"status": "ok"}
