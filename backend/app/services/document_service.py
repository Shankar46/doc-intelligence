"""
Orchestration layer. Ties together file validation -> OCR ->
extraction -> financial validation -> persistence, and assembles the
final response matching the mandatory JSON contract (spec 5.2).

This is intentionally the ONLY place that calls multiple services in
sequence -- routes stay thin, services stay single-purpose.
"""
import logging
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.repositories.document_repository import DocumentRepository
from app.services.document_validation_service import validate_file, DocumentValidationError, _detect_kind
from app.services.ocr_service import extract_text
from app.services.extraction_service import extract_fields
from app.services.financial_validation_service import run_validation

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def process_document(db: Session, file_bytes: bytes, filename: str,
                      content_type: str, document_type: str) -> dict:
    start = time.perf_counter()
    repo = DocumentRepository(db)

    # 1. File validation (spec 4.1) -- failures here become FAILED responses, not exceptions,
    #    so the client always gets a structured JSON body rather than a bare error.
    try:
        file_validation = validate_file(file_bytes, filename, content_type)
    except DocumentValidationError as exc:
        logger.warning("File validation failed for '%s': %s", filename, exc.message)
        result = _build_failed_response(filename, document_type, exc.code, exc.message, start)
        repo.save_result(filename, document_type, "FAILED", result)
        return result

    # 2. OCR / text extraction
    try:
        kind = _detect_kind(content_type, filename)
        ocr_result = extract_text(file_bytes, kind)
    except Exception as exc:
        logger.exception("OCR failed for '%s'", filename)
        result = _build_failed_response(filename, document_type, "OCR_FAILED", str(exc), start)
        repo.save_result(filename, document_type, "FAILED", result)
        return result

    # 3. AI field & table extraction (spec 4.2 / 4.3)
    try:
        extracted_data = extract_fields(ocr_result.pages, document_type, page_layouts=ocr_result.layouts)
    except Exception as exc:
        logger.exception("Extraction failed for '%s'", filename)
        result = _build_failed_response(filename, document_type, "EXTRACTION_FAILED", str(exc), start)
        repo.save_result(filename, document_type, "FAILED", result)
        return result

    # 4. Financial validation (spec 4.4)
    validation_result = run_validation(document_type, extracted_data)

    # 5. Processing status is about document processing, not financial arithmetic.
    # A readable/supported document that was successfully OCRed and parsed is
    # PROCESSING PASS even when one or more financial validation checks fail.
    # Validation failures are isolated in validation.overall_status/checks.
    processing_status = "PASS"

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    result = {
        "document_name": filename,
        "document_type": document_type,
        "processing_status": processing_status,
        "file_validation": file_validation.model_dump(),
        "extracted_data": extracted_data,
        "validation": validation_result,
        "processing_metadata": {
            "ocr_used": ocr_result.ocr_used,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": elapsed_ms,
        },
    }

    repo.save_result(filename, document_type, processing_status, result)
    logger.info("Processed '%s' as %s -> %s", filename, document_type, processing_status)
    return result


def _build_failed_response(filename: str, document_type: str, code: str, message: str, start: float) -> dict:
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "document_name": filename,
        "document_type": document_type,
        "processing_status": "FAILED",
        "file_validation": {
            "file_type": "unknown", "is_supported": False, "is_readable": False,
            "page_count": 0, "status": "FAILED", "reason": message,
        },
        "extracted_data": {},
        "validation": {"checks": [], "overall_status": "NOT_APPLICABLE", "issues": [message]},
        "processing_metadata": {
            "ocr_used": False,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": elapsed_ms,
        },
        "error_code": code,
    }
