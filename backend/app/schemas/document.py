"""
Request/response schemas for document upload and listing.
These mirror the exact JSON shapes required by the case study spec (section 4.1 / 5.2).
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    invoice = "invoice"
    balance_sheet = "balance_sheet"
    profit_and_loss = "profit_and_loss"
    cash_flow_statement = "cash_flow_statement"


class ProcessingStatus(str, Enum):
    PASS = "PASS"
    FAILED = "FAILED"


class FileValidation(BaseModel):
    file_type: str
    is_supported: bool
    is_readable: bool
    page_count: int
    status: str  # "PASS" | "FAILED"
    reason: Optional[str] = None  # populated when status == FAILED


class ProcessingMetadata(BaseModel):
    ocr_used: bool
    processed_at: str
    processing_time_ms: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class DocumentListItem(BaseModel):
    document_name: str
    document_type: str
    processing_status: str
    processed_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    count: int
