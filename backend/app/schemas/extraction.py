"""
Schemas for extracted field values, evidence/grounding, and financial
validation checks. These compose into the full ProcessedDocumentResponse
that every /process call and /documents/{name} GET must return.
"""
from typing import Any, Optional
from pydantic import BaseModel

from app.schemas.document import FileValidation, ProcessingMetadata


class Evidence(BaseModel):
    source_text: Optional[str] = None
    page_number: Optional[int] = None


class ExtractedField(BaseModel):
    """A single extracted value with optional confidence + evidence.

    Missing values MUST be represented as value=None — never fabricate
    a plausible-looking number. Don't drop the field either; keep the key
    with a null value so the evaluator can see it was checked and absent.
    """
    value: Any = None
    confidence: Optional[float] = None  # OPTIONAL per spec
    page_number: Optional[int] = None
    evidence: Optional[Evidence] = None


class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class ValidationCheck(BaseModel):
    """One financial validation rule's outcome.

    status is one of PASS | FAIL | NOT_APPLICABLE.
    Use NOT_APPLICABLE (never invent inputs) when a required field for
    this check isn't present in the extracted data.
    """
    name: str
    formula: str
    operands: dict[str, Any]
    calculated_value: Optional[float] = None
    reported_value: Optional[float] = None
    variance: Optional[float] = None
    status: str  # PASS | FAIL | NOT_APPLICABLE


class ValidationResult(BaseModel):
    checks: list[ValidationCheck]
    overall_status: str  # PASS | FAIL | NOT_APPLICABLE
    issues: list[str] = []


class ProcessedDocumentResponse(BaseModel):
    document_name: str
    document_type: str
    processing_status: str  # PASS | FAILED
    overall_confidence: Optional[float] = None  # OPTIONAL
    file_validation: FileValidation
    extracted_data: dict[str, Any]  # field_name -> ExtractedField dict, plus "line_items": [...]
    validation: ValidationResult
    processing_metadata: ProcessingMetadata
