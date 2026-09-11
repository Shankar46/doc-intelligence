"""Smoke-test the checked-in New Dataset files end to end.

This is intentionally lightweight: it verifies that every dataset document can
be OCR/text-extracted, semantically parsed, and financially validated without a
processing exception. It also prints a compact per-file summary when run with
``pytest -s`` so extraction gaps are easy to debug.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

import pytest

from app.services.document_validation_service import validate_file
from app.services.financial_validation_service import run_validation
from app.services.extraction_service import extract_fields
from app.services.ocr_service import extract_text


DATASET_ROOT = Path(__file__).resolve().parents[2] / "New Dataset"


DATASET_TYPES = {
    "Balance Sheet": "balance_sheet",
    "Cash Flows": "cash_flow_statement",
    "Profit & Loss": "profit_and_loss",
    "Invoices": "invoice",
}


def _dataset_cases() -> list[tuple[Path, str]]:
    cases: list[tuple[Path, str]] = []
    if not DATASET_ROOT.exists():
        return cases
    for folder, document_type in DATASET_TYPES.items():
        for path in sorted((DATASET_ROOT / folder).glob("*")):
            if path.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"}:
                cases.append((path, document_type))
    return cases


@pytest.mark.parametrize("path,document_type", _dataset_cases(), ids=lambda x: x.name if isinstance(x, Path) else x)
def test_new_dataset_document_extracts_without_processing_error(monkeypatch, path: Path, document_type: str):
    monkeypatch.setattr("app.services.extraction_service.settings.use_llm_fallback", False)

    file_bytes = path.read_bytes()
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    validation = validate_file(file_bytes, path.name, content_type)

    kind = "pdf" if path.suffix.lower() == ".pdf" else "image"
    ocr = extract_text(file_bytes, kind)
    extracted = extract_fields(ocr.pages, document_type, page_layouts=ocr.layouts)
    financial = run_validation(document_type, extracted)
    quality = extracted.get("extraction_quality", {})

    print(
        f"{path.parent.name}/{path.name}: status={validation.status} "
        f"ocr_used={ocr.ocr_used} validation={financial['overall_status']} "
        f"missing={quality.get('required_fields_missing', [])}"
    )

    assert validation.status == "PASS"
    assert extracted
    assert "extraction_quality" in extracted
    assert financial["overall_status"] in {"PASS", "FAIL", "NOT_APPLICABLE"}
