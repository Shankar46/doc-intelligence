"""Print extraction and validation diagnostics for files in ``New Dataset``."""
from __future__ import annotations

import argparse
import mimetypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.document_validation_service import validate_file  # noqa: E402
from app.services.extraction_service import extract_fields, settings  # noqa: E402
from app.services.financial_validation_service import run_validation  # noqa: E402
from app.services.ocr_service import extract_text  # noqa: E402


DATASET_TYPES = {
    "Balance Sheet": "balance_sheet",
    "Cash Flows": "cash_flow_statement",
    "Profit & Loss": "profit_and_loss",
    "Invoices": "invoice",
}


KEY_FIELDS = {
    "invoice": ["invoice_number", "invoice_date", "vendor_name", "customer_name", "subtotal", "tax_amount", "shipping_and_handling", "total_amount"],
    "balance_sheet": ["total_capital_and_liabilities", "total_assets"],
    "profit_and_loss": ["total_income", "total_expenditure", "consolidated_net_profit_before_minority_interest", "minority_interest", "consolidated_net_profit_attributable_to_group"],
    "cash_flow_statement": ["operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "net_change_in_cash", "opening_cash", "closing_cash"],
}


def iter_cases(dataset_root: Path):
    for folder, document_type in DATASET_TYPES.items():
        for path in sorted((dataset_root / folder).glob("*")):
            if path.suffix.lower() in {".pdf", ".jpg", ".jpeg", ".png"}:
                yield path, document_type


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "New Dataset")
    parser.add_argument("--only-problems", action="store_true")
    parser.add_argument("--name-contains", default="", help="Only process files whose name contains this text")
    parser.add_argument("--sources", action="store_true", help="Print source evidence for key fields")
    parser.add_argument("--llm", action="store_true", help="Enable configured LLM fallback")
    args = parser.parse_args()

    settings.use_llm_fallback = args.llm
    for path, document_type in iter_cases(args.dataset_root):
        if args.name_contains and args.name_contains.lower() not in path.name.lower():
            continue
        file_bytes = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        validation = validate_file(file_bytes, path.name, content_type)
        kind = "pdf" if path.suffix.lower() == ".pdf" else "image"
        ocr = extract_text(file_bytes, kind)
        extracted = extract_fields(ocr.pages, document_type, page_layouts=ocr.layouts)
        financial = run_validation(document_type, extracted)
        quality = extracted.get("extraction_quality", {})
        failed = [c for c in financial["checks"] if c["status"] == "FAIL" and not c["name"].endswith("_comparative")]
        missing = quality.get("required_fields_missing", [])
        if args.only_problems and not failed and not missing and financial["overall_status"] != "NOT_APPLICABLE":
            continue

        print(f"\n{path.parent.name}/{path.name}")
        print(f"  file={validation.status} ocr_used={ocr.ocr_used} validation={financial['overall_status']} missing={missing}")
        for field in KEY_FIELDS.get(document_type, []):
            info = extracted.get(field, {})
            if isinstance(info, dict):
                print(f"  {field}: {info.get('value')!r}")
                if args.sources:
                    source = (info.get("evidence") or {}).get("source_text")
                    if source:
                        print(f"    source: {source}")
        if document_type == "invoice":
            print(f"  line_items: {len(extracted.get('line_items', []))}")
        for check in failed:
            print(
                f"  FAIL {check['name']}: calculated={check['calculated_value']} "
                f"reported={check['reported_value']} variance={check['variance']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
