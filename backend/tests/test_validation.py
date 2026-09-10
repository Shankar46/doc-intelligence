"""Basic tests for financial validation logic (spec section 11)."""
from app.services.financial_validation_service import run_validation


def test_invoice_validation_pass():
    extracted = {
        "subtotal": {"value": 12500.00},
        "tax_amount": {"value": 625.00},
        "discount": {"value": 0.00},
        "total_amount": {"value": 13125.00},
        "line_items": [
            {"description": "Service A", "quantity": 1, "unit_price": 12500.00, "amount": 12500.00}
        ],
    }
    result = run_validation("invoice", extracted)
    assert result["overall_status"] == "PASS"


def test_invoice_validation_fail_on_mismatch():
    extracted = {
        "subtotal": {"value": 12500.00},
        "tax_amount": {"value": 625.00},
        "discount": {"value": 0.00},
        "total_amount": {"value": 99999.00},  # wrong on purpose
    }
    result = run_validation("invoice", extracted)
    assert result["overall_status"] == "FAIL"


def test_balance_sheet_not_applicable_when_fields_missing():
    extracted = {"total_assets": {"value": None}}
    result = run_validation("balance_sheet", extracted)
    assert result["overall_status"] == "NOT_APPLICABLE"


def test_balance_sheet_pass():
    extracted = {
        "total_assets": {"value": 1000.0},
        "total_liabilities": {"value": 600.0},
        "total_equity": {"value": 400.0},
    }
    result = run_validation("balance_sheet", extracted)
    assert result["overall_status"] == "PASS"
