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


def test_profit_and_loss_validation_pass():
    extracted = {
        "revenue": {"value": 50000.0},
        "cost_of_sales": {"value": 20000.0},
        "gross_profit": {"value": 30000.0},
        "operating_expenses": {"value": 10000.0},
        "operating_profit": {"value": 20000.0},
        "tax": {"value": 4000.0},
        "net_profit": {"value": 16000.0},
    }
    result = run_validation("profit_and_loss", extracted)
    assert result["overall_status"] == "PASS"


def test_cash_flow_validation_pass():
    extracted = {
        "operating_cash_flow": {"value": 15000.0},
        "investing_cash_flow": {"value": -5000.0},
        "financing_cash_flow": {"value": -2000.0},
        "opening_cash": {"value": 10000.0},
        "net_change_in_cash": {"value": 8000.0},
        "closing_cash": {"value": 18000.0},
    }
    result = run_validation("cash_flow_statement", extracted)
    assert result["overall_status"] == "PASS"



def test_invoice_shipping_is_included_in_total_formula():
    extracted = {
        "subtotal": {"value": 135.0},
        "tax_amount": {"value": 12.48},
        "shipping_and_handling": {"value": 10.0},
        "discount": {"value": 0.0},
        "total_amount": {"value": 157.48},
    }
    result = run_validation("invoice", extracted)
    check = next(c for c in result["checks"] if c["name"] == "invoice_total_check")
    assert check["status"] == "PASS"
    assert check["calculated_value"] == 157.48


def test_invoice_arithmetic_failure_does_not_mean_processing_failure():
    # Validation is intentionally independent from document processing status.
    # This mirrors the service contract: a readable parsed document can have
    # validation.overall_status=FAIL without becoming processing_status=FAILED.
    extracted = {
        "subtotal": {"value": 135.0},
        "tax_amount": {"value": 12.48},
        "shipping_and_handling": {"value": 10.0},
        "total_amount": {"value": 999.0},
    }
    result = run_validation("invoice", extracted)
    assert result["overall_status"] == "FAIL"
    assert any(c["status"] == "FAIL" for c in result["checks"])


def test_invoice_dynamic_solver_discovers_split_taxes_and_shipping():
    extracted = {
        "subtotal": {"value": 1000.0},
        "cgst": {"value": 90.0},
        "sgst": {"value": 90.0},
        "shipping_and_handling": {"value": 20.0},
        "total_amount": {"value": 1200.0},
        "discovered_fields": {
            "CGST": {"value": 90.0},
            "SGST": {"value": 90.0},
            "Shipping Charge": {"value": 20.0},
        },
    }
    result = run_validation("invoice", extracted)
    check = next(c for c in result["checks"] if c["name"] == "invoice_total_check")
    assert check["status"] == "PASS"
    assert check["calculated_value"] == 1200.0
    assert len(check["operands"]["components"]) >= 3


def test_invoice_cash_paid_change_is_independent_check():
    extracted = {
        "subtotal": {"value": 100.0},
        "tax_amount": {"value": 10.0},
        "total_amount": {"value": 110.0},
        "cash_paid": {"value": 200.0},
        "change_amount": {"value": 90.0},
    }
    result = run_validation("invoice", extracted)
    check = next(c for c in result["checks"] if c["name"] == "cash_change_check")
    assert check["status"] == "PASS"
