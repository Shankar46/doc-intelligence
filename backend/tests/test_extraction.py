"""Tests for extraction service and Hugging Face / DeepSeek-R1 output cleaning."""
import json
from app.services.extraction_service import _clean_json_response, _get_llm_client, OpenAI
from app.core.config import settings


def test_clean_json_response_with_deepseek_r1_think_block():
    raw_response = """<think>
Step 1: Parse document text.
Step 2: Locate invoice number and amounts.
Found invoice_number: INV-1001, total_amount: 1500.00
</think>
```json
{
  "fields": {
    "invoice_number": {"value": "INV-1001", "page_number": 1, "source_text": "Invoice # INV-1001"},
    "total_amount": {"value": 1500.00, "page_number": 1, "source_text": "Total: 1500.00"}
  },
  "line_items": []
}
```"""
    cleaned = _clean_json_response(raw_response)
    data = json.loads(cleaned)
    assert "fields" in data
    assert data["fields"]["invoice_number"]["value"] == "INV-1001"
    assert data["fields"]["total_amount"]["value"] == 1500.00


def test_get_llm_client_huggingface(monkeypatch):
    if OpenAI is None:
        import pytest
        pytest.skip("optional OpenAI SDK not installed in this environment")
    monkeypatch.setattr(settings, "llm_provider", "huggingface")
    monkeypatch.setattr(settings, "huggingface_api_key", "hf_test_token_12345")
    monkeypatch.setattr(settings, "huggingface_model", "deepseek-ai/DeepSeek-R1")

    client, model_name = _get_llm_client()
    assert model_name == "deepseek-ai/DeepSeek-R1"
    assert str(client.base_url).rstrip("/") == "https://router.huggingface.co/hf-inference/v1"


def test_balance_sheet_scanned_layout_regression():
    from app.services.extraction_service import extract_fields
    ocr = """Consolidated Balance Sheet\nAs at March 31, 2021\n% in '000\nAs at 31-Mar-21\nAs at 31-Mar-20\nCAPITAL AND LIABILITIES\nCapital\n5,512,776\n5,483,286\nReserves and surplus\n2,092,589,110\n1,758,103,766\nMinority interest\n6,327,647\n5,766,413\nDeposits\n13,337,208,758\n11,462,071,336\nBorrowings\n1,776,967,487\n1,868,343,231\nOther liabilities and provisions\n776,460,664\n708,536,341\nTotal\n17,995,066,442\n15,808,304,373\nASSETS\nCash and balances with Reserve Bank of India\n973,703,555\n722,110,033\nBalances with banks and money at call and short notice\n239,021,709\n157,291,086\nInvestments\n4,388,231,117\n3,893,049,519\nAdvances\n11,852,835,198\n10,436,708,771\nFixed assets\n10\n50,995,631\n46,268,558\nOther assets\n11\n490,279,232\n552,876,406\nTotal\n17,995,066,442\n15,808,304,373"""
    data = extract_fields([ocr], "balance_sheet")
    assert data["total_assets"]["value"] == 17995066442
    assert data["total_assets__comparative"]["value"] == 15808304373
    assert data["fixed_assets"]["value"] == 50995631
    assert data["fixed_assets__comparative"]["value"] == 46268558


def test_balance_sheet_does_not_expose_invoice_style_line_items():
    from app.services.extraction_service import extract_fields
    text = ["Consolidated Balance Sheet\nAs at 31-Mar-21 As at 31-Mar-20\nCapital 5,512,776 5,483,286\nTotal 17,995,066,442 15,808,304,373\nTotal 17,995,066,442 15,808,304,373"]
    data = extract_fields(text, "balance_sheet")
    assert "line_items" not in data


def test_empty_invoice_line_items_are_removed(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es, "_call_llm", lambda *args, **kwargs: {"fields": {}, "line_items": [{}, {"description": None, "quantity": None, "unit_price": None, "amount": None}]})
    data = es.extract_fields(["Invoice"], "invoice")
    assert "line_items" not in data


def test_invoice_ocr_label_value_separation_without_llm(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["Invoice\nInvoice Number:\nINV-2026-77\nDate:\n2026-02-10\nFrom:\nAcme Pvt Ltd\nBilled To:\nGlobal Tech Ltd\nCurrency:\nINR\nSubtotal:\n10,000\nGST (18%):\n1,800\nDiscount:\n0\nTotal Amount Due:\n11,800"]
    data = es.extract_fields(text, "invoice")
    assert data["invoice_number"]["value"] == "INV-2026-77"
    assert data["invoice_date"]["value"] == "2026-02-10"
    assert data["vendor_name"]["value"] == "Acme Pvt Ltd"
    assert data["customer_name"]["value"] == "Global Tech Ltd"
    assert data["subtotal"]["value"] == 10000
    assert data["tax_amount"]["value"] == 1800
    assert data["total_amount"]["value"] == 11800


def test_cashflow_ocr_label_value_separation_without_llm(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["Consolidated Cash Flow Statement\nNet Cash Provided by Operating Activities\n45,000\nNet Cash Used in Investing Activities\n(15,000)\nNet Cash Used in Financing Activities\n(5,000)\nNet Increase in Cash\n25,000\nCash at Beginning of Year\n30,000\nCash at End of Year\n55,000"]
    data = es.extract_fields(text, "cash_flow_statement")
    assert data["operating_cash_flow"]["value"] == 45000
    assert data["investing_cash_flow"]["value"] == -15000
    assert data["financing_cash_flow"]["value"] == -5000
    assert data["net_change_in_cash"]["value"] == 25000
    assert data["opening_cash"]["value"] == 30000
    assert data["closing_cash"]["value"] == 55000


def test_invoice_rejects_tax_id_as_tax_and_parses_decimal_comma_total(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["Invoice no: 94404257\nDate of issue 07/03/2013\nClient: Cruz PLC\nTales from the Buffalo Bills\nTax Id: 964-99-8203\nTotal $ 126,27"]
    data = es.extract_fields(text, "invoice")
    assert data["invoice_number"]["value"] == "94404257"
    assert data["invoice_date"]["value"] == "07/03/2013"
    assert data["customer_name"]["value"] == "Cruz PLC"
    assert data["vendor_name"]["value"] is None
    assert data["tax_amount"]["value"] is None
    assert data["currency"]["value"] == "$"
    assert data["total_amount"]["value"] == 126.27
