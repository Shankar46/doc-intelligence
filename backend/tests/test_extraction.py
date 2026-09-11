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
    monkeypatch.setattr(settings, "huggingface_base_url", "https://router.huggingface.co/hf-inference/v1")

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


def test_invoice_common_ocr_variants(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["TAX INVOICE\nBill No: INV/77\nInvoice Date: 15 Aug 2026\nSupplier: Acme Technologies Pvt Ltd\nBill To: XYZ Retail\nUSD\nDescription Qty Unit Price Amount\nConsulting 2 500.00 1,000.00\nGST (18%): 180.00\nSub Total: 1,000.00\nGrand Total: 1,180.00"]
    data = es.extract_fields(text, "invoice")
    assert data["invoice_number"]["value"] == "INV/77"
    assert data["invoice_date"]["value"] == "15 Aug 2026"
    assert data["vendor_name"]["value"] == "Acme Technologies Pvt Ltd"
    assert data["customer_name"]["value"] == "XYZ Retail"
    assert data["tax_amount"]["value"] == 180
    assert data["subtotal"]["value"] == 1000
    assert data["total_amount"]["value"] == 1180
    assert data["line_items"][0]["amount"] == 1000


def test_profit_and_loss_bank_style_comparative(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["Consolidated Profit and Loss Account\nInterest Earned 1000 900\nOther Income 200 180\nTotal Income 1200 1080\nInterest Expended 400 360\nOperating Expenses 300 270\nProvisions and Contingencies 100 90\nTotal Expenditure 800 720\nConsolidated Net Profit before Minority Interest 400 360\nMinority Interest 20 10\nConsolidated Net Profit attributable to the Group 380 350"]
    data = es.extract_fields(text, "profit_and_loss")
    assert data["interest_earned"]["value"] == 1000
    assert data["interest_earned__comparative"]["value"] == 900
    assert data["total_expenditure"]["value"] == 800
    assert data["consolidated_net_profit_attributable_to_group"]["value"] == 380
    from app.services.financial_validation_service import run_validation
    validation = run_validation("profit_and_loss", data)
    assert validation["overall_status"] == "PASS"
    assert any(c["name"] == "income_reconciliation_check_comparative" for c in validation["checks"])


def test_profit_and_loss_generic_aliases(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["Statement of Profit and Loss\nTotal Revenue: $150,000\nCost of Goods Sold: $60,000\nGross Profit: $90,000\nOperating Expenses: $35,000\nOperating Income: $55,000\nIncome Tax Expense: $11,000\nNet Income: $44,000"]
    data = es.extract_fields(text, "profit_and_loss")
    assert data["revenue"]["value"] == 150000
    assert data["cost_of_sales"]["value"] == 60000
    assert data["gross_profit"]["value"] == 90000
    assert data["operating_profit"]["value"] == 55000
    assert data["tax"]["value"] == 11000
    assert data["net_profit"]["value"] == 44000


def test_cashflow_negative_parentheses_and_comparative(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["Cash Flow Statement\nNet Cash Provided by Operating Activities 45,000 40,000\nNet Cash Used in Investing Activities (15,000) (12,000)\nNet Cash Used in Financing Activities (5,000) (3,000)\nFX Translation Adjustment 0 0\nNet Increase in Cash 25,000 25,000\nCash at Beginning of Year 30,000 5,000\nCash at End of Year 55,000 30,000"]
    data = es.extract_fields(text, "cash_flow_statement")
    assert data["investing_cash_flow"]["value"] == -15000
    assert data["investing_cash_flow__comparative"]["value"] == -12000
    assert data["closing_cash"]["value"] == 55000
    from app.services.financial_validation_service import run_validation
    validation = run_validation("cash_flow_statement", data)
    assert validation["overall_status"] == "PASS"
    assert any(c["name"] == "closing_cash_check_comparative" for c in validation["checks"])


def test_invoice_layout_regression_with_unlabeled_entities_and_shipping(monkeypatch):
    import app.services.extraction_service as es
    from app.services.financial_validation_service import run_validation
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["""TAYLOR CERTIFIED PROCESSING INC
Invoice
THE ORTHOTIC GROUP INC.
Invoice Details: 6825
Date: Nov 03 2022
Description Qty Unit Price Amount
Item 01 3M 80 Adhesive 2 8.00 16.00
3M 570 Seam Sealer WHT 12oz 12 cartridges/case 10760667-011PW 4 14.00 56.00
Item 03 Film Tape 5 6.00 30.00
Item 04 Hot Melt 4 7.00 28.00
Nitrile 18756 Medical Glove BLK Medium Powder/Latex Free 100/BX 5 3.00 15.00
Subtotal $135.00
Shipping and Handling $10.00
Sales Tax 12.48
Total Due $157.48"""]
    data = es.extract_fields(text, "invoice")
    assert data["invoice_number"]["value"] == "6825"
    assert data["vendor_name"]["value"] == "TAYLOR CERTIFIED PROCESSING INC"
    assert data["customer_name"]["value"] == "THE ORTHOTIC GROUP INC."
    assert data["shipping_and_handling"]["value"] == 10
    assert len(data["line_items"]) == 5
    assert data["line_items"][0]["quantity"] == 2
    assert data["line_items"][1]["quantity"] == 4
    assert data["line_items"][4]["quantity"] == 5
    # The supplied figures deliberately expose an arithmetic inconsistency:
    # 16+56+30+28+15 = 145, while the reported subtotal is 135. The system
    # must flag this rather than silently changing a source value.
    validation = run_validation("invoice", data)
    assert validation["overall_status"] == "FAIL"
    assert any(c["name"] == "line_items_sum_check" and c["status"] == "FAIL" for c in validation["checks"])
    assert data["extraction_quality"]["manual_review_required"] is False


def test_bank_pnl_uses_bank_required_schema(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["""Consolidated Profit and Loss Account
Interest Earned 1000 900
Other Income 200 180
Total Income 1200 1080
Interest Expended 500 450
Operating Expenses 300 280
Provisions and Contingencies 100 90
Total Expenditure 900 820
Consolidated Net Profit before Minority Interest 300 260
Minority Interest 10 8
Consolidated Net Profit attributable to the Group 290 252"""]
    data = es.extract_fields(text, "profit_and_loss")
    assert data["extraction_quality"]["required_fields_missing"] == []
    assert data["extraction_quality"]["manual_review_required"] is False
    assert len(data["financial_line_items"]) >= 10


def test_dynamic_discovered_fields_keep_unknown_labels(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["Invoice Number: INV-900\nCustomer Reference: PO-7788\nPayment Terms: Net 30\nDue Date: 2026-09-30\nSubtotal: 500\nTax: 90\nTotal Due: 590"]
    data = es.extract_fields(text, "invoice")
    discovered = data["discovered_fields"]
    assert discovered["customer reference"]["value"] == "PO-7788"
    assert discovered["payment terms"]["value"] == "Net 30"
    assert discovered["due date"]["value"] == "2026-09-30"


def test_dynamic_ocr_spacing_matches_statement_labels(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["Balance Sheet\nOtherassets 100 90\nFixedassets 200 180\nTotalassets 300 270"]
    data = es.extract_fields(text, "balance_sheet")
    assert data["other_assets"]["value"] == 100
    assert data["fixed_assets"]["value"] == 200
    assert data["total_assets"]["value"] == 300
    assert data["total_assets__comparative"]["value"] == 270


def test_llm_dynamic_discovery_is_single_grounded_merge(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", True)
    calls = {"count": 0}

    def fake_call(*args, **kwargs):
        calls["count"] += 1
        return {"fields": {"total_amount": {"value": 590, "page_number": 1, "source_text": "Total Due: 590"}},
                "discovered_fields": {"payment_terms": {"value": "Net 30", "page_number": 1, "source_text": "Payment Terms: Net 30"}},
                "line_items": []}

    monkeypatch.setattr(es, "_call_llm", fake_call)
    data = es.extract_fields(["Invoice Number: INV-1\nPayment Terms: Net 30\nTotal Due: 590"], "invoice")
    assert calls["count"] == 1
    assert data["total_amount"]["value"] == 590
    assert data["discovered_fields"]["payment terms"]["value"] == "Net 30"


def test_invoice_shipping_field_is_canonical_and_dynamic_extras_remain(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["""TAX INVOICE
Supplier: Acme Medical Pvt Ltd
Bill To: XYZ Hospital
Invoice Details: 6825
Date: Nov 03 2022
Description Qty Unit Price Amount
Bandage 2 8.00 16.00
Subtotal 16.00
Shipping and Handling 10.00
Sales Tax 1.60
Total Due 27.60
Payment Terms: Net 30"""]
    data = es.extract_fields(text, "invoice")
    assert data["shipping_and_handling"]["value"] == 10
    assert data["invoice_number"]["value"] == "6825"
    assert data["discovered_fields"]["payment terms"]["value"] == "Net 30"


def test_layout_aware_invoice_table_uses_x_columns_not_embedded_numbers(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    layout = [
        {"text":"Description", "left":10, "top":10, "width":70, "height":10},
        {"text":"Qty", "left":120, "top":10, "width":20, "height":10},
        {"text":"Unit Price", "left":160, "top":10, "width":50, "height":10},
        {"text":"Amount", "left":230, "top":10, "width":45, "height":10},
        {"text":"3M", "left":10, "top":30, "width":20, "height":10},
        {"text":"570", "left":35, "top":30, "width":20, "height":10},
        {"text":"12oz", "left":60, "top":30, "width":30, "height":10},
        {"text":"100/BX", "left":92, "top":30, "width":35, "height":10},
        {"text":"2", "left":120, "top":30, "width":10, "height":10},
        {"text":"8.00", "left":160, "top":30, "width":30, "height":10},
        {"text":"16.00", "left":230, "top":30, "width":35, "height":10},
    ]
    text = ["Description Qty Unit Price Amount\n3M 570 12oz 100/BX 2 8.00 16.00"]
    data = es.extract_fields(text, "invoice", page_layouts=[layout])
    assert len(data["line_items"]) == 1
    item = data["line_items"][0]
    assert item["quantity"] == 2
    assert item["unit_price"] == 8
    assert item["amount"] == 16
    assert "570" in item["description"]
    assert "12oz" in item["description"]



def test_layout_invoice_merges_wrapped_ocr_rows_and_keeps_numeric_description(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    layout = [
        {"text":"#", "left":10, "top":10, "width":10, "height":10},
        {"text":"Description", "left":40, "top":10, "width":70, "height":10},
        {"text":"Quantity", "left":150, "top":10, "width":45, "height":10},
        {"text":"Price", "left":220, "top":10, "width":35, "height":10},
        {"text":"Total", "left":280, "top":10, "width":35, "height":10},
        {"text":"02", "left":10, "top":30, "width":15, "height":10},
        {"text":"3M", "left":40, "top":30, "width":20, "height":10},
        {"text":"570", "left":65, "top":30, "width":25, "height":10},
        {"text":"Seam", "left":95, "top":30, "width":30, "height":10},
        {"text":"Sealer", "left":40, "top":44, "width":35, "height":10},
        {"text":"WHT", "left":80, "top":44, "width":25, "height":10},
        {"text":"12oz", "left":110, "top":44, "width":30, "height":10},
        {"text":"4", "left":150, "top":44, "width":10, "height":10},
        {"text":"14.00", "left":220, "top":44, "width":35, "height":10},
        {"text":"56.00", "left":280, "top":44, "width":35, "height":10},
        {"text":"03", "left":10, "top":65, "width":15, "height":10},
        {"text":"Film", "left":40, "top":65, "width":25, "height":10},
        {"text":"5", "left":150, "top":65, "width":10, "height":10},
        {"text":"6.00", "left":220, "top":65, "width":35, "height":10},
        {"text":"30.00", "left":280, "top":65, "width":35, "height":10},
    ]
    data = es.extract_fields(["# Description Quantity Price Total"], "invoice", page_layouts=[layout])
    # This synthetic OCR layout has a wrapped second row. The parser must merge it.
    assert [(x["quantity"], x["unit_price"], x["amount"]) for x in data["line_items"]] == [(4,14,56),(5,6,30)]
    assert "570" in data["line_items"][0]["description"]
    assert "12oz" in data["line_items"][0]["description"]


def test_batch2_0499_ocr_quantity_repair_and_shipping_formula(monkeypatch):
    import app.services.extraction_service as es
    from app.services.financial_validation_service import run_validation
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    # This mirrors the failure shown by the user's OCR: the first quantity is
    # misread as 5 even though 5*8 != 16. The parser must repair it to 2 from
    # the same row's amount/price, while keeping all embedded description data.
    text = ["""INVOICE
THE ORTHOTIC GROUP INC. 160 Markland Street Invoices ap@ohi.net MARKHAM, Ontario, Canada L6C 0C6
# Description Quantity Price Total
01 3M 80 Rubber&Vinyl Spray Adh. Yellow 24 fl oz. 6 cans/case 5 $8.00 $16.00
02 3M 570 Seam Sealer WHT 12oz 12 cartridges/case 10760667-011PW 4 $14.00 $56.00
03 0485 D/C FilmTp. 1/2'x98'-1/64 44 rolls/case 5 $6.00 $30.00
04 3M 3762LM-PG Hot Melt TAN 1\" x 3\" 22 pounds/case 4 $7.00 $28.00
05 Nitrile 18756 Medical Glove BLK Medium Powder/Latex Free | 100/BX 5 $3.00 $15.00
Subtotal: $135.00
Sales Tax 8%: $12.48
Shipping and Handling: $10.00
Total Due: $157.48
Invoice Details:
Invoice#: 6825
Invoice date: Nov 03, 2022
Due date: Dec 03, 2022"""]
    data = es.extract_fields(text, "invoice")
    assert len(data["line_items"]) == 5
    assert [(x["quantity"], x["unit_price"], x["amount"]) for x in data["line_items"]] == [(2,8,16),(4,14,56),(5,6,30),(4,7,28),(5,3,15)]
    assert "570" in data["line_items"][1]["description"]
    assert "12oz" in data["line_items"][1]["description"]
    assert "100/BX" in data["line_items"][4]["description"]
    assert data["shipping_and_handling"]["value"] == 10
    validation = run_validation("invoice", data)
    total = next(c for c in validation["checks"] if c["name"] == "invoice_total_check")
    assert total["status"] == "PASS"
    subtotal = next(c for c in validation["checks"] if c["name"] == "line_items_sum_check")
    assert subtotal["status"] == "FAIL"
    assert subtotal["calculated_value"] == 145
    assert subtotal["reported_value"] == 135


def test_invoice_source_content_batch2_0499_regression(monkeypatch):
    import app.services.extraction_service as es
    from app.services.financial_validation_service import run_validation
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    text = ["""INVOICE
THE ORTHOTIC GROUP INC. 160 Markland Street Invoices ap@ohi.net MARKHAM, Ontario, Canada L6C 0C6
# Description Quantity Price Total
01 3M 80 Rubber&Vinyl Spray Adh. Yellow 24 fl oz. 6 cans/case 2 $8.00 $16.00
02 3M 570 Seam Sealer WHT 12oz 12 cartridges/case 10760667-011PW 4 $14.00 $56.00
03 0485 D/C FilmTp. 1/2'x98'-1/64 44 rolls/case 5 $6.00 $30.00
04 3M 3762LM-PG Hot Melt TAN 1\" x 3\" 22 pounds/case 4 $7.00 $28.00
05 Nitrile 18756 Medical Glove BLK Medium Powder/Latex Free | 100/BX 5 $3.00 $15.00
Subtotal: $135.00
Sales Tax 8%: $12.48
Shipping and Handling: $10.00
Total Due: $157.48
Invoice Details:
Invoice#: 6825
Invoice date: Nov 03, 2022
Due date: Dec 03, 2022
Terms and conditions:
Please send payment within 30 days of receiving this invoice."""]
    data = es.extract_fields(text, "invoice")
    assert data["invoice_number"]["value"] == "6825"
    assert data["invoice_date"]["value"] == "Nov 03, 2022"
    assert data["vendor_name"]["value"] == "THE ORTHOTIC GROUP INC."
    assert data["subtotal"]["value"] == 135
    assert data["tax_amount"]["value"] == 12.48
    assert data["shipping_and_handling"]["value"] == 10
    assert data["total_amount"]["value"] == 157.48
    assert len(data["line_items"]) == 5
    assert [(x["quantity"], x["unit_price"], x["amount"]) for x in data["line_items"]] == [(2,8,16),(4,14,56),(5,6,30),(4,7,28),(5,3,15)]
    validation = run_validation("invoice", data)
    total_check = next(c for c in validation["checks"] if c["name"] == "invoice_total_check")
    assert total_check["status"] == "PASS"
    subtotal_check = next(c for c in validation["checks"] if c["name"] == "line_items_sum_check")
    assert subtotal_check["status"] == "FAIL"
    assert subtotal_check["calculated_value"] == 145
    assert subtotal_check["reported_value"] == 135


def test_locale_parenthesis_artifact_is_negative():
    from app.services.extraction_service import _parse_money
    assert _parse_money("122,449,924()") == -122449924
    assert _parse_money("(11,577,570)") == -11577570


def test_dynamic_page_chrome_is_removed_for_multi_page_statements():
    import app.services.extraction_service as es
    pages = ["Annual Report 2020-21\nPage 1 of 2\nAssets 100", "Annual Report 2020-21\nPage 2 of 2\nLiabilities 100"]
    cleaned = es._strip_repeated_page_chrome(pages)
    assert "Annual Report" not in cleaned[0]
    assert "Page 1" not in cleaned[0]
    assert "Assets 100" in cleaned[0]
    assert "Liabilities 100" in cleaned[1]


def test_statement_aliases_recover_real_dataset_labels(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    cash_flow = es.extract_fields(["""Cash flows from operating activities:
Net cash flow (used in) / from operating activities (1,200.50) 900.00
Cash flows from investing activities:
Net cash flow used in investing activities (300.00) (200.00)
Cash flows from financing activities:
Net cash flow from / (used in) financing activities 1,500.00 1,000.00
Net increase in cash and cash equivalents 0.00 1,700.00
Cash and cash equivalents as at April 1st 121,272.51 119,572.51
Cash and cash equivalents as at March 31st 121,272.51 121,272.51"""], "cash_flow_statement")
    assert cash_flow["operating_cash_flow"]["value"] == -1200.50
    assert cash_flow["opening_cash"]["value"] == 121272.51
    assert cash_flow["closing_cash"]["value"] == 121272.51

    profit_and_loss = es.extract_fields(["""INCOME
Total 407,994.77 204,666.10
II EXPENDITURE
Total 342,548.27 158,517.40
Consolidated Net Profit for the year before minorities' interest 65,446.50 46,148.70"""], "profit_and_loss")
    assert profit_and_loss["total_income"]["value"] == 407994.77
    assert profit_and_loss["total_expenditure"]["value"] == 342548.27
    assert profit_and_loss["consolidated_net_profit_before_minority_interest"]["value"] == 65446.50


def test_balance_sheet_missing_total_is_repaired_from_visible_components(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    data = es.extract_fields(["""CAPITAL AND LIABILITIES
Capital 100
ASSETS
Cash and balances with Reserve Bank of India 200
Balances with banks and money at call and short notice 300
Investments 400
Advances 500
Fixed assets 600
Other assets 700"""], "balance_sheet")
    assert data["total_assets"]["value"] == 2700


def test_receipt_gst_aliases_are_mapped_to_invoice_totals(monkeypatch):
    import app.services.extraction_service as es
    monkeypatch.setattr(es.settings, "use_llm_fallback", False)
    data = es.extract_fields(["""TAX INVOICE
Date: 14-02-2018 13:02:42
Total (Excluding GST): 28.58
GST payable (6%): 1.72
Total (Inclusive of GST): 30.30"""], "invoice")
    assert data["subtotal"]["value"] == 28.58
    assert data["tax_amount"]["value"] == 1.72
    assert data["total_amount"]["value"] == 30.30
