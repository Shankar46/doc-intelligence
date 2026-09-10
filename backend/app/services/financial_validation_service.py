"""
Financial calculation validation (spec section 4.4). One function per
document type. Every check returns PASS / FAIL / NOT_APPLICABLE -- never
invents a missing operand. Tolerance is a small absolute epsilon to
allow for rounding; tune per your test data.
"""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

TOLERANCE = 1.0  # absolute currency-unit tolerance for rounding differences


def _num(extracted: dict, field: str) -> Optional[float]:
    """Pull a numeric value out of an ExtractedField dict, else None."""
    entry = extracted.get(field)
    if not entry:
        return None
    value = entry.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check(name: str, formula: str, operands: dict[str, Any],
           calculated: Optional[float], reported: Optional[float]) -> dict:
    if calculated is None or reported is None:
        return {
            "name": name, "formula": formula, "operands": operands,
            "calculated_value": calculated, "reported_value": reported,
            "variance": None, "status": "NOT_APPLICABLE",
        }
    variance = round(calculated - reported, 2)
    status = "PASS" if abs(variance) <= TOLERANCE else "FAIL"
    return {
        "name": name, "formula": formula, "operands": operands,
        "calculated_value": round(calculated, 2), "reported_value": round(reported, 2),
        "variance": variance, "status": status,
    }


def validate_invoice(extracted: dict) -> list[dict]:
    checks = []
    subtotal = _num(extracted, "subtotal")
    tax = _num(extracted, "tax_amount")
    discount = _num(extracted, "discount") or 0.0
    total = _num(extracted, "total_amount")

    if subtotal is not None and tax is not None:
        calculated_total = subtotal + tax - discount
        checks.append(_check(
            "invoice_total_check", "subtotal + tax_amount - discount",
            {"subtotal": subtotal, "tax_amount": tax, "discount": discount},
            calculated_total, total,
        ))
    else:
        checks.append(_check(
            "invoice_total_check", "subtotal + tax_amount - discount",
            {"subtotal": subtotal, "tax_amount": tax, "discount": discount},
            None, total,
        ))

    line_items = extracted.get("line_items") or []
    if line_items:
        line_total_sum = 0.0
        has_all_amounts = True
        for item in line_items:
            amount = item.get("amount")
            qty = item.get("quantity")
            price = item.get("unit_price")
            if amount is None:
                has_all_amounts = False
                continue
            line_total_sum += amount
            # per-line quantity * unit_price ≈ amount check
            if qty is not None and price is not None:
                checks.append(_check(
                    f"line_item_check[{item.get('description', '?')}]",
                    "quantity * unit_price", {"quantity": qty, "unit_price": price},
                    qty * price, amount,
                ))
        if has_all_amounts:
            checks.append(_check(
                "line_items_sum_check", "sum(line_items.amount)",
                {"line_item_count": len(line_items)},
                line_total_sum, subtotal,
            ))

    return checks


def validate_balance_sheet(extracted: dict) -> list[dict]:
    assets = _num(extracted, "total_assets")
    liabilities = _num(extracted, "total_liabilities")
    equity = _num(extracted, "total_equity")

    calculated = None
    if liabilities is not None and equity is not None:
        calculated = liabilities + equity

    return [_check(
        "balance_sheet_check", "total_liabilities + total_equity ≈ total_assets",
        {"total_liabilities": liabilities, "total_equity": equity},
        calculated, assets,
    )]


def validate_profit_and_loss(extracted: dict) -> list[dict]:
    revenue = _num(extracted, "revenue")
    cogs = _num(extracted, "cost_of_sales")
    gross_profit = _num(extracted, "gross_profit")
    opex = _num(extracted, "operating_expenses")
    operating_profit = _num(extracted, "operating_profit")
    tax = _num(extracted, "tax")
    net_profit = _num(extracted, "net_profit")

    checks = []

    calc_gross = (revenue - cogs) if (revenue is not None and cogs is not None) else None
    checks.append(_check(
        "gross_profit_check", "revenue - cost_of_sales", {"revenue": revenue, "cost_of_sales": cogs},
        calc_gross, gross_profit,
    ))

    calc_operating = (gross_profit - opex) if (gross_profit is not None and opex is not None) else None
    checks.append(_check(
        "operating_profit_check", "gross_profit - operating_expenses",
        {"gross_profit": gross_profit, "operating_expenses": opex},
        calc_operating, operating_profit,
    ))

    calc_net = (operating_profit - tax) if (operating_profit is not None and tax is not None) else None
    checks.append(_check(
        "net_profit_check", "operating_profit - tax",
        {"operating_profit": operating_profit, "tax": tax},
        calc_net, net_profit,
    ))

    return checks


def validate_cash_flow(extracted: dict) -> list[dict]:
    operating = _num(extracted, "operating_cash_flow")
    investing = _num(extracted, "investing_cash_flow")
    financing = _num(extracted, "financing_cash_flow")
    opening = _num(extracted, "opening_cash")
    net_change = _num(extracted, "net_change_in_cash")
    closing = _num(extracted, "closing_cash")

    checks = []

    if operating is not None and investing is not None and financing is not None:
        calc_net_change = operating + investing + financing
    else:
        calc_net_change = None
    checks.append(_check(
        "net_change_check",
        "operating_cash_flow + investing_cash_flow + financing_cash_flow",
        {"operating_cash_flow": operating, "investing_cash_flow": investing, "financing_cash_flow": financing},
        calc_net_change, net_change,
    ))

    calc_closing = (opening + net_change) if (opening is not None and net_change is not None) else None
    checks.append(_check(
        "closing_cash_check", "opening_cash + net_change_in_cash",
        {"opening_cash": opening, "net_change_in_cash": net_change},
        calc_closing, closing,
    ))

    return checks


VALIDATORS = {
    "invoice": validate_invoice,
    "balance_sheet": validate_balance_sheet,
    "profit_and_loss": validate_profit_and_loss,
    "cash_flow_statement": validate_cash_flow,
}


def run_validation(document_type: str, extracted: dict) -> dict:
    validator = VALIDATORS.get(document_type)
    if not validator:
        return {"checks": [], "overall_status": "NOT_APPLICABLE", "issues": [f"Unknown document type: {document_type}"]}

    checks = validator(extracted)
    statuses = [c["status"] for c in checks]

    if any(s == "FAIL" for s in statuses):
        overall = "FAIL"
    elif all(s == "NOT_APPLICABLE" for s in statuses):
        overall = "NOT_APPLICABLE"
    else:
        overall = "PASS"

    issues = [f"{c['name']} failed: calculated={c['calculated_value']} reported={c['reported_value']}"
              for c in checks if c["status"] == "FAIL"]

    return {"checks": checks, "overall_status": overall, "issues": issues}
