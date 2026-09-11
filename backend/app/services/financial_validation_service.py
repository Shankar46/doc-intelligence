"""Financial validation rules required by the case study."""
import logging
from typing import Any, Optional
logger = logging.getLogger(__name__)
TOLERANCE = 1.0


def _num(extracted: dict, field: str) -> Optional[float]:
    entry = extracted.get(field)
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _check(name: str, formula: str, operands: dict[str, Any], calculated: Optional[float], reported: Optional[float]) -> dict:
    if calculated is None or reported is None:
        return {"name": name, "formula": formula, "operands": operands,
                "calculated_value": calculated, "reported_value": reported,
                "variance": None, "status": "NOT_APPLICABLE"}
    variance = round(calculated - reported, 2)
    return {"name": name, "formula": formula, "operands": operands,
            "calculated_value": round(calculated, 2), "reported_value": round(reported, 2),
            "variance": variance, "status": "PASS" if abs(variance) <= TOLERANCE else "FAIL"}


def validate_invoice(extracted: dict) -> list[dict]:
    checks = []
    subtotal, tax, discount, total = (_num(extracted, x) for x in ("subtotal", "tax_amount", "discount", "total_amount"))
    # Missing discount means zero only when subtotal/tax/total are otherwise present.
    discount = 0.0 if discount is None else discount
    if subtotal is not None and total is not None:
        if tax is not None:
            checks.append(_check("invoice_total_check", "subtotal + tax_amount - discount",
                                 {"subtotal": subtotal, "tax_amount": tax, "discount": discount}, subtotal + tax - discount, total))
        else:
            checks.append(_check("invoice_total_check", "subtotal - discount (tax included/not separately shown)",
                                 {"subtotal": subtotal, "discount": discount}, subtotal - discount, total))
    else:
        checks.append(_check("invoice_total_check", "subtotal + tax_amount - discount",
                             {"subtotal": subtotal, "tax_amount": tax, "discount": discount}, None, total))

    items = extracted.get("line_items") or []
    if items:
        amounts = []
        for item in items:
            if not isinstance(item, dict):
                continue
            amount, qty, price = item.get("amount"), item.get("quantity"), item.get("unit_price")
            if amount is not None:
                try: amounts.append(float(amount))
                except (TypeError, ValueError): pass
            if qty is not None and price is not None and amount is not None:
                checks.append(_check(f"line_item_check[{item.get('description', '?')}]", "quantity * unit_price",
                                     {"quantity": qty, "unit_price": price}, float(qty) * float(price), float(amount)))
        if amounts and subtotal is not None:
            checks.append(_check("line_items_sum_check", "sum(line_items.amount)",
                                 {"line_item_count": len(amounts)}, sum(amounts), subtotal))
    return checks


def _bs_period(extracted: dict, suffix: str = "") -> list[dict]:
    def n(base): return _num(extracted, base + suffix)
    assets, cap_liab = n("total_assets"), n("total_capital_and_liabilities")
    checks = [_check("balance_sheet_check" + ("_comparative" if suffix else ""),
                     ("comparative " if suffix else "") + "total_capital_and_liabilities ≈ total_assets",
                     {"total_capital_and_liabilities": cap_liab}, cap_liab, assets)]
    liab_fields = ["capital", "reserves_and_surplus", "minority_interest", "deposits", "borrowings", "other_liabilities_and_provisions"]
    asset_fields = ["cash_and_balances_with_reserve_bank_of_india", "balances_with_banks_and_money_at_call_and_short_notice", "investments", "advances", "fixed_assets", "other_assets"]
    lv = [n(f) for f in liab_fields]; av = [n(f) for f in asset_fields]
    if all(v is not None for v in lv) and cap_liab is not None:
        checks.append(_check("balance_sheet_liabilities_components_check" + ("_comparative" if suffix else ""),
                             "sum(capital, reserves_and_surplus, minority_interest, deposits, borrowings, other_liabilities_and_provisions) ≈ total_capital_and_liabilities",
                             dict(zip(liab_fields, lv)), sum(lv), cap_liab))
    if all(v is not None for v in av) and assets is not None:
        checks.append(_check("balance_sheet_assets_components_check" + ("_comparative" if suffix else ""),
                             "sum(asset components) ≈ total_assets", dict(zip(asset_fields, av)), sum(av), assets))
    return checks


def validate_balance_sheet(extracted: dict) -> list[dict]:
    checks = _bs_period(extracted)
    # A comparative period is validated independently when all required totals exist.
    if _num(extracted, "total_assets__comparative") is not None or _num(extracted, "total_capital_and_liabilities__comparative") is not None:
        checks.extend(_bs_period(extracted, "__comparative"))
    # For older generic statements, use liabilities + equity only if those fields are actually present.
    if _num(extracted, "total_capital_and_liabilities") is None:
        liabilities, equity, assets = _num(extracted, "total_liabilities"), _num(extracted, "total_equity"), _num(extracted, "total_assets")
        checks = [_check("balance_sheet_check", "total_liabilities + total_equity ≈ total_assets",
                         {"total_liabilities": liabilities, "total_equity": equity},
                         liabilities + equity if liabilities is not None and equity is not None else None, assets)]
    return checks


def validate_profit_and_loss(extracted: dict) -> list[dict]:
    n = lambda f: _num(extracted, f)
    # Support the exact case-study reconciliation fields when they are present.
    # For generic P&L documents, retain the common revenue/COGS/profit checks.
    if not any(n(f) is not None for f in ["interest_earned", "other_income", "total_income", "interest_expended", "total_expenditure"]):
        revenue, cogs, gross, opex, op_profit, tax, net = [n(f) for f in ["revenue", "cost_of_sales", "gross_profit", "operating_expenses", "operating_profit", "tax", "net_profit"]]
        return [
            _check("gross_profit_check", "revenue - cost_of_sales", {"revenue": revenue, "cost_of_sales": cogs}, revenue - cogs if revenue is not None and cogs is not None else None, gross),
            _check("operating_profit_check", "gross_profit - operating_expenses", {"gross_profit": gross, "operating_expenses": opex}, gross - opex if gross is not None and opex is not None else None, op_profit),
            _check("net_profit_check", "operating_profit - tax", {"operating_profit": op_profit, "tax": tax}, op_profit - tax if op_profit is not None and tax is not None else None, net),
        ]
    checks = []
    pairs = [
        ("income_reconciliation_check", "interest_earned + other_income ≈ total_income", ["interest_earned", "other_income"], "total_income"),
        ("expenditure_reconciliation_check", "interest_expended + operating_expenses + provisions_and_contingencies ≈ total_expenditure", ["interest_expended", "operating_expenses", "provisions_and_contingencies"], "total_expenditure"),
        ("net_profit_before_minority_check", "total_income - total_expenditure ≈ consolidated_net_profit_before_minority_interest", ["total_income", "total_expenditure"], "consolidated_net_profit_before_minority_interest"),
        ("net_profit_attributable_check", "profit_before_minority_interest - minority_interest ≈ consolidated_net_profit_attributable_to_group", ["consolidated_net_profit_before_minority_interest", "minority_interest"], "consolidated_net_profit_attributable_to_group"),
        ("appropriation_check", "current_profit + brought_forward_profit ≈ total_available_for_appropriation", ["current_profit", "brought_forward_profit"], "total_available_for_appropriation"),
    ]
    for name, formula, operands, reported_field in pairs:
        vals = [n(x) for x in operands]; reported = n(reported_field)
        calculated = None if any(v is None for v in vals) else sum(vals) if len(vals) == 2 and name != "net_profit_attributable_check" else None
        if name == "net_profit_before_minority_check" and all(v is not None for v in vals): calculated = vals[0] - vals[1]
        elif name == "net_profit_attributable_check" and all(v is not None for v in vals): calculated = vals[0] - vals[1]
        elif name in {"income_reconciliation_check", "expenditure_reconciliation_check", "appropriation_check"} and all(v is not None for v in vals): calculated = sum(vals)
        checks.append(_check(name, formula, dict(zip(operands, vals)), calculated, reported))
    return checks


def validate_cash_flow(extracted: dict) -> list[dict]:
    n = lambda f: _num(extracted, f)
    op, inv, fin, fx = n("operating_cash_flow"), n("investing_cash_flow"), n("financing_cash_flow"), n("fx_translation_adjustment")
    net, opening, closing = n("net_change_in_cash"), n("opening_cash"), n("closing_cash")
    amalg = n("cash_acquired_on_amalgamation") or 0.0
    other = n("other_adjustments") or 0.0
    calc = op + inv + fin + (fx or 0.0) if op is not None and inv is not None and fin is not None else None
    checks = [_check("net_change_check", "operating_cash_flow + investing_cash_flow + financing_cash_flow + FX / Translation Adjustment",
                     {"operating_cash_flow": op, "investing_cash_flow": inv, "financing_cash_flow": fin, "fx_translation_adjustment": fx}, calc, net)]
    calc_close = opening + net + amalg + other if opening is not None and net is not None else None
    checks.append(_check("closing_cash_check", "opening_cash + net_change_in_cash + applicable_adjustments",
                         {"opening_cash": opening, "net_change_in_cash": net, "cash_acquired_on_amalgamation": amalg, "other_adjustments": other}, calc_close, closing))
    return checks

VALIDATORS = {"invoice": validate_invoice, "balance_sheet": validate_balance_sheet, "profit_and_loss": validate_profit_and_loss, "cash_flow_statement": validate_cash_flow}

def run_validation(document_type: str, extracted: dict) -> dict:
    validator = VALIDATORS.get(document_type)
    if not validator:
        return {"checks": [], "overall_status": "NOT_APPLICABLE", "issues": [f"Unknown document type: {document_type}"]}
    checks = validator(extracted)
    statuses = [c["status"] for c in checks]
    overall = "FAIL" if any(s == "FAIL" for s in statuses) else ("NOT_APPLICABLE" if all(s == "NOT_APPLICABLE" for s in statuses) else "PASS")
    issues = [f"{c['name']} failed: calculated={c['calculated_value']} reported={c['reported_value']}" for c in checks if c["status"] == "FAIL"]
    return {"checks": checks, "overall_status": overall, "issues": issues}
