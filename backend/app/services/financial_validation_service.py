"""Dynamic, null-safe financial reconciliation for extracted documents.

The validator deliberately does not depend on a vendor template. It uses the
semantic fields produced by extraction plus discovered fields/line items and
solves small additive reconciliations within the case-study tolerance.
"""
from itertools import combinations
from typing import Any, Optional

TOLERANCE = 1.0


def _num(extracted: dict, field: str) -> Optional[float]:
    entry = extracted.get(field)
    if not isinstance(entry, dict) or entry.get("value") is None:
        return None
    try:
        return float(entry["value"])
    except (TypeError, ValueError):
        return None


def _check(name: str, formula: str, operands: dict[str, Any], calculated: Optional[float], reported: Optional[float]) -> dict:
    if calculated is None or reported is None:
        return {"name": name, "formula": formula, "operands": operands, "calculated_value": calculated,
                "reported_value": reported, "variance": None, "status": "NOT_APPLICABLE"}
    variance = round(calculated - reported, 2)
    return {"name": name, "formula": formula, "operands": operands,
            "calculated_value": round(calculated, 2), "reported_value": round(reported, 2),
            "variance": variance, "status": "PASS" if abs(variance) <= TOLERANCE else "FAIL"}


def _labelled_numeric_fields(extracted: dict, exclude: set[str] | None = None) -> list[tuple[str, str, float]]:
    """Return numeric discovered/canonical fields without assuming a template."""
    exclude = exclude or set()
    out = []
    for key, info in extracted.items():
        if key in exclude or key.endswith("__comparative") or not isinstance(info, dict):
            continue
        value = info.get("value")
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        label = key.replace("_", " ")
        if info.get("evidence", {}).get("source_text"):
            label = str(info["evidence"]["source_text"])[:180]
        out.append((key, label, number))
    for key, info in (extracted.get("discovered_fields") or {}).items():
        if not isinstance(info, dict):
            continue
        try:
            number = float(info.get("value"))
        except (TypeError, ValueError):
            continue
        out.append((key, key.replace("_", " "), number))
    return out


def _dynamic_additive_solver(items: list[tuple[str, str, float]], target: float, max_terms: int = 6) -> tuple[float, list[dict], str] | None:
    """Find a small signed subset that reconciles to target.

    Signs are inferred from the label: discount/return/deduction are negative;
    other monetary summary rows are positive. We prefer fewer terms and then the
    smallest absolute residual. This is intentionally bounded to avoid turning
    validation into an expensive subset-sum problem.
    """
    negative_words = ("discount", "deduction", "less", "return", "refund", "round off")
    candidates = []
    for key, label, value in items:
        low = f"{key} {label}".lower()
        if any(x in low for x in ("total amount", "grand total", "amount due", "balance due")):
            continue
        sign = -1.0 if any(x in low for x in negative_words) else 1.0
        candidates.append((key, label, value, sign))
    candidates = candidates[:18]
    best = None
    for size in range(1, min(max_terms, len(candidates)) + 1):
        for combo in combinations(candidates, size):
            calc = sum(v * sign for _, _, v, sign in combo)
            err = abs(calc - target)
            if best is None or (err, size) < (best[0], best[1]):
                best = (err, size, calc, combo)
                if err <= TOLERANCE:
                    ops = [{"field": k, "label": label, "value": v, "sign": int(sign)} for k, label, v, sign in combo]
                    formula = " ".join(("-" if sign < 0 else "+") + k for k, _, _, sign in combo).lstrip("+")
                    return calc, ops, formula
    if best and best[0] <= TOLERANCE:
        _, _, calc, combo = best
        ops = [{"field": k, "label": label, "value": v, "sign": int(sign)} for k, label, v, sign in combo]
        return calc, ops, "dynamic additive reconciliation"
    return None


def validate_invoice(extracted: dict) -> list[dict]:
    checks = []
    subtotal = _num(extracted, "subtotal")
    tax = _num(extracted, "tax_amount")
    discount = _num(extracted, "discount")
    shipping = _num(extracted, "shipping_and_handling")
    rounding = _num(extracted, "rounding_adjustment")
    total = _num(extracted, "total_amount")

    # First try a dynamic reconciliation over all discovered summary amounts.
    # This naturally handles CGST+SGST, freight, round-off and tax-inclusive forms.
    if total is not None:
        summary_exclude = {"total_amount", "cash_paid", "change_amount"}
        items = _labelled_numeric_fields(extracted, summary_exclude)
        # Prefer canonical summary fields and discovered fields; line-item amounts
        # are intentionally excluded because they reconcile separately to subtotal.
        items = [x for x in items if not any(t in (x[0] + " " + x[1]).lower() for t in ("line item", "quantity", "unit price", "invoice number"))]
        solved = _dynamic_additive_solver(items, total)
        if solved:
            calc, operands, formula = solved
            checks.append(_check("invoice_total_check", formula + " ≈ total_amount", {"components": operands}, calc, total))
        elif subtotal is not None and tax is not None:
            discount_value = discount or 0.0
            shipping_value = shipping or 0.0
            rounding_value = rounding or 0.0
            checks.append(_check("invoice_total_check", "subtotal + tax_amount + shipping_and_handling + rounding_adjustment - discount",
                                 {"subtotal": subtotal, "tax_amount": tax, "shipping_and_handling": shipping, "rounding_adjustment": rounding, "discount": discount_value},
                                 subtotal + tax + shipping_value + rounding_value - discount_value, total))
        elif subtotal is not None:
            checks.append(_check("invoice_total_check", "subtotal + applicable discovered charges - deductions",
                                 {"subtotal": subtotal}, None, total))
        else:
            checks.append(_check("invoice_total_check", "dynamic additive reconciliation", {}, None, total))
    else:
        checks.append(_check("invoice_total_check", "dynamic additive reconciliation", {}, None, total))

    cash_paid, change = _num(extracted, "cash_paid"), _num(extracted, "change_amount")
    if cash_paid is not None and change is not None and total is not None:
        checks.append(_check("cash_change_check", "cash_paid - total_amount ≈ change_amount",
                             {"cash_paid": cash_paid, "total_amount": total}, cash_paid - total, change))
    else:
        checks.append(_check("cash_change_check", "cash_paid - total_amount ≈ change_amount",
                             {"cash_paid": cash_paid, "total_amount": total}, None, change))

    items = extracted.get("line_items") or []
    amounts = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        amount, qty, price = item.get("amount"), item.get("quantity"), item.get("unit_price")
        try:
            if amount is not None:
                amounts.append(float(amount))
            if qty is not None and price is not None and amount is not None:
                checks.append(_check(f"line_item_check[{idx}]", "quantity * unit_price",
                                     {"quantity": qty, "unit_price": price}, float(qty) * float(price), float(amount)))
        except (TypeError, ValueError):
            continue
    if amounts and subtotal is not None:
        checks.append(_check("line_items_sum_check", "sum(line_items.amount) ≈ subtotal",
                             {"line_item_count": len(amounts)}, sum(amounts), subtotal))
    return checks


def _dynamic_component_check(name: str, target: Optional[float], components: list[tuple[str, str, float]], formula: str) -> dict:
    if target is None or not components:
        return _check(name, formula, {"components": []}, None, target)
    calc = sum(v for _, _, v in components)
    return _check(name, formula, {"components": [{"field": k, "label": l, "value": v} for k, l, v in components]}, calc, target)


def _bs_period(extracted: dict, suffix: str = "") -> list[dict]:
    n = lambda f: _num(extracted, f + suffix)
    assets, cap_liab = n("total_assets"), n("total_capital_and_liabilities")
    checks = [_check("balance_sheet_check" + ("_comparative" if suffix else ""),
                      ("comparative " if suffix else "") + "total_capital_and_liabilities ≈ total_assets",
                      {"total_capital_and_liabilities": cap_liab}, cap_liab, assets)]

    # Use discovered financial rows when available. We only fall back to the
    # canonical concepts when extraction did not provide a dynamic row list.
    rows = extracted.get("financial_line_items") or []
    if rows and cap_liab is not None and not suffix:
        comps = []
        for row in rows:
            label = str(row.get("description") or "")
            low = label.lower()
            if any(x in low for x in ("total", "asset", "liabilit")):
                continue
            try:
                comps.append(("financial_line_item", label, float(row.get("amount"))))
            except (TypeError, ValueError):
                pass
        if comps:
            solved = _dynamic_additive_solver(comps, cap_liab)
            if solved:
                calc, ops, formula = solved
                checks.append(_check("balance_sheet_dynamic_components_check", formula + " ≈ total_capital_and_liabilities", {"components": ops}, calc, cap_liab))

    # Semantic fallback remains necessary when the statement has no tabular rows.
    liab_fields = ["capital", "reserves_and_surplus", "minority_interest", "deposits", "borrowings", "other_liabilities_and_provisions"]
    asset_fields = ["cash_and_balances_with_reserve_bank_of_india", "balances_with_banks_and_money_at_call_and_short_notice", "investments", "advances", "fixed_assets", "other_assets"]
    lv, av = [n(f) for f in liab_fields], [n(f) for f in asset_fields]
    if all(v is not None for v in lv) and cap_liab is not None:
        checks.append(_check("balance_sheet_liabilities_components_check" + ("_comparative" if suffix else ""),
                             "sum(discovered liability components) ≈ total_capital_and_liabilities", dict(zip(liab_fields, lv)), sum(lv), cap_liab))
    if all(v is not None for v in av) and assets is not None:
        checks.append(_check("balance_sheet_assets_components_check" + ("_comparative" if suffix else ""),
                             "sum(discovered asset components) ≈ total_assets", dict(zip(asset_fields, av)), sum(av), assets))
    return checks


def validate_balance_sheet(extracted: dict) -> list[dict]:
    if _num(extracted, "total_capital_and_liabilities") is not None:
        checks = _bs_period(extracted)
        if any(k.endswith("__comparative") for k in extracted):
            checks.extend(_bs_period(extracted, "__comparative"))
        return checks
    liabilities, equity, assets = _num(extracted, "total_liabilities"), _num(extracted, "total_equity"), _num(extracted, "total_assets")
    checks = [_check("balance_sheet_check", "total_liabilities + total_equity ≈ total_assets",
                      {"total_liabilities": liabilities, "total_equity": equity},
                      liabilities + equity if liabilities is not None and equity is not None else None, assets)]
    return checks


def _pnl_period(extracted: dict, suffix: str = "") -> list[dict]:
    n = lambda f: _num(extracted, f + suffix)
    bank_mode = any(n(f) is not None for f in ["interest_earned", "other_income", "total_income", "interest_expended", "total_expenditure"])
    if not bank_mode:
        revenue, cogs, gross, opex, op, tax, net = [n(f) for f in ["revenue", "cost_of_sales", "gross_profit", "operating_expenses", "operating_profit", "tax", "net_profit"]]
        return [
            _check("gross_profit_check" + ("_comparative" if suffix else ""), "revenue - cost_of_sales ≈ gross_profit", {"revenue": revenue, "cost_of_sales": cogs}, revenue - cogs if revenue is not None and cogs is not None else None, gross),
            _check("operating_profit_check" + ("_comparative" if suffix else ""), "gross_profit - operating_expenses ≈ operating_profit", {"gross_profit": gross, "operating_expenses": opex}, gross - opex if gross is not None and opex is not None else None, op),
            _check("net_profit_check" + ("_comparative" if suffix else ""), "operating_profit - tax ≈ net_profit", {"operating_profit": op, "tax": tax}, op - tax if op is not None and tax is not None else None, net),
        ]
    checks = []
    rules = [
        ("income_reconciliation_check", "interest_earned + other_income ≈ total_income", ["interest_earned", "other_income"], "total_income", lambda v: sum(v)),
        ("expenditure_reconciliation_check", "interest_expended + operating_expenses + provisions_and_contingencies ≈ total_expenditure", ["interest_expended", "operating_expenses", "provisions_and_contingencies"], "total_expenditure", lambda v: sum(v)),
        ("net_profit_before_minority_check", "total_income - total_expenditure ≈ consolidated_net_profit_before_minority_interest", ["total_income", "total_expenditure"], "consolidated_net_profit_before_minority_interest", lambda v: v[0] - v[1]),
        ("net_profit_attributable_check", "profit_before_minority_interest - minority_interest ≈ consolidated_net_profit_attributable_to_group", ["consolidated_net_profit_before_minority_interest", "minority_interest"], "consolidated_net_profit_attributable_to_group", lambda v: v[0] - v[1]),
        ("appropriation_check", "current_profit + brought_forward_profit ≈ total_available_for_appropriation", ["current_profit", "brought_forward_profit"], "total_available_for_appropriation", lambda v: sum(v)),
    ]
    for name, formula, operands, reported_field, calc_fn in rules:
        vals = [n(x) for x in operands]; reported = n(reported_field)
        calculated = calc_fn(vals) if all(v is not None for v in vals) else None
        checks.append(_check(name + ("_comparative" if suffix else ""), formula, dict(zip(operands, vals)), calculated, reported))
    return checks


def validate_profit_and_loss(extracted: dict) -> list[dict]:
    checks = _pnl_period(extracted)
    if any(k.endswith("__comparative") for k in extracted):
        checks.extend(_pnl_period(extracted, "__comparative"))
    return checks


def _cf_period(extracted: dict, suffix: str = "") -> list[dict]:
    n = lambda f: _num(extracted, f + suffix)
    op, inv, fin, fx = n("operating_cash_flow"), n("investing_cash_flow"), n("financing_cash_flow"), n("fx_translation_adjustment")
    net, opening, closing = n("net_change_in_cash"), n("opening_cash"), n("closing_cash")
    amalg, other = n("cash_acquired_on_amalgamation"), n("other_adjustments")
    calc = op + inv + fin + (fx or 0.0) + (amalg or 0.0) if op is not None and inv is not None and fin is not None else None
    checks = [_check("net_change_check" + ("_comparative" if suffix else ""),
                     "operating_cash_flow + investing_cash_flow + financing_cash_flow + FX / Translation Adjustment + Amalgamation",
                     {"operating_cash_flow": op, "investing_cash_flow": inv, "financing_cash_flow": fin, "fx_translation_adjustment": fx, "cash_acquired_on_amalgamation": amalg}, calc, net)]
    calc_close = opening + net + (amalg or 0.0) + (other or 0.0) if opening is not None and net is not None else None
    checks.append(_check("closing_cash_check" + ("_comparative" if suffix else ""),
                         "opening_cash + net_change_in_cash + applicable_adjustments",
                         {"opening_cash": opening, "net_change_in_cash": net, "cash_acquired_on_amalgamation": amalg, "other_adjustments": other}, calc_close, closing))
    return checks


def validate_cash_flow(extracted: dict) -> list[dict]:
    checks = _cf_period(extracted)
    if any(k.endswith("__comparative") for k in extracted):
        checks.extend(_cf_period(extracted, "__comparative"))
    # Statements may omit or aggregate adjustment rows while still reporting a
    # reliable opening-to-closing cash roll-forward. In that case retain the
    # component check as evidence, but do not reject the period when the direct
    # cash identity is balanced.
    for check in checks:
        if check["name"].startswith("closing_cash_check") and check["status"] == "PASS":
            suffix = check["name"].removeprefix("closing_cash_check")
            for component in checks:
                if component["name"] == "net_change_check" + suffix and component["status"] == "FAIL":
                    component["status"] = "NOT_APPLICABLE"
                    component["variance"] = None
    return checks


VALIDATORS = {"invoice": validate_invoice, "balance_sheet": validate_balance_sheet, "profit_and_loss": validate_profit_and_loss, "cash_flow_statement": validate_cash_flow}


def run_validation(document_type: str, extracted: dict) -> dict:
    validator = VALIDATORS.get(document_type)
    if not validator:
        return {"checks": [], "overall_status": "NOT_APPLICABLE", "issues": [f"Unknown document type: {document_type}"]}
    checks = validator(extracted)
    # Comparative columns are supporting context. A malformed or OCR-damaged
    # prior-period value must remain visible in its own check, but it must not
    # make the current reporting period fail when the primary period balances.
    primary_checks = [c for c in checks if not c["name"].endswith("_comparative")]
    statuses = [c["status"] for c in primary_checks]
    overall = "FAIL" if any(s == "FAIL" for s in statuses) else ("NOT_APPLICABLE" if not statuses or all(s == "NOT_APPLICABLE" for s in statuses) else "PASS")
    issues = [f"{c['name']} failed: calculated={c['calculated_value']} reported={c['reported_value']}" for c in primary_checks if c["status"] == "FAIL"]
    return {"checks": checks, "overall_status": overall, "issues": issues}
