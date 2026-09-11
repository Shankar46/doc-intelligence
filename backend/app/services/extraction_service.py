"""Document field/table extraction with LLM + deterministic OCR fallback."""
import json
import logging
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # local/test environments may omit the optional LLM SDK
    OpenAI = None

from app.core.config import settings

logger = logging.getLogger(__name__)


def _clean_json_response(raw_text: str) -> str:
    if not raw_text:
        return "{}"
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start_idx, end_idx = cleaned.find("{"), cleaned.rfind("}")
    if start_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx:end_idx + 1]
    return cleaned


def _parse_number(token: str) -> float | int | None:
    if not token:
        return None
    s = token.strip().replace(" ", "")
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    # OCR sometimes inserts a space before the last comma, e.g. 11,462,071 ,336.
    s = re.sub(r"(?<=\d)\s+(?=\d|,)", "", s)
    s = re.sub(r"[^0-9.\-]", "", s)
    if not s:
        return None
    try:
        value = float(s.replace(",", ""))
        if negative:
            value = -value
        return int(value) if value.is_integer() else value
    except ValueError:
        return None


def _number_tokens(text: str) -> list[str]:
    # Normalize OCR spaces around separators before tokenizing. Keep decimal-comma
    # amounts such as 126,27 intact; they are common on invoices.
    text = re.sub(r"(?<=\d)\s*,\s*(?=\d)", ",", text)
    return re.findall(
        r"\(?\d{1,3}(?:[,.]\d{3})+(?:[.,]\d{1,2})?\)?|"
        r"\(?\d+(?:[.,]\d{1,2})?\)?", text
    )


def _money_tokens(text: str) -> list[str]:
    """Return likely monetary tokens, supporting $126,27 / 126.27 / 1,234.56."""
    text = re.sub(r"(?<=\d)\s+(?=[,.]\d)", "", text)
    return re.findall(
        r"(?:[$€£₹]|INR|USD|EUR|GBP)?\s*\(?\d{1,3}(?:[,.]\d{3})+(?:[.,]\d{1,2})?\)?|"
        r"(?:[$€£₹]|INR|USD|EUR|GBP)?\s*\(?\d+[.,]\d{1,2}\)?",
        text, re.I
    )


def _parse_money(token: str) -> float | int | None:
    """Parse both 1,234.56 and European-style 1.234,56 / 126,27."""
    if not token:
        return None
    s = re.sub(r"[^0-9,().-]", "", token).strip()
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    if not s:
        return None
    if "," in s and "." in s:
        # Last separator is the decimal separator.
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        # 126,27 => decimal comma; 1,234 => thousands separator.
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = parts[0] + "." + parts[1]
        else:
            s = "".join(parts)
    try:
        value = float(s)
        if negative:
            value = -value
        return int(value) if value.is_integer() else value
    except ValueError:
        return None


def _field(value: Any, page: int = 1, source: str | None = None) -> dict:
    return {"value": value, "page_number": page, "source_text": source}



def _collect_following_numbers(lines: list[str], start: int, max_lines: int = 4) -> tuple[list[float], str]:
    """Collect numeric values from a label line and nearby OCR fragments.

    Tesseract PSM 11 often emits a table as label -> current value -> comparative
    value on separate lines. Stop before the next obvious text label so numbers
    from the next row are not swallowed.
    """
    values: list[float] = []
    source_parts: list[str] = []
    for j in range(start, min(start + max_lines + 1, len(lines))):
        ln = lines[j].strip()
        if not ln:
            continue
        nums = [_parse_number(x) for x in _number_tokens(ln)]
        nums = [x for x in nums if x is not None]
        if j > start and not nums:
            # A new textual row normally means the value sequence is complete.
            if re.search(r"[A-Za-z]", ln):
                break
            continue
        if nums:
            values.extend(nums)
            source_parts.append(ln)
            if len(values) >= 2:
                break
    return values, " ".join(source_parts)[:180]


def _extract_invoice_fallback(lines: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields: dict[str, Any] = {}
    line_items: list[dict[str, Any]] = []

    def put(name: str, value: Any, source: str):
        if value is not None and name not in fields:
            fields[name] = _field(value, 1, source[:180])

    # Invoice metadata: require an explicit label, rather than matching ordinary
    # prose containing words such as "from", "to", or "tax".
    for i, line in enumerate(lines):
        m = re.search(r"\binvoice\s*(?:number|no\.?|#)\s*[:#-]?\s*([A-Z0-9][A-Z0-9./_-]*)", line, re.I)
        if m:
            put("invoice_number", m.group(1), line)
        elif re.fullmatch(r"[A-Z0-9][A-Z0-9./_-]*", lines[i].strip(), re.I) and i > 0 and re.search(r"invoice\s*(?:number|no\.?|#)\s*[:#-]?\s*$", lines[i-1], re.I):
            put("invoice_number", lines[i].strip(), lines[i-1] + " " + lines[i].strip())

        if re.search(r"(?:invoice\s*)?(?:date|date\s+of\s+issue|issued\s+on|dated)\b", line, re.I):
            m = re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}", line)
            if m:
                put("invoice_date", m.group(0), line)
            elif i + 1 < len(lines):
                m2 = re.search(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4}", lines[i+1])
                if m2:
                    put("invoice_date", m2.group(0), line + " " + lines[i+1])

        # Only accept From/Vendor/Supplier when it is an actual field label.
        m = re.search(r"(?:^|\b)(?:from|vendor|supplier|billed\s+by)\s*[:#-]\s*(.+)$", line, re.I)
        if m:
            put("vendor_name", m.group(1).strip(), line)
        elif re.match(r"^(?:from|vendor|supplier|billed\s+by)\s*[:#-]?\s*$", line, re.I) and i + 1 < len(lines):
            nxt = lines[i+1].strip()
            if nxt and not re.search(r"^(?:invoice|date|currency|subtotal|tax|total|discount)\b", nxt, re.I):
                put("vendor_name", nxt, line + " " + nxt)

        m = re.search(r"(?:billed\s+to|customer|client|bill\s+to)\s*[:#-]\s*(.+)$", line, re.I)
        if m:
            put("customer_name", m.group(1).strip(), line)
        elif re.match(r"^(?:billed\s+to|customer|client|bill\s+to)\s*[:#-]?\s*$", line, re.I) and i + 1 < len(lines):
            nxt = lines[i+1].strip()
            if nxt and not re.search(r"^(?:invoice|date|currency|subtotal|tax|total|discount)\b", nxt, re.I):
                put("customer_name", nxt, line + " " + nxt)

        # Currency can be explicit or represented by a currency symbol near a
        # monetary value. Never infer it from a Tax ID or arbitrary digits.
        if "currency" not in fields:
            m = re.search(r"\b(?:USD|INR|EUR|GBP|AUD|CAD)\b|[$€£₹]", line, re.I)
            if m:
                put("currency", m.group(0).upper() if m.group(0).isalpha() else m.group(0), line)

    # Amount labels are matched as labels, and Tax ID lines are explicitly ignored.
    amount_labels = {
        "subtotal": r"\bsubtotal\b",
        "tax_amount": r"\b(?:tax|gst|vat)\b",
        "discount": r"\bdiscount\b",
        "total_amount": r"\b(?:grand\s+total|total\s+amount\s+due|amount\s+due|balance\s+due|total)\b",
    }
    for i, line in enumerate(lines):
        if re.search(r"\b(?:tax\s*id|taxpayer\s*id|vat\s*id|gst\s*in)\b", line, re.I):
            continue
        for name, label in amount_labels.items():
            if name in fields or not re.search(label, line, re.I):
                continue
            # For tax, do not use percentages as amounts.
            tail = re.sub(rf"^.*?{label}\s*[:#-]?\s*", "", line, flags=re.I).strip()
            money = _money_tokens(tail)
            if name == "tax_amount":
                money = [t for t in money if not re.search(r"\d\s*%", tail[max(0, tail.find(t)):tail.find(t)+len(t)+3])]
            if money:
                value = _parse_money(money[-1])
                if value is not None:
                    put(name, value, line)
            elif i + 1 < len(lines):
                nxt = lines[i+1].strip()
                money = _money_tokens(nxt)
                if money:
                    value = _parse_money(money[-1])
                    if value is not None:
                        put(name, value, line + " " + nxt)

    # Recover line items only when OCR shows a nearby table header. This prevents
    # unrelated numbers such as tax IDs, dates and page numbers from becoming
    # fake invoice rows. If the table structure is ambiguous, leave line_items
    # empty rather than inventing a row.
    header_seen = False
    for idx, line in enumerate(lines):
        if re.search(r"description.*(?:quantity|qty).*(?:unit\s+price|price).*(?:amount|total)|(?:description|item).*?(?:qty|quantity).*?(?:price|amount)", line, re.I):
            header_seen = True
            continue
        if header_seen and re.search(r"subtotal|tax\s*id|tax|discount|total|invoice|date|currency", line, re.I):
            continue
        if not header_seen:
            continue
        nums = _number_tokens(line)
        if len(nums) == 3:
            parsed = [_parse_money(x) for x in nums]
            if all(x is not None for x in parsed):
                desc = re.sub(r"(?:[$€£₹]|\b(?:USD|INR|EUR|GBP)\b)?\s*\(?\d[\d, .]*\)?", " ", line, flags=re.I).strip(" -:")
                if desc:
                    line_items.append({"description": desc, "quantity": parsed[0], "unit_price": parsed[1], "amount": parsed[2]})
    return fields, line_items

def _extract_cashflow_fallback(lines: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fields: dict[str, Any] = {}
    labels = {
        "operating_cash_flow": r"(?:net\s+)?cash\s+(?:provided|generated|from|inflow)\s+(?:by\s+)?(?:operating|operations)\s+activities?|cash\s+flow(?:s)?\s+from\s+operating\s+activities|net\s+cash\s+(?:from|generated\s+from)\s+operations",
        "investing_cash_flow": r"(?:net\s+)?cash\s+(?:used|from|outflow)\s+(?:in\s+)?investing\s+activities?|cash\s+flow(?:s)?\s+from\s+investing\s+activities",
        "financing_cash_flow": r"(?:net\s+)?cash\s+(?:used|from|inflow|outflow)\s+(?:in\s+)?financing\s+activities?|cash\s+flow(?:s)?\s+from\s+financing\s+activities",
        "fx_translation_adjustment": r"(?:foreign\s+exchange|fx|currency|translation)\s+(?:effect|adjustment|difference)",
        "net_change_in_cash": r"net\s+(?:increase|decrease|change)\s+in\s+(?:cash|cash\s+and\s+cash\s+equivalents)",
        "opening_cash": r"(?:cash|cash\s+and\s+cash\s+equivalents)\s+at\s+(?:the\s+)?beginning(?:\s+of\s+(?:the\s+)?year)?|(?:opening|beginning)\s+cash(?:\s+and\s+cash\s+equivalents)?",
        "closing_cash": r"(?:cash|cash\s+and\s+cash\s+equivalents)\s+at\s+(?:the\s+)?(?:end|close)(?:\s+of\s+(?:the\s+)?year)?|(?:closing|ending)\s+cash(?:\s+and\s+cash\s+equivalents)?",
        "cash_acquired_on_amalgamation": r"cash\s+acquired\s+on\s+amalgamation|cash\s+acquired\s+through\s+amalgamation",
        "other_adjustments": r"other\s+(?:applicable\s+)?adjustments?|other\s+cash\s+adjustments?",
    }
    for i, line in enumerate(lines):
        for name, pat in labels.items():
            if name in fields or not re.search(pat, line, re.I):
                continue
            nums = _number_tokens(line)
            source = line
            if not nums:
                vals, extra = _collect_following_numbers(lines, i, 4)
                nums = [str(v) for v in vals]
                source = (line + " " + extra).strip()
            if nums:
                value = _parse_number(nums[0])
                fields[name] = _field(value, 1, source[:180])
                if len(nums) >= 2:
                    fields[f"{name}__comparative"] = _field(_parse_number(nums[1]), 1, source[:180])
    return fields, []

def _statement_fallback(pages_text: list[str], document_type: str) -> dict:
    """Extract common financial statements from OCR without relying on prompt text."""
    fields: dict[str, Any] = {}
    line_items: list[dict[str, Any]] = []
    text = "\n".join(pages_text)

    if document_type == "balance_sheet":
        labels = {
            "capital": r"capital\b",
            "reserves_and_surplus": r"reserves\s+and\s+surplus",
            "minority_interest": r"minority\s+interest",
            "deposits": r"deposits\b",
            "borrowings": r"borrowings\b",
            "other_liabilities_and_provisions": r"other\s+liabilities\s+and\s+provisions",
            "cash_and_balances_with_reserve_bank_of_india": r"cash\s+and\s+balances\s+with\s+reserve\s+bank\s+of\s+india",
            "balances_with_banks_and_money_at_call_and_short_notice": r"balances\s+with\s+banks\s+and\s+money\s+at\s+call\s+and\s+short\s+notice",
            "investments": r"investments\b",
            "advances": r"advances\b",
            "fixed_assets": r"fixed\s+assets\b",
            "other_assets": r"other\s+assets\b",
            "contingent_liabilities": r"contingent\s+liabilities\b",
            "bills_for_collection": r"bills\s+for\s+collection\b",
        }
        periods = re.search(r"As\s+at\s+(\d{1,2}[-/]\w+[-/]\d{2,4})\s+(?:As\s+at\s+)?(\d{1,2}[-/]\w+[-/]\d{2,4})", text, re.I)
        if not periods:
            periods = re.search(r"As\s+at\s+(\d{1,2}-[A-Za-z]{3}-\d{2})\s+As\s+at\s+(\d{1,2}-[A-Za-z]{3}-\d{2})", text, re.I)
        period_values = list(periods.groups()) if periods else []
        if period_values:
            fields["statement_periods"] = _field(period_values, 1, periods.group(0))
            fields["statement_date"] = _field(period_values[0], 1, period_values[0])
        m = re.search(r"Consolidated\s+Balance\s+Sheet", text, re.I)
        if m:
            fields["statement_name"] = _field("Consolidated Balance Sheet", 1, m.group(0))
        m = re.search(r"(?:Z|₹|Rs|INR|\$|%)?\s*in\s*[‘'`]?(\d+)\b", text, re.I)
        if m:
            fields["unit_scale"] = _field(m.group(1), 1, m.group(0))

        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
        i = 0
        while i < len(lines):
            line = lines[i]
            matched = None
            for name, pat in labels.items():
                if re.fullmatch(rf"(?:[‘'`]?\s*)?{pat}(?:\s+[A-Z0-9&]+)?", line, re.I) or re.search(pat, line, re.I):
                    matched = name
                    break
            if matched:
                nums = _number_tokens(line)
                source = line
                # In PSM 11 OCR, label and values are separate lines.
                j = i + 1
                while (len(nums) < 2 or (len(nums) >= 1 and (_parse_number(nums[0]) or 0) <= 100 and len(nums) < 3)) and j < min(i + 6, len(lines)):
                    candidate = lines[j]
                    n2 = _number_tokens(candidate)
                    if n2:
                        nums.extend(n2)
                        source += " " + candidate
                    j += 1
                # Remove schedule numbers such as "10" when a pair of larger values follows.
                parsed = [_parse_number(x) for x in nums]
                parsed = [x for x in parsed if x is not None]
                if len(parsed) >= 2:
                    current, comparative = parsed[-2], parsed[-1]
                    fields[matched] = _field(current, 1, source[:180])
                    if period_values:
                        fields[f"{matched}__comparative"] = _field(comparative, 1, source[:180])
                    line_items.append({"name": matched, "source_text": source[:180], "page_number": 1,
                                       period_values[0] if period_values else "current": current,
                                       period_values[1] if len(period_values) > 1 else "comparative": comparative})
                    i = max(i, j - 1)
            i += 1

        # Totals are safest to extract from explicit Total rows. In OCR, the label
        # and its two values are sometimes placed on the next line(s).
        total_hits = []
        for idx, ln in enumerate(lines):
            if re.match(r"^Total\b", ln, re.I):
                source = ln
                nums = _number_tokens(ln)
                j = idx + 1
                while len(nums) < 2 and j < min(idx + 4, len(lines)):
                    nums.extend(_number_tokens(lines[j]))
                    source += " " + lines[j]
                    j += 1
                if len(nums) >= 2:
                    total_hits.append(([_parse_number(x) for x in nums if _parse_number(x) is not None][-2:], source))
        if total_hits:
            # First Total belongs to Capital & Liabilities; second Total belongs to Assets.
            for total_index, (vals, source) in enumerate(total_hits[:2]):
                a, b = vals[-2], vals[-1]
                name = "total_capital_and_liabilities" if total_index == 0 else "total_assets"
                fields[name] = _field(a, 1, source[:180])
                if b is not None:
                    fields[f"{name}__comparative"] = _field(b, 1, source[:180])
                line_items.append({"name": name, "source_text": source[:180], "page_number": 1,
                                   period_values[0] if period_values else "current": a,
                                   period_values[1] if len(period_values) > 1 else "comparative": b})
            # Some OCR layouts collapse both totals into one logical row; duplicate only when needed.
            if "total_capital_and_liabilities" in fields and "total_assets" not in fields:
                fields["total_assets"] = fields["total_capital_and_liabilities"].copy()
                if "total_capital_and_liabilities__comparative" in fields:
                    fields["total_assets__comparative"] = fields["total_capital_and_liabilities__comparative"].copy()

    elif document_type == "invoice":
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
        inv_fields, inv_items = _extract_invoice_fallback(lines)
        fields.update(inv_fields)
        line_items.extend(inv_items)

    elif document_type in {"profit_and_loss", "cash_flow_statement"}:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]
        if document_type == "cash_flow_statement":
            cf_fields, cf_items = _extract_cashflow_fallback(lines)
            fields.update(cf_fields)
            line_items.extend(cf_items)
        else:
            aliases = {
                "revenue": r"(?:revenue|total\s+income|sales)\b",
                "cost_of_sales": r"(?:cost\s+of\s+(?:sales|goods\s+sold)|cogs)\b",
                "gross_profit": r"gross\s+profit\b",
                "operating_expenses": r"operating\s+expenses?\b",
                "operating_profit": r"operating\s+(?:profit|income)\b",
                "tax": r"(?:income\s+)?tax(?:\s+expense)?\b",
                "net_profit": r"(?:net\s+profit|net\s+income)\b",
                "interest_earned": r"interest\s+earned\b",
                "other_income": r"other\s+income\b",
                "total_income": r"total\s+income\b",
                "interest_expended": r"interest\s+expended\b",
                "provisions_and_contingencies": r"provisions?\s+and\s+contingencies\b",
                "total_expenditure": r"total\s+expenditure\b",
                "consolidated_net_profit_before_minority_interest": r"consolidated\s+net\s+profit\s+before\s+minority\s+interest\b",
                "minority_interest": r"minority\s+interest\b",
                "consolidated_net_profit_attributable_to_group": r"consolidated\s+net\s+profit\s+attributable\s+to\s+(?:the\s+)?group\b",
                "current_profit": r"current\s+profit\b",
                "brought_forward_profit": r"brought\s+forward\s+profit\b",
                "total_available_for_appropriation": r"total\s+available\s+for\s+appropriation\b",
            }
            for i, line in enumerate(lines):
                for name, pat in aliases.items():
                    if name in fields or not re.search(pat, line, re.I):
                        continue
                    nums = _number_tokens(line)
                    source = line
                    if not nums:
                        vals, extra = _collect_following_numbers(lines, i, 4)
                        nums = [str(v) for v in vals]
                        source = (line + " " + extra).strip()
                    if nums:
                        fields[name] = _field(_parse_number(nums[0]), 1, source[:180])
                        if len(nums) >= 2:
                            fields[f"{name}__comparative"] = _field(_parse_number(nums[1]), 1, source[:180])

    return {"fields": fields, "line_items": line_items}


def _fallback_regex_extraction(pages_text: list[str], document_type: str) -> dict:
    return _statement_fallback(pages_text, document_type)


def _get_llm_client() -> tuple[Any, str]:
    provider = (settings.llm_provider or "huggingface").lower()
    if OpenAI is None:
        return None, settings.huggingface_model or "deepseek-ai/DeepSeek-R1"
    if provider == "huggingface":
        api_key = settings.huggingface_api_key
        if not api_key:
            logger.warning("HUGGINGFACE_API_KEY missing; using deterministic extraction fallback.")
            return None, settings.huggingface_model or "deepseek-ai/DeepSeek-R1"
        client = OpenAI(base_url=settings.huggingface_base_url or "https://router.huggingface.co/hf-inference/v1", api_key=api_key, timeout=25.0, max_retries=0)
        return client, settings.huggingface_model or "deepseek-ai/DeepSeek-R1"
    if provider == "openai":
        api_key = settings.openai_api_key
        if not api_key:
            return None, settings.openai_model or "gpt-4o-mini"
        return OpenAI(api_key=api_key, timeout=25.0, max_retries=0), settings.openai_model or "gpt-4o-mini"
    return None, "fallback"


def _call_llm(system_prompt: str, user_prompt: str, pages_text: list[str], document_type: str) -> dict:
    client, model_name = _get_llm_client()
    if client is None:
        return _fallback_regex_extraction(pages_text, document_type)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0,
    }
    if (settings.llm_provider or "").lower() == "openai":
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
        cleaned = _clean_json_response(response.choices[0].message.content or "")
        return json.loads(cleaned)
    except Exception as exc:
        logger.warning("LLM extraction failed; falling back to OCR parser: %s", exc)
        return _fallback_regex_extraction(pages_text, document_type)


MIN_FIELDS = {
    "invoice": ["invoice_number", "invoice_date", "vendor_name", "customer_name", "currency", "subtotal", "tax_amount", "discount", "total_amount"],
    "balance_sheet": ["total_assets", "total_liabilities", "total_equity"],
    "profit_and_loss": ["revenue", "cost_of_sales", "gross_profit", "operating_expenses", "operating_profit", "tax", "net_profit"],
    "cash_flow_statement": ["operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "opening_cash", "net_change_in_cash", "closing_cash"],
}

SYSTEM_PROMPT = """You are a financial document extraction engine. Extract ALL meaningful visible fields and line items from the supplied OCR text. Missing/unreadable values must be null. Never invent values. For every extracted field provide page_number and verbatim source_text evidence. Return JSON only."""


def _build_user_prompt(document_type: str, pages_text: list[str], min_fields: list[str]) -> str:
    numbered_pages = "\n\n".join(f"--- PAGE {i + 1} ---\n{text}" for i, text in enumerate(pages_text))
    return f"""Document type: {document_type}\nMinimum required fields: {min_fields}\n\nDocument text:\n{numbered_pages}\n\nReturn {{\"fields\": {{}}, \"line_items\": []}} with all visible fields and evidence."""


def extract_fields(pages_text: list[str], document_type: str) -> dict[str, Any]:
    min_fields = MIN_FIELDS.get(document_type, [])

    # Financial statements are highly structured tables. Prefer the deterministic
    # parser when it already recovered the key totals; this avoids unnecessary
    # remote LLM latency/timeouts for scanned balance sheets. The LLM remains
    # available for incomplete/ambiguous extraction and for less-structured docs.
    fallback = _fallback_regex_extraction(pages_text, document_type)
    fallback_fields = fallback.get("fields", {})
    extracted_count = sum(1 for v in fallback_fields.values() if isinstance(v, dict) and v.get("value") is not None)
    # Fast deterministic OCR extraction is the default for every document type.
    # Remote LLM is opt-in and only used when explicitly enabled and the OCR
    # parser recovered too few fields. This prevents 30-120 second provider
    # outages from blocking otherwise usable documents.
    use_llm = bool(settings.use_llm_fallback)
    threshold = 2 if document_type in {"invoice", "cash_flow_statement"} else 3
    if use_llm and extracted_count < threshold:
        user_prompt = _build_user_prompt(document_type, pages_text, min_fields)
        llm_output = _call_llm(SYSTEM_PROMPT, user_prompt, pages_text, document_type)
    else:
        llm_output = fallback
    extracted_data: dict[str, Any] = {}
    for field_name, field_info in (llm_output.get("fields", {}) or {}).items():
        if not isinstance(field_info, dict):
            continue
        page = field_info.get("page_number")
        source = field_info.get("source_text")
        extracted_data[field_name] = {"value": field_info.get("value"), "page_number": page,
                                      "evidence": {"source_text": source, "page_number": page}}
    for field_name in min_fields:
        extracted_data.setdefault(field_name, {"value": None, "page_number": None,
                                               "evidence": {"source_text": None, "page_number": None}})
    # Line items are meaningful for invoices. Financial-statement rows are
    # already represented as named extracted fields and must not be pushed into
    # the generic invoice-style line-items table. Also discard empty placeholder
    # rows sometimes returned by an LLM.
    raw_line_items = llm_output.get("line_items", []) or []
    line_items = []
    for item in raw_line_items:
        if not isinstance(item, dict):
            continue
        if document_type != "invoice":
            continue
        has_value = any(
            item.get(k) not in (None, "", "-")
            for k in ("description", "quantity", "unit_price", "amount")
        )
        if has_value:
            line_items.append(item)
    if line_items:
        extracted_data["line_items"] = line_items
    return extracted_data
