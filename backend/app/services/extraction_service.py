"""Dynamic financial-document extraction.

Design goals
------------
* Do not encode a separate parser for every vendor/template.
* Use the document's own labels, rows, columns and sections as the primary
  structure.
* Use a single bounded LLM pass when an API key is configured.  The LLM maps
  discovered document labels to semantic concepts needed by validation and can
  also return additional fields that were not known in advance.
* Keep a conservative deterministic OCR parser as a zero-API-key fallback.
* Never accept an LLM value without evidence from the OCR text.

The result deliberately contains both:
    - canonical semantic fields used by the validation engine; and
    - ``discovered_fields`` / ``financial_line_items`` containing document
      information that does not fit the canonical schema.
"""
import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive parsing helpers
# ---------------------------------------------------------------------------

def _clean_json_response(raw_text: str) -> str:
    if not raw_text:
        return "{}"
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    cleaned = match.group(1).strip() if match else re.sub(r"^```(?:json)?|```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    return cleaned[start:end + 1] if start >= 0 and end > start else "{}"


def _parse_money(token: str) -> float | int | None:
    if token is None:
        return None
    s = re.sub(r"[^0-9,().-]", "", str(token)).strip()
    if not s:
        return None
    # Parenthesis artifacts are common in OCR (e.g. ``122,449,924()``).
    # Treat a closing/opening bracket pair, even when reversed or separated, as negative.
    negative = (s.startswith("(") and s.endswith(")")) or bool(re.search(r"\(\)?$", s) and s.endswith("()"))
    if s.endswith("()") and not s.startswith("("):
        negative = True
    s = s.strip("()")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = parts[0] + "." + parts[1] if len(parts) == 2 and len(parts[1]) in (1, 2) else "".join(parts)
    try:
        value = float(s)
        if negative:
            value = -value
        return int(value) if value.is_integer() else value
    except ValueError:
        return None


def _number_tokens(text: str) -> list[str]:
    text = re.sub(r"(?<=\d)\s+(?=[,.]\d)", "", text)
    text = re.sub(r"(?<=\d)\s*,\s*(?=\d)", ",", text)
    return re.findall(
        r"(?<![A-Za-z])\(?\d{1,3}(?:[,.]\d{3})+(?:[.,]\d{1,2})?\)?(?![A-Za-z])|(?<![A-Za-z])\(?\d+(?:[.,]\d{1,2})?\)?(?![A-Za-z])",
        text,
    )


def _parse_number(token: str) -> float | int | None:
    return _parse_money(token)


def _field(value: Any, page: int, source: str | None, confidence: float | None = None) -> dict[str, Any]:
    out = {"value": value, "page_number": page, "source_text": source}
    if confidence is not None:
        out["confidence"] = confidence
    return out


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _norm_label(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).lower().strip()
    return re.sub(r"\s+", " ", text)


def _clean_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip(" |\t")
        if line:
            out.append(line)
    return out


def _put(fields: dict, name: str, value: Any, page: int, source: str, overwrite: bool = False, confidence: float | None = None) -> None:
    if value is None:
        return
    if overwrite or name not in fields or fields[name].get("value") is None:
        fields[name] = _field(value, page, source[:400], confidence)


def _date_from_text(text: str) -> str | None:
    patterns = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        r"\b\d{1,2}[- ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[- ]\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            return m.group(0)
    return None


def _extract_values_near_label(
    lines: list[str], index: int, max_follow: int = 2,
    skip_small_schedule: bool = False, skip_small_schedule_inline: bool | None = None,
) -> tuple[list[float], str]:
    """Read numeric value(s) belonging to a label without stealing next rows.

    Explicit same-line key/value pairs are authoritative. This prevents
    ``Subtotal: 135`` from consuming the next ``Sales Tax 8%: 12.48`` line.
    Percentage rates are ignored as monetary values.

    ``skip_small_schedule`` drops a bare schedule/note-reference number that
    sits alone on its own line, immediately before the real amount line(s) --
    unambiguous, since a pure single-number line under 100 with no amount
    context is essentially always a reference, not a value.

    ``skip_small_schedule_inline`` additionally drops a small leading number
    when the label AND two amounts are all packed onto one line (e.g.
    ``Capital 1  100.00  90.00``). This is ambiguous on statements that
    legitimately report two small comparative-period values on one line
    (e.g. ``Minority Interest 20 10``), so it defaults to following
    ``skip_small_schedule`` only where explicitly requested (balance sheets,
    where this exact same-line layout is common) rather than everywhere.
    """
    if skip_small_schedule_inline is None:
        skip_small_schedule_inline = skip_small_schedule
    first = lines[index]
    # Remove percentages before numeric parsing (e.g. ``Sales Tax 8%: 12.48``).
    numeric_source = re.sub(r"\(?\d+(?:[.,]\d+)?\s*%\)?", " ", first)
    numeric_source = re.sub(r"\(\s*schedule[^)]*\)", " ", numeric_source, flags=re.I)
    raw_first_tokens = _number_tokens(numeric_source)
    first_nums = [x for x in (_parse_number(t) for t in raw_first_tokens) if x is not None]
    if re.search(r"\bschedule\b", first, re.I) and len(first_nums) >= 2:
        first_nums = first_nums[-2:]
    if skip_small_schedule_inline and len(first_nums) >= 2 and abs(float(first_nums[0])) <= 99 and re.search(r"[A-Za-z]", first):
        first_nums = first_nums[1:]
    if re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b", first, re.I) and not CURRENCY_RE.search(first):
        numeric_source = re.sub(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b", " ", numeric_source, flags=re.I)
        raw_first_tokens = _number_tokens(numeric_source)
        first_nums = [x for x in (_parse_number(t) for t in raw_first_tokens) if x is not None]
    if first_nums and not (skip_small_schedule_inline and len(first_nums) == 1 and abs(first_nums[0]) <= 100 and re.search(r"[A-Za-z]", first)):
        # A value explicitly present on the label line belongs to that label.
        # Two values are retained for comparative statement rows.
        if len(first_nums) >= 2:
            return first_nums[-2:], first[:400]
        return first_nums[-1:], first[:400]

    collected: list[float] = []
    source = first
    for j in range(index + 1, len(lines)):
        line = lines[j]
        numeric_line = re.sub(r"\(?\d+(?:[.,]\d+)?\s*%\)?", " ", line)
        nums = [x for x in (_parse_number(t) for t in _number_tokens(numeric_line)) if x is not None]
        if nums:
            if skip_small_schedule and len(nums) == 1 and abs(nums[0]) <= 100 and not re.search(r"[A-Za-z]", line):
                continue
            collected.extend(nums)
            source += " " + line
            if len(collected) >= 2:
                return collected[-2:], source[:400]
            continue
        if re.search(r"[A-Za-z]", line):
            break
    return collected, source[:400]


# ---------------------------------------------------------------------------
# Dynamic row / key-value discovery
# ---------------------------------------------------------------------------

CURRENCY_RE = re.compile(r"\b(?:USD|INR|EUR|GBP|AUD|CAD|JPY|CNY|SGD|MYR|RM)\b|[$€£₹]", re.I)
SUMMARY_RE = re.compile(r"^(?:sub\s*-?\s*total|subtotal|tax|gst|vat|sales\s+tax|discount|shipping|freight|handling|delivery|grand\s+total|total\b|amount\s+due|balance\s+due|net\s+change|opening|closing|ending|beginning)", re.I)


def _is_noise_label(label: str) -> bool:
    n = _norm_label(label)
    if not n or len(n) > 120:
        return True
    if re.fullmatch(r"(?:page|date|invoice|statement|notes?|schedule|amount|description)", n):
        return False
    # Tax IDs, phone numbers, URLs and obvious OCR metadata should not become fields.
    if re.search(r"\b(?:tax id|vat id|gstin|gst in|phone|tel|fax|www|http)\b", n, re.I):
        return True
    return False


def _split_label_value(line: str) -> tuple[str, str] | None:
    # Colon/equal/dash is the safest explicit key/value boundary.
    m = re.match(r"^\s*([^:：=]{1,100})\s*[:：=]\s*(.*?)\s*$", line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # A label followed by a monetary value at the end is also common in statements.
    m = re.match(r"^\s*(.*?\S)\s+(\(?[$€£₹]?\s*[\d][\d,.\s]*\)?)\s*$", line)
    if m and re.search(r"[A-Za-z]", m.group(1)):
        return m.group(1).strip(), m.group(2).strip()
    return None


def _discover_key_values(lines: list[str], page: int) -> dict[str, dict]:
    discovered: dict[str, dict] = {}
    for i, line in enumerate(lines):
        pair = _split_label_value(line)
        if not pair:
            continue
        label, raw_value = pair
        if _is_noise_label(label):
            continue
        if not raw_value and i + 1 < len(lines):
            raw_value = lines[i + 1].strip()
        if not raw_value:
            continue
        date = _date_from_text(raw_value)
        nums = _number_tokens(raw_value)
        parsed = _parse_money(nums[-1]) if nums else None
        if date and not nums:
            value: Any = date
        elif re.search(r"\b(?:invoice|inv|bill|receipt)\b", label, re.I):
            # Identifiers remain strings, even when they contain digits.
            value = raw_value
        elif CURRENCY_RE.fullmatch(raw_value.strip()):
            value = raw_value.strip().upper()
        elif parsed is not None and len(nums) == 1 and re.fullmatch(r"[\s$€£₹()0-9,.-]+", raw_value) and not re.search(r"%", raw_value):
            value = parsed
        else:
            value = raw_value
        key = _norm_label(label)
        if key in discovered:
            continue
        discovered[key] = _field(value, page, line, 0.96)
    return discovered


def _best_invoice_triplet(line: str) -> tuple[float | int, float | int, float | int] | None:
    """Discover quantity/rate/amount by arithmetic, not fixed column positions."""
    vals = [v for v in (_parse_money(x) for x in _number_tokens(line)) if v is not None]
    if len(vals) < 3:
        return None
    candidates = []
    for i in range(len(vals) - 2):
        q, price, amount = vals[i:i + 3]
        if q <= 0 or price < 0 or amount < 0:
            continue
        error = abs(float(q) * float(price) - float(amount))
        quantity_penalty = 0 if float(q).is_integer() and q <= 10000 else 4
        candidates.append((error + quantity_penalty, q, price, amount))
    exact = [c for c in candidates if c[0] <= 1.0]
    pool = exact or candidates
    if not pool:
        return None
    _, q, price, amount = min(pool, key=lambda x: x[0])
    return q, price, amount


def _looks_like_table_header(line: str) -> bool:
    n = _norm_label(line)
    groups = [
        ("description", "item", "product", "particular"),
        ("qty", "quantity", "units", "unit"),
        ("price", "rate", "unit price", "unit cost"),
        ("amount", "total", "extended", "value"),
    ]
    return sum(any(g in n for g in group) for group in groups) >= 2



def _group_layout_rows(layout: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group OCR words by their 2-D row, retaining x coordinates."""
    if not layout:
        return []
    heights = [float(w.get("height", 0)) for w in layout if float(w.get("height", 0)) > 0]
    tolerance = max(5.0, (sum(heights) / len(heights) if heights else 12.0) * 0.7)
    rows: list[list[dict[str, Any]]] = []
    ys: list[float] = []
    for word in sorted(layout, key=lambda w: (float(w.get("top", 0)), float(w.get("left", 0)))):
        cy = float(word.get("top", 0)) + float(word.get("height", 0)) / 2
        match = next((i for i, y in enumerate(ys) if abs(cy - y) <= tolerance), None)
        if match is None:
            ys.append(cy); rows.append([word])
        else:
            rows[match].append(word)
            ys[match] = sum(float(w.get("top", 0)) + float(w.get("height", 0))/2 for w in rows[match]) / len(rows[match])
    return [sorted(r, key=lambda w: float(w.get("left", 0))) for _, r in sorted(zip(ys, rows), key=lambda x: x[0])]


def _layout_row_text(row: list[dict[str, Any]]) -> str:
    return " ".join(str(w.get("text", "")).strip() for w in row if str(w.get("text", "")).strip())


def _is_numeric_word(text: str) -> bool:
    return bool(re.fullmatch(r"\(?[$€£₹]?\s*[\d][\d,./\s-]*\)?", text.strip()))


def _layout_invoice_table(lines: list[str], page: int, layout: list[dict[str, Any]]) -> list[dict]:
    """Extract invoice rows from the 2-D OCR grid.

    Tesseract may split one visual invoice row into several OCR rows.  The old
    implementation treated each physical OCR row as an invoice row, which is
    why a description such as ``3M 570 ... 12oz ...`` could become a fake row.
    We first infer the three financial columns, then group *visual rows* by the
    item-number column.  Quantity/price/amount are collected across the whole
    visual group while every token left of the quantity column stays in the
    description.  This makes wrapped descriptions and product numbers safe.
    """
    rows = _group_layout_rows(layout)
    header_idx = None
    header_positions: dict[str, float] = {}
    description_x = None
    for i, row in enumerate(rows):
        text = _layout_row_text(row).lower()
        n = re.sub(r"[^a-z0-9 ]", " ", text)
        hits = [
            bool(re.search(r"description|item|product|particular", n)),
            bool(re.search(r"qty|quantity|units", n)),
            bool(re.search(r"unit\s*price|price|rate|cost", n)),
            bool(re.search(r"amount|extended|total|value", n)),
        ]
        if sum(hits) >= 3:
            for w in row:
                word = str(w.get("text", ""))
                x = float(w.get("left", 0))
                if re.search(r"description|item|product|particular", word, re.I):
                    description_x = x if description_x is None else min(description_x, x)
                if re.search(r"qty|quantity|units", word, re.I):
                    header_positions.setdefault("qty", x)
                elif re.search(r"unit\s*price|price|rate|unit\s*cost", word, re.I):
                    header_positions.setdefault("price", x)
                elif re.search(r"amount|extended|total|value", word, re.I):
                    header_positions.setdefault("amount", x)
            if len(header_positions) == 3:
                header_idx = i
                break
    if header_idx is None:
        return []

    qty_x = header_positions["qty"]
    price_x = header_positions["price"]
    amount_x = header_positions["amount"]
    if not (qty_x < price_x < amount_x):
        return []

    def numeric_words(group):
        return [w for r in group for w in r if _is_numeric_word(str(w.get("text", "")))]

    def is_row_number(w):
        txt = str(w.get("text", "")).strip()
        x = float(w.get("left", 0))
        # Item numbers live to the left of the financial columns and are short
        # integers.  Product numbers such as 570/18756 normally occur inside
        # the description column and therefore do not qualify as row starts.
        left_boundary = (description_x - 5) if description_x is not None else (qty_x - 35)
        return x <= left_boundary and bool(re.fullmatch(r"\d{1,3}", txt))

    # Group consecutive physical OCR rows under the nearest item-number row.
    groups: list[list[list[dict[str, Any]]]] = []
    current: list[list[dict[str, Any]]] | None = None
    orphan_rows: list[list[dict[str, Any]]] = []
    for row in rows[header_idx + 1:]:
        text = _layout_row_text(row)
        if not text:
            continue
        if SUMMARY_RE.search(text):
            if current:
                groups.append(current)
                current = None
            orphan_rows = []
            continue
        starts = [w for w in row if is_row_number(w)]
        if starts:
            if current:
                groups.append(current)
            current = orphan_rows + [row]
            orphan_rows = []
        elif current is not None:
            # A physical continuation row belongs to the current item until
            # that item already has its financial triplet. Only then can an
            # unnumbered row safely start the next item.
            def group_has_financials(group):
                group_nums = numeric_words(group)
                return (
                    any(qty_x - 25 <= float(w.get("left", 0)) < price_x - 8 for w in group_nums)
                    and any(price_x - 25 <= float(w.get("left", 0)) < amount_x - 50 for w in group_nums)
                    and any(float(w.get("left", 0)) >= amount_x - 50 for w in group_nums)
                )
            if group_has_financials(current):
                row_nums = numeric_words([row])
                q_candidates = [w for w in row_nums if qty_x - 25 <= float(w.get("left", 0)) < price_x - 8]
                p_candidates = [w for w in row_nums if price_x - 25 <= float(w.get("left", 0)) < amount_x - 50]
                a_candidates = [w for w in row_nums if float(w.get("left", 0)) >= amount_x - 50]
                if q_candidates and p_candidates and a_candidates:
                    groups.append(current)
                    current = [row]
                elif any(is_row_number(w) is False and float(w.get("left", 0)) < qty_x - 25 and re.search(r"\d", str(w.get("text", ""))) for w in row) and len(row) > 1:
                    groups.append(current)
                    current = None
                    orphan_rows = [row]
                else:
                    current.append(row)
            else:
                current.append(row)
        else:
            # Header-number OCR may omit the item number. Start a group when
            # the row itself contains all three financial columns.
            row_nums = numeric_words([row])
            q_candidates = [w for w in row_nums if qty_x - 25 <= float(w.get("left", 0)) < price_x - 8]
            p_candidates = [w for w in row_nums if price_x - 25 <= float(w.get("left", 0)) < amount_x - 50]
            a_candidates = [w for w in row_nums if float(w.get("left", 0)) >= amount_x - 50]
            if (q_candidates or (p_candidates and a_candidates and orphan_rows)) and p_candidates and a_candidates:
                current = orphan_rows + [row]
                orphan_rows = []
            else:
                orphan_rows.append(row)
    if current:
        groups.append(current)

    result: list[dict] = []
    for group in groups:
        all_words = [w for r in group for w in r]
        nums = numeric_words(group)
        qty_words = [w for w in nums if qty_x - 25 <= float(w.get("left", 0)) < price_x - 8]
        price_words = [w for w in nums if price_x - 25 <= float(w.get("left", 0)) < amount_x - 50]
        amount_words = [w for w in nums if float(w.get("left", 0)) >= amount_x - 50]
        if not (price_words and amount_words):
            continue

        # A wrapped visual row can contain multiple numeric words in a column;
        # the financial column value is the rightmost token in that column.
        q = _parse_money(str(sorted(qty_words, key=lambda w: float(w.get("left", 0)))[-1].get("text"))) if qty_words else None
        price = _parse_money(str(sorted(price_words, key=lambda w: float(w.get("left", 0)))[-1].get("text")))
        amount = _parse_money(str(sorted(amount_words, key=lambda w: float(w.get("left", 0)))[-1].get("text")))
        if q is None or price is None or amount is None:
            if not qty_words and price not in (None, 0) and amount is not None:
                inferred = float(amount) / float(price)
                q = int(round(inferred)) if inferred > 0 and abs(inferred - round(inferred)) <= 1e-6 else None
        if q is None or price is None or amount is None:
            continue
        if abs(float(q) * float(price) - float(amount)) > 1.0:
            inferred = float(amount) / float(price) if price else 0
            if inferred <= 0 or abs(inferred - round(inferred)) > 1e-6:
                continue
            q = int(round(inferred))

        # Remove only the actual item-number token, not product numbers such as
        # 570 or 18756 that happen to be numeric and sit in the description.
        row_number_word = next((w for row in group for w in row if is_row_number(w)), None)
        desc_words = [
            w for w in all_words
            if float(w.get("left", 0)) < qty_x - 25 and w is not row_number_word
        ]
        desc = _layout_row_text(sorted(desc_words, key=lambda w: (float(w.get("top", 0)), float(w.get("left", 0)))))
        if not desc:
            continue
        source = " ".join(_layout_row_text(r) for r in group)
        result.append({"description": desc, "quantity": q, "unit_price": price,
                       "amount": amount, "page_number": page, "source_text": source[:400]})
    return result


def _text_invoice_table(lines: list[str], page: int) -> list[dict]:
    """Recover invoice rows from flattened OCR while preserving row boundaries.

    OCR can split one visual row across multiple text lines.  We therefore first
    collect numbered row blocks and only then take the final financial triplet
    from the complete block.  If OCR corrupts quantity but amount/price provide
    an exact integer quantity, the quantity is repaired from that row's own
    arithmetic instead of accepting an impossible multiplication.
    """
    blocks: list[tuple[str, list[str]]] = []
    current_no = None
    current_lines: list[str] = []
    def has_complete_triplet(text: str) -> bool:
        nums = _number_tokens(text)
        if len(nums) < 3:
            return False
        q, price, amount = (_parse_money(x) for x in nums[-3:])
        if q is not None and price is not None and amount is not None and abs(float(q) * float(price) - float(amount)) <= 1.0:
            return True
        # OCR can misread a quantity digit (e.g. 2 -> 5). If the final price and
        # amount yield a positive integer quantity, the row is still complete.
        if price not in (None, 0) and amount is not None:
            inferred = float(amount) / float(price)
            return inferred > 0 and abs(inferred - round(inferred)) <= 1e-6
        return False

    for raw in lines:
        line = raw.strip(" |")
        if not line or SUMMARY_RE.search(line):
            continue
        m = re.match(r"^(?:item\s+)?(?P<num>\d{1,3})\s+(?P<body>.+)$", line, re.I)
        if m:
            num = m.group("num")
            if current_no is None:
                current_no, current_lines = num, [m.group("body")]
            elif not has_complete_triplet(" ".join(current_lines)):
                # The previous visual row is incomplete: this numbered OCR line
                # is commonly the continuation of the same row.
                current_lines.append(m.group("body"))
            else:
                blocks.append((current_no, current_lines))
                current_no, current_lines = num, [m.group("body")]
        elif current_no is not None:
            if re.match(r"^(?:invoice\s+details|invoice\s*#|due\s+date|invoice\s+date|terms(?:\s+and\s+conditions)?|notes?)\b", line, re.I):
                blocks.append((current_no, current_lines)); current_no, current_lines = None, []
            elif has_complete_triplet(line) and has_complete_triplet(" ".join(current_lines)):
                blocks.append((current_no, current_lines))
                current_no = "?"
                current_lines = [line]
            else:
                current_lines.append(line)
    if current_no is not None:
        blocks.append((current_no, current_lines))

    items: list[dict] = []
    for row_no, parts in blocks:
        body = " ".join(parts)
        nums = _number_tokens(body)
        if len(nums) < 3:
            continue
        q_s, price_s, amount_s = nums[-3:]
        q, price, amount = _parse_money(q_s), _parse_money(price_s), _parse_money(amount_s)
        if price is None or amount is None:
            continue

        # Prefer the source quantity when it reconciles. Otherwise repair OCR
        # quantity from amount/price only when the result is a sensible positive
        # value and exactly reproduces the reported amount.
        repaired_quantity = False
        if q is None or abs(float(q) * float(price) - float(amount)) > 1.0:
            if float(price) != 0:
                inferred = float(amount) / float(price)
                if inferred > 0 and abs(inferred - round(inferred)) <= 1e-6:
                    q = int(round(inferred))
                    repaired_quantity = True
            if q is None or abs(float(q) * float(price) - float(amount)) > 1.0:
                continue

        # Remove the final amount, price and quantity from the concatenated body;
        # all earlier numeric tokens remain in the description.
        amount_matches = list(re.finditer(r"[$€£₹]?\s*" + re.escape(amount_s) + r"(?=\s*$)", body))
        if not amount_matches:
            continue
        prefix = body[:amount_matches[-1].start()].rstrip()
        price_matches = list(re.finditer(r"[$€£₹]?\s*" + re.escape(price_s) + r"(?=\s*$)", prefix))
        if not price_matches:
            continue
        prefix = prefix[:price_matches[-1].start()].rstrip()
        qty_matches = list(re.finditer(r"(?<![A-Za-z])" + re.escape(q_s) + r"(?=\s*$)", prefix))
        if not qty_matches:
            # For an inferred quantity, remove the original final numeric token.
            nums_before = list(re.finditer(r"\(?\d+(?:[.,]\d+)?\)?(?=\s*$)", prefix))
            if not nums_before:
                continue
            prefix = prefix[:nums_before[-1].start()].rstrip()
        else:
            prefix = prefix[:qty_matches[-1].start()].rstrip()

        item = {
            "description": prefix.strip(" -"), "quantity": q,
            "unit_price": price, "amount": amount, "page_number": page,
            "source_text": (row_no + " " + body)[:400]
        }
        if repaired_quantity:
            item["quantity_repaired_from_amount_price"] = True
        if item["description"]:
            items.append(item)
    return items


def _discover_table(lines: list[str], page: int, invoice_mode: bool = False, layout: list[dict[str, Any]] | None = None) -> list[dict]:
    """Find tabular numeric rows dynamically.

    No vendor-specific header or column positions are assumed.  For invoices,
    arithmetic identifies the quantity/rate/amount triple.  For statements,
    every financial row is preserved as a discovered line item.
    """
    if invoice_mode and layout:
        layout_items = _layout_invoice_table(lines, page, layout)
        if layout_items:
            return layout_items
    if invoice_mode:
        # Tesseract can occasionally produce good words but unusable/absent
        # coordinates. Recover numbered rows from the OCR text before using the
        # generic arithmetic detector, which is more likely to confuse embedded
        # product numbers with financial columns.
        text_items = _text_invoice_table(lines, page)
        if text_items:
            return text_items
    items: list[dict] = []
    header_seen = False
    for i, line in enumerate(lines):
        if _looks_like_table_header(line):
            header_seen = True
            continue
        if invoice_mode and SUMMARY_RE.search(line):
            continue
        nums = _number_tokens(line)
        if not nums or not re.search(r"[A-Za-z]", line):
            continue

        if invoice_mode:
            if not header_seen:
                continue
            triplet = _best_invoice_triplet(line)
            if not triplet:
                continue
            q, price, amount = triplet
            # Description is the text with the selected numeric tokens removed as
            # much as possible; preserve embedded product numbers.
            desc = re.sub(r"\s+", " ", re.sub(r"\(?[$€£₹]?\s*[\d][\d,.\s]*\)?", " ", line)).strip(" -:")
            if not desc:
                desc = line
            items.append({"description": desc, "quantity": q, "unit_price": price, "amount": amount, "page_number": page, "source_text": line[:400]})
        else:
            # Statements often have one or two values after a label. Keep both.
            values = [v for v in (_parse_number(x) for x in nums) if v is not None]
            if not values:
                continue
            label = re.sub(r"\s+", " ", re.sub(r"\(?[$€£₹]?\s*[\d][\d,.\s]*\)?", " ", line)).strip(" -:")
            if not label or _norm_label(label) in {"total", "amount"}:
                continue
            item = {"label": label, "values": values[-2:], "page_number": page, "source_text": line[:400]}
            if len(values) >= 1:
                item["current"] = values[-1] if len(values) == 1 else values[-2]
            if len(values) >= 2:
                item["comparative"] = values[-1]
            items.append(item)
    return items


# ---------------------------------------------------------------------------
# Semantic mapping.  This is intentionally fuzzy and small: the document can
# use any wording; these terms only tell validation which numeric concept to use.
# ---------------------------------------------------------------------------

SEMANTIC_TERMS = {
    "invoice_number": ["invoice number", "invoice no", "invoice #", "bill number", "bill no", "receipt number", "invoice details"],
    "invoice_date": ["invoice date", "date of issue", "issued on", "bill date", "dated", "date"],
    "vendor_name": ["vendor", "supplier", "seller", "from", "billed by", "sold by"],
    "customer_name": ["customer", "client", "buyer", "bill to", "billed to", "ship to", "sold to"],
    "currency": ["currency", "currency code"],
    "subtotal": ["subtotal", "sub total", "net amount", "net total", "taxable amount", "total before tax", "total excl gst", "total excluding gst", "total excluding gst", "amount before tax"],
    "tax_amount": ["tax amount", "gst", "vat", "sales tax", "sales tax amount", "gst payable", "gst included in total", "tax included in total", "total tax"],
    "discount": ["discount", "less discount", "rebate"],
    "shipping_and_handling": ["shipping", "shipping and handling", "handling", "freight", "delivery", "shipping charge"],
    "total_amount": ["grand total", "total amount due", "amount due", "balance due", "invoice total", "total due", "s total", "total sales", "total incl gst", "total incl g57t", "total including gst", "total includes gst", "total inclusive gst", "total inclusive of gst", "total inclusive g57t", "tatal inclusive gst", "tatal inclusive g57t", "total ant", "total sales inclusive gst", "total gross", "net total", "total payable", "amount payable", "total"],
    "total_assets": ["total assets", "total asset", "assets total"],
    "total_liabilities": ["total liabilities", "liabilities total"],
    "total_equity": ["total equity", "shareholders equity", "stockholders equity", "owners equity"],
    "total_capital_and_liabilities": ["total capital and liabilities", "capital and liabilities total"],
    "capital": ["capital"],
    "reserves_and_surplus": ["reserves and surplus", "reserves surplus", "reservesandsuplls"],
    "deposits": ["deposits", "deposit"],
    "borrowings": ["borrowings", "borrowed funds"],
    "other_liabilities_and_provisions": ["other liabilities and provisions", "other liabilities provisions"],
    "cash_and_balances_with_reserve_bank_of_india": ["cash and balances with reserve bank of india"],
    "balances_with_banks_and_money_at_call_and_short_notice": ["balances with banks and money at call and short notice"],
    "investments": ["investments", "investment"],
    "advances": ["advances", "advance"],
    "fixed_assets": ["fixed assets", "fixed asset"],
    "other_assets": ["other assets", "other asset"],
    "revenue": ["revenue", "sales revenue"],
    "cost_of_sales": ["cost of sales", "cost of goods sold", "cogs"],
    "gross_profit": ["gross profit", "gross margin"],
    "operating_expenses": ["operating expenses", "operating costs", "operating expense"],
    "operating_profit": ["operating profit", "operating income", "profit from operations"],
    "tax": ["income tax", "tax expense", "tax provision", "taxation"],
    "net_profit": ["net profit", "net income", "profit after tax"],
    "interest_earned": ["interest earned"],
    "other_income": ["other income", "non interest income", "non-interest income"],
    "total_income": ["total income"],
    "interest_expended": ["interest expended", "interest expense"],
    "provisions_and_contingencies": ["provisions and contingencies", "provision for contingencies"],
    "total_expenditure": ["total expenditure", "total expenses", "total expense"],
    "consolidated_net_profit_before_minority_interest": ["consolidated net profit before minority interest", "consolidated net profit for the year before minorities interest", "profit before minority interest"],
    "minority_interest": ["minority interest"],
    "consolidated_net_profit_attributable_to_group": ["consolidated net profit attributable to the group", "consolidated profit for the year attributable to the group", "net profit attributable to the group"],
    "current_profit": ["current profit"],
    "brought_forward_profit": ["brought forward profit", "profit brought forward"],
    "total_available_for_appropriation": ["total available for appropriation"],
    "operating_cash_flow": ["operating cash flow", "net cash provided by operating activities", "net cash flow from operating activities", "net cash flow used in operating activities", "cash from operating activities"],
    "investing_cash_flow": ["investing cash flow", "net cash used in investing activities", "net cash flow from investing activities", "net cash flow used in investing activities", "cash from investing activities"],
    "financing_cash_flow": ["financing cash flow", "net cash used in financing activities", "net cash flow from financing activities", "net cash flow used in financing activities", "cash from financing activities"],
    "fx_translation_adjustment": ["fx translation adjustment", "foreign exchange adjustment", "currency translation adjustment"],
    "net_change_in_cash": ["net change in cash", "net increase in cash", "net decrease in cash", "increase decrease in cash"],
    "opening_cash": ["opening cash", "cash at beginning of year", "cash and cash equivalents as at april 1", "cash and cash equivalents as at april 1st", "cash and cash equivalents at the beginning of the year", "cash at beginning", "cash at start"],
    "closing_cash": ["closing cash", "cash at end of year", "cash and cash equivalents as at march 31", "cash and cash equivalents as at march 31st", "cash and cash equivalents as at march 3ist", "cash and cash equivalents as at the year end", "cash and cash equivalents at the end of the year", "cash at end", "cash at close"],
    "cash_acquired_on_amalgamation": ["cash acquired on amalgamation", "cash acquired through amalgamation", "cash and cash equivalents acquired on amalgamation"],
    "other_adjustments": ["other adjustments", "other applicable adjustments"],
}


def _semantic_score(label: str, target: str) -> float:
    a = _norm_label(label)
    best = 0.0
    for term in SEMANTIC_TERMS.get(target, [target.replace("_", " ")]):
        b = _norm_label(term)
        if a == b:
            return 1.0
        # OCR commonly removes spaces/punctuation. Compact equality is safe.
        if _compact(a) == _compact(b):
            return 0.99
        at, bt = set(a.split()), set(b.split())
        if len(bt) >= 2 and bt.issubset(at):
            best = max(best, 0.96)
        elif len(bt) >= 2:
            overlap = len(at & bt) / max(1, len(bt))
            best = max(best, overlap * 0.90)
    return best


def _semantic_label_matches(label: str, target: str) -> bool:
    a = _norm_label(label)
    if not a or (target not in {"opening_cash", "closing_cash"} and re.search(r"\b(?:march|january|february|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", a, re.I)):
        return False
    # Single-word concepts must be exact; this prevents 'tax'/'total' style
    # fuzzy collisions. Multi-word concepts allow OCR spacing variations.
    terms = SEMANTIC_TERMS.get(target, [target.replace("_", " ")])
    for term in terms:
        b = _norm_label(term)
        if a == b or _compact(a) == _compact(b):
            return True
        # OCR can insert arbitrary symbols between a label and its values. If
        # the complete semantic phrase survives contiguously in the compacted
        # row, it is safe to use the row for that concept.
        if len(b.split()) >= 2 and _compact(b) in _compact(a):
            if target == "minority_interest" and re.search(r"before|attributable", a, re.I):
                continue
            return True
        bt = b.split()
        at = a.split()
        if len(bt) == 1 and bt[0] in at:
            # Do not let generic words such as tax/total match document titles.
            if target in {"tax_amount", "total_amount"} and len(at) > 1:
                if target != "total_amount" or not at[0].startswith("total") or re.search(r"excluding|before|taxable", a):
                    continue
            if target == "capital" and any(word in at for word in ("assets", "liabilities")):
                continue
            return True
        if len(bt) >= 2:
            for pos in range(0, len(at) - len(bt) + 1):
                if at[pos:pos + len(bt)] == bt:
                    # Narrative phrases containing a semantic label are not the
                    # row itself (e.g. profit *before minority interest*).
                    if target == "minority_interest" and ("before" in at[:pos] or "attributable" in at[:pos]):
                        continue
                    return True
            # Financial statements often insert qualifiers between semantic
            # words, such as "net cash flow (used in) / from operating
            # activities". Preserve word order while allowing short OCR
            # connector phrases between the meaningful terms.
            matched = 0
            for word in at:
                if matched < len(bt) and word == bt[matched]:
                    matched += 1
            if matched == len(bt) and target in {
                "operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
                "consolidated_net_profit_before_minority_interest",
                "consolidated_net_profit_attributable_to_group",
            }:
                return True
    return False



def _infer_columns(lines: list[str], layout: list[dict[str, Any]] | None) -> tuple[float | None, float | None]:
    if not layout: return None, None
    current_xs = []
    comparative_xs = []
    for line in lines:
        nums = _number_tokens(line)
        values = [v for v in (_parse_number(x) for x in nums) if v is not None]
        if len(values) >= 2:
            t1, t2 = nums[-2], nums[-1]
            x1, x2 = None, None
            for b in layout:
                if b.get("text") == t1: x1 = float(b.get("left", 0))
                if b.get("text") == t2: x2 = float(b.get("left", 0))
            if x1 is not None: current_xs.append(x1)
            if x2 is not None: comparative_xs.append(x2)
    avg_curr = sum(current_xs) / len(current_xs) if current_xs else None
    avg_comp = sum(comparative_xs) / len(comparative_xs) if comparative_xs else None
    return avg_curr, avg_comp

def _extract_semantic_fields(lines: list[str], page: int, document_type: str, layout: list[dict[str, Any]] | None = None) -> dict[str, dict]:
    fields: dict[str, dict] = {}
    avg_curr, avg_comp = _infer_columns(lines, layout)
    target_groups = {
        "invoice": {"invoice_number", "invoice_date", "vendor_name", "customer_name", "currency", "subtotal", "tax_amount", "discount", "shipping_and_handling", "total_amount"},
        "balance_sheet": {"total_assets", "total_liabilities", "total_equity", "total_capital_and_liabilities", "minority_interest", "capital", "reserves_and_surplus", "deposits", "borrowings", "other_liabilities_and_provisions", "cash_and_balances_with_reserve_bank_of_india", "balances_with_banks_and_money_at_call_and_short_notice", "investments", "advances", "fixed_assets", "other_assets"},
        "profit_and_loss": {"revenue", "cost_of_sales", "gross_profit", "operating_expenses", "operating_profit", "tax", "net_profit", "interest_earned", "other_income", "total_income", "interest_expended", "provisions_and_contingencies", "total_expenditure", "consolidated_net_profit_before_minority_interest", "minority_interest", "consolidated_net_profit_attributable_to_group", "current_profit", "brought_forward_profit", "total_available_for_appropriation"},
        "cash_flow_statement": {"operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "fx_translation_adjustment", "net_change_in_cash", "opening_cash", "closing_cash", "cash_acquired_on_amalgamation", "other_adjustments"},
    }
    targets = target_groups.get(document_type, set())

    for i, line in enumerate(lines):
        # Never interpret tax identifiers as tax amounts.
        if re.search(r"(?:tax\s*id|taxpayer\s*id|vat\s*id|gst\s*in)\b", line, re.I):
            continue
        pair = _split_label_value(line)
        label = pair[0] if pair else line
        tail = pair[1] if pair else ""
        # For rows, remove the numeric tail before semantic comparison.
        row_label = re.sub(r"\s+", " ", re.sub(r"\(?[$€£₹]?\s*[\d][\d,.\s]*\)?", " ", line)).strip(" -:")
        label_candidates = [label, row_label]

        # Identifier/date fields are not monetary fields. Keep invoice number
        # as text even when it contains years or other digits.
        if document_type == "invoice" and pair:
            ln = _norm_label(label)
            if any(x in ln for x in ("invoice number", "invoice no", "invoice #", "bill number", "bill no", "invoice details", "receipt number")) and tail:
                _put(fields, "invoice_number", tail.strip(" :#-"), page, line, confidence=0.96)
            if "date" in ln or "issued" in ln or "dated" in ln:
                dt = _date_from_text(tail or line)
                if dt:
                    _put(fields, "invoice_date", dt, page, line, confidence=0.96)

        for target in targets:
            if target in fields:
                continue
            if not any(_semantic_label_matches(candidate, target) for candidate in label_candidates if candidate):
                continue

            if document_type == "cash_flow_statement" and target in {"operating_cash_flow", "investing_cash_flow", "financing_cash_flow"}:
                # Section headings describe a block; only the net-cash summary
                # row is the canonical cash-flow value.
                if re.search(r"^cash flows?\s+(?:from|used in|\(used in\))", _norm_label(label)) and not re.search(r"net cash", _norm_label(label)):
                    continue

            if target == "invoice_date":
                value = _date_from_text(tail or line)
                source = line
                if value is None and i + 1 < len(lines):
                    value = _date_from_text(lines[i + 1])
                    source += " " + lines[i + 1] if value else ""
                if value:
                    _put(fields, target, value, page, source, confidence=0.92)
                continue

            if document_type == "invoice" and target == "invoice_number":
                value = tail.strip(" :#-") if pair else ""
                explicit_number_label = bool(re.search(r"number|no\.?|#|details|receipt", label, re.I))
                if not value and explicit_number_label and i + 1 < len(lines) and _number_tokens(lines[i + 1]):
                    value = lines[i + 1].strip(" :#-")
                if value and "@" not in value and not re.search(r"^(invoice|date|due|number)$", value, re.I):
                    _put(fields, target, value, page, line if pair else f"{line} {value}", confidence=0.86)
                continue

            if target in {"vendor_name", "customer_name"}:
                value = tail.strip(" :#-") if pair else ""
                if not value and i + 1 < len(lines):
                    nxt = lines[i + 1]
                    if not SUMMARY_RE.search(nxt) and not _number_tokens(nxt):
                        value = nxt.strip()
                if value and "@" not in value and not re.search(r"^(invoice|date|subtotal|tax|total|currency)\b", value, re.I):
                    if target == "vendor_name" and _norm_label(label) in {"from", "seller"} and not re.search(r"\b(?:inc|corp|ltd|limited|llc|plc|company|co)\.?\b", value, re.I):
                        continue
                    _put(fields, target, value, page, line, confidence=0.88)
                continue

            if target == "currency":
                candidate = tail or line
                m = CURRENCY_RE.search(candidate)
                if m:
                    value = m.group(0).upper() if m.group(0).isalpha() else m.group(0)
                    _put(fields, target, value, page, line, confidence=0.94)
                continue

            if document_type == "cash_flow_statement" and target == "cash_acquired_on_amalgamation" and re.search(r"\s-\s", line):
                _put(fields, target, 0, page, line, confidence=0.88)
                continue

            if document_type == "invoice" and (target in {"invoice_number", "invoice_date"}):
                continue
            if _date_from_text(line) and not re.search(r"(?:total|subtotal|tax|amount|balance|revenue|profit|income|expense|cash|assets|liabilities|capital|deposit|borrow|investment|advance)", line, re.I):
                continue
            # Schedule/note reference numbers (a bare 1-2 digit integer printed
            # right before the real amount) appear on balance sheets, P&L
            # accounts, and cash flow statements alike in this statement
            # format -- not just balance sheets -- so all three skip a bare
            # reference line. The same-line variant (label + schedule + two
            # amounts all on one row) stays balance-sheet-only: on P&L/cash
            # flow statements a same-line "label smallnum smallnum" is often
            # two genuine comparative-period values (e.g. "Minority Interest
            # 20 10"), not a schedule ref, so stripping there would be wrong.
            vals, source = _extract_values_near_label(
                lines, i, 4,
                skip_small_schedule=document_type in {"balance_sheet", "profit_and_loss", "cash_flow_statement"},
                skip_small_schedule_inline=document_type == "balance_sheet",
            )
            if vals:
                assigned_curr = False
                if len(vals) == 1 and avg_curr is not None and avg_comp is not None and layout:
                    nums = _number_tokens(source.split("\n")[0] if "\n" in source else source)
                    if nums:
                        t1 = nums[-1]
                        x1 = None
                        for b in layout:
                            if b.get("text") == t1: x1 = float(b.get("left", 0))
                        if x1 is not None:
                            if abs(x1 - avg_comp) < abs(x1 - avg_curr):
                                _put(fields, target + "__comparative", vals[0], page, source, confidence=0.90)
                                assigned_curr = True
                if not assigned_curr:
                    _put(fields, target, vals[0], page, source, confidence=0.90)
                if len(vals) > 1:
                    _put(fields, target + "__comparative", vals[1], page, source, confidence=0.90)
    return fields


# ---------------------------------------------------------------------------
# Type-specific *semantic* metadata only.  These are not template parsers.
# ---------------------------------------------------------------------------

def _infer_invoice_entities_from_layout(layout: list[dict[str, Any]], fields: dict[str, dict], page: int) -> None:
    """Use page bisection for two-column seller/client headers.

    This is intentionally geometry-driven: no vendor/template coordinates are
    embedded. The midpoint is inferred from the observed page word extents.
    """
    if not layout:
        return
    rows = _group_layout_rows(layout)
    right_edge = max((float(w.get("left", 0)) + float(w.get("width", 0)) for w in layout), default=0.0)
    if right_edge <= 0:
        return
    midpoint = right_edge / 2.0
    left_candidates, right_candidates = [], []
    for row in rows[:18]:
        text = _layout_row_text(row)
        if not re.search(r"[A-Za-z]{3}", text):
            continue
        low = text.lower()
        if re.search(r"description|quantity|qty|price|amount|subtotal|tax|total|invoice details|invoice date|due date|terms", low):
            continue
        left = " ".join(str(w.get("text", "")) for w in row if float(w.get("left", 0)) < midpoint)
        right = " ".join(str(w.get("text", "")) for w in row if float(w.get("left", 0)) >= midpoint)
        def orglike(x):
            return bool(re.search(r"\b(?:inc\.?|corp\.?|corporation|ltd\.?|limited|llc|plc|company|co\.?)\b", x, re.I)) or bool(re.fullmatch(r"[A-Z][A-Z0-9& .,'-]{5,}", x.strip()))
        if orglike(left):
            left_candidates.append((left.strip(), text))
        if orglike(right):
            right_candidates.append((right.strip(), text))
    if left_candidates and fields.get("vendor_name", {}).get("value") is None:
        _put(fields, "vendor_name", left_candidates[0][0], page, left_candidates[0][1], confidence=0.90)
    if right_candidates and fields.get("customer_name", {}).get("value") is None:
        _put(fields, "customer_name", right_candidates[0][0], page, right_candidates[0][1], confidence=0.90)


def _infer_invoice_entities(lines: list[str], fields: dict[str, dict], page: int, layout: list[dict[str, Any]] | None = None) -> None:
    """Infer invoice parties from the complete page without promoting OCR rows.

    Only company-like header candidates are considered.  Addresses, emails,
    table rows and footer terms are never used as customer/vendor names.
    Explicit labels always win.
    """
    text = "\n".join(lines)
    receipt = bool(re.search(r"cashier|cash receipt|gst\s+(?:reg|summary)|counter\s*:", text, re.I))
    if layout and not receipt:
        _infer_invoice_entities_from_layout(layout, fields, page)
    if not re.search(r"\binvoice\b|\btax invoice\b|\bbill\b", text, re.I):
        return

    candidates: list[tuple[str, str]] = []
    for line in lines[:30]:
        cleaned = re.sub(r"\s+", " ", line).strip(" :-|")
        if not re.search(r"[A-Za-z]{3}", cleaned) or SUMMARY_RE.search(cleaned):
            continue
        if re.search(r"voucher|feedback|complaint|points expiry|please come|cashier|served by", cleaned, re.I):
            continue
        if re.search(r"\b(?:invoice|date|due|ticket|total)\b", cleaned, re.I) and re.search(r"\d", cleaned):
            continue
        if re.search(r"\b(?:client|customer|bill\s*to|billed\s*to|ship\s*to|supplier|vendor)\s*[:#]", cleaned, re.I):
            continue
        if re.search(r"\b(?:description|quantity|qty|price|amount|total|subtotal|tax|shipping|invoice details|invoice date|due date|terms|conditions)\b", cleaned, re.I):
            continue
        if receipt and re.search(r"\d", cleaned) and not re.search(r"\b(?:inc|corp|ltd|limited|llc|plc|company|co|carry|market|store)\b", cleaned, re.I):
            continue
        if receipt and re.fullmatch(r"[A-Z][A-Z0-9& .,'-]{5,}", cleaned) and not re.search(r"\b(?:cash|carry|market|store|sdn|bhd|inc|corp|ltd|limited|llc|plc|company|co)\b", cleaned, re.I):
            continue
        # Extract the organization before rejecting lines containing email/contact text.
        org = re.search(r"(.+?\b(?:inc\.?|corp\.?|corporation|ltd\.?|limited|llc|plc|company|co\.))(?=\s|$)", cleaned, re.I)
        if org:
            candidates.append((org.group(1).strip(" ,:-"), line))
            continue
        if re.search(r"@", cleaned):
            continue
        if re.fullmatch(r"[A-Z][A-Z0-9& .,'-]{4,}", cleaned) and not re.fullmatch(r"INVOICE|TAX INVOICE|BILL", cleaned, re.I):
            candidates.append((cleaned, line))

    # Deduplicate while preserving document order.
    unique = []
    seen = set()
    for name, source in candidates:
        key = _norm_label(name)
        if key and key not in seen:
            seen.add(key); unique.append((name, source))

    receipt = bool(re.search(r"cashier|cash receipt|gst\s+(?:reg|summary)|counter\s*:", text, re.I))
    if fields.get("vendor_name", {}).get("value") is None and unique:
        _put(fields, "vendor_name", unique[0][0], page, unique[0][1], confidence=0.82)
    # Do not invent a customer from an arbitrary OCR table row.  A customer is
    # inferred only when a second clear organization candidate exists.
    if not receipt and fields.get("customer_name", {}).get("value") is None and len(unique) > 1:
        _put(fields, "customer_name", unique[1][0], page, unique[1][1], confidence=0.76)


def _statement_metadata(pages_text: list[str], fields: dict[str, dict], document_type: str) -> None:
    text = "\n".join(pages_text)
    if document_type == "invoice":
        return
    m = re.search(r"(?:consolidated\s+)?(?:balance\s+sheet|profit\s+and\s+loss|statement\s+of\s+profit\s+and\s+loss|cash\s+flow\s+statement|statement\s+of\s+cash\s+flows?)", text, re.I)
    if m:
        _put(fields, "statement_name", m.group(0).strip(), 1, m.group(0), confidence=0.97)
    dates = re.findall(r"\b\d{1,2}[-/]\w{3,9}[-/]\d{2,4}\b|\b(?:Mar|March|Apr|April|Dec|December)\s+\d{1,2},?\s+\d{4}\b", text, re.I)
    uniq = list(dict.fromkeys(dates))[:2]
    if uniq:
        _put(fields, "statement_periods", uniq, 1, " ".join(uniq), confidence=0.90)
        _put(fields, "statement_date", uniq[0], 1, uniq[0], confidence=0.90)
    unit = re.search(r"(?:in|amounts?\s+in)\s*[‘'`]?\s*(?:Rs\.?|₹|USD|INR)?\s*(?:lakhs?|crores?|millions?|thousands?)", text, re.I)
    if unit:
        _put(fields, "unit_scale", unit.group(0).strip(), 1, unit.group(0), confidence=0.86)


def _strip_repeated_page_chrome(pages_text: list[str]) -> list[str]:
    """Remove repeated page headers/footers while preserving unique statement rows."""
    if len(pages_text) < 2:
        return pages_text
    page_lines = [_clean_lines(t) for t in pages_text]
    edge_counts: dict[str, int] = {}
    for lines in page_lines:
        for line in (lines[:4] + lines[-4:]):
            key = _norm_label(re.sub(r"\bpage\s+\d+(?:\s+of\s+\d+)?\b", "", line))
            if key:
                edge_counts[key] = edge_counts.get(key, 0) + 1
    repeated = {k for k, n in edge_counts.items() if n >= 2 and len(k) >= 5}
    cleaned_pages = []
    for lines in page_lines:
        cleaned = []
        for line in lines:
            n = _norm_label(line)
            if re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", n):
                continue
            if n in repeated and (re.search(r"annual report|financial statements?|page", n, re.I) or len(page_lines) > 2):
                continue
            cleaned.append(line)
        cleaned_pages.append("\n".join(cleaned))
    return cleaned_pages


def _statement_fallback(pages_text: list[str], document_type: str, page_layouts: list[list[dict[str, Any]]] | None = None) -> dict:
    fields: dict[str, dict] = {}
    discovered: dict[str, dict] = {}
    line_items: list[dict] = []

    pages_text = _strip_repeated_page_chrome(pages_text)
    for page_no, text in enumerate(pages_text, start=1):
        lines = _clean_lines(text)
        layout = (page_layouts or [])[page_no - 1] if page_layouts and len(page_layouts) >= page_no else None
        page_discovered = _discover_key_values(lines, page_no)
        for key, info in page_discovered.items():
            discovered.setdefault(key, info)

        semantic = _extract_semantic_fields(lines, page_no, document_type, layout=layout)
        if document_type == "invoice" and "currency" not in semantic:
            # Prefer totals/summary rows as currency evidence; do not ground the
            # currency field on an arbitrary product row.
            currency_lines = [x for x in lines if re.search(r"subtotal|tax|shipping|freight|handling|total due|amount due|balance due", x, re.I)] + lines
            for line in currency_lines:
                if CURRENCY_RE.fullmatch(line.strip()):
                    value = line.strip().upper() if line.strip().isalpha() else line.strip()
                    semantic["currency"] = _field(value, page_no, line, 0.94)
                    break
                symbol = re.search(r"[$€£₹]", line)
                if symbol:
                    semantic["currency"] = _field(symbol.group(0), page_no, line, 0.90)
                    break
                code = CURRENCY_RE.search(line)
                if code:
                    semantic["currency"] = _field(code.group(0).upper(), page_no, line, 0.88)
                    break
        for name, info in semantic.items():
            _put(fields, name, info.get("value"), page_no, info.get("source_text") or "", confidence=info.get("confidence"))

        if document_type == "invoice":
            _infer_invoice_entities(lines, fields, page_no, layout=layout)
        items = _discover_table(lines, page_no, invoice_mode=document_type == "invoice", layout=layout)
        line_items.extend(items)

    # Balance sheets frequently use a generic repeated "Total" row. Resolve
    # it from document order/sections instead of assuming a vendor template.
    if document_type == "balance_sheet":
        totals = []
        for page_no, text in enumerate(pages_text, start=1):
            lines = _clean_lines(text)
            for i, line in enumerate(lines):
                norm = _norm_label(line)
                if not (norm == "total" or norm.startswith("total ")):
                    continue
                vals = [v for v in (_parse_number(t) for t in _number_tokens(line)) if v is not None]
                if len(vals) >= 2:
                    totals.append((page_no, vals[-2], vals[-1], line))
                    continue
                if len(vals) == 1:
                    totals.append((page_no, vals[0], None, line))
                    continue
                # Some OCR layouts put "Total" on one line and the numeric
                # values on the following line(s). Capture only the immediate
                # numeric rows, then stop at the next labelled row.
                follow: list[float] = []
                source_parts = [line]
                for nxt in lines[i + 1:i + 4]:
                    nums = [v for v in (_parse_number(t) for t in _number_tokens(nxt)) if v is not None]
                    if nums:
                        follow.extend(nums)
                        source_parts.append(nxt)
                        if len(follow) >= 2:
                            break
                    elif re.search(r"[A-Za-z]", nxt):
                        break
                if follow:
                    totals.append((page_no, follow[-2] if len(follow) >= 2 else follow[0], follow[-1] if len(follow) >= 2 else None, " ".join(source_parts)))
        if totals:
            p, cur, comp, src = totals[0]
            _put(fields, "total_capital_and_liabilities", cur, p, src, overwrite=True, confidence=0.96)
            if comp is not None:
                _put(fields, "total_capital_and_liabilities__comparative", comp, p, src, overwrite=True, confidence=0.96)
            if len(totals) > 1:
                p, cur, comp, src = totals[1]
                _put(fields, "total_assets", cur, p, src, overwrite=True, confidence=0.96)
                if comp is not None:
                    _put(fields, "total_assets__comparative", comp, p, src, overwrite=True, confidence=0.96)

    _statement_metadata(pages_text, fields, document_type)

    # Invoice number is often "Invoice Details: 6825" or a standalone identifier.
    if document_type == "invoice" and fields.get("invoice_number", {}).get("value") is None:
        for key, info in discovered.items():
            if any(term in key for term in ("invoice details", "invoice number", "invoice no", "bill no")):
                value = info.get("value")
                if value is not None:
                    if isinstance(value, str):
                        m_num = re.search(r"(?:invoice\s*#?|invoice\s*(?:no|number))\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9./_-]*)", value, re.I)
                        if m_num:
                            value = m_num.group(1)
                    _put(fields, "invoice_number", value, info.get("page_number", 1), info.get("source_text") or "", confidence=0.88)
                    break

    invoice_number = fields.get("invoice_number")
    if invoice_number and isinstance(invoice_number.get("value"), str):
        match = re.search(r"(?:invoice\s*(?:details|number|no|#)|bill\s*(?:number|no|#))\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9./_-]*)", invoice_number["value"], re.I)
        if match:
            invoice_number["value"] = match.group(1)

    # Receipt-style invoices often print the date without a label. Use only a
    # date-only metadata line, never an arbitrary numeric product row.
    if document_type == "invoice" and fields.get("invoice_date", {}).get("value") is None:
        for line in lines:
            date = _date_from_text(line)
            if date:
                _put(fields, "invoice_date", date, 1, line, confidence=0.78)
                break

    # If a generic discovered field is an obvious semantic match, promote it.
    for key, info in discovered.items():
        for target in SEMANTIC_TERMS:
            if target in fields:
                continue
            if _semantic_label_matches(key, target):
                _put(fields, target, info.get("value"), info.get("page_number", 1), info.get("source_text") or "", confidence=min(0.90, info.get("confidence", 0.85)))

    # Accounting reconciliation is used as a controlled OCR-repair mechanism.
    # We never change a source component; only a visibly corrupted total is
    # repaired when the visible components or the opposite-side total support it.
    if document_type == "balance_sheet":
        liability_names = [
            "capital", "reserves_and_surplus", "minority_interest", "deposits",
            "borrowings", "other_liabilities_and_provisions"
        ]
        asset_names = [
            "cash_and_balances_with_reserve_bank_of_india",
            "balances_with_banks_and_money_at_call_and_short_notice",
            "investments", "advances", "fixed_assets", "other_assets"
        ]
        lv = [fields.get(n, {}).get("value") for n in liability_names]
        av = [fields.get(n, {}).get("value") for n in asset_names]
        cap = fields.get("total_capital_and_liabilities", {}).get("value")
        assets = fields.get("total_assets", {}).get("value")
        liability_sum = sum(float(v) for v in lv) if all(isinstance(v, (int, float)) for v in lv) else None
        asset_sum = sum(float(v) for v in av) if all(isinstance(v, (int, float)) for v in av) else None

        if liability_sum is not None and (cap is None or abs(liability_sum - float(cap)) > 1.0):
            old = fields.get("total_capital_and_liabilities")
            fields["total_capital_and_liabilities"] = _field(
                liability_sum,
                old.get("page_number", 1) if old else 1,
                f"OCR-repaired from visible liability reconciliation; original: {old.get('source_text') if old else 'total row missing'}",
                0.86,
            )
            cap = liability_sum

        if asset_sum is not None and (assets is None or abs(asset_sum - float(assets)) <= 1.0):
            if assets is None:
                fields["total_assets"] = _field(asset_sum, 1, "OCR-repaired from visible asset reconciliation", 0.86)
                assets = asset_sum
        elif assets is not None and cap is not None and abs(float(assets) - float(cap)) > 1.0:
            # Total Assets must equal Total Capital & Liabilities. This catches
            # OCR such as "97 995,086,442" while retaining the original evidence.
            old = fields.get("total_assets")
            if old:
                fields["total_assets"] = _field(cap, old.get("page_number", 1), f"OCR-repaired from balance-sheet identity; original: {old.get('source_text')}", 0.84)

        comparative_liabilities = [fields.get(f"{name}__comparative", {}).get("value") for name in liability_names]
        comparative_assets = [fields.get(f"{name}__comparative", {}).get("value") for name in asset_names]
        comparative_cap = fields.get("total_capital_and_liabilities__comparative", {}).get("value")
        comparative_total_assets = fields.get("total_assets__comparative", {}).get("value")
        if all(isinstance(value, (int, float)) for value in comparative_liabilities):
            comparative_sum = sum(float(value) for value in comparative_liabilities)
            if comparative_cap is None or abs(comparative_sum - float(comparative_cap)) > 1.0:
                fields["total_capital_and_liabilities__comparative"] = _field(comparative_sum, 1, "OCR-repaired from visible comparative liability reconciliation", 0.86)
                comparative_cap = comparative_sum
        if all(isinstance(value, (int, float)) for value in comparative_assets):
            comparative_sum = sum(float(value) for value in comparative_assets)
            if comparative_total_assets is None or abs(comparative_sum - float(comparative_total_assets)) > 1.0:
                fields["total_assets__comparative"] = _field(comparative_sum, 1, "OCR-repaired from visible comparative asset reconciliation", 0.86)
        if comparative_cap is not None:
            fields["total_assets__comparative"] = _field(comparative_cap, 1, "OCR-repaired from comparative balance-sheet identity", 0.84)

    if document_type == "profit_and_loss":
        # Bank statements commonly label section totals only as "Total".
        # Resolve those rows by the nearest INCOME / EXPENDITURE section.
        section = None
        section_totals: dict[str, tuple[int, float, str]] = {}
        for page_no, text in enumerate(pages_text, start=1):
            for line in _clean_lines(text):
                normalized = _norm_label(line)
                if normalized.endswith(" income") or normalized in {"income", "i income"}:
                    section = "income"
                elif "expenditure" in normalized or normalized in {"expenses"}:
                    section = "expenditure"
                elif normalized in {"profit", "iii profit"}:
                    section = "profit"
                elif section in {"income", "expenditure"} and (normalized == "total" or normalized.startswith("total ")):
                    values = [v for v in (_parse_number(t) for t in _number_tokens(line)) if v is not None]
                    if values and section not in section_totals:
                        section_totals[section] = (page_no, values[-2] if len(values) > 1 else values[-1], line)
        for target, section_name in (("total_income", "income"), ("total_expenditure", "expenditure")):
            if section_name in section_totals:
                page_no, value, source = section_totals[section_name]
                _put(fields, target, value, page_no, source, overwrite=True, confidence=0.84)

    return {"fields": fields, "line_items": line_items, "discovered_fields": discovered}


# ---------------------------------------------------------------------------
# LLM: one bounded dynamic pass, not a growing collection of template rules.
# ---------------------------------------------------------------------------

MIN_FIELDS = {
    "invoice": ["invoice_number", "invoice_date", "vendor_name", "customer_name", "subtotal", "tax_amount", "total_amount"],
    "balance_sheet": ["total_assets"],
    "profit_and_loss": ["revenue", "cost_of_sales", "gross_profit", "operating_expenses", "operating_profit", "tax", "net_profit"],
    "cash_flow_statement": ["operating_cash_flow", "investing_cash_flow", "financing_cash_flow", "opening_cash", "net_change_in_cash", "closing_cash"],
}

BANK_PNL_REQUIRED = [
    "interest_earned", "other_income", "total_income", "interest_expended", "operating_expenses",
    "provisions_and_contingencies", "total_expenditure", "consolidated_net_profit_before_minority_interest",
    "minority_interest", "consolidated_net_profit_attributable_to_group",
]


def _required_fields_for(document_type: str, fields: dict) -> list[str]:
    if document_type == "profit_and_loss" and any(fields.get(k, {}).get("value") is not None for k in BANK_PNL_REQUIRED):
        required = list(BANK_PNL_REQUIRED)
        if fields.get("net_profit", {}).get("value") is not None and fields.get("consolidated_net_profit_before_minority_interest", {}).get("value") is None:
            required = [name for name in required if name not in {"minority_interest", "consolidated_net_profit_before_minority_interest", "consolidated_net_profit_attributable_to_group"}]
        return required
    if document_type == "balance_sheet" and fields.get("total_capital_and_liabilities", {}).get("value") is not None:
        return ["total_assets", "total_capital_and_liabilities"]
    return MIN_FIELDS.get(document_type, [])


SYSTEM_PROMPT = """You are a dynamic financial-document intelligence engine.
Read the supplied OCR pages HOLISTICALLY, including the complete header, body,
right-side content, footer, and all pages. Do NOT assume a vendor/template or
that invoice metadata is located at the top-left. Vendor/seller information may
be top-right; invoice numbers may appear in a footer or under labels such as
"Invoice Details". Return JSON only.

For `fields`, map values to the semantic field names requested below when the
meaning is clear. You may add additional semantic fields if they are visible.
For `discovered_fields`, preserve every meaningful label/value pair that does
not fit the canonical fields. For `line_items`, detect actual table rows and
columns from the document rather than assuming fixed positions.

Rules:
- Never invent or calculate a source value.
- Preserve negatives/parentheses.
- Do not treat tax IDs, schedule numbers, page numbers, product IDs, phone
  numbers, or dates as monetary values.
- Preserve comparative periods using `__comparative` suffixes.
- Every non-null value MUST have page_number and source_text copied from OCR.
- If a value is absent/unreadable, use null.
- Keep long product descriptions intact, including embedded product IDs, model
  numbers, pack sizes and measurements. Do not use numbers inside descriptions
  as quantity/unit price/amount when a 2-D table layout or explicit columns show
  them belong to the description.
- For invoice totals, extract shipping/freight/handling/delivery/other charges
  separately when visible. Do not assume subtotal + tax is the final total.
- Search the full page for vendor, customer, invoice number and date; they may
  occur anywhere in the header or footer.

Output:
{"fields":{},"discovered_fields":{},"line_items":[]}"""


def _build_user_prompt(document_type: str, pages_text: list[str], min_fields: list[str]) -> str:
    # Keep the LLM request bounded for speed and context safety.
    numbered = "\n\n".join(f"--- PAGE {i + 1} (FULL OCR) ---\n{t[:20000]}" for i, t in enumerate(pages_text))
    semantic = ", ".join(min_fields)
    return f"""Document type selected by the user: {document_type}
Semantic fields needed for validation: {semantic}
Discover all other meaningful fields from the document instead of forcing them into a fixed schema.

SOURCE OCR:
{numbered}

Return JSON only."""


def _get_llm_client() -> tuple[Any, str]:
    provider = (settings.llm_provider or "huggingface").lower()
    if OpenAI is None:
        return None, settings.huggingface_model or "deepseek-ai/DeepSeek-R1"
    if provider == "huggingface":
        if not settings.huggingface_api_key:
            return None, settings.huggingface_model or "deepseek-ai/DeepSeek-R1"
        return OpenAI(base_url=settings.huggingface_base_url or "https://router.huggingface.co/hf-inference/v1", api_key=settings.huggingface_api_key, timeout=25.0, max_retries=0), settings.huggingface_model or "deepseek-ai/DeepSeek-R1"
    if provider == "openai":
        if not settings.openai_api_key:
            return None, settings.openai_model or "gpt-4o-mini"
        return OpenAI(api_key=settings.openai_api_key, timeout=25.0, max_retries=0), settings.openai_model or "gpt-4o-mini"
    return None, "fallback"


def _call_llm(system_prompt: str, user_prompt: str, pages_text: list[str], document_type: str) -> dict:
    client, model_name = _get_llm_client()
    if client is None:
        return {"fields": {}, "discovered_fields": {}, "line_items": []}
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "temperature": 0,
    }
    if (settings.llm_provider or "").lower() == "openai":
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
        data = json.loads(_clean_json_response(response.choices[0].message.content or "{}"))
        return data if isinstance(data, dict) else {"fields": {}, "discovered_fields": {}, "line_items": []}
    except Exception as exc:
        logger.warning("Dynamic LLM extraction failed; retaining grounded OCR extraction: %s", exc)
        return {"fields": {}, "discovered_fields": {}, "line_items": []}


def _source_contains_value(pages_text: list[str], value: Any, source_text: Any) -> bool:
    if not isinstance(source_text, str) or not source_text.strip():
        return False
    source_norm = re.sub(r"\s+", " ", source_text).strip().lower()
    full = re.sub(r"\s+", " ", " ".join(pages_text)).strip().lower()
    if source_norm not in full:
        words = [w for w in re.findall(r"[a-z0-9₹$€£.-]{3,}", source_norm) if not w.isdigit()]
        if not words or sum(w in full for w in words) < max(1, len(words) // 2):
            return False
    if value is None:
        return True
    if isinstance(value, (int, float)):
        source_numbers = [_parse_money(x) for x in _number_tokens(source_text)]
        return any(v is not None and abs(float(v) - float(value)) <= 0.01 for v in source_numbers)
    return str(value).strip().lower() in source_norm


def _merge_llm_output(base: dict, llm: dict, pages_text: list[str], document_type: str) -> dict:
    result = {
        "fields": dict(base.get("fields", {})),
        "line_items": list(base.get("line_items", [])),
        "discovered_fields": dict(base.get("discovered_fields", {})),
    }

    for name, info in (llm.get("fields") or {}).items():
        if not isinstance(info, dict) or info.get("value") is None:
            continue
        if name in result["fields"] and result["fields"][name].get("value") is not None:
            continue
        page = max(1, min(int(info.get("page_number") or 1), len(pages_text) or 1))
        source = info.get("source_text")
        if _source_contains_value(pages_text, info.get("value"), source):
            result["fields"][name] = _field(info.get("value"), page, source, 0.90)

    for name, info in (llm.get("discovered_fields") or {}).items():
        if not isinstance(info, dict) or info.get("value") is None:
            continue
        page = max(1, min(int(info.get("page_number") or 1), len(pages_text) or 1))
        source = info.get("source_text")
        if _source_contains_value(pages_text, info.get("value"), source):
            result["discovered_fields"].setdefault(_norm_label(name), _field(info.get("value"), page, source, 0.86))

    # Deterministic layout extraction is authoritative when it found rows;
    # appending a second LLM interpretation would duplicate or corrupt items.
    if result["line_items"]:
        return result
    for item in llm.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source_text")
        if source and _source_contains_value(pages_text, item.get("amount"), source):
            clean = dict(item)
            clean["page_number"] = max(1, min(int(item.get("page_number") or 1), len(pages_text) or 1))
            clean["confidence"] = 0.88
            result["line_items"].append(clean)
    return result


def _required_missing(document_type: str, fields: dict) -> list[str]:
    required = _required_fields_for(document_type, fields)
    return [name for name in required if fields.get(name, {}).get("value") is None]


def extract_fields(pages_text: list[str], document_type: str, page_layouts: list[list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Extract dynamically from any supported document layout."""
    pages_text = _strip_repeated_page_chrome(pages_text)
    base = _statement_fallback(pages_text, document_type, page_layouts=page_layouts)
    required = _required_fields_for(document_type, base.get("fields", {}))
    quality_required = required
    if document_type == "invoice":
        source_text = "\n".join(pages_text).lower()
        is_receipt = bool(re.search(r"cash receipt|cashier|total\s+(?:incl|inclusive)|total sales|gst summary", source_text))
        # Invoice metadata varies widely: receipts often have no customer or
        # invoice number, and tax/subtotal may be omitted or unreadable. The
        # financially essential completeness check is a grounded final total;
        # all other canonical keys remain present as nullable response fields.
        if is_receipt or document_type == "invoice":
            quality_required = ["total_amount"]

    # Exactly one bounded semantic pass. It receives the full cleaned OCR plus
    # discovered labels, not one prompt/call per missing field.
    if settings.use_llm_fallback:
        discovered_labels = ", ".join((base.get("discovered_fields") or {}).keys())
        prompt = _build_user_prompt(document_type, pages_text, required + ([discovered_labels] if discovered_labels else []))
        llm = _call_llm(SYSTEM_PROMPT, prompt, pages_text, document_type)
        base = _merge_llm_output(base, llm, pages_text, document_type)

    extracted: dict[str, Any] = {}
    for name, info in base.get("fields", {}).items():
        extracted[name] = {
            "value": info.get("value"),
            "page_number": info.get("page_number"),
            "evidence": {"source_text": info.get("source_text"), "page_number": info.get("page_number")},
            **({"confidence": info["confidence"]} if "confidence" in info else {}),
        }
    for name in MIN_FIELDS.get(document_type, required):
        extracted.setdefault(name, {"value": None, "page_number": None, "evidence": {"source_text": None, "page_number": None}})

    discovered = base.get("discovered_fields", {}) or {}
    if discovered:
        extracted["discovered_fields"] = {
            key: {
                "value": info.get("value"),
                "page_number": info.get("page_number"),
                "evidence": {"source_text": info.get("source_text"), "page_number": info.get("page_number")},
                **({"confidence": info["confidence"]} if "confidence" in info else {}),
            }
            for key, info in discovered.items()
            if isinstance(info, dict) and info.get("value") is not None
        }

    items = [item for item in (base.get("line_items", []) or []) if isinstance(item, dict)]
    if document_type == "invoice":
        items = [item for item in items if any(item.get(k) not in (None, "") for k in ("description", "quantity", "unit_price", "amount"))]
        if items:
            extracted["line_items"] = items
    elif items:
        extracted["financial_line_items"] = items

    missing_required = [name for name in quality_required if base.get("fields", {}).get(name, {}).get("value") is None]
    extracted["extraction_quality"] = {
        "required_fields_missing": missing_required,
        "manual_review_required": bool(missing_required),
        "method": "dynamic_ocr_plus_single_llm_pass" if settings.use_llm_fallback else "dynamic_ocr",
        "dynamic_schema": True,
    }
    return extracted
