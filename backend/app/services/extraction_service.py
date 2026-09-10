"""
AI-based field & table extraction (spec section 4.2 / 4.3).

Design:
  - One prompt template per document type, listing the MINIMUM required
    fields from the spec, but explicitly instructing the model to also
    capture every other visible field/line item it can find.
  - The model is instructed to return strict JSON matching our schema,
    including page_number + a short source_text snippet as evidence for
    each field, and to use null (never invent) for anything not present.
  - Swap _call_llm's body to use Gemini/Anthropic instead of OpenAI if
    you prefer a different free-tier provider -- the rest of the file
    doesn't need to change.
"""
import json
import logging
import re
from typing import Any

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)


def _clean_json_response(raw_text: str) -> str:
    """Clean LLM output by stripping <think> reasoning blocks and markdown code fences."""
    if not raw_text:
        return "{}"
    # Strip DeepSeek-R1 reasoning trace block <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Match JSON block inside ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    # Find the outermost json object {...}
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    return cleaned


def _get_llm_client() -> tuple[OpenAI, str]:
    """Returns (client, model_name) based on configured LLM_PROVIDER."""
    provider = (settings.llm_provider or "huggingface").lower()

    if provider == "huggingface":
        api_key = settings.huggingface_api_key
        if not api_key:
            raise RuntimeError(
                "HUGGINGFACE_API_KEY is not set in environment or .env file. "
                "Please set HUGGINGFACE_API_KEY to use deepseek-ai/DeepSeek-R1 or other Hugging Face models."
            )
        base_url = settings.huggingface_base_url or "https://router.huggingface.co/hf-inference/v1"
        client = OpenAI(base_url=base_url, api_key=api_key)
        return client, settings.huggingface_model or "deepseek-ai/DeepSeek-R1"

    elif provider == "openai":
        api_key = settings.openai_api_key
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment or .env file.")
        client = OpenAI(api_key=api_key)
        return client, settings.openai_model or "gpt-4o-mini"

    else:
        raise RuntimeError(
            f"Unsupported or unconfigured LLM provider '{settings.llm_provider}'. "
            "Supported providers: huggingface, openai."
        )


def _call_llm(system_prompt: str, user_prompt: str) -> dict:
    client, model_name = _get_llm_client()

    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    if (settings.llm_provider or "").lower() == "openai":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content or ""

    cleaned = _clean_json_response(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("LLM returned non-JSON content. Raw: %s | Cleaned: %s", raw[:500], cleaned[:500])
        raise

MIN_FIELDS = {
    "invoice": [
        "invoice_number", "invoice_date", "vendor_name", "customer_name",
        "currency", "subtotal", "tax_amount", "discount", "total_amount",
    ],
    "balance_sheet": [
        "total_assets", "total_liabilities", "total_equity",
    ],
    "profit_and_loss": [
        "revenue", "cost_of_sales", "gross_profit", "operating_expenses",
        "operating_profit", "tax", "net_profit",
    ],
    "cash_flow_statement": [
        "operating_cash_flow", "investing_cash_flow", "financing_cash_flow",
        "opening_cash", "net_change_in_cash", "closing_cash",
    ],
}

SYSTEM_PROMPT = """You are a financial document extraction engine.
You will be given the OCR/text content of a financial document, page by page.

Rules:
- Extract ALL meaningful fields and line items visible in the document, not just
  the minimum list provided -- headers, dates, parties, currencies, every
  financial line item, comparative-period values, and tables.
- If a value is not present or not legible, set its "value" to null. NEVER
  invent, guess, or infer a number that is not supported by the text.
- For every field, include which page it came from ("page_number") and a short
  verbatim snippet of the source text that supports it ("source_text"), when available.
- Numbers must be plain numbers (no currency symbols/commas) in the "value" field.
- Return ONLY valid JSON matching the exact schema described in the user message.
  No markdown, no commentary, no code fences.
"""


def _build_user_prompt(document_type: str, pages_text: list[str], min_fields: list[str]) -> str:
    numbered_pages = "\n\n".join(
        f"--- PAGE {i + 1} ---\n{text}" for i, text in enumerate(pages_text)
    )
    return f"""Document type: {document_type}
Minimum required fields (still extract everything else visible too): {min_fields}

Document text:
{numbered_pages}

Return JSON with this exact top-level shape:
{{
  "fields": {{
    "<field_name>": {{"value": <number|string|null>, "page_number": <int|null>, "source_text": "<snippet|null>"}}
    ... one entry per field/value found, using the minimum fields as required keys
        plus any additional fields you found ...
  }},
  "line_items": [
    {{"description": "...", "quantity": <number|null>, "unit_price": <number|null>, "amount": <number|null>}}
    ... only for invoices / statements with tabular line items, else [] ...
  ]
}}
"""
def extract_fields(pages_text: list[str], document_type: str) -> dict[str, Any]:
    """
    Returns: {"extracted_data": {field: {...}}, "line_items": [...]}
    Field entries follow the ExtractedField shape (value/confidence/page_number/evidence).
    """
    min_fields = MIN_FIELDS.get(document_type, [])
    user_prompt = _build_user_prompt(document_type, pages_text, min_fields)

    llm_output = _call_llm(SYSTEM_PROMPT, user_prompt)

    extracted_data: dict[str, Any] = {}
    for field_name, field_info in llm_output.get("fields", {}).items():
        extracted_data[field_name] = {
            "value": field_info.get("value"),
            "page_number": field_info.get("page_number"),
            "evidence": {
                "source_text": field_info.get("source_text"),
                "page_number": field_info.get("page_number"),
            },
        }

    # ensure every minimum-required field is present, even if the model omitted it
    for field_name in min_fields:
        extracted_data.setdefault(field_name, {
            "value": None, "page_number": None,
            "evidence": {"source_text": None, "page_number": None},
        })

    line_items = llm_output.get("line_items", []) or []
    if line_items:
        extracted_data["line_items"] = line_items

    return extracted_data
