"""Tests for extraction service and Hugging Face / DeepSeek-R1 output cleaning."""
import json
from app.services.extraction_service import _clean_json_response, _get_llm_client
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
    monkeypatch.setattr(settings, "llm_provider", "huggingface")
    monkeypatch.setattr(settings, "huggingface_api_key", "hf_test_token_12345")
    monkeypatch.setattr(settings, "huggingface_model", "deepseek-ai/DeepSeek-R1")

    client, model_name = _get_llm_client()
    assert model_name == "deepseek-ai/DeepSeek-R1"
    assert str(client.base_url).rstrip("/") == "https://router.huggingface.co/hf-inference/v1"
