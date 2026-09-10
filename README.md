# Document Intelligence Platform

AI-powered document extraction, financial validation, and API platform
for invoices, balance sheets, P&L statements, and cash flow statements.

> **TODO before submission:** fill in every `[FILL IN]` below, add real
> deployment URLs, and remove this note.

## 1. Solution Overview & Architecture

[FILL IN — 3-5 sentences + link/embed the architecture diagram in docs/architecture.png]

Flow: Upload → File Validation → OCR/Text Extraction → AI Field Extraction →
Financial Validation → Persistence (SQLite) → Dashboard / REST API.

## 2. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Built-in Swagger/OpenAPI, async, fast to build |
| DB | SQLite (SQLAlchemy) | Zero-setup persistence, easy to swap for Postgres |
| OCR | PyMuPDF (native text) + Tesseract (scanned fallback) | Free, local, no rate limits under deadline |
| Extraction LLM | Hugging Face (`deepseek-ai/DeepSeek-R1`) | High-performance reasoning model via Hugging Face Inference Router |
| Frontend | Jinja2 + vanilla JS + CSS | No build step, matches spec's "HTML/CSS, JS optional" requirement |
| Deployment | [FILL IN — Render/Railway/Koyeb] | Free tier, simple `Procfile` deploy |

## 3. Local Setup

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env   # fill in your own API key locally
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000` for the dashboard and `http://localhost:8000/docs` for Swagger.

## 4. Environment Variables

See `.env.example` at repo root — no real secrets are committed. Minimum
to run: `DATABASE_URL` (defaults to local SQLite) and one of
`HUGGINGFACE_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` depending on
`LLM_PROVIDER` (set to `huggingface` for `deepseek-ai/DeepSeek-R1`).


## 5. Deployed URLs

- Frontend: [FILL IN]
- Backend API base: [FILL IN]
- Swagger/OpenAPI: [FILL IN]/docs
- GitHub repo: [FILL IN]

## 6. API Examples

**POST /api/v1/documents/process**
```bash
curl -X POST "$BASE_URL/api/v1/documents/process" \
  -F "file=@sample_invoice.pdf" \
  -F "document_type=invoice"
```

**GET /api/v1/documents/{document_name}**
```bash
curl "$BASE_URL/api/v1/documents/sample_invoice.pdf"
```

**GET /api/v1/documents**
```bash
curl "$BASE_URL/api/v1/documents"
```

## 7. OCR / LLM Provider Used

[FILL IN — which OCR path was actually exercised (native text vs Tesseract) and which LLM model/provider, including free-tier notes.]

## 8. Confidence Scoring

[FILL IN if implemented — how it's calculated and how to interpret it. Optional per spec.]

## 9. Financial Validation Rules & Tolerance

Implemented in `backend/app/services/financial_validation_service.py`.
Tolerance: `TOLERANCE = 1.0` currency unit (absolute), to allow for
rounding in source documents. [Adjust and justify if you change this.]

- Invoice: `subtotal + tax_amount - discount ≈ total_amount`; per-line
  `quantity × unit_price ≈ amount`; `sum(line items) ≈ subtotal`.
- Balance Sheet: `total_liabilities + total_equity ≈ total_assets`.
- P&L: `revenue - cost_of_sales ≈ gross_profit`; `gross_profit - operating_expenses ≈ operating_profit`; `operating_profit - tax ≈ net_profit`.
- Cash Flow: `operating + investing + financing ≈ net_change_in_cash`; `opening_cash + net_change_in_cash ≈ closing_cash`.

Any check with a missing operand returns `NOT_APPLICABLE`, never a guessed value.

## 10. Database / Persistence

SQLite via SQLAlchemy (`backend/app/models/document.py`). Each processing
run is stored as a full JSON snapshot in `processed_documents.result_json`;
`GET /api/v1/documents/{name}` returns the most recent row for that name.

## 11. Known Limitations

[FILL IN — e.g. synchronous processing only, single active LLM provider, simple SQLite dedup-by-name-only.]

## 12. What I'd Change for Production

[FILL IN — e.g. async job queue, Postgres, per-document versioning, stronger confidence calibration, retry/backoff on LLM calls, virus scanning uploads, rate limiting.]

## 13. AI Coding Assistants Used

[FILL IN — which assistant(s), and roughly where: e.g. "Claude used for backend scaffolding, prompt design for extraction, and README drafting; all logic reviewed and tested manually."]
