# Document Intelligence Platform

AI-powered document extraction, financial validation, persistence, and REST API platform for invoices, balance sheets, P&L statements, and cash flow statements.

## 1. Solution Overview & Architecture

The application accepts PDF/JPG/PNG financial documents, validates file integrity and page count, extracts native PDF text or uses adaptive Tesseract OCR for scanned documents, then performs document-type-specific structured extraction with evidence. Financial relationships are validated using only fields actually present in the source, and processed results are persisted in SQLite for dashboard/API retrieval.

Flow: Upload → File Validation → OCR/Text Extraction → AI/Deterministic Field & Table Extraction → Financial Validation → Persistence (SQLite) → Dashboard / REST API.

Architecture diagram: `docs/architecture.png`

**Important:** document type is selected explicitly by the user in the frontend and sent as request metadata. Automatic document classification is intentionally not required by the case study. Always choose the correct type before processing a file.

## 2. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Built-in Swagger/OpenAPI and simple REST implementation |
| DB | SQLite (SQLAlchemy) | Zero-setup persistent store; easy to replace with Postgres |
| OCR | PyMuPDF (native text) + adaptive Tesseract | Free/local OCR and works with scanned PDFs/images |
| Extraction LLM | Hugging Face (`deepseek-ai/DeepSeek-R1`) | Optional structured extraction provider |
| Extraction fallback | Deterministic OCR parser | Allows useful local processing when an LLM key is unavailable |
| Frontend | Jinja2 + vanilla JS + CSS | No frontend build step |
| Deployment | Render/Railway/Koyeb compatible | `Procfile`/`render.yaml` included |

## 3. Local Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp ../.env.example ../.env
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000` for the dashboard and `http://localhost:8000/docs` for Swagger.

## 4. Environment Variables

See `.env.example`. Never commit real API keys. `DATABASE_URL` defaults to local SQLite. If `LLM_PROVIDER=huggingface`, provide `HUGGINGFACE_API_KEY`; if no LLM key is configured, the deterministic OCR fallback is used.

## 5. Deployed URLs

Populate these after deployment before final submission:

- Frontend: `<LIVE_FRONTEND_URL>`
- Backend API base: `<LIVE_BACKEND_API_URL>`
- Swagger/OpenAPI: `<LIVE_BACKEND_API_URL>/docs`
- Public GitHub repository: `<PUBLIC_GITHUB_REPOSITORY_URL>`

## 6. API Examples

**POST /api/v1/documents/process**

```bash
curl -X POST "$BASE_URL/api/v1/documents/process" \
  -F "file=@sample_invoice.pdf" \
  -F "document_type=invoice"
```

Supported `document_type` values:

- `invoice`
- `balance_sheet`
- `profit_and_loss`
- `cash_flow_statement`

**GET /api/v1/documents/{document_name}**

```bash
curl "$BASE_URL/api/v1/documents/sample_invoice.pdf"
```

**GET /api/v1/documents**

```bash
curl "$BASE_URL/api/v1/documents"
```

**GET /api/v1/health**

```bash
curl "$BASE_URL/api/v1/health"
```

## 7. OCR / LLM Provider Used

Native PDFs are read with PyMuPDF. Pages with little/no native text are rasterized at 250 DPI and processed using fast adaptive Tesseract OCR (table-friendly PSM first, extra passes only when needed). Financial table scans prefer a table-preserving OCR layout, with additional OCR variants used when useful. The optional LLM provider is Hugging Face with `deepseek-ai/DeepSeek-R1`. If the provider/key is unavailable, extraction falls back to document-type-specific OCR parsing rather than parsing the extraction prompt.

The 2021 scanned Consolidated Balance Sheet was specifically regression-tested. Its visible totals are `17,995,066,442` for 31-Mar-21 and `15,808,304,373` for 31-Mar-20.

## 8. Confidence Scoring

Confidence scoring is not implemented because it is optional in the case study. Evidence is provided for extracted values with source text and page number where available.

## 9. Financial Validation Rules & Tolerance

Implemented in `backend/app/services/financial_validation_service.py`.

Tolerance: `TOLERANCE = 1.0` currency unit to allow small rounding differences.

- **Invoice:** `subtotal + tax_amount - discount ≈ total_amount`; line `quantity × unit_price ≈ amount`; line-item sum reconciles to subtotal when available. If tax is not separately shown, the check uses subtotal/discount without inventing tax.
- **Balance Sheet:** `total_capital_and_liabilities ≈ total_assets`, plus component-level asset and capital/liability reconciliation when all required components are available. Each comparative period is checked independently.
- **P&L:** `interest_earned + other_income ≈ total_income`; `interest_expended + operating_expenses + provisions_and_contingencies ≈ total_expenditure`; `total_income - total_expenditure ≈ consolidated_net_profit_before_minority_interest`; `profit_before_minority_interest - minority_interest ≈ consolidated_net_profit_attributable_to_group`; appropriation reconciliation when present. A generic revenue/COGS fallback is retained for conventional P&Ls.
- **Cash Flow:** operating + investing + financing + FX/translation adjustment ≈ net increase/change in cash; opening cash + net change + applicable amalgamation/other adjustments ≈ closing cash. Parentheses are treated as negative values.

Any check with a missing required operand returns `NOT_APPLICABLE`; values are never invented.

## 10. Database / Persistence

SQLite via SQLAlchemy (`backend/app/models/document.py`). Each processing run is stored as a full JSON snapshot. `GET /api/v1/documents/{name}` returns the latest result for that document name, and the dashboard uses the list endpoint to display persisted records.

## 11. Known Limitations

- Processing is synchronous and designed for documents up to three pages.
- Tesseract accuracy depends on scan quality and layout; adaptive OCR reduces but does not eliminate OCR errors.
- The LLM provider is optional and can be replaced without changing the API contract.
- SQLite is suitable for the assessment but a production deployment should use managed Postgres or an equivalent database.
- The supplied assessment dataset contains Balance Sheet PDFs; real Invoice/P&L/Cash Flow documents should be added for final end-to-end accuracy evaluation.

## 12. What I'd Change for Production

Use managed Postgres, asynchronous/background processing, stronger OCR/table extraction, LLM retry/backoff and provider fallback, calibrated confidence scoring, document versioning, malware scanning, rate limiting, authentication, structured observability/metrics, and a larger labelled evaluation dataset for measured extraction accuracy.

## 13. AI Coding Assistants Used

ChatGPT was used for debugging, extraction/prompt design, OCR troubleshooting, validation-rule implementation, test design, documentation review, and code improvement. All submitted logic was reviewed and tested against the assessment requirements and supplied Balance Sheet documents.

### Extraction performance

The application uses deterministic OCR extraction by default for fast, reliable processing. Set `USE_LLM_FALLBACK=true` only when you want the optional remote LLM fallback for documents the OCR parser cannot recover. Remote failures do not block the deterministic path.
