# Document Intelligence Platform

AI-powered document extraction, financial validation, persistence, and REST API platform for invoices, balance sheets, P&L statements, and cash flow statements.

## 1. Solution Overview & Architecture

The application accepts PDF/JPG/PNG financial documents, validates file integrity and page count, extracts native PDF text or uses adaptive Tesseract OCR for scanned documents, then performs **dynamic structure-aware extraction** with evidence. The extractor discovers key/value pairs, statement rows, table headers, line items, comparative periods, and additional fields from the actual document instead of maintaining a parser for each vendor/template. A single optional LLM pass maps discovered content to validation concepts and preserves unknown fields. Financial relationships are validated using only grounded values actually present in the source, and processed results are persisted in SQLite for dashboard/API retrieval.

Flow: Upload → File Validation → OCR/Text Extraction → 2-D Layout-Aware Field & Table Extraction → Grounded LLM Semantic Discovery → Financial Validation → Persistence (SQLite) → Dashboard / REST API.

Architecture diagram: `docs/architecture.png`

**Important:** document type is selected explicitly by the user in the frontend and sent as request metadata. Automatic document classification is intentionally not required by the case study. Always choose the correct type before processing a file.

## 2. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI | Built-in Swagger/OpenAPI and simple REST implementation |
| DB | SQLite (SQLAlchemy) | Zero-setup persistent store; easy to replace with Postgres |
| OCR | PyMuPDF (native text) + adaptive Tesseract | Free/local OCR and works with scanned PDFs/images |
| Extraction LLM | Hugging Face (`deepseek-ai/DeepSeek-R1`) | One bounded semantic extraction pass when configured |
| Extraction engine | Dynamic row/key-value/table discovery + semantic mapping | Adapts to unseen labels and layouts without vendor-specific parsers |
| Extraction fallback | Dynamic deterministic OCR parser | Allows useful local processing when an LLM key is unavailable |
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

See `.env.example`. Never commit real API keys. `DATABASE_URL` defaults to local SQLite. If `LLM_PROVIDER=huggingface`, provide `HUGGINGFACE_API_KEY` to enable LLM completion; if no LLM key is configured, the deterministic OCR extractor is used.

## 5. Deployed URLs

Populate these after deployment before final submission:

- Frontend: `NOT_DEPLOYED`
- Backend API base: `NOT_DEPLOYED`
- Swagger/OpenAPI: `<LIVE_BACKEND_API_URL>/docs`
- Public GitHub repository: `NOT_PUBLISHED`

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

Native PDFs are read with PyMuPDF. Scanned pages are rasterized at 200 DPI and processed using adaptive Tesseract OCR; a second layout pass is used only when the primary result is weak. The extraction layer then discovers the document structure dynamically: explicit key/value pairs, noisy OCR labels, financial rows, repeated totals, comparative values, and invoice tables. When configured, one bounded LLM pass performs semantic mapping and discovers additional fields; it is not called once per missing field. If no provider/key is available, the dynamic OCR extractor remains usable.

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

The application uses a dynamic OCR-first pipeline. It does not create a separate LLM request for every field: when `USE_LLM_FALLBACK=true` and an API key is configured, one bounded LLM pass receives the page OCR and maps semantic fields while also returning unknown/discovered fields and table rows. Existing OCR-grounded values are not overwritten. LLM values are accepted only when their evidence can be found in the supplied OCR. If no key is configured, the dynamic deterministic extractor remains fully usable.


## Dynamic extraction architecture

The extraction pipeline is intentionally not vendor/template-specific. OCR retains 2-D word coordinates, and invoice table extraction uses the actual column-header positions to keep numbers embedded in product descriptions out of quantity/unit-price/amount fields. The LLM receives the complete OCR for all pages, including headers and footers, and any non-canonical fields are preserved under `discovered_fields`.

Invoice total validation supports subtotal + tax + shipping/handling/other charges - discount. Financial validation failures are reported under `validation`; they do not change `processing_status` for a readable document that was successfully parsed.

Presentation deliverables: `docs/solution_presentation.pptx` and `docs/solution_presentation.pdf`.

## Extraction audit

The latest extraction corrections and regression findings are documented in `docs/extraction_audit_v10.md`.

### v12 invoice OCR regression fix

The invoice fallback now preserves embedded numeric product data and uses the final reconciled quantity/unit-price/amount triplet when OCR loses row coordinates. Numbered rows and unnumbered-but-arithmetic rows are supported, while invoice footer sections are prevented from being appended to the last line-item description. Same-line financial labels no longer consume values from the next row, and percentage rates are excluded from tax amounts. The invoice total check includes shipping/handling, and processing status remains `PASS` when financial validation fails.

### v13 invoice extraction hardening

The v13 parser fixes the remaining failure mode where Tesseract splits one visual invoice row into multiple physical OCR rows. Invoice rows are grouped by the 2-D item-number/column layout before financial values are assigned, so wrapped descriptions are merged instead of becoming extra line items. The text fallback also repairs an OCR-corrupted quantity only when the row's reported amount divided by price gives an exact positive integer (for example `5 × 8 != 16` is repaired to quantity `2`). Embedded product numbers such as `570`, `12oz`, `10760667-011PW`, and `100/BX` remain in descriptions. Currency evidence is grounded on summary rows, and vendor inference ignores invoice labels, contact emails, addresses, and arbitrary table rows.

The v13 regression suite contains **35 passed, 1 skipped** tests locally.
