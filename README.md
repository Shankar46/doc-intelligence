# Document Intelligence Platform

AI-powered document extraction, financial validation, persistence, and REST API platform for invoices, balance sheets, P&L statements, and cash flow statements.

## 1. Deployed Application Links (Evaluator Please Note)

- **Frontend Application:** `https://doc-intelligence-nb5c.onrender.com`
- **Backend API Base URL:** `https://doc-intelligence-nb5c.onrender.com/api/v1`
- **Swagger / OpenAPI Docs:** `https://doc-intelligence-nb5c.onrender.com/docs`
- **Public GitHub Repository:** `https://github.com/Shankar46/doc-intelligence`

## 2. Solution Overview and Architecture

The application accepts financial documents (PDF/JPG/PNG) and extracts native text or uses OCR for scanned documents. It performs dynamic structure-aware extraction, discovering key/value pairs, statement rows, and tables from the document instead of relying on vendor-specific templates. The processed results are mapped, mathematically validated (ensuring totals match line items), and stored in SQLite, making them accessible via a dashboard and REST API. 

**Note:** Always select the correct document type in the frontend before processing a file.

## 3. Technology Stack & Reason for Major Choices

- **API:** FastAPI (Chosen for high performance, async support, and auto-generated Swagger UI).
- **Database:** SQLite & SQLAlchemy (Chosen for simplicity and zero-configuration persistence during prototype phase).
- **OCR/Document Parsing:** PyMuPDF & Tesseract (Chosen for fast, free-tier offline native parsing and robust OCR capabilities).
- **Extraction Model:** Gemini (Chosen for its robust document intelligence, JSON-schema adherence, and multi-modal understanding).
- **Frontend:** Jinja2 + Vanilla JS + CSS (Chosen to keep the stack monolithic and easy to deploy without needing a separate Node.js build pipeline).

## 4. Local Setup Instructions

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000` for the dashboard and `http://localhost:8000/docs` for the API documentation.

## 5. Environment Variables

See `.env.example`. 
- `DATABASE_URL` defaults to local SQLite. 
- `LLM_PROVIDER` should be set to `huggingface`.
- `HUGGINGFACE_API_KEY` is required for the AI extraction process.

## 6. API Usage Examples

**Process Document:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/process" \
  -F "file=@sample_invoice.pdf" \
  -F "document_type=invoice"
```
*(Supported types: `invoice`, `balance_sheet`, `profit_and_loss`, `cash_flow_statement`)*

**Get Document Result:**
```bash
curl "http://localhost:8000/api/v1/documents/sample_invoice.pdf"
```

**List Documents:**
```bash
curl "http://localhost:8000/api/v1/documents"
```

## 7. Financial Validation Rules & Tolerance

- **Invoice:** Subtotal + Tax - Discount ≈ Total
- **Balance Sheet:** Total Assets ≈ Total Capital & Liabilities
- **Profit & Loss:** Total Income - Total Expenditure ≈ Profit
- **Cash Flow:** Net change in cash ≈ Closing cash - Opening cash

*(A numerical tolerance of `1.0` currency unit is applied for rounding differences).*

## 8. Database / Persistence Approach

Processed documents are saved into an SQLite database (`app.db`). We store the full structured JSON extraction alongside the file name, status, and document type. This means the DB effectively acts as a versioned cache of the processed response, making dashboard listing and API retrieval extremely fast and preventing data drift.

## 9. Known Limitations

- **Concurrency:** The SQLite database is configured for simple concurrent access, but a production system under heavy load would require PostgreSQL to prevent write-locking.
- **Large Documents:** Extremely dense documents larger than 3 pages may hit AI context-window limitations or timeout the synchronous API request.

## 10. Production Improvements

- **Async Processing:** Shift from synchronous API processing to a webhook/polling model using Celery and Redis to handle large batches of documents without timing out the HTTP request.
- **Cloud Storage:** Migrate document uploads from local disk to AWS S3 / Google Cloud Storage.
- **Managed Database:** Migrate from SQLite to a managed PostgreSQL cluster (e.g., AWS RDS).

## 11. AI Coding Assistants Used

Generative AI (Gemini) was used during development as permitted by the instructions. It was primarily utilized for writing boilerplate FastAPI scaffolding, styling the CSS for the frontend dashboard, and writing unit tests to cover the extraction logic.
