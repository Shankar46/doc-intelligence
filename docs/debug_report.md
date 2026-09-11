# Debug & Verification Report

## Original defect

The local extraction fallback searched the complete LLM user prompt. Because the prompt itself contained field names and examples, deterministic regex could extract schema/schedule text as if it were document data. This caused values such as `assets = 10`, `liabilities = 12`, `invoice_number = Minimum`, and signature text as `customer_name`.

## Fix

- Fallback extraction now receives the actual OCR/document text separately from the prompt.
- Balance Sheet extraction uses a dedicated table parser.
- Schedule numbers are never interpreted as `total_liabilities` or `total_equity`.
- OCR number parsing handles spaces around comma groups and parentheses for negatives.
- A targeted OCR pass repairs incomplete Balance Sheet Total rows when full-page OCR is damaged by table borders.
- Comparative periods and values are retained.
- Financial statement line items are displayed in a dedicated frontend table.
- Validation rules now follow the case-study formulas.

## Real supplied-document verification

| File | Current Assets | Current Capital & Liabilities | Comparative Assets | Validation |
|---|---:|---:|---:|---|
| Consolidated Balance Sheet 2017.pdf | 8,923,441,607 | 8,923,441,607 | 7,622,123,264 | PASS |
| Consolidated Balance Sheet 2018.pdf | 11,031,861,695 | 11,031,861,695 | 8,923,441,607 | PASS |
| Consolidated Balance Sheet 2019.pdf | 12,928,057,065 | 12,928,057,065 | 11,031,861,695 | PASS |
| Consolidated Balance Sheet 2020.pdf | 15,808,304,373 | 15,808,304,373 | 12,928,057,065 | PASS |
| Consolidated Balance Sheet 2021.pdf | 17,995,066,442 | 17,995,066,442 | 15,808,304,373 | PASS |

All five supplied documents are scanned/image-based PDFs and were successfully OCR-processed locally.

## API verification

Verified locally with the supplied 2020 Balance Sheet:

- `POST /api/v1/documents/process` → HTTP 200, correct extracted totals, validation PASS
- `GET /api/v1/documents/{document_name}` → HTTP 200, persisted latest result
- `GET /api/v1/documents` → dashboard list endpoint available
- `GET /api/v1/health` → HTTP 200
- `GET /docs` → HTTP 200 Swagger/OpenAPI page
- unsupported `.txt` upload → HTTP 400 structured error

## Automated tests

```text
15 passed, 1 skipped
```

The skipped test only concerns the optional OpenAI-compatible SDK in the current execution environment; the dependency remains in `backend/requirements.txt`.

## Security cleanup

The real-looking Hugging Face token previously present in `.env.example` was removed. The final package does not include the Git history. If the old repository history has already been pushed anywhere, rotate/revoke that token and rewrite the public Git history before submission.

## Remaining submission actions

Deployment cannot be completed without access to the candidate's deployment/GitHub accounts. Before submitting, publish the repository, deploy the Docker application, configure `HUGGINGFACE_API_KEY` as a deployment secret, verify the live URLs, and replace the deployment placeholders in the README.
