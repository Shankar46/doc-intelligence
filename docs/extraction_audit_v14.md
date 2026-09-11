# Extraction Audit — Dynamic v14

## Implementation status

This version changes the pipeline from template/static parsing toward a document-driven dynamic architecture.

- OCR preserves word-level `(left, top, width, height, confidence)` coordinates for scanned documents.
- Native PDFs use PyMuPDF word coordinates.
- Physical OCR rows are reconstructed from Y coordinates.
- Invoice columns are discovered from semantic header anchors and X coordinates rather than fixed vendor positions.
- Wrapped invoice rows are grouped into logical items.
- Numeric product identifiers and pack sizes remain in descriptions when their X positions belong to the description column.
- Header/entity separation can use page bisection for two-column seller/client layouts.
- Locale-aware number parsing supports decimal/thousands separator inference and OCR parenthesis artifacts.
- Repeated multi-page page chrome is removed before semantic matching.
- Key/value pairs are dynamically discovered across the complete document.
- One bounded LLM pass can map discovered content to validation concepts and preserve unknown fields.
- LLM values are accepted only when grounded in OCR evidence.
- Invoice validation uses a bounded additive solver to discover combinations such as subtotal + CGST + SGST + shipping - discount + round-off.
- Cash-paid/change reconciliation is independent.
- Balance-sheet/P&L/cash-flow validation remains period-aware and null-safe.
- Processing status is decoupled from financial validation status.

## Regression suite

`pytest backend/tests` → **39 passed, 1 skipped** in the build environment.

The suite includes regression coverage for:

1. Layout-aware invoice column extraction.
2. Wrapped OCR invoice rows.
3. Embedded product/model/pack-size numbers.
4. OCR quantity repair using amount / price arithmetic.
5. Shipping-aware invoice totals.
6. Split-tax dynamic additive reconciliation.
7. Cash paid versus change.
8. Parenthesis/comma locale normalization.
9. Repeated multi-page header/footer removal.
10. Dynamic unknown/discovered fields.
11. Comparative balance-sheet and P&L extraction.
12. Processing-status/validation separation.

## Important source-data note

The known `batch2-0499` content contains a genuine source inconsistency: the five visible line amounts sum to 145 while the printed subtotal is 135. The printed invoice total of 157.48 is consistent with 135 + 12.48 + 10. Therefore a correct system should pass the total reconciliation while failing the line-items-to-subtotal check; it must not silently alter source values.

## Deployment

`Dockerfile`, `render.yaml`, and `.env.example` are present. No live deployment URL or public GitHub repository is claimed in this package because those external resources have not been provisioned in this environment.
