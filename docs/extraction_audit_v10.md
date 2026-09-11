# Extraction Audit and Corrections — Dynamic v10

## Root cause found

The previous implementation had become too dependent on a growing set of document-type labels and layout-specific branches. That approach is slow to maintain and can fail on a new vendor/template even when the document is otherwise readable. Invoice OCR can also place product numbers, package sizes, quantities, unit prices and amounts on the same line.

The redesign changes the extraction model from **template parsing** to **dynamic document understanding**. Canonical semantic concepts are retained only where the validation contract needs them; everything else is discovered from the document and preserved.

The previous implementation also treated optional invoice `discount` as a required field and treated generic Profit & Loss fields as mandatory even for bank-style statements that use the HDFC-specific schema.

## Corrections

### Invoice
- Added `Invoice Details` as an invoice-number label variant.
- Added unlabeled company-name fallback for clear legal entity names.
- Added shipping/handling/freight/delivery charge extraction.
- Added `shipping_and_handling` to invoice financial validation.
- Tax IDs are excluded from tax extraction.
- Tax percentages are stripped before tax amount parsing.
- Decimal-comma amounts are supported.
- Line-item quantity/unit-price/amount selection now uses arithmetic consistency (`quantity * unit_price ~= amount`) instead of blindly selecting the last three numeric tokens.
- Long descriptions containing numbers such as product IDs and package sizes are preserved.
- Table-header detection no longer mistakes ordinary `item`, `invoice`, or `date` metadata for the actual table header.
- Optional discount is no longer treated as a missing required field.

### Profit & Loss
- Supports both generic corporate P&L and bank-style HDFC P&L schemas.
- Bank-style required-field detection is conditional on the detected bank schema.
- Added common aliases such as `profit from operations`, `taxation`, `profit after tax`, and `gross margin`.
- Comparative values are retained independently.
- Added `financial_line_items` so visible rows beyond the minimum schema are preserved.

### Balance Sheet
- Existing HDFC-style extraction retained.
- Comparative values and schedule-number handling retained.
- Added `financial_line_items` for visible statement rows beyond the canonical fields.
- Generic balance-sheet required-field handling no longer incorrectly requires `total_liabilities` and `total_equity` when the source uses a different presentation such as `Total Capital & Liabilities`.

### Cash Flow
- Existing operating/investing/financing/FX/opening/net-change/closing extraction retained.
- Parenthesized values remain negative.
- Added `financial_line_items` for visible statement rows beyond the canonical fields.

## Important invoice audit observation

The supplied review states these five line amounts: 16, 56, 30, 28 and 15. Those values sum to **145**, not 135. The same review states that the printed subtotal is 135 and that subtotal + tax 12.48 + shipping 10 = total 157.48.

The implementation deliberately does **not** hardcode a correction to 135 or 145. If the source really contains those five amounts and a subtotal of 135, the correct behavior is to flag the line-item/subtotal reconciliation as a validation failure. This is preferable to silently altering a source value.

## Test result

`24 passed, 1 skipped`.

The skipped test concerns optional LLM SDK availability in the local test environment.

## Interpretation

A successful financial validation does not prove extraction is complete. The dashboard/API should expose extracted values, evidence and missing-field/manual-review indicators separately from financial validation. A document can have mathematically consistent totals while still missing vendor/customer/line-item fields.

## Dynamic extraction redesign

### Structure discovery
- Explicit `label: value` / `label = value` pairs are discovered without a fixed field list.
- OCR-noisy labels are normalized with whitespace/punctuation-insensitive matching.
- Financial rows are preserved as `financial_line_items` for statements.
- Invoice table headers are detected from column semantics; quantity/rate/amount are selected using arithmetic consistency instead of fixed positions.
- Unknown fields such as payment terms, customer references, due dates, PO numbers, and other vendor-specific fields are retained under `discovered_fields`.

### Semantic mapping
The validator still needs concepts such as `total_assets`, `total_income`, or `closing_cash`. These are semantic targets, not document templates. The same target can be populated from different labels/layouts. The extractor also preserves the original discovered label and evidence so the system does not discard information that is outside the validation schema.

### LLM behavior
When an API key is configured, the system makes one bounded LLM extraction call for the whole document instead of making one request per missing field. The LLM may discover additional fields and tables, but every accepted value must be grounded in page evidence. This makes processing more predictable and avoids a large number of repeated requests.

### OCR performance
The scanned-page OCR path was changed to 200 DPI with PSM 6 as the primary pass and PSM 11 only when the primary OCR is weak. Preprocessing is limited to one additional retry. This removes the previous four-pass worst case on difficult scans.

## Regression test result

`27 passed, 1 skipped`. Three additional tests cover dynamic unknown-field discovery, OCR-spacing tolerance, and single-call grounded LLM merging.
