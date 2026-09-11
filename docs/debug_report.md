# Extraction and validation debug report — v8

## Root cause found
The earlier implementation treated deterministic regex extraction as the primary source for every document. This was too conservative for invoice layouts and too fragile for OCR-fragmented financial statements. It also allowed ambiguous labels to capture nearby values (for example `Minority Interest` inside a longer row) and could select an OCR-corrupted total.

## v8 corrections
- Invoice extraction now supports label/value separation, common label variants, currency symbols/codes, decimal-comma amounts, tax percentages versus tax amounts, explicit tax-included wording, and flexible invoice table headers.
- Invoice metadata matching is anchored to actual field labels so prose such as `Tales from ...` cannot become a vendor name. Tax IDs are excluded from tax extraction.
- P&L extraction now distinguishes generic revenue/COGS statements from the case-study bank-style formulas and supports comparative values. Ambiguous occurrences such as `profit before minority interest` are not mistaken for the standalone `minority interest` row.
- Cash-flow extraction supports operating/investing/financing activities, bracketed negatives, FX adjustments, opening/closing cash, applicable adjustments, and comparative values.
- Balance-sheet extraction handles OCR labels with missing spaces (`Reservesandsurplus`, `Otherassets`), standalone schedule identifiers, current/comparative values, explicit generic totals, and all required HDFC-style component rows.
- Balance-sheet OCR repair uses an accounting identity only when the asset components and capital/liability total reconcile exactly; this repairs an obvious OCR digit corruption rather than inventing a financial value.
- LLM completion is now a second layer for missing required fields. Existing grounded OCR values are never overwritten. LLM values require source evidence that is present in the OCR text.
- Extraction output now includes `extraction_quality.required_fields_missing` and `manual_review_required`, allowing the dashboard to identify incomplete extraction separately from financial validation.
- Financial validation is period-aware for comparative P&L and cash-flow data and remains null-safe (`NOT_APPLICABLE` when required operands are unavailable).

## Verification
- Automated tests: **22 passed, 1 skipped**.
- The supplied scanned 2021 Balance Sheet was reprocessed after the changes: required HDFC-style components and both totals reconcile with **PASS**.
- The remaining `total_liabilities` and `total_equity` fields are left `null` because the HDFC-style source presents `Total Capital & Liabilities`, not separate labels for those fields.

## Dataset limitation
The supplied real source dataset contains Balance Sheet PDFs. The repository's invoice/P&L/cash-flow examples are sample structured outputs rather than equivalent real source documents. Therefore those three categories are covered by robust parser tests and sample schemas, but true source-to-output accuracy for them should be measured again when real invoice/P&L/cash-flow documents are available.
