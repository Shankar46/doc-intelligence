# Extraction Audit v11 — Dynamic Layout-Aware Fixes

## Changes

1. **2-D OCR data retained**
   - Tesseract now uses `image_to_data` and preserves word coordinates/confidence.
   - Native PDF pages also expose PyMuPDF word coordinates when available.
   - OCR text is reconstructed from spatial rows rather than relying only on a flattened character stream.

2. **Layout-aware invoice table extraction**
   - Detects Description, Quantity, Unit Price/Rate, and Amount/Extended columns from the actual table header.
   - Numeric tokens physically located in the description column remain in the description, including product IDs, measurements, and pack sizes.
   - Wrapped description rows can be carried into the next numeric row.
   - Generic arithmetic fallback remains available for unusual layouts.

3. **Holistic invoice metadata discovery**
   - LLM instructions explicitly require searching the complete page, including top-right headers and bottom/footer areas.
   - `Invoice Details: 6825` and similar identifier labels are treated as invoice metadata, not money.

4. **Shipping/handling support**
   - `InvoiceExtractedData.shipping_and_handling` is now an explicit canonical schema field.
   - Invoice total validation uses `subtotal + tax + shipping/handling - discount`.

5. **Processing vs. validation status**
   - A successfully readable and parsed document now receives `processing_status = PASS`.
   - Arithmetic discrepancies remain visible in `validation.checks` and `validation.overall_status = FAIL`.
   - File/OCR/extraction exceptions can still produce `processing_status = FAILED`.

6. **Regression coverage**
   - 35 tests pass and 1 optional LLM test is skipped locally.
   - Added a synthetic 2-D invoice layout regression proving embedded numbers such as `570`, `12oz`, and `100/BX` do not become quantity/price values.
   - Added shipping formula and processing/validation separation tests.

## Important limitation

The actual `batch2-0499.jpg` image is not included in the v10 source archive used for this revision, so its real OCR output cannot be re-run end-to-end here. The reported failure mode is covered by a coordinate-aware regression test; the actual image should still be used for final acceptance testing.
