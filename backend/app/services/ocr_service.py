"""
Text extraction. Strategy:
  1. Native PDF -> pull text directly via PyMuPDF (fast, exact, no OCR errors).
  2. If a PDF page has ~no extractable text (i.e. it's a scanned image) or
     the input is a JPG/PNG -> rasterize/open as image and run Tesseract OCR.

Returns per-page text plus a flag telling the caller whether OCR was used,
which is required in `processing_metadata.ocr_used`.
"""
import io
import logging
import fitz
import pytesseract
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

MIN_CHARS_FOR_NATIVE_TEXT = 20  # below this, assume the page is scanned


class OCRResult:
    def __init__(self, pages: list[str], ocr_used: bool):
        self.pages = pages          # list of text, index 0 = page 1
        self.ocr_used = ocr_used
        self.full_text = "\n".join(pages)


def extract_text(file_bytes: bytes, kind: str) -> OCRResult:
    if kind == "pdf":
        return _extract_from_pdf(file_bytes)
    return _extract_from_image(file_bytes)


def _extract_from_pdf(file_bytes: bytes) -> OCRResult:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages_text = []
    ocr_used = False

    for page_index in range(doc.page_count):
        page = doc[page_index]
        native_text = page.get_text().strip()

        if len(native_text) >= MIN_CHARS_FOR_NATIVE_TEXT:
            pages_text.append(native_text)
            continue

        # Likely a scanned page -> rasterize and OCR it
        logger.info("Page %d has little native text, falling back to OCR", page_index + 1)
        pix = page.get_pixmap(dpi=250)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text = pytesseract.image_to_string(img)
        pages_text.append(ocr_text.strip())
        ocr_used = True

    return OCRResult(pages=pages_text, ocr_used=ocr_used)


def _extract_from_image(file_bytes: bytes) -> OCRResult:
    img = Image.open(io.BytesIO(file_bytes))
    text = pytesseract.image_to_string(img)
    return OCRResult(pages=[text.strip()], ocr_used=True)
