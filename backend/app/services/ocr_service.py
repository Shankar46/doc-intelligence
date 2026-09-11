"""Native PDF text extraction with adaptive Tesseract OCR for scanned pages."""
import io
import logging
import re
import fitz
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from app.core.config import settings

logger = logging.getLogger(__name__)
if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

MIN_CHARS_FOR_NATIVE_TEXT = 20


class OCRResult:
    def __init__(self, pages: list[str], ocr_used: bool):
        self.pages = pages
        self.ocr_used = ocr_used
        self.full_text = "\n".join(pages)


def extract_text(file_bytes: bytes, kind: str) -> OCRResult:
    if kind == "pdf":
        return _extract_from_pdf(file_bytes)
    return _extract_from_image(file_bytes)


def _ocr_best(img: Image.Image) -> str:
    """Fast adaptive OCR for financial documents.

    Start with the table-friendly PSM 11 pass. Only run extra OCR passes when
    the first pass is clearly weak; this avoids spending minutes running four
    full-page Tesseract passes on every scanned document.
    """
    # 250 DPI is normally sufficient for clean financial scans and is materially
    # faster than 300 DPI while keeping small table text readable.
    candidates = []

    def run(psm: int, image: Image.Image) -> str:
        try:
            return pytesseract.image_to_string(image, config=f"--psm {psm}").strip()
        except Exception as exc:
            logger.warning("Tesseract PSM %s failed: %s", psm, exc)
            return ""

    # Best first pass for sparse/table-heavy annual-report pages.
    first = run(11, img)
    if first:
        score = _ocr_score(first)
        candidates.append((first, score))
        # A strong financial-table OCR result is good enough; do not perform
        # several expensive fallback passes unnecessarily.
        if score >= 120 and re.search(r"(?:total|assets|liabilities|invoice|revenue|profit|cash)", first, re.I):
            return first

    # One conventional layout pass helps when PSM 11 fragments a table.
    second = run(6, img)
    if second:
        candidates.append((second, _ocr_score(second)))
        if _ocr_score(second) >= 120:
            return second

    # Only preprocess when the normal passes were weak.
    try:
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray).filter(ImageFilter.SHARPEN)
        processed = run(11, gray)
        if processed:
            candidates.append((processed, _ocr_score(processed) + 2))
    except Exception as exc:
        logger.warning("Preprocessed OCR failed: %s", exc)

    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[1])[0]

def _ocr_score(text: str) -> int:
    if not text:
        return -999
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    financial_terms = re.findall(
        r"\b(?:total|capital|assets|liabilities|deposits|borrowings|investments|advances|revenue|tax|profit|cash|invoice|subtotal|gst|vat)\b",
        text, re.I,
    )
    numeric_lines = sum(1 for line in lines if len(re.findall(r"\d[\d, .]*", line)) >= 1)
    paired_numeric_lines = sum(1 for line in lines if len(re.findall(r"\d[\d, .]*", line)) >= 2)
    score = len(financial_terms) * 5 + numeric_lines + paired_numeric_lines * 8 + min(len(lines), 80) // 10
    if re.search(r"consolidated\s+balance\s+sheet", text, re.I): score += 100
    if re.search(r"capital\s+and\s+liabilities", text, re.I): score += 60
    if re.search(r"invoice|subtotal|amount\s+due", text, re.I): score += 50
    return score


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
        logger.info("Page %d has little native text; using adaptive Tesseract OCR", page_index + 1)
        pix = page.get_pixmap(dpi=250, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages_text.append(_ocr_best(img))
        ocr_used = True
    return OCRResult(pages=pages_text, ocr_used=ocr_used)


def _extract_from_image(file_bytes: bytes) -> OCRResult:
    img = Image.open(io.BytesIO(file_bytes))
    return OCRResult(pages=[_ocr_best(img)], ocr_used=True)
