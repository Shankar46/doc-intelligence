"""Native text extraction plus layout-aware OCR for scanned financial documents.

The OCR layer now keeps both readable page text and 2-D word coordinates.  The
coordinates are intentionally exposed to extraction so invoice tables can be
parsed by column position instead of flattening every number into one string.
"""
import io
import logging
import re
from statistics import median
from typing import Any

import fitz
import pytesseract
from PIL import Image, ImageOps, ImageFilter

from app.core.config import settings

logger = logging.getLogger(__name__)
if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

MIN_CHARS_FOR_NATIVE_TEXT = 20


class OCRResult:
    def __init__(self, pages: list[str], ocr_used: bool, layouts: list[list[dict[str, Any]]] | None = None):
        self.pages = pages
        self.ocr_used = ocr_used
        self.full_text = "\n".join(pages)
        # One list of word boxes per page. Native PDF pages use PyMuPDF words;
        # scanned pages use Tesseract image_to_data output.
        self.layouts = layouts or [[] for _ in pages]


def _ocr_score(text: str) -> int:
    if not text:
        return -999
    terms = re.findall(
        r"\b(?:invoice|subtotal|tax|gst|vat|total|balance|assets|liabilities|capital|deposits|borrowings|revenue|income|profit|expenditure|cash|operating|investing|financing)\b",
        text,
        re.I,
    )
    lines = [x for x in text.splitlines() if x.strip()]
    numeric = sum(bool(re.search(r"\d", x)) for x in lines)
    paired = sum(len(re.findall(r"\d[\d,.() -]*", x)) >= 2 for x in lines)
    financial_rows = sum(
        bool(re.search(r"(?:invoice|subtotal|tax|gst|vat|total|assets|liabilities|capital|reserves|deposits|borrowings|investments|advances|revenue|income|profit|expenditure|operating|investing|financing|cash)", line, re.I))
        and bool(re.search(r"\d", line))
        for line in lines
    )
    score = len(terms) * 5 + numeric + paired * 10 + financial_rows * 20 + min(len(lines), 100) // 5
    if re.search(r"invoice|bill|subtotal|grand\s+total", text, re.I):
        score += 60
    if re.search(r"balance\s+sheet|capital\s+and\s+liabilities", text, re.I):
        score += 80
    if re.search(r"cash\s+flow", text, re.I):
        score += 50
    if re.search(r"profit\s+and\s+loss|statement\s+of\s+profit", text, re.I):
        score += 50
    return score


def _normalise_word(word: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(word.get("text", "")).strip(),
        "left": float(word.get("left", 0)),
        "top": float(word.get("top", 0)),
        "width": float(word.get("width", 0)),
        "height": float(word.get("height", 0)),
        "confidence": float(word.get("confidence", -1)),
    }


def _data_to_layout(data: dict[str, list[Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    count = len(data.get("text", []))
    for i in range(count):
        text = str(data["text"][i] or "").strip()
        if not text:
            continue
        conf_raw = data.get("conf", [-1] * count)[i]
        try:
            conf = float(conf_raw)
        except (TypeError, ValueError):
            conf = -1.0
        words.append(_normalise_word({
            "text": text,
            "left": data.get("left", [0] * count)[i],
            "top": data.get("top", [0] * count)[i],
            "width": data.get("width", [0] * count)[i],
            "height": data.get("height", [0] * count)[i],
            "confidence": conf,
        }))
    return words


def _layout_to_text(words: list[dict[str, Any]]) -> str:
    """Reconstruct readable lines from 2-D OCR words.

    We group by baseline first and only then sort horizontally.  This avoids
    the old character-stream behaviour while preserving the complete text.
    """
    if not words:
        return ""
    heights = [w["height"] for w in words if w["height"] > 0]
    tolerance = max(5.0, (median(heights) if heights else 12.0) * 0.65)
    rows: list[list[dict[str, Any]]] = []
    row_y: list[float] = []
    for word in sorted(words, key=lambda w: (w["top"], w["left"])):
        cy = word["top"] + word["height"] / 2
        match = None
        for idx, y in enumerate(row_y):
            if abs(cy - y) <= tolerance:
                match = idx
                break
        if match is None:
            row_y.append(cy)
            rows.append([word])
        else:
            rows[match].append(word)
            row_y[match] = sum(w["top"] + w["height"] / 2 for w in rows[match]) / len(rows[match])
    ordered = sorted(zip(row_y, rows), key=lambda x: x[0])
    return "\n".join(" ".join(w["text"] for w in sorted(row, key=lambda x: x["left"])) for _, row in ordered)


def _ocr_candidate(img: Image.Image, psm: int) -> tuple[str, list[dict[str, Any]]]:
    try:
        data = pytesseract.image_to_data(img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
        layout = _data_to_layout(data)
        return _layout_to_text(layout), layout
    except Exception as exc:
        logger.warning("Tesseract PSM %s failed: %s", psm, exc)
        return "", []


def _ocr_best(img: Image.Image) -> tuple[str, list[dict[str, Any]]]:
    """Fast adaptive OCR with layout data attached to the selected result."""
    primary_text, primary_layout = _ocr_candidate(img, 6)
    if primary_text and _ocr_score(primary_text) >= 90:
        return primary_text, primary_layout

    alternate_text, alternate_layout = _ocr_candidate(img, 11)
    candidates = [(x, y, _ocr_score(x)) for x, y in ((primary_text, primary_layout), (alternate_text, alternate_layout)) if x]
    best_score = max((score for _, _, score in candidates), default=-999)
    if best_score < 70:
        try:
            gray = ImageOps.grayscale(img)
            gray = ImageOps.autocontrast(gray).filter(ImageFilter.SHARPEN)
            retry_text, retry_layout = _ocr_candidate(gray, 6)
            if retry_text:
                candidates.append((retry_text, retry_layout, _ocr_score(retry_text) + 3))
        except Exception as exc:
            logger.warning("Preprocessed OCR failed: %s", exc)
    return max(candidates, key=lambda x: x[2])[:2] if candidates else ("", [])


def _native_layout(page: fitz.Page) -> list[dict[str, Any]]:
    # PyMuPDF returns x0,y0,x1,y1,text,block,line,word.
    words = []
    for item in page.get_text("words"):
        if len(item) < 5 or not str(item[4]).strip():
            continue
        x0, y0, x1, y1, text = item[:5]
        words.append(_normalise_word({
            "text": text,
            "left": x0,
            "top": y0,
            "width": x1 - x0,
            "height": y1 - y0,
            "confidence": 100,
        }))
    return words


def _extract_from_pdf(file_bytes: bytes) -> OCRResult:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages, layouts, used = [], [], False
    for page_index in range(doc.page_count):
        page = doc[page_index]
        native = page.get_text().strip()
        native_layout = _native_layout(page)
        if len(native) >= MIN_CHARS_FOR_NATIVE_TEXT:
            pages.append(native)
            layouts.append(native_layout)
        else:
            pix = page.get_pixmap(dpi=200, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text, layout = _ocr_best(img)
            pages.append(text)
            layouts.append(layout)
            used = True
    return OCRResult(pages, used, layouts)


def _extract_from_image(file_bytes: bytes) -> OCRResult:
    img = Image.open(io.BytesIO(file_bytes))
    text, layout = _ocr_best(img)
    return OCRResult([text], True, [layout])


def extract_text(file_bytes: bytes, kind: str) -> OCRResult:
    return _extract_from_pdf(file_bytes) if kind == "pdf" else _extract_from_image(file_bytes)
