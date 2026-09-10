"""
Input-control layer (spec section 4.1 / 3). This checks the file itself
is a supported, readable, non-corrupted PDF/JPG/PNG within the page
limit — it does NOT classify document type or look at content meaning.
"""
import logging
import fitz  # PyMuPDF
from PIL import Image
import io

from app.core.config import settings
from app.schemas.document import FileValidation

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/jpg": "image",
    "image/png": "image",
}


class DocumentValidationError(Exception):
    """Raised for a validation failure; carries a machine-readable code."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _detect_kind(content_type: str, filename: str) -> str:
    if content_type in SUPPORTED_MIME_TYPES:
        return SUPPORTED_MIME_TYPES[content_type]
    # fall back to extension since browsers/clients sometimes send generic content-types
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith((".jpg", ".jpeg", ".png")):
        return "image"
    raise DocumentValidationError(
        "UNSUPPORTED_FILE_TYPE", "Only PDF / JPG / PNG documents are supported."
    )


def validate_file(file_bytes: bytes, filename: str, content_type: str) -> FileValidation:
    """
    Runs all pre-extraction checks. Raises DocumentValidationError on
    failure (caught by the route and turned into a controlled error
    response) or returns a FileValidation with status=PASS.
    """
    if not file_bytes:
        raise DocumentValidationError("EMPTY_FILE", "The uploaded file is empty.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise DocumentValidationError(
            "FILE_TOO_LARGE", f"File exceeds the {settings.max_file_size_mb}MB limit."
        )

    kind = _detect_kind(content_type, filename)

    if kind == "pdf":
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page_count = doc.page_count
            if page_count == 0:
                raise DocumentValidationError("CORRUPTED_FILE", "PDF has no pages.")
        except DocumentValidationError:
            raise
        except Exception as exc:
            logger.warning("Failed to open PDF '%s': %s", filename, exc)
            raise DocumentValidationError("CORRUPTED_FILE", "PDF could not be read; it may be corrupted.")

        if page_count > settings.max_pages:
            raise DocumentValidationError(
                "PAGE_LIMIT_EXCEEDED",
                f"Document has {page_count} pages; the limit is {settings.max_pages}.",
            )

        return FileValidation(
            file_type="application/pdf",
            is_supported=True,
            is_readable=True,
            page_count=page_count,
            status="PASS",
        )

    # image
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # raises if truncated/corrupt
    except Exception as exc:
        logger.warning("Failed to open image '%s': %s", filename, exc)
        raise DocumentValidationError("CORRUPTED_FILE", "Image could not be read; it may be corrupted.")

    return FileValidation(
        file_type=content_type or "image",
        is_supported=True,
        is_readable=True,
        page_count=1,
        status="PASS",
    )
