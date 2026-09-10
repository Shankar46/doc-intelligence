"""
Persistence layer. Nothing in here knows about HTTP, OCR, or LLMs —
it only knows how to save/fetch ProcessedDocument rows. Keeping this
isolated makes it trivial to swap SQLite for Postgres later.
"""
import logging
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.document import ProcessedDocument

logger = logging.getLogger(__name__)


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_result(self, document_name: str, document_type: str,
                     processing_status: str, result_json: dict) -> ProcessedDocument:
        record = ProcessedDocument(
            document_name=document_name,
            document_type=document_type,
            processing_status=processing_status,
            result_json=result_json,
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        logger.info("Saved processing result for '%s' (status=%s)", document_name, processing_status)
        return record

    def get_latest_by_name(self, document_name: str) -> ProcessedDocument | None:
        return (
            self.db.query(ProcessedDocument)
            .filter(ProcessedDocument.document_name == document_name)
            .order_by(desc(ProcessedDocument.created_at))
            .first()
        )

    def list_latest_per_document(self) -> list[ProcessedDocument]:
        """Returns the latest record for every distinct document_name.

        Simple approach for SQLite: pull all rows ordered by recency and
        keep the first occurrence of each name. Fine at assignment scale;
        for larger datasets use a window function / GROUP BY subquery.
        """
        rows = (
            self.db.query(ProcessedDocument)
            .order_by(desc(ProcessedDocument.created_at))
            .all()
        )
        seen = set()
        latest = []
        for row in rows:
            if row.document_name not in seen:
                latest.append(row)
                seen.add(row.document_name)
        return latest
