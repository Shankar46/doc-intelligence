"""
Persistence model for a processed document. We store the full
structured response as JSON so the API contract and the DB row never
drift apart — the DB is effectively a versioned cache of API responses.
"""
from sqlalchemy import Column, Integer, String, DateTime, JSON, func

from app.core.database import Base


class ProcessedDocument(Base):
    __tablename__ = "processed_documents"

    id = Column(Integer, primary_key=True, index=True)
    document_name = Column(String, index=True, nullable=False)
    document_type = Column(String, nullable=False)
    processing_status = Column(String, nullable=False)  # PASS | FAILED
    result_json = Column(JSON, nullable=False)  # full structured response
    created_at = Column(DateTime(timezone=True), server_default=func.now())
