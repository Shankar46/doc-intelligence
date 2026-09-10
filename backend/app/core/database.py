"""
SQLAlchemy engine/session setup. SQLite by default (zero external
dependency) — swap DATABASE_URL to Postgres/MySQL without code changes.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import settings

# Ensure sqlite directory exists when using a local file DB
if settings.database_url.startswith("sqlite"):
    db_path = settings.database_url.split("///")[-1]
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create tables on startup. For a real prod app, use Alembic migrations instead."""
    from app.models import document  # noqa: F401 (ensures models are registered)
    Base.metadata.create_all(bind=engine)
