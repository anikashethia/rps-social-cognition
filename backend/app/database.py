"""SQLite database connection and session factory."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./rps.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_all() -> None:
    """Create all tables if they don't exist, then apply incremental migrations."""
    Base.metadata.create_all(bind=engine)
    _migrate()


def _migrate() -> None:
    """Add new columns to existing tables that predate them."""
    with engine.connect() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(trials)")).fetchall()}
        if "level" not in cols:
            conn.execute(text("ALTER TABLE trials ADD COLUMN level INTEGER"))
            conn.commit()


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
