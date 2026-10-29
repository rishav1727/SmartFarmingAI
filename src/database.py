import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    create_engine, Column, String, Boolean,
    Float, Integer, Text, DateTime, func
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ── Connection string ─────────────────────────────────────────────────────────
# Production  → set DATABASE_URL env var (PostgreSQL from Supabase / Render / Railway)
# Local dev   → falls back to SQLite file in project root
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./smartfarming.db")

# Render / Railway give postgres:// or postgresql:// — SQLAlchemy 2.0 defaults to psycopg (v3).
# Explicitly use postgresql+psycopg2:// to use psycopg2-binary reliably.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

if "neon.tech" in DATABASE_URL or "postgresql" in DATABASE_URL:
    db_host = DATABASE_URL.split("@")[-1].split("/")[0] if "@" in DATABASE_URL else "PostgreSQL"
    print(f"[Database] Active backend: Neon Cloud PostgreSQL ({db_host})")
else:
    print("[Database] Active backend: Local SQLite")

_connect_args = {"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class Contribution(Base):
    """Community-submitted labeled image for model retraining."""
    __tablename__ = "contributions"

    id               = Column(String(36),  primary_key=True, default=lambda: str(uuid.uuid4()))
    image_url        = Column(Text,         nullable=False)   # Public URL (Supabase / local)
    storage_path     = Column(Text)                           # Internal storage path / key
    label            = Column(String(200),  nullable=False)   # Disease class label
    crop_type        = Column(String(100))                    # Optional crop hint
    contributor_name = Column(String(200))                    # Optional
    contributor_email= Column(String(200))                    # Optional
    verified         = Column(Boolean,      default=False)    # Admin-approved label
    used_in_training = Column(Boolean,      default=False)    # Already included in a run
    notes            = Column(Text)                           # Admin notes
    submitted_at     = Column(DateTime,     default=datetime.utcnow)


class ModelVersion(Base):
    """Tracks every model training run and its performance metrics."""
    __tablename__ = "model_versions"

    id                  = Column(Integer,      primary_key=True, autoincrement=True)
    version             = Column(String(50),   nullable=False)   # e.g. "v1.0", "v2.1"
    accuracy            = Column(Float)                           # Val accuracy %
    val_loss            = Column(Float)
    trained_on_samples  = Column(Integer)                         # Total training images used
    new_contributions   = Column(Integer)                         # Community images in this run
    hf_repo             = Column(String(200))                     # HuggingFace repo  (optional)
    weights_filename    = Column(String(200))                     # "best_model.pth"
    is_active           = Column(Boolean,      default=False)     # Currently deployed
    notes               = Column(Text)
    created_at          = Column(DateTime,     default=datetime.utcnow)


# ── Helpers ───────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables (safe to call on every startup)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
