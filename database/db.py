"""
db.py
SQLAlchemy ORM models and session management for PCB Smart Maintenance System.
Supports PostgreSQL for production and automatic fallback to SQLite for local development.
"""

from datetime import datetime
import json
import os
from pathlib import Path
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    equipment_id = Column(String(50), nullable=False, index=True)
    image_path = Column(String(255))
    defect_class = Column(String(50), nullable=False)
    bbox_coords = Column(Text)  # JSON string: [x1, y1, x2, y2]
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        coords = []
        if self.bbox_coords:
            try:
                coords = json.loads(self.bbox_coords)
            except Exception:
                coords = self.bbox_coords
        return {
            "id": self.id,
            "equipment_id": self.equipment_id,
            "image_path": self.image_path,
            "defect_class": self.defect_class,
            "bbox_coords": coords,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class RULEstimate(Base):
    __tablename__ = "rul_estimates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    equipment_id = Column(String(50), nullable=False, index=True)
    health_score = Column(Float, nullable=False)
    rul_cycles = Column(Integer, nullable=False)
    defect_count = Column(Integer, default=0)
    predicted_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "equipment_id": self.equipment_id,
            "health_score": round(self.health_score, 4),
            "rul_cycles": self.rul_cycles,
            "defect_count": self.defect_count,
            "predicted_at": self.predicted_at.isoformat() if self.predicted_at else None,
        }


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'source', 'adapted'
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    map_score = Column(Float)
    f1_score = Column(Float)
    inference_time_ms = Column(Float)
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "map_score": self.map_score,
            "f1_score": self.f1_score,
            "inference_time_ms": self.inference_time_ms,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
        }


_global_engine = None
_global_session_factory = None


def init_db(database_url: str = None):
    """
    Initializes database connection and creates tables.
    Defaults to DATABASE_URL env var or PostgreSQL.
    Falls back to SQLite if PostgreSQL connection fails.
    """
    global _global_engine, _global_session_factory

    if database_url is None:
        database_url = os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost/pcb_maintenance"
        )

    try:
        engine = create_engine(database_url, echo=False)
        # Verify connection
        with engine.connect() as conn:
            pass
        print(f"✓ Connected to database: {database_url.split('@')[-1]}")
    except Exception as e:
        print(f"Notice: Could not connect to primary database ({e}).")
        # Ensure local database folder exists
        db_dir = Path("database")
        db_dir.mkdir(parents=True, exist_ok=True)
        fallback_url = f"sqlite:///{db_dir.resolve() / 'pcb_maintenance.db'}"
        print(f"✓ Initializing local SQLite fallback database: {fallback_url}")
        engine = create_engine(fallback_url, echo=False, connect_args={"check_same_thread": False})

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    _global_engine = engine
    _global_session_factory = session_factory
    return session_factory


def get_db_session():
    """Returns a new database session"""
    global _global_session_factory
    if _global_session_factory is None:
        init_db()
    return _global_session_factory()


if __name__ == "__main__":
    session_maker = init_db()
    session = session_maker()
    print("Database tables initialized successfully.")
    session.close()

