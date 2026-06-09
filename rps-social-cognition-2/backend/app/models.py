"""SQLAlchemy ORM models for the RPS social cognition task."""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    participant_id = Column(String, nullable=False)
    session_number = Column(Integer, nullable=False)
    mode = Column(String, nullable=False)  # "dev", "behavioral", or "scanner"
    config_index = Column(Integer, nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    anchor_t_ms = Column(Float, nullable=True)

    trials = relationship("Trial", back_populates="session")
    triggers = relationship("Trigger", back_populates="session")


class Trial(Base):
    __tablename__ = "trials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    block = Column(Integer, nullable=False)
    agent = Column(String, nullable=False)
    trial_in_block = Column(Integer, nullable=False)
    trial_global = Column(Integer, nullable=False)
    participant_choice = Column(Integer, nullable=True)
    agent_choice = Column(Integer, nullable=False)
    outcome = Column(String, nullable=False)
    points_delta = Column(Integer, nullable=False)
    points_cumulative = Column(Integer, nullable=False)
    rt_ms = Column(Float, nullable=True)
    onset_ms = Column(Float, nullable=False)
    iti_duration_ms = Column(Float, nullable=False)
    block_onset_ms = Column(Float, nullable=False)

    session = relationship("Session", back_populates="trials")


class Trigger(Base):
    __tablename__ = "triggers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    tr_number = Column(Integer, nullable=False)
    t_ms = Column(Float, nullable=False)

    session = relationship("Session", back_populates="triggers")
