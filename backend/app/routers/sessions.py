"""Session management endpoints."""

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ..database import get_db
from ..models import Session

router = APIRouter(prefix="/sessions", tags=["sessions"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class SessionCreate(BaseModel):
    participant_id: str
    session_number: int = 1
    mode: str  # "dev", "behavioral", or "scanner"
    config_index: int


class SessionOut(BaseModel):
    session_id: int
    participant_id: str
    mode: str
    config_index: int

    model_config = {"from_attributes": True}


class SessionDetail(BaseModel):
    session_id: int
    participant_id: str
    session_number: int
    mode: str
    config_index: int
    created_at: str
    anchor_t_ms: float | None

    model_config = {"from_attributes": True}


class AnchorUpdate(BaseModel):
    anchor_t_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("", response_model=SessionOut)
def create_session(body: SessionCreate, db: DBSession = Depends(get_db)):
    session = Session(
        participant_id=body.participant_id,
        session_number=body.session_number,
        mode=body.mode,
        config_index=body.config_index,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionOut(
        session_id=session.id,
        participant_id=session.participant_id,
        mode=session.mode,
        config_index=session.config_index,
    )


@router.patch("/{session_id}/anchor", response_model=SessionOut)
def set_anchor(session_id: int, body: AnchorUpdate, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.anchor_t_ms = body.anchor_t_ms
    db.commit()
    db.refresh(session)
    return SessionOut(
        session_id=session.id,
        participant_id=session.participant_id,
        mode=session.mode,
        config_index=session.config_index,
    )


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: int, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDetail(
        session_id=session.id,
        participant_id=session.participant_id,
        session_number=session.session_number,
        mode=session.mode,
        config_index=session.config_index,
        created_at=session.created_at.isoformat(),
        anchor_t_ms=session.anchor_t_ms,
    )
