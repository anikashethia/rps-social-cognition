"""Scanner TR trigger logging endpoints."""

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ..database import get_db
from ..models import Session, Trigger

router = APIRouter(prefix="/sessions/{session_id}/triggers", tags=["triggers"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class TriggerCreate(BaseModel):
    tr_number: int
    t_ms: float


class TriggerOut(BaseModel):
    id: int
    session_id: int
    tr_number: int
    t_ms: float

    model_config = {"from_attributes": True}


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("", response_model=TriggerOut)
def create_trigger(
    session_id: int, body: TriggerCreate, db: DBSession = Depends(get_db)
):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    trigger = Trigger(session_id=session_id, **body.model_dump())
    db.add(trigger)
    db.commit()
    db.refresh(trigger)
    return trigger
