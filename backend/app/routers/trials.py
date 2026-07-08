"""Trial logging endpoints."""

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ..database import get_db
from ..models import Session, Trial

router = APIRouter(prefix="/sessions/{session_id}/trials", tags=["trials"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class TrialCreate(BaseModel):
    block: int
    agent: str
    trial_in_block: int
    trial_global: int
    participant_choice: int | None = None
    agent_choice: int
    outcome: str
    points_delta: int
    points_cumulative: int
    rt_ms: float | None = None
    onset_ms: float
    feedback_onset_ms: float
    iti_duration_ms: float
    block_onset_ms: float
    condition: str  # "friendly" or "neutral"
    level: int | None = None  # CHASE level (0, 1, or 2)
    is_noisy: bool | None = None
    bot_attr_r: float | None = None
    bot_attr_p: float | None = None
    bot_attr_s: float | None = None
    p_attr_r: float | None = None
    p_attr_p: float | None = None
    p_attr_s: float | None = None


class TrialOut(BaseModel):
    id: int
    session_id: int
    block: int
    agent: str
    trial_in_block: int
    trial_global: int
    participant_choice: int | None
    agent_choice: int
    outcome: str
    points_delta: int
    points_cumulative: int
    rt_ms: float | None
    onset_ms: float
    feedback_onset_ms: float
    iti_duration_ms: float
    block_onset_ms: float
    condition: str
    level: int | None
    is_noisy: bool | None
    bot_attr_r: float | None
    bot_attr_p: float | None
    bot_attr_s: float | None
    p_attr_r: float | None
    p_attr_p: float | None
    p_attr_s: float | None

    model_config = {"from_attributes": True}


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("", response_model=TrialOut)
def create_trial(
    session_id: int, body: TrialCreate, db: DBSession = Depends(get_db)
):
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    trial = Trial(session_id=session_id, **body.model_dump())
    db.add(trial)
    db.commit()
    db.refresh(trial)
    return trial
