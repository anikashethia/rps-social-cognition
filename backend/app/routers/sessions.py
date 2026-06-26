"""Session management endpoints."""

import json
import os

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ..database import get_db
from ..models import ParticipantRegistration, Session

_ROTATION_FILE = os.path.join(os.path.dirname(__file__), "../rotations/rotation.json")

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


class ResolvedBlock(BaseModel):
    avatar_id: str
    condition: str
    level: int


class SessionRotation(BaseModel):
    blocks: list[ResolvedBlock]


@router.get("/{session_id}/rotation", response_model=SessionRotation)
def get_session_rotation(session_id: int, db: DBSession = Depends(get_db)):
    """
    Return the resolved block schedule for this session.
    Combines the config's (condition, level) ordering with the participant's
    registered friendly/neutral avatar IDs.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    reg = db.get(ParticipantRegistration, session.participant_id)
    if reg is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Participant '{session.participant_id}' has no avatar registration. "
                "Register them at POST /api/participants/{id}/registration before starting."
            ),
        )

    with open(_ROTATION_FILE) as f:
        rotation_data = json.load(f)

    config = rotation_data.get("configs", {}).get(str(session.config_index))
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Config index {session.config_index} not found in rotation table",
        )

    avatar_map = {
        "friendly": reg.friendly_avatar_id,
        "neutral": reg.neutral_avatar_id,
    }

    blocks = [
        ResolvedBlock(
            avatar_id=avatar_map[b["condition"]],
            condition=b["condition"],
            level=b["level"],
        )
        for b in config["blocks"]
    ]
    return SessionRotation(blocks=blocks)
