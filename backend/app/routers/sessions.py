"""Session management endpoints."""

import random

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from ..database import get_db
from ..models import ParticipantRegistration, Session

router = APIRouter(prefix="/sessions", tags=["sessions"])


# ── Block order generation ────────────────────────────────────────────────────


def generate_block_order(session_id: int) -> list[dict]:
    """
    Generate a random 6-block order, seeded from session_id for reproducibility.

    Matches Buergi et al.'s mn_RPS_config.m constraints:
      - All 6 (condition, level) combinations appear exactly once
      - First block is not level 2 (participants learn structure before hardest level)
      - No two consecutive blocks share the same level

    Uses rejection sampling; typically resolves in <5 iterations.
    """
    rng = random.Random(session_id)
    all_blocks = [
        {"condition": cond, "level": lvl}
        for cond in ("friendly", "neutral")
        for lvl in (0, 1, 2)
    ]
    while True:
        blocks = all_blocks.copy()
        rng.shuffle(blocks)
        if blocks[0]["level"] >= 2:
            continue
        if any(blocks[i]["level"] == blocks[i + 1]["level"] for i in range(5)):
            continue
        return blocks


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class SessionCreate(BaseModel):
    participant_id: str
    session_number: int = 1
    mode: str  # "dev", "behavioral", or "scanner"
    config_index: int = 1  # stored for reference; block order is generated from session_id


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
    Return the resolved 6-block schedule for this session.

    Block order is generated randomly from session_id (reproducible, unique per session),
    matching Buergi et al.'s mn_RPS_config.m constraints. Avatar IDs are resolved from
    the participant's IOS-based registration.
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
                "Register them at PUT /api/participants/{id}/registration before starting."
            ),
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
        for b in generate_block_order(session_id)
    ]
    return SessionRotation(blocks=blocks)
