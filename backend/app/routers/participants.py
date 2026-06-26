"""Participant registration endpoints (IOS-based avatar assignment)."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from ..database import get_db
from ..models import ParticipantRegistration

router = APIRouter(prefix="/participants", tags=["participants"])


class RegistrationCreate(BaseModel):
    friendly_avatar_id: str
    neutral_avatar_id: str


class RegistrationOut(BaseModel):
    participant_id: str
    friendly_avatar_id: str
    neutral_avatar_id: str
    registered_at: str

    model_config = {"from_attributes": True}


@router.put("/{participant_id}/registration", response_model=RegistrationOut)
def upsert_registration(
    participant_id: str,
    body: RegistrationCreate,
    db: DBSession = Depends(get_db),
):
    reg = db.get(ParticipantRegistration, participant_id)
    if reg is None:
        reg = ParticipantRegistration(participant_id=participant_id)
        db.add(reg)
    reg.friendly_avatar_id = body.friendly_avatar_id
    reg.neutral_avatar_id = body.neutral_avatar_id
    reg.registered_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(reg)
    return RegistrationOut(
        participant_id=reg.participant_id,
        friendly_avatar_id=reg.friendly_avatar_id,
        neutral_avatar_id=reg.neutral_avatar_id,
        registered_at=reg.registered_at.isoformat(),
    )


@router.get("/{participant_id}/registration", response_model=RegistrationOut)
def get_registration(participant_id: str, db: DBSession = Depends(get_db)):
    reg = db.get(ParticipantRegistration, participant_id)
    if reg is None:
        raise HTTPException(
            status_code=404,
            detail=f"No registration found for participant '{participant_id}'",
        )
    return RegistrationOut(
        participant_id=reg.participant_id,
        friendly_avatar_id=reg.friendly_avatar_id,
        neutral_avatar_id=reg.neutral_avatar_id,
        registered_at=reg.registered_at.isoformat(),
    )
