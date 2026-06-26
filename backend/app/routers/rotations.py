"""Rotation config endpoints."""

import json
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/rotations", tags=["rotations"])

_ROTATION_FILE = os.path.join(os.path.dirname(__file__), "../rotations/rotation.json")


class BlockConfig(BaseModel):
    condition: str
    level: int


class RotationConfig(BaseModel):
    blocks: list[BlockConfig]


@router.get("/{config_index}", response_model=RotationConfig)
def get_rotation(config_index: int):
    with open(_ROTATION_FILE) as f:
        data = json.load(f)
    config = data.get("configs", {}).get(str(config_index))
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"Config index {config_index} not found in rotation table",
        )
    return RotationConfig(blocks=config["blocks"])
