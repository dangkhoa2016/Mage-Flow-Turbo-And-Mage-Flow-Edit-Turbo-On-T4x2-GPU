from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    steps: int = Field(default=4, ge=4, le=4)
    width: int = Field(default=1024, ge=1024, le=1024, multiple_of=16)
    height: int = Field(default=1024, ge=1024, le=1024, multiple_of=16)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must be a non-empty string")
        return value


class GenerationResponse(BaseModel):
    id: str
    status: Literal["completed"]
    model: Literal["mage-flow-turbo"] = "mage-flow-turbo"
    device: Literal["cuda:0"] = "cuda:0"
    seed: int
    width: int
    height: int
    elapsed_seconds: float
    output: str


class EditResponse(BaseModel):
    id: str
    status: Literal["completed"]
    model: Literal["mage-flow-edit-turbo"] = "mage-flow-edit-turbo"
    device: Literal["cuda:1"] = "cuda:1"
    seed: int
    elapsed_seconds: float
    output: str
