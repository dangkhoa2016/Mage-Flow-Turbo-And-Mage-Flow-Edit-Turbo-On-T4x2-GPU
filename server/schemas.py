from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    steps: int = Field(default=4, ge=4, le=4, strict=True)
    width: int = Field(default=1024, ge=1024, le=1024, multiple_of=16, strict=True)
    height: int = Field(default=1024, ge=1024, le=1024, multiple_of=16, strict=True)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must be a non-empty string")
        return value


class _WorkerOutputModel(BaseModel):
    id: str = Field(min_length=1)
    status: Literal["completed"]
    seed: int = Field(ge=0, le=2**32 - 1)
    elapsed_seconds: float = Field(ge=0.0)
    output: str = Field(min_length=1)


class GenerationResponse(_WorkerOutputModel):
    model: Literal["mage-flow-turbo"] = "mage-flow-turbo"
    device: Literal["cuda:0"] = "cuda:0"
    width: int = Field(default=1024, ge=1024, le=1024)
    height: int = Field(default=1024, ge=1024, le=1024)


class EditResponse(_WorkerOutputModel):
    model: Literal["mage-flow-edit-turbo"] = "mage-flow-edit-turbo"
    device: Literal["cuda:1"] = "cuda:1"