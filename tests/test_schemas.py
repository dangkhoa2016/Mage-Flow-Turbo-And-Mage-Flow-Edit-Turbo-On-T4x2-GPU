import pytest
from pydantic import ValidationError
from server.schemas import GenerationRequest


def test_generation_defaults():
    request = GenerationRequest(prompt='hello')
    assert request.seed == 42
    assert request.steps == 4
    assert request.width == 1024
    assert request.height == 1024


def test_generation_rejects_empty_prompt():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt='')


def test_generation_requires_multiple_of_16():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt='hello', width=1001)


def test_generation_restricts_to_proven_steps():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt='hello', steps=8)
    with pytest.raises(ValidationError):
        GenerationRequest(prompt='hello', steps=100)


def test_generation_restricts_to_proven_resolution():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt='hello', width=2048, height=2048)
    with pytest.raises(ValidationError):
        GenerationRequest(prompt='hello', width=256, height=256)


def test_generation_rejects_whitespace_only_prompt():
    with pytest.raises(ValidationError, match="non-empty"):
        GenerationRequest(prompt='   ')
    with pytest.raises(ValidationError, match="non-empty"):
        GenerationRequest(prompt='\t\n')
    with pytest.raises(ValidationError, match="non-empty"):
        GenerationRequest(prompt='  \n  \t  ')
