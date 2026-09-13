from __future__ import annotations

from typing import Protocol


class T2IWorker(Protocol):
    ready: bool
    device: str

    def generate(self, *, prompt: str, seed: int, steps: int, width: int, height: int) -> dict:
        ...


class EditWorker(Protocol):
    ready: bool
    device: str

    def edit(self, *, image_bytes: bytes, prompt: str, seed: int) -> dict:
        ...
