from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ServiceState:
    t2i_ready: bool = False
    edit_ready: bool = False
    coordinator_ready: bool = True

    @property
    def ready(self) -> bool:
        return self.coordinator_ready and self.t2i_ready and self.edit_ready

    def as_dict(self) -> dict:
        return {**asdict(self), "ready": self.ready}
