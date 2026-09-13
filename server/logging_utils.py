from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

BAR = "━" * 46


def stage_start(number: str, name: str) -> float:
    print(BAR, flush=True)
    print(f"[STAGE {number}] {name}", flush=True)
    print(BAR, flush=True)
    return time.perf_counter()


def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def log_pass(message: str) -> None:
    print(f"[PASS] {message}", flush=True)


def log_warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def log_fail(message: str) -> None:
    print(f"[FAIL] {message}", flush=True)


def heartbeat(started_at: float, state: str) -> None:
    elapsed = int(time.perf_counter() - started_at)
    print(f"[HEARTBEAT] elapsed={elapsed}s state={state}", flush=True)


def stage_end(started_at: float, result: str) -> None:
    elapsed = time.perf_counter() - started_at
    print(f"[PASS] {result}", flush=True)
    print(f"[TIME] {elapsed:.2f} s", flush=True)


@contextmanager
def timed_stage(number: str, name: str, result: str):
    started_at = stage_start(number, name)
    try:
        yield started_at
    except Exception as exc:
        log_fail(f"{result}: {type(exc).__name__}: {exc}")
        raise
    else:
        stage_end(started_at, result)
