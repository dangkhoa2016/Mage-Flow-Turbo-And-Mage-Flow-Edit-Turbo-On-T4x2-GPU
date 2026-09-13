"""Runtime safety call-site wiring and audit hook for the public Mage-Flow server.

Routes the real Mage inference call-sites (``model.txt_enc.screen_text`` /
``model.txt_enc.screen_edit``) to the GGUF safety adapter
(``server/safety/gguf_safety.py``) and emits one JSONL audit record per real
screen call to the path in ``MAGE_SAFETY_AUDIT``.

Scope guarantees:
  - Conditioning / diffusion / denoise / scheduler / VAE math: UNTOUCHED.
  - Frozen policy strings and the production verdict parser: UNTOUCHED.
  - Prompt construction used for Mage conditioning: UNTOUCHED.
  - HF/PyTorch conditioning encoder: UNTOUCHED (never reloaded, never replaced).
  - No HF safety fallback exists: the GGUF adapter fail-closes to BLOCK on any
    transport / HTTP / schema / parser / vision error.
  - Seed handling and image preprocessing: untouched (screener is side-effect free).
  - The worker refuses to start unless MAGE_SAFETY_BACKEND is exactly 'gguf'.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from server.safety.gguf_safety import GGUFSafetyAdapter

_AUDIT_PATH = os.environ.get("MAGE_SAFETY_AUDIT", "").strip()
_audit_lock = threading.Lock()
_local = threading.local()


def set_current_request_id(request_id: str) -> None:
    """Tie audit records to the real worker request currently being processed."""
    _local.request_id = request_id


def clear_current_request_id() -> None:
    _local.request_id = None


def _audit_path() -> str:
    return os.environ.get("MAGE_SAFETY_AUDIT", _AUDIT_PATH).strip()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class _AuditScreener:
    """Wraps a GGUFSafetyAdapter so every real screen call is audited."""

    def __init__(self, adapter: GGUFSafetyAdapter, request_type: str):
        self._adapter = adapter
        self._request_type = request_type

    def screen_text(self, prompt: str, max_new_tokens: int = 160) -> Any:
        start = time.time()
        verdict = self._adapter.screen_text(prompt, max_new_tokens=max_new_tokens)
        self._audit("screen_text", "TEXT", prompt, verdict, start)
        return verdict

    def screen_edit(self, prompt: str, ref_images: Any, max_new_tokens: int = 192) -> Any:
        start = time.time()
        verdict = self._adapter.screen_edit(prompt, ref_images, max_new_tokens=max_new_tokens)
        self._audit("screen_edit", "MULTIMODAL", prompt, verdict, start)
        return verdict

    def _audit(self, call_name: str, modal: str, prompt: str, verdict: Any, start: float) -> None:
        end = time.time()
        counters = self._adapter.counters()
        entry = {
            "request_id": getattr(_local, "request_id", None),
            "request_type": self._request_type,
            "safety_backend": "gguf",
            "gguf_server_url": self._adapter.client.base_url,
            "call": call_name,
            "modal": modal,
            "safety_call_count": counters["GGUF_CALLS"],
            "hf_fallback_call_count": counters["HF_FALLBACK_CALLS"],
            "fail_closed_count": counters["FAIL_CLOSED_COUNT"],
            "safety_start_utc": _utc_now(),
            "safety_end_utc": _utc_now(),
            "safety_wall_seconds": round(end - start, 6),
            "raw_completion": getattr(verdict, "raw", ""),
            "normalized_verdict": "ALLOW" if not verdict.violates else "BLOCK",
            "categories": list(verdict.categories or []),
            "reason": verdict.reason,
        }
        line = json.dumps(entry, ensure_ascii=False)
        print(f"[SAFETY-AUDIT] {line}", flush=True)
        path = _audit_path()
        if path:
            with _audit_lock, open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


# --- non-semantic per-stage timing probes -----------------------------------
# Wrap vendor `mage_flow.pipeline` module-level stage boundaries at worker
# import time (runtime composition; NO vendor file edits, NO computation
# change). `_encode_texts_packed` / `_encode_edits_packed` measure conditioning,
# `_velocity` measures the denoise loop forwards (one call per denoise step),
# `_decode_one` measures the single VAE decode.

_TICKS: dict[str, float] = {}
_COUNTS: dict[str, int] = {}


def _make_probe(module: Any, name: str, kind: str) -> None:
    original = getattr(module, name, None)
    if original is None:
        return

    def probe(*args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            key = f"{kind}_{name}"
            _TICKS[key] = _TICKS.get(key, 0.0) + (time.monotonic() - t0)
            _COUNTS[key] = _COUNTS.get(key, 0) + 1

    probe.__name__ = original.__name__
    setattr(module, name, probe)


def install_timing_probes(request_type: str) -> None:
    import mage_flow.pipeline as mp

    kind = request_type.lower()
    for name in ("_encode_texts_packed", "_encode_edits_packed", "_velocity", "_decode_one"):
        _make_probe(mp, name, kind)
    print(
        f"[TIMING] probes installed request_type={request_type} wrapped=encode/velocity/decode (non-semantic)",
        flush=True,
    )


def timing_report(request_type: str) -> dict:
    """Return per-stage seconds + call counts; also prints a [TIMING] line."""
    kind = request_type.lower()
    report: dict[str, dict] = {"seconds": {}, "calls": {}}
    for name in ("_encode_texts_packed", "_encode_edits_packed", "_velocity", "_decode_one"):
        key = f"{kind}_{name}"
        if key in _TICKS:
            report["seconds"][name] = round(_TICKS[key], 3)
            report["calls"][name] = _COUNTS[key]
    print(f"[TIMING] report {report}", flush=True)
    return report


def install_and_audit_safety(pipeline: Any, request_type: str) -> GGUFSafetyAdapter:
    """Install the GGUF safety adapter on the real call-sites.

    Hard error if ``MAGE_SAFETY_BACKEND`` is anything other than
    ``gguf``: a silent HF safety fallback is forbidden.
    Returns the adapter so the worker can expose counters.
    """
    backend = os.environ.get("MAGE_SAFETY_BACKEND", "hf").strip().lower()
    if backend != "gguf":
        raise RuntimeError(
            "MAGE_SAFETY_BACKEND must be 'gguf' for the public server; "
            f"got {backend!r}; HF safety fallback is forbidden in GGUF mode"
        )
    adapter = GGUFSafetyAdapter.from_env()
    screener = _AuditScreener(adapter, request_type)
    text_encoder = pipeline.model.txt_enc
    text_encoder.screen_text = screener.screen_text
    text_encoder.screen_edit = screener.screen_edit
    print(
        f"[SAFETY] backend=gguf installed request_type={request_type} "
        f"url={adapter.client.base_url} audit={_audit_path() or '(stdout only)'}",
        flush=True,
    )
    return adapter
