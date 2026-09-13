"""Minimal GGUF (Qwen3-VL / llama.cpp) safety adapter.

GGUF safety integration contract:

- Reuses the EXACT production policy text and semantic system/user message
  structure (``mage_flow.models.modules.mage_text``) as far as the llama.cpp
  OpenAI-compatible chat API permits.
- Verdicts are normalized by the EXACT production parser
  (``_extract_json_object`` -> ``FilterVerdict``); a GGUF verdict therefore has
  the same allow/block semantics as the HF path.
- FAIL-CLOSED: any transport, HTTP, timeout, schema, parser, or vision error
  maps to the production blocked/reject outcome (``FilterVerdict(violates=True)``).
- GGUF mode NEVER calls the HF safety path; ``HF_FALLBACK_CALLS`` therefore
  stays 0 by construction (there is no fallback branch).
- Prompt/KV caching is not relied upon; explicitly disabled with
  ``cache_prompt=false`` per request.

Configuration surface:

    MAGE_SAFETY_BACKEND=hf|gguf
    MAGE_GGUF_SAFETY_URL=http://127.0.0.1:8081
    MAGE_GGUF_SAFETY_TIMEOUT_SECONDS=<finite timeout>
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import urllib.error
import urllib.request
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mage_flow.models.modules.mage_text import FilterVerdict

DEFAULT_URL = os.environ.get("MAGE_GGUF_SAFETY_URL", "http://127.0.0.1:8081")
DEFAULT_TIMEOUT_S = float(os.environ.get("MAGE_GGUF_SAFETY_TIMEOUT_SECONDS", "15"))


@lru_cache(maxsize=1)
def _mage_text_contract() -> tuple[str, str, Any, Any]:
    """Lazy-load the exact production Mage filters so CPU-only import stays safe.

    Returns ``(CONTENT_FILTER_EDIT_SYSTEM, CONTENT_FILTER_SYSTEM, FilterVerdict,
    _extract_json_object)``. ``mage_flow`` exists only inside the GPU runtime,
    so the contract is resolved on first real safety use, never at import time.
    """
    from mage_flow.models.modules.mage_text import (
        CONTENT_FILTER_EDIT_SYSTEM,
        CONTENT_FILTER_SYSTEM,
        FilterVerdict,
        _extract_json_object,
    )

    return (CONTENT_FILTER_EDIT_SYSTEM, CONTENT_FILTER_SYSTEM, FilterVerdict, _extract_json_object)


def _text_policy() -> str:
    return _mage_text_contract()[1]


def _edit_policy() -> str:
    return _mage_text_contract()[0]


def _b64_png_data_uri(image: Any) -> str:
    """Serialize a PIL image to a ``data:image/png;base64,...`` URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_gguf_messages(
    prompt: str,
    images: list[Any] | None = None,
    policy: str | None = None,
    edit: bool = False,
) -> list[dict[str, Any]]:
    """Build llama.cpp chat messages with the EXACT production system/user
    semantic structure and policy text.

    Text:  system=CONTENT_FILTER_SYSTEM,
           user  = "Prompt to classify:\n<prompt>"
    Edit:  system=CONTENT_FILTER_EDIT_SYSTEM,
           user  = [{"type":"image_url",...} xN, {"type":"text","text": the exact
                    production edit-instruction sentence}]
    """
    messages: list[dict[str, Any]]
    if images:
        user_content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": uri}} for uri in images]
        instruction = (prompt or "").strip() or "(no textual instruction)"
        user_content.append(
            {
                "type": "text",
                "text": (
                    f"There {'is' if len(images) == 1 else 'are'} {len(images)} source "
                    f"image(s) above. Edit instruction: {instruction}\n"
                    "Classify this edit request."
                ),
            }
        )
        messages = [
            {"role": "system", "content": policy or _edit_policy()},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "system", "content": policy or _text_policy()},
            {"role": "user", "content": f"Prompt to classify:\n{prompt}"},
        ]
    return messages


def _extract_completion_text(data: dict) -> str:
    """Extract the assistant text from an OpenAI-style chat completion.

    Raises ValueError when the expected content shape is missing (schema
    fault), so the caller can map it to the production BLOCK outcome.
    """
    try:
        choices = data["choices"]
        message = choices[0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"chat-completion response missing expected choices/message shape: {exc}") from exc
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    raise ValueError(f"chat-completion response content has unexpected shape: {type(content).__name__}")


class GGUFSafetyClient:
    """Thin, fail-closed HTTP client for the qualified llama-server."""

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_S,
        model_name: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self._model_name = model_name
        self._model_lock = threading.Lock()
        self.last_http_status: int | None = None

    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def model_name(self) -> str:
        if self._model_name:
            return self._model_name
        with self._model_lock:
            if self._model_name:
                return self._model_name
            try:
                with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=self.timeout_seconds) as resp:
                    payload = json.loads(resp.read().decode())
                self._model_name = payload["models"][0]["id"]
            except Exception:
                self._model_name = "qwen3-vl"
        return self._model_name

    def chat(self, messages: list[dict], max_new_tokens: int) -> tuple[str, int]:
        """POST a chat completion. Returns (assistant_text, http_status).

        Raises on transport/HTTP/schema failure; the caller maps to BLOCK.
        """
        request = urllib.request.Request(
            self.chat_url(),
            data=json.dumps(
                {
                    "model": self.model_name(),
                    "messages": messages,
                    "temperature": 0,  # mirrors HF do_sample=False (greedy)
                    "max_tokens": max_new_tokens,
                    "cache_prompt": False,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as resp:
                self.last_http_status = resp.status
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            self.last_http_status = exc.code
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            raise RuntimeError(f"llama-server HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            self.last_http_status = None
            raise RuntimeError(f"llama-server unreachable ({exc.reason!r}) on {self.chat_url()}") from exc
        except Exception as exc:
            self.last_http_status = None
            raise RuntimeError(f"llama-server transport/body failure: {type(exc).__name__}: {exc}") from exc
        return _extract_completion_text(payload), self.last_http_status


class GGUFSafetyAdapter:
    """Production-compatible safety screeners backed by the GGUF server.

    Exposes ``screen_text`` / ``screen_edit`` with the same signature and
    return type (``FilterVerdict``) as ``TextEncoder``, so the Mage call-site
    ``model.txt_enc.screen_text(...)`` can be routed here unchanged.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_S,
        model_name: str | None = None,
    ):
        self.client = GGUFSafetyClient(base_url=base_url, timeout_seconds=timeout_seconds, model_name=model_name)
        self._lock = threading.Lock()
        self.gguf_calls = 0
        self.hf_fallback_calls = 0
        self.fail_closed_count = 0

    def counters(self) -> dict:
        with self._lock:
            return {
                "GGUF_CALLS": self.gguf_calls,
                "HF_FALLBACK_CALLS": self.hf_fallback_calls,
                "FAIL_CLOSED_COUNT": self.fail_closed_count,
            }

    def last_http_status(self) -> int | None:
        return self.client.last_http_status

    def _parse_verdict(self, text: str, prefix: str, max_new_tokens: int) -> FilterVerdict:
        """Production parser + FAIL-CLOSED mapping. Never calls HF."""
        _edit_system, _system, verdict_type, extract_json = _mage_text_contract()
        try:
            parsed = extract_json(text)
            violates = bool(parsed.get("violates", False))
            cats = [c for c in (parsed.get("categories", []) or []) if isinstance(c, str)]
            reason = str(parsed.get("reason", "")).strip()
            return verdict_type(violates, cats, reason, text)
        except Exception as exc:
            with self._lock:
                self.fail_closed_count += 1
            return verdict_type(
                True,
                ["policy"],
                f"{prefix} unparseable verdict (blocked): {type(exc).__name__}: {exc}",
                text,
            )

    def raw_screen_messages(self, messages: list[dict], max_new_tokens: int) -> FilterVerdict:
        """Screen with caller-supplied messages (used by fault injection)."""
        prefix = "gguf safety error"
        with self._lock:
            self.gguf_calls += 1
        try:
            text, _status = self.client.chat(messages, max_new_tokens=max_new_tokens)
        except Exception as exc:
            with self._lock:
                self.fail_closed_count += 1
            _edit_system, _system, verdict_type, _extract_json = _mage_text_contract()
            return verdict_type(True, ["policy"], f"{prefix} (blocked): {type(exc).__name__}: {exc}", "")
        return self._parse_verdict(text, prefix, max_new_tokens)

    def screen_text(self, prompt: str, max_new_tokens: int = 160) -> FilterVerdict:
        messages = make_gguf_messages(prompt=prompt, edit=False)
        return self.raw_screen_messages(messages, max_new_tokens=max_new_tokens)

    def screen_edit(self, prompt: str, ref_images: Any, max_new_tokens: int = 192) -> FilterVerdict:
        from PIL import Image

        pils = [ref_images] if isinstance(ref_images, Image.Image) else list(ref_images)
        pils = [p.convert("RGB") for p in pils if p is not None]
        if not pils:
            # Production screen_edit falls back to screen_text when no image is
            # given; the GGUF mode fallback is still GGUF (never HF).
            return self.screen_text(prompt, max_new_tokens=max_new_tokens)
        uris = [_b64_png_data_uri(p.pil_image if hasattr(p, "pil_image") else p) for p in pils]
        messages = make_gguf_messages(prompt=prompt, images=uris, edit=True)
        return self.raw_screen_messages(messages, max_new_tokens=max_new_tokens)

    @classmethod
    def from_env(cls) -> GGUFSafetyAdapter:
        return cls(
            base_url=os.environ.get("MAGE_GGUF_SAFETY_URL", DEFAULT_URL),
            timeout_seconds=float(os.environ.get("MAGE_GGUF_SAFETY_TIMEOUT_SECONDS", "15")),
        )

    def install_on(self, text_encoder: Any) -> str:
        """Route ``text_encoder.screen_text`` / ``screen_edit`` to this adapter.

        This is the ONLY call-site routing change: the Mage conditioning /
        diffusion code is untouched. Returns the installed backend name.
        """
        text_encoder.screen_text = self.screen_text
        text_encoder.screen_edit = self.screen_edit
        return "gguf"


def _make_installed_adapter() -> GGUFSafetyAdapter | None:
    backend = os.environ.get("MAGE_SAFETY_BACKEND", "hf").strip().lower()
    if backend == "hf":
        return None
    if backend == "gguf":
        return GGUFSafetyAdapter.from_env()
    raise ValueError(f"MAGE_SAFETY_BACKEND must be 'hf' or 'gguf'; got {backend!r}")


def install_safety_backend(pipeline: Any) -> str:
    """Minimal worker call-site routing for the safety backend.

    ``MAGE_SAFETY_BACKEND=hf``   -> no-op (production HF path).
    ``MAGE_SAFETY_BACKEND=gguf`` -> routes ``txt_enc.screen_text/screen_edit``
                                    through the GGUF adapter (never HF).
    Returns the installed backend name.
    """
    adapter = _make_installed_adapter()
    if adapter is None:
        return "hf"
    return adapter.install_on(pipeline.model.txt_enc)
