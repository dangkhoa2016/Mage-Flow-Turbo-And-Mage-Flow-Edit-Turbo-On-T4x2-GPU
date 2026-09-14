# REST API contract

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](api.vi.md)

Base URL inside Kaggle:

```text
http://127.0.0.1:8090
```

When a Cloudflare Quick Tunnel is enabled, it forwards only to the authenticated coordinator. Internal worker ports `8101` and `8102` remain localhost-only and must never be exposed directly.

## Authentication

`GET /health` is a minimal unauthenticated liveness check.

The following endpoints require:

```http
Authorization: Bearer <MAGE_FLOW_API_TOKEN>
```

- `GET /ready`
- `GET /v1/info`
- `POST /v1/images/generations`
- `POST /v1/images/edits`

The Kaggle workflow generates a strong temporary token per session and does not print the token value.

## GET /health

Returns coordinator liveness only:

```json
{"status":"ok","coordinator":"healthy"}
```

## GET /ready

Overall readiness is fail-closed. `ready=true` only when both T2I and Edit workers report ready on their required devices.

## GET /v1/info

Returns the public runtime contract and current readiness for both workers. The expected routing contract is:

```text
T2I  -> cuda:0
Edit -> cuda:1
CPU fallback -> false
```

## POST /v1/images/generations

JSON body:

```json
{
  "prompt": "A cinematic mountain lake at sunrise",
  "seed": 42,
  "steps": 4,
  "width": 1024,
  "height": 1024
}
```

`width` and `height` must be multiples of 16. The public T2I path uses Mage-Flow Turbo on physical `cuda:0`. The public API is restricted to the GPU-proven acceptance profile: `steps=4`, `width=1024`, `height=1024`.

Successful response shape:

```json
{
  "id": "img_...",
  "status": "completed",
  "model": "mage-flow-turbo",
  "device": "cuda:0",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "elapsed_seconds": 18.42,
  "output": "data:image/png;base64,..."
}
```

The output is a PNG data URL so remote clients do not depend on Kaggle-local output paths.

## POST /v1/images/edits

Multipart form fields:

- `image`: source image file (PNG/JPEG/WebP, maximum 8 MiB)
- `prompt`: edit instruction
- `seed`: optional integer, default `42`

The public coordinator enforces a maximum upload size of **8 MiB** before forwarding. Oversized uploads return `413`, unsupported media types return `415`, empty uploads return `400`.

Example successful response shape:

```json
{
  "id": "img_...",
  "status": "completed",
  "model": "mage-flow-edit-turbo",
  "device": "cuda:1",
  "seed": 42,
  "elapsed_seconds": 11.25,
  "output": "data:image/png;base64,..."
}
```

The Edit runtime uses the frozen public parameters, including `steps=4`, `cfg=1.0`, `prompt_template="mage-flow-edit"`, `vl_cond_long_edge=384`, and no automatic CPU fallback.

## Error policy

- `400/422`: invalid user request (e.g. empty prompt, whitespace-only prompt, malformed JSON, missing required fields)
- `401`: missing Bearer token
- `403`: invalid Bearer token
- `413`: image upload exceeds the public limit
- `415`: unsupported image media type
- `502`: coordinator reached a worker but the worker request failed
- `503`: required worker is not ready
- `5xx`: unexpected runtime/model failure

No public endpoint silently falls back to CPU. Whitespace-only prompts are rejected at the public boundary before any worker invocation.
