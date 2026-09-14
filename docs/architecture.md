# Architecture

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](architecture.vi.md)

```text
                   public/authenticated boundary
                            |
                    REST API coordinator
                    system Python
                    127.0.0.1:8090
                         /      \
                        /        \
           localhost internal    localhost internal
                HTTP :8101            HTTP :8102
                    |                      |
               T2I worker             Edit worker
        validated Mage .venv     validated Mage .venv
               cuda:0                 cuda:1
           Mage-Flow Turbo       Mage-Flow Edit Turbo
```

The coordinator deliberately does not import the heavy Mage/PyTorch model runtime. Each model worker runs inside the restored, validated Mage virtual environment while the public FastAPI coordinator remains in the normal Kaggle Python environment.

Internal worker interfaces bind to localhost only and are never exposed directly through a Quick Tunnel.

Overall readiness remains fail-closed: `/ready` becomes ready only when both workers identify themselves correctly and are healthy on their required devices. No automatic CPU fallback is allowed.

## T2I worker

Mage-Flow Turbo runs on `cuda:0`:

- local Kaggle model mount; no model download when the mount exists,
- validated runtime Python from the restored `.venv`,
- strict T4 x2 hardware gate before model load,
- transformer and VAE on `cuda:0`, text encoder on CPU for the accepted runtime layout,
- upstream prompt screening and Gaussian-Shading behavior preserved,
- synchronous generation serialized inside the worker,
- generated PNG returned as `data:image/png;base64,...`.

## Edit worker

Mage-Flow Edit Turbo runs on `cuda:1`:

- local Kaggle model mount,
- validated restored Mage runtime,
- transformer and VAE on `cuda:1`, text encoder on CPU,
- frozen public inference settings: `steps=4`, `cfg=1.0`, `prompt_template="mage-flow-edit"`, `vl_cond_long_edge=384`, blank negative prompt, maximum size `1024`,
- source image transported through the coordinator to the localhost worker,
- generated PNG returned as a data URL.

## Resident service model

`scripts/run_public_acceptance.sh` starts or reuses healthy resident T2I/Edit workers. Live acceptance does not reload the models. The accepted acceptance records resident-worker reuse and zero runtime re-extraction during the reuse path.

## Authentication boundary

- `GET /health` is a minimal unauthenticated liveness endpoint.
- `GET /ready` and all `/v1/*` endpoints require `Authorization: Bearer <MAGE_FLOW_API_TOKEN>`.
- The workflow generates or reuses a temporary per-session token.
- A Cloudflare Quick Tunnel, when enabled, forwards only to the authenticated coordinator.

## Production claim

This repository demonstrates a production-style topology and fail-closed behavior. It does not claim SLA-backed availability, high availability, or multi-tenant isolation.
