# Mage-Flow-Turbo and Mage-Flow-Edit-Turbo on T4x2 GPU

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-1.0.0-orange)
![Python](https://img.shields.io/badge/python-3.10--3.14-blue)
![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU)
![Runtime](https://img.shields.io/badge/runtime-Kaggle%20NVIDIA%20T4%C3%972-20BEFF?logo=kaggle&logoColor=white)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

Production-style public REST API demo for Mage-Flow Turbo and Mage-Flow Edit Turbo on Kaggle NVIDIA T4 x2.

## Target runtime

- Mage-Flow-Turbo text-to-image worker: `cuda:0`
- Mage-Flow-Edit-Turbo image-edit worker: `cuda:1`
- REST coordinator: localhost CPU process on `127.0.0.1:8090`
- Model workers: restored and validated Mage runtime `.venv`
- Internal worker endpoints: `127.0.0.1:8101` (T2I) and `127.0.0.1:8102` (Edit)
- Optional Cloudflare Quick Tunnel for the authenticated coordinator only
- Temporary Bearer-token protection for `/ready` and `/v1/*`
- No CPU fallback in the intended T4 x2 workflow

## Public API

`GET /health` is minimal unauthenticated liveness. `/ready` and `/v1/*` require `Authorization: Bearer <MAGE_FLOW_API_TOKEN>`.

- `GET /health`
- `GET /ready`
- `GET /v1/info`
- `POST /v1/images/generations`
- `POST /v1/images/edits`

See [docs/api.md](docs/api.md) for the complete contract.

## Current implementation status

The T4 x2 GPU public acceptance surface has completed local qualification:

- T2I runtime worker is pinned to `cuda:0` and refuses CPU fallback.
- Edit runtime worker is pinned to `cuda:1` and refuses CPU fallback.
- Text encoders run on CPU while the T2I/Edit transformers and VAEs remain on their assigned GPUs.
- Workers bind only to localhost; the FastAPI coordinator communicates with them over internal HTTP.
- Successful T2I/Edit responses return PNG data URLs (`data:image/png;base64,...`).
- `scripts/run_public_acceptance.sh` validates the runtime/model mounts and T4 x2 topology, starts or reuses both resident workers plus the coordinator, runs live REST acceptance, and records reuse/device-isolation evidence.
- Live REST acceptance covers `/health`, `/ready`, `/v1/info`, T2I generation, and Edit generation.
- Resident-worker reuse is accepted; runtime re-extraction is not required.
- Vendor Mage source remains unmodified by this public project.
- CPU-safe test suite: configured in GitHub Actions for the declared Python matrix.
- Bilingual Kaggle notebook stages `00`–`19` pass offline structural validation.

Historical/local T4 x2 public acceptance has completed for the frozen public acceptance surface. This is distinct from the formal GPU qualification gate in the current canonical support/publication chain: that formal gate has **not** been executed. This repository is the canonical public publication tree. A fresh public notebook `Run All` from a clean saved-version context, and saved-version verification, are still pending publication steps and are not claimed yet.

## Run the public acceptance surface on Kaggle T4 x2

After the validated Mage runtime and both Kaggle model mounts are available:

```bash
./scripts/run_public_acceptance.sh
```

The orchestrator creates or reuses a per-session API token under `.runtime/api_token`, verifies the T4 x2 topology, starts or reuses the two model workers, starts the authenticated coordinator, waits for readiness, and runs live REST acceptance.

On a machine without the required GPU topology, the orchestrator fails closed with `GPU_ACCELERATOR_REQUIRED` instead of falling back to CPU.

Stop the services with:

```bash
./scripts/stop.sh
```

## Validation

CPU-safe validation does not load the models or require a GPU. The full release-gate set is enforced locally and in CI:

```bash
python -m pytest -q
python scripts/validate_bilingual_docs.py
python scripts/validate_notebook.py
python scripts/validate_docs_links.py
python scripts/validate_repository.py
python scripts/validate_build_artifacts.py
python -m pip check
python -m compileall -q server scripts examples
bash -n scripts/*.sh
shellcheck scripts/*.sh
ruff check .
ruff format --check .
mypy server scripts
python -m build
pip-audit -r requirements.txt
```

Expected repository-level validation includes a passing CPU-safe pytest suite plus:

```text
[PASS] BILINGUAL_DOCUMENTATION_PAIRING_VALID
[PASS] NOTEBOOK_STRUCTURE_VALID
[PASS] DOCS_LINKS_VALID
[PASS] REPOSITORY_HYGIENE_VALID
[PASS] BUILD_ARTIFACTS_VALID
```

## Publication state

- Historical/local GPU public-acceptance evidence: complete
- Formal canonical-chain GPU qualification: not executed
- CPU-safe validation: 475/475 PASS (clean-HOME hermetic, 2026-09-16 validation closeout)
- Bilingual source documentation: included
- Public notebook structural validation: complete
- Public notebook live `Run All`: pending
- Public saved version verification: pending
- Open-source license: MIT
- GitHub repository: published
- GitHub Actions CI and community metadata: configured
- Python distribution: Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU
- Python package version: 1.0.0
- Canonical support Dataset: `dangkhoa2016/mage-flow-t4x2-runtime-support` (datasetId `12144721`, version `2`)
- `v1.0.0`: not yet released

`v1.0.0` remains gated on a fresh public Kaggle `Run All` and saved-version verification; the Kaggle publication steps are not claimed yet.

## Documentation

- [REST API](docs/api.md) — complete HTTP contract
- [Architecture](docs/architecture.md) — system design and runtime topology
- [Implementation plan](docs/implementation-plan.md) — staged delivery plan
- [Kaggle runtime](docs/kaggle.md) — Kaggle T4 x2 runtime notes
- [Limitations](docs/limitations.md) — known limits of the public demo
- [CPU-safe validation](docs/t2i-cpu-validation.md) — offline validation that does not require a GPU

## License

Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>

Full license terms are available in the [LICENSE](LICENSE) file.
