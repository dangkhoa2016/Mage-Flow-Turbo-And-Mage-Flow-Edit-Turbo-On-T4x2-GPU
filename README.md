# Mage-Flow-Turbo and Mage-Flow-Edit-Turbo on T4x2 GPU

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-1.0.0-orange)
![Python](https://img.shields.io/badge/python-3.10--3.14-blue)
![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU)
![Runtime](https://img.shields.io/badge/runtime-Kaggle%20NVIDIA%20T4%C3%972-20BEFF?logo=kaggle&logoColor=white)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

Official public Kaggle NVIDIA T4 x2 workflow and authenticated REST API for Mage-Flow Turbo and Mage-Flow Edit Turbo.

## v1.0.0 release

- GitHub Release: [v1.0.0](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/releases/tag/v1.0.0)
- Official Kaggle evidence: [Saved Version 352899347](https://www.kaggle.com/code/dangkhoa2016/mage-flow-turbo-mage-flow-edit-turbo-t4x2-demo?scriptVersionId=352899347)
- Frozen release commit: `f6bcbb4f8d577e6cd576e2ff48438710f89a93a5`
- Exact-tag CI: [run 36226798319](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/actions/runs/36226798319)
- Official T2I evidence: 1024×1024, 4 steps, seed 42, ~94.741 s
- Official Edit evidence: max size 1024, 4 steps, CFG 1.0, ~150.826 s

The `v1.0.0` tag is bound to the frozen release commit above. The Kaggle Saved Version is the external runtime evidence for that exact revision; later changes on `main` do not alter the published release authority.

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
- `scripts/run_public_acceptance.sh` validates the runtime/model mounts and T4 x2 topology, starts or reuses both resident workers plus the coordinator, and runs control-plane acceptance covering health, readiness, security, runtime info, and device routing.
- The notebook performs exactly one real T2I inference and one real Edit inference in later stages, validating PNG outputs, latency metadata, and output checksums without duplicating generation inside the orchestrator.
- Resident-worker reuse is accepted; runtime re-extraction is not required.
- Vendor Mage source remains unmodified by this public project.
- CPU-safe test suite: configured in GitHub Actions for the declared Python matrix.
- The bilingual Kaggle notebook uses an unnumbered introduction followed by stages `01`–`19`; offline structural validation passes.

The public T4 x2 acceptance workflow has been exercised successfully on the supported runtime. Release `v1.0.0` is now published from exact commit `f6bcbb4f8d577e6cd576e2ff48438710f89a93a5` and is backed by the verified public Kaggle Saved Version 352899347. The release evidence remains external to the source tree so the published source revision stays immutable.

## Run the public acceptance surface on Kaggle T4 x2

For the public notebook workflow, start from a clean Kaggle session. In the Notebook editor, open the right-hand **Settings** pane, select **NVIDIA T4 x2** under **Session options**, enable **Internet**, then open **Input → Add Input**.

Attach these four read-only inputs **before** Save & Run All. Search the exact name and choose the result owned by `dangkhoa2016`:

| Type | Kaggle resource | Expected mount root |
| :--- | :--- | :--- |
| Dataset | `dangkhoa2016/mage-flow-t4x2-runtime-support` | `/kaggle/input/datasets/dangkhoa2016/mage-flow-t4x2-runtime-support` |
| Model | `dangkhoa2016/mage-flow-community-mage-flow-turbo` | `/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-turbo/pytorch/default/1` |
| Model | `dangkhoa2016/mage-flow-community-mage-flow-edit-turbo` | `/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-edit-turbo/pytorch/default/1` |
| Model | `dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf` | `/kaggle/input/models/dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf/gguf/q4-k-m/1` |

For Models, use **Add Input → Models** (or **Add Models** if Kaggle shows that shortcut). If Kaggle asks you to accept a license or confirm a model variation, complete that step first. Confirm that the **Input** pane lists exactly one Dataset and three Models. Do not manually upload, rename, or copy them into `/kaggle/working`; the notebook validates their read-only `/kaggle/input` mounts.

Import notebooks/mage-flow-turbo-and-mage-flow-edit-turbo-on-t4x2-gpu.ipynb and use **Save & Run All**. The notebook clones the public repository, records the exact commit used by the run, verifies and applies the pinned optimization patch, restores the validated runtime from the attached support Dataset, and then invokes scripts/run_public_acceptance.sh.

The orchestrator creates or reuses a per-session API token under `.runtime/api_token`, verifies the T4 x2 topology, starts or reuses the two model workers sequentially to limit peak system RAM, starts the authenticated coordinator, waits for readiness, and runs control-plane acceptance.

On a machine without the required GPU topology, the orchestrator fails closed with `GPU_ACCELERATOR_REQUIRED` instead of falling back to CPU.

Stop the services with:

```bash
./scripts/stop.sh
```

## Validation

CPU-safe validation does not load the models or require a GPU. The full release-check set is enforced locally and in CI:

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

- T4 x2 public acceptance workflow: validated on the supported runtime
- Exact-revision GPU verification: PASS for `v1.0.0` at `f6bcbb4f8d577e6cd576e2ff48438710f89a93a5`
- GitHub Release: [v1.0.0](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/releases/tag/v1.0.0)
- Official Kaggle Saved Version: [352899347](https://www.kaggle.com/code/dangkhoa2016/mage-flow-turbo-mage-flow-edit-turbo-t4x2-demo?scriptVersionId=352899347)
- CPU-safe validation: PASS across the declared Python 3.10–3.14 GitHub Actions matrix
- Bilingual source documentation: included
- Public notebook structural validation: complete
- Release evidence: recorded externally in the final Kaggle Saved Version and linked from the GitHub Release
- Open-source license: MIT
- GitHub repository: published
- GitHub Actions CI and community metadata: configured
- Python distribution: Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU
- Python package version: 1.0.0
- Canonical support Dataset: `dangkhoa2016/mage-flow-t4x2-runtime-support` (datasetId `12144721`, version `2`)
- Release state: `v1.0.0` is frozen at the exact verified release commit; post-release documentation changes on `main` do not rewrite that tag or its Kaggle evidence.

Kaggle execution evidence is intentionally external to the source tree so verifying a Saved Version never requires a self-referential source commit.

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
