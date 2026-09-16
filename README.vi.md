# Mage-Flow Turbo + Edit Turbo — Kaggle T4x2 REST API Demo

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

Project public dạng production-style cho REST API của Mage-Flow Turbo và Mage-Flow Edit Turbo trên Kaggle NVIDIA T4 x2.

## Runtime mục tiêu

- Worker text-to-image Mage-Flow-Turbo: `cuda:0`
- Worker image-edit Mage-Flow-Edit-Turbo: `cuda:1`
- REST coordinator: tiến trình CPU trên localhost tại `127.0.0.1:8090`
- Model worker: dùng Mage runtime `.venv` đã được restore và xác minh
- Endpoint nội bộ của worker: `127.0.0.1:8101` (T2I) và `127.0.0.1:8102` (Edit)
- Cloudflare Quick Tunnel là tùy chọn và chỉ expose coordinator đã có authentication
- Bảo vệ tạm thời bằng Bearer token cho `/ready` và `/v1/*`
- Không fallback sang CPU trong workflow T4 x2 mục tiêu

## Public API

`GET /health` là liveness tối giản không cần authentication. `/ready` và `/v1/*` yêu cầu `Authorization: Bearer <MAGE_FLOW_API_TOKEN>`.

- `GET /health`
- `GET /ready`
- `GET /v1/info`
- `POST /v1/images/generations`
- `POST /v1/images/edits`

Xem [docs/api.vi.md](docs/api.vi.md) để biết contract đầy đủ.

## Trạng thái implementation hiện tại

Public candidate T4 x2 GPU đã hoàn tất qualification local:

- T2I runtime worker được pin vào `cuda:0` và từ chối CPU fallback.
- Edit runtime worker được pin vào `cuda:1` và từ chối CPU fallback.
- Text encoder chạy trên CPU, còn transformer và VAE của T2I/Edit nằm trên GPU đã được chỉ định.
- Các worker chỉ bind localhost; FastAPI coordinator giao tiếp với worker qua HTTP nội bộ.
- Response T2I/Edit thành công trả PNG data URL (`data:image/png;base64,...`).
- `scripts/run_public_candidate.sh` xác minh runtime/model mount và topology T4 x2, khởi động hoặc tái sử dụng hai resident worker cùng coordinator, chạy live REST acceptance và ghi lại bằng chứng reuse/device isolation.
- Live REST acceptance kiểm tra `/health`, `/ready`, `/v1/info`, T2I generation và Edit generation.
- Resident-worker reuse đã được chấp nhận; không cần giải nén lại runtime.
- Mage vendor source không bị project public này sửa đổi.
- Bộ test CPU-safe: được cấu hình trong GitHub Actions cho ma trận Python đã khai báo.
- Kaggle notebook song ngữ ở các stage `00`–`19` PASS structural validation offline.

GPU qualification đã hoàn tất cho local public candidate. Repository này là corrected publication candidate. Public notebook `Run All` từ clean saved-version context và saved-version verification vẫn là các bước publication chưa hoàn tất, vì vậy chưa được tuyên bố PASS.

## Chạy public candidate trên Kaggle T4 x2

Sau khi Mage runtime đã được xác minh và hai Kaggle model mount đã sẵn sàng:

```bash
./scripts/run_public_candidate.sh
```

Orchestrator tạo hoặc tái sử dụng API token theo session trong `.runtime/api_token`, kiểm tra topology T4 x2, khởi động hoặc tái sử dụng hai model worker, khởi động coordinator có authentication, chờ readiness và chạy live REST acceptance.

Trên máy không có đúng topology GPU yêu cầu, orchestrator fail-closed với `GPU_ACCELERATOR_REQUIRED` thay vì fallback sang CPU.

Dừng các service bằng:

```bash
./scripts/stop.sh
```

## Xác minh

Validation CPU-safe không load model và không cần GPU. Toàn bộ các cổng release-gate được thực thi cục bộ và trên CI:

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

Validation ở mức repository bao gồm bộ pytest CPU-safe PASS và:

```text
[PASS] BILINGUAL_DOCUMENTATION_PAIRING_VALID
[PASS] NOTEBOOK_STRUCTURE_VALID
[PASS] DOCS_LINKS_VALID
[PASS] REPOSITORY_HYGIENE_VALID
[PASS] BUILD_ARTIFACTS_VALID
```

## Trạng thái publication

- GPU public-candidate qualification: hoàn tất
- CPU-safe validation: 331/331 PASS (hermetic clean-HOME, 2026-09-16 v11-hardening closeout)
- Source documentation song ngữ: đã có
- Public notebook structural validation: hoàn tất
- Public notebook live `Run All`: đang chờ
- Public saved version verification: đang chờ
- Open-source license: MIT
- GitHub Actions CI và community metadata: đã cấu hình
- Tạo official GitHub repository: đang chờ
- `v1.0.0`: chưa release

## Tài liệu

- [REST API](docs/api.vi.md) — contract HTTP đầy đủ
- [Kiến trúc](docs/architecture.vi.md) — thiết kế hệ thống và runtime topology
- [Kế hoạch implementation](docs/implementation-plan.vi.md) — kế hoạch triển khai theo giai đoạn
- [Runtime Kaggle](docs/kaggle.vi.md) — ghi chú runtime Kaggle T4 x2
- [Giới hạn](docs/limitations.vi.md) — các giới hạn đã biết của demo public
- [Validation CPU-safe](docs/t2i-cpu-validation.vi.md) — validation offline không cần GPU

## Giấy phép

Copyright (c) 2026 Đăng Khoa <i.am@dangkhoa.dev>

Toàn văn giấy phép có trong file [LICENSE](LICENSE).
