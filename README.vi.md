# Mage-Flow-Turbo and Mage-Flow-Edit-Turbo on T4x2 GPU

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-1.0.0-orange)
![Python](https://img.shields.io/badge/python-3.10--3.14-blue)
![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU)
![Runtime](https://img.shields.io/badge/runtime-Kaggle%20NVIDIA%20T4%C3%972-20BEFF?logo=kaggle&logoColor=white)

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

Workflow Kaggle NVIDIA T4 x2 public chính thức cùng REST API có authentication cho Mage-Flow Turbo và Mage-Flow Edit Turbo.

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

Public acceptance surface T4 x2 GPU đã hoàn tất qualification local:

- T2I runtime worker được pin vào `cuda:0` và từ chối CPU fallback.
- Edit runtime worker được pin vào `cuda:1` và từ chối CPU fallback.
- Text encoder chạy trên CPU, còn transformer và VAE của T2I/Edit nằm trên GPU đã được chỉ định.
- Các worker chỉ bind localhost; FastAPI coordinator giao tiếp với worker qua HTTP nội bộ.
- Response T2I/Edit thành công trả PNG data URL (`data:image/png;base64,...`).
- `scripts/run_public_acceptance.sh` xác minh runtime/model mount và topology T4 x2, khởi động hoặc tái sử dụng hai resident worker cùng coordinator, rồi chạy control-plane acceptance cho health, readiness, security, runtime info và device routing.
- Notebook thực hiện đúng một inference T2I và một inference Edit thật ở các stage sau, xác minh PNG output, latency metadata và output checksum mà không generate trùng bên trong orchestrator.
- Resident-worker reuse đã được chấp nhận; không cần giải nén lại runtime.
- Mage vendor source không bị project public này sửa đổi.
- Bộ test CPU-safe: được cấu hình trong GitHub Actions cho ma trận Python đã khai báo.
- Kaggle notebook song ngữ dùng phần giới thiệu không đánh số, sau đó là các stage `01`–`19`; structural validation offline PASS.

Workflow public acceptance trên T4 x2 đã được chạy thành công trên runtime được hỗ trợ. Các bước kiểm tra release được thực hiện trên đúng revision được chọn để publish trước khi tạo version tag. Public notebook Run All từ clean saved-version context cùng saved-version verification vẫn thuộc checklist publication.

## Chạy public acceptance surface trên Kaggle T4 x2

Đối với public notebook workflow, hãy bắt đầu từ clean Kaggle session. Trong Notebook editor, mở panel **Settings** bên phải, chọn **NVIDIA T4 x2** ở **Session options**, bật **Internet**, sau đó mở **Input → Add Input**.

Attach đủ bốn input read-only dưới đây **trước** khi Save & Run All. Hãy tìm đúng tên và chọn kết quả có owner `dangkhoa2016`:

| Loại | Kaggle resource | Mount root mong đợi |
| :--- | :--- | :--- |
| Dataset | `dangkhoa2016/mage-flow-t4x2-runtime-support` | `/kaggle/input/datasets/dangkhoa2016/mage-flow-t4x2-runtime-support` |
| Model | `dangkhoa2016/mage-flow-community-mage-flow-turbo` | `/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-turbo/pytorch/default/1` |
| Model | `dangkhoa2016/mage-flow-community-mage-flow-edit-turbo` | `/kaggle/input/models/dangkhoa2016/mage-flow-community-mage-flow-edit-turbo/pytorch/default/1` |
| Model | `dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf` | `/kaggle/input/models/dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf/gguf/q4-k-m/1` |

Đối với Model, dùng **Add Input → Models** (hoặc **Add Models** nếu Kaggle hiển thị shortcut này). Nếu Kaggle yêu cầu chấp nhận license hoặc xác nhận model variation, hãy hoàn tất bước đó trước. Kiểm tra mục **Input** phải hiển thị đúng một Dataset và ba Model. Không upload thủ công, đổi tên hoặc copy chúng vào `/kaggle/working`; notebook sẽ kiểm tra các mount read-only dưới `/kaggle/input`.

Import notebooks/mage-flow-turbo-and-mage-flow-edit-turbo-on-t4x2-gpu.ipynb rồi dùng **Save & Run All**. Notebook sẽ clone public repository, ghi lại chính xác commit được dùng cho lần chạy, xác minh và áp dụng pinned optimization patch, restore validated runtime từ support Dataset đã attach, sau đó gọi scripts/run_public_acceptance.sh.

Orchestrator tạo hoặc tái sử dụng API token theo session trong `.runtime/api_token`, kiểm tra topology T4 x2, khởi động hoặc tái sử dụng hai model worker theo thứ tự tuần tự để giới hạn peak system RAM, khởi động coordinator có authentication, chờ readiness rồi chạy control-plane acceptance.

Trên máy không có đúng topology GPU yêu cầu, orchestrator fail-closed với `GPU_ACCELERATOR_REQUIRED` thay vì fallback sang CPU.

Dừng các service bằng:

```bash
./scripts/stop.sh
```

## Xác minh

Validation CPU-safe không load model và không cần GPU. Toàn bộ release checks được thực thi cục bộ và trên CI:

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

- Workflow public acceptance trên T4 x2: đã được xác minh trên runtime được hỗ trợ
- Xác minh GPU cho đúng revision: được thực hiện trong checklist release trước khi tạo tag
- CPU-safe validation: PASS trên toàn bộ GitHub Actions matrix Python 3.10–3.14 đã khai báo
- Source documentation song ngữ: đã có
- Public notebook structural validation: hoàn tất
- Release evidence: được ghi nhận bên ngoài source tree trong Kaggle Saved Version cuối cùng và được liên kết từ GitHub Release
- Open-source license: MIT
- GitHub repository: đã public
- GitHub Actions CI và community metadata: đã cấu hình
- Python distribution: Mage-Flow-Turbo-And-Mage-Flow-Edit-Turbo-On-T4x2-GPU
- Python package version: 1.0.0
- Canonical support Dataset: `dangkhoa2016/mage-flow-t4x2-runtime-support` (datasetId `12144721`, version `2`)
- Chính sách release: chỉ tạo tag `v1.0.0` sau khi Kaggle Saved Version cuối cùng được xác minh trên đúng release revision.

Execution evidence trên Kaggle được giữ bên ngoài source tree để việc xác minh Saved Version không bao giờ yêu cầu một source commit tự tham chiếu.

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
