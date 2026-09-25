# Tích hợp source T2I — validation CPU-only

> 🌐 Language / Ngôn ngữ: [English](t2i-cpu-validation.md) | **Tiếng Việt**

Tài liệu này ghi lại phạm vi source validation CPU-safe. Nó không thay thế GPU acceptance. Các bước kiểm tra GPU cho release được thực hiện riêng trên đúng revision được chọn để publish.

Ngày ghi nhận: 2026-09-16. Đây là historical validation snapshot: các con số dưới đây phản ánh revision được ghi nhận lúc đó (474 tests), không phải test count hiện tại của branch. Validation authority hiện tại là GitHub Actions run mới nhất của release revision.

## Kết quả

```text
Python compile        PASS
Repository tests      474/474 PASS (hermetic clean-HOME)
Ruff lint/format      PASS
mypy server+scripts   PASS
ShellCheck            PASS (scripts/*.sh)
pip-audit             PASS (không có lỗ hổng đã biết)
Build artifacts       PASS (wheel + sdist, LICENSE, version parity)
Bilingual docs        PASS
Docs links            PASS (offline)
Repository hygiene    PASS (UTF-8, final newline, exec bits)
Notebook structure    PASS
Notebook compiles     PASS (mọi code cell; không có assert gates)
GPU used              NO
Model loaded          NO
T2I GPU acceptance    PASS (historical/local T4 x2 acceptance)
Edit GPU acceptance   PASS (historical/local T4 x2 acceptance)
```

## CPU-safe validation bao phủ những gì

- strict routing contract `cuda:0` cho T2I,
- không cấu hình CPU fallback,
- command construction cho validated runtime,
- worker transport chỉ trên localhost,
- worker readiness identity check,
- forwarding generation request,
- dimension validation,
- fail-closed behavior khi worker unavailable,
- bounded public upload reads và rejection ảnh malformed,
- contract `model_path` trên CLI `service_health`,
- permission file API token của orchestrator và Git identity hermetic,
- PID identity check trước mọi `kill` ở lifecycle script,
- newline termination của notebook code cell, compile và metadata checks,
- các assert gate của notebook bị loại bỏ để dùng explicit raise,
- import torch giới hạn tại cell preflight GPU stage 03,
- coordinator success/error mapping,
- rejection ở public boundary cho prompt rỗng và prompt whitespace-only mà không gọi worker,
- security checks không-load-model chi phí thấp (401/415/422) trước mọi inference,
- dual-worker service và acceptance helper,
- parity cấu trúc song ngữ EN/VI và docs-link validation offline,
- repository hygiene, final-newline gates và build-artifact/version parity,
- dependency vulnerability audit (pip-audit) trên các dependency đã khai báo,
- cấu trúc notebook song ngữ mà không load Mage/model/GPU.

## Ranh giới xác minh

CPU-only validation chứng minh source wiring và fail-closed behavior. Real model load, device residency và image generation/editing cần workflow Kaggle NVIDIA T4 x2 và được xác minh riêng cho revision được chọn để release.
