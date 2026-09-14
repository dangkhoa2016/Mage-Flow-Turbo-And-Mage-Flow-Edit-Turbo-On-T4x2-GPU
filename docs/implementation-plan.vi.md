# Kế hoạch triển khai và trạng thái hiện tại

> 🌐 Language / Ngôn ngữ: [English](implementation-plan.md) | **Tiếng Việt**

## Phase A — nền tảng project sạch

Trạng thái: **hoàn tất**.

- Local Git project trên branch `main`.
- Public source được tách khỏi các artifact qualification/evidence nội bộ.
- Packaging, test, lifecycle script và documentation nằm trong một repository có thể review.
- License đã chọn: MIT (xem `LICENSE`).

## Phase B1 — tích hợp T2I runtime

Trạng thái: **hoàn tất; CPU tests PASS; GPU acceptance PASS**.

- Routing nghiêm ngặt: `cuda:0`, không CPU fallback.
- Localhost worker tại `127.0.0.1:8101`.
- Mage `.venv` đã restore và xác minh được tách khỏi environment của coordinator.
- Kaggle Mage-Flow-Turbo model mount local được truyền trực tiếp vào runtime.
- T2I transformer/VAE trên `cuda:0`; text encoder đặt trên CPU theo layout đã được acceptance.
- Real REST generation acceptance đã hoàn tất ở `1024x1024`.

## Phase B2 — tích hợp Edit runtime

Trạng thái: **hoàn tất; CPU tests PASS; GPU acceptance PASS**.

- Routing nghiêm ngặt: `cuda:1`, không CPU fallback.
- Localhost worker tại `127.0.0.1:8102`.
- Ảnh nguồn được truyền qua coordinator có authentication.
- Edit parameter đã freeze: `steps=4`, `cfg=1.0`, `prompt_template="mage-flow-edit"`, `vl_cond_long_edge=384`, negative prompt rỗng, kích thước tối đa `1024`.
- Edit transformer/VAE trên `cuda:1`; text encoder đặt trên CPU theo layout đã được acceptance.
- Real REST edit acceptance đã hoàn tất.

## Phase C — dual-worker orchestration và acceptance

Trạng thái: **hoàn tất**.

- `scripts/run_public_acceptance.sh` xác minh runtime, model mount, source commit, token handling và topology T4 x2.
- Worker resident đang healthy được tái sử dụng thay vì reload.
- Coordinator readiness fail-closed cho đến khi cả hai worker ready.
- Live acceptance kiểm tra health, readiness, runtime info, T2I generation, Edit generation, image validity, latency và output checksum.
- Public surface đã được acceptance ghi nhận resident reuse, không re-extract runtime và không sửa vendor source.

## Phase D — Kaggle notebook song ngữ

Trạng thái: **structural validation PASS; public live Run All đang chờ**.

Notebook dùng các stage `00`–`19`. Mỗi code cell đều có phần hướng dẫn English/Vietnamese song ngữ ngay trước đó. Offline validation kiểm tra stage structure, public-safe label, không CPU fallback và không import model framework nặng ở cell scope.

## Phase E — publication

Trạng thái: **đang chờ**.

Source-level cannonicalization closeout đã hoàn tất; công việc còn lại chỉ liên quan publication và không yêu cầu GPU re-qualification.

Các bước chỉ liên quan publication còn lại:

1. Freeze publication metadata và repository description.
2. Tạo official GitHub repository và push clean rewritten history.
3. Chạy public Kaggle notebook end-to-end từ clean saved-version context.
4. Xác minh public saved notebook version.
5. Chỉ tag/release `v1.0.0` sau khi publication checks PASS.

Không cần rerun GPU qualification cho các bước metadata/publication này.
