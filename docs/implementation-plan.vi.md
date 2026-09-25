# Kế hoạch triển khai và trạng thái hiện tại

> 🌐 Language / Ngôn ngữ: [English](implementation-plan.md) | **Tiếng Việt**

## Phase A — nền tảng project sạch

Trạng thái: **hoàn tất**.

- Local Git project trên branch `main`.
- Repository chỉ chứa public source, test, documentation và các asset phục vụ release.
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
- Control-plane acceptance kiểm tra health, readiness, security, runtime info và device routing; các stage notebook sau đó thực hiện một inference T2I thật và một inference Edit thật rồi xác minh image validity, latency metadata và output checksum.
- Public surface đã được acceptance ghi nhận resident reuse, không re-extract runtime và không sửa vendor source.

## Phase D — Kaggle notebook song ngữ

Trạng thái: **structural validation PASS; release execution evidence được lưu bên ngoài source tree**.

Notebook dùng phần giới thiệu song ngữ không đánh số, sau đó là các stage `01`–`19`. Mỗi code cell đều có phần hướng dẫn English/Vietnamese song ngữ ngay trước đó. Offline validation kiểm tra stage structure, public-safe label, không CPU fallback và không import model framework nặng ở cell scope. Kaggle Saved Version cuối cùng là release evidence và được ghi nhận có chủ đích bên ngoài source tree.

## Phase E — publication

Trạng thái: **source hoàn tất; cần external release evidence trước khi tag**.

Bố cục public source đã hoàn tất. T4 x2 acceptance đã được chạy trên runtime được hỗ trợ. Execution evidence cuối cùng được tạo dưới dạng Kaggle Saved Version trên đúng revision được chọn để release và được liên kết từ GitHub Release thay vì ghi ngược trở lại source metadata.

Trình tự publication:

1. Freeze publication metadata và repository history.
2. Chạy public Kaggle notebook end-to-end từ clean saved-version context.
3. Xác minh Saved Version bind đúng frozen release revision.
4. Tag/release `v1.0.0` mà không thay đổi source tree đã được xác minh.
5. Liên kết Saved Version cùng log/output evidence từ GitHub Release.

Cách này tránh vòng lặp tự tham chiếu trong đó việc ghi trạng thái run thành công vào source lại làm thay đổi chính revision vừa được xác minh.
