# Tích hợp source T2I — validation CPU-only

> 🌐 Language / Ngôn ngữ: [English](t2i-cpu-validation.md) | **Tiếng Việt**

Tài liệu này ghi lại phạm vi source validation CPU-safe. Nó không thay thế GPU acceptance; qualification T4 x2 riêng biệt đã hoàn tất thành công cho public candidate đã freeze.

Ngày ghi nhận: 2026-09-15. Snapshot CPU-safe validation lịch sử cho commit này: các con số dưới đây phản ánh trạng thái test repository tại snapshot đó (100 tests), không phải con số closeout cuối cùng.

## Kết quả

```text
Python compile        PASS
Repository tests      100/100 PASS (hermetic clean-HOME)
Notebook structure    PASS
Notebook compiles     PASS (mọi code cell)
GPU used              NO
Model loaded          NO
T2I GPU acceptance    PASS (separate T4 x2 qualification)
Edit GPU acceptance   PASS (separate T4 x2 qualification)
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
- newline termination của notebook code cell và compile checks,
- coordinator success/error mapping,
- rejection ở public boundary cho prompt rỗng và prompt chỉ chứa khoảng trắng mà không gọi worker,
- dual-worker service và acceptance helper,
- cấu trúc notebook song ngữ mà không load Mage/model/GPU.

## Ranh giới xác minh

CPU-only validation chứng minh source wiring và fail-closed behavior. Real model load, device residency và image generation/editing được chứng minh bởi separate accepted Kaggle NVIDIA T4 x2 run, không phải bởi CPU-safe test path này.
