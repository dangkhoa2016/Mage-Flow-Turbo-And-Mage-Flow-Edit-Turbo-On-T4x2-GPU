# T2I source integration — CPU-only validation

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](t2i-cpu-validation.vi.md)

This document records the CPU-safe source-validation scope. It does not substitute for GPU acceptance; the separate T4 x2 qualification has already completed successfully for the frozen public candidate.

Record date: 2026-09-15. Final corrected publication snapshot at HEAD: the counts below reflect the repository test state at this commit (104 tests).

## Result

```text
Python compile        PASS
Repository tests      104/104 PASS (clean-HOME hermetic)
Notebook structure    PASS
Notebook compiles     PASS (every code cell)
GPU used              NO
Model loaded          NO
T2I GPU acceptance    PASS (separate T4 x2 qualification)
Edit GPU acceptance   PASS (separate T4 x2 qualification)
```

## What CPU-safe validation covers

- strict `cuda:0` routing contract for T2I,
- no CPU fallback configuration,
- validated-runtime command construction,
- localhost-only worker transport,
- worker readiness identity checks,
- generation request forwarding,
- dimension validation,
- fail-closed behavior when a worker is unavailable,
- bounded public upload reads and malformed-image rejection,
- `service_health` CLI `model_path` contract,
- orchestrator API-token file permissions and hermetic Git identity,
- PID identity checks before any lifecycle `kill`,
- notebook code-cell newline termination and compile checks,
- coordinator success and error mapping,
- public-boundary rejection of empty and whitespace-only prompts without invoking workers,
- dual-worker service and acceptance helpers,
- bilingual notebook structure without loading Mage/models/GPU.

## Boundary

CPU-only validation proves source wiring and fail-closed behavior. Real model load, device residency, and image generation/editing are established by the separate accepted Kaggle NVIDIA T4 x2 run, not by this CPU-safe test path.