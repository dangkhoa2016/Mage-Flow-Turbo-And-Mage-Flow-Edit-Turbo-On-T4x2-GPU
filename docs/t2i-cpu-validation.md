# T2I source integration — CPU-only validation

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](t2i-cpu-validation.vi.md)

This document records the CPU-safe source-validation scope. It does not substitute for GPU acceptance; the separate T4 x2 qualification has already completed successfully for the frozen public candidate.

Record date: 2026-09-16. v11-hardening closeout at HEAD: the counts below reflect the repository validation state at this commit (331 tests).

## Result

```text
Python compile        PASS
Repository tests      331/331 PASS (clean-HOME hermetic)
Ruff lint/format      PASS
mypy server+scripts   PASS
ShellCheck            PASS (scripts/*.sh)
pip-audit             PASS (no known vulnerabilities)
Build artifacts       PASS (wheel + sdist, LICENSE, version parity)
Bilingual docs        PASS
Docs links            PASS (offline)
Repository hygiene    PASS (UTF-8, final newline, exec bits)
Notebook structure    PASS
Notebook compiles     PASS (every code cell; no assert gates)
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
- notebook code-cell newline termination, compile and metadata checks,
- notebook assert gates rejected in favour of explicit raises,
- torch imports restricted to the stage-03 GPU preflight cell,
- coordinator success and error mapping,
- public-boundary rejection of empty and whitespace-only prompts without invoking workers,
- cheap no-model security checks (401/415/422) before any inference,
- dual-worker service and acceptance helpers,
- bilingual EN/VI structural parity and offline docs-link validation,
- repository hygiene, final-newline gates and build-artifact/version parity,
- dependency vulnerability audit (pip-audit) over the declared dependencies,
- bilingual notebook structure without loading Mage/models/GPU.

## Boundary

CPU-only validation proves source wiring and fail-closed behavior. Real model load, device residency, and image generation/editing are established by the separate accepted Kaggle NVIDIA T4 x2 run, not by this CPU-safe test path.
