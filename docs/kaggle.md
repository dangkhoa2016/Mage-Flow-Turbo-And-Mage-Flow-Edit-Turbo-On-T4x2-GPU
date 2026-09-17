# Kaggle runtime contract

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](kaggle.vi.md)

- Accelerator: NVIDIA T4 x2.
- Internet: ON — Stage 01 fetches the exact public project source from GitHub at a pinned commit.
- Three read-only Kaggle inputs attached:
  - `dangkhoa2016/mage-flow-community-mage-flow-turbo` (T2I model).
  - `dangkhoa2016/mage-flow-community-mage-flow-edit-turbo` (Edit model).
  - `dangkhoa2016/mage-flow-t4x2-runtime-cache` (canonical Mage runtime archive).
- Stage 01 verifies the attached canonical runtime archive (exact size + SHA-256), restores or reuses the runtime cache, materializes the exact public GitHub project source at the pinned SHA, verifies runtime provenance, and only then imports project modules.
- `RUNTIME_ROOT` existence is the runtime reuse guard: `BOOTSTRAP_RUNTIME_RESTORE_COUNT=0` means the verified runtime was already present; `=1` means it was restored in this notebook session.
- T2I worker binds to `127.0.0.1:8101` and uses `cuda:0`.
- Edit worker binds to `127.0.0.1:8102` and uses `cuda:1`.
- REST coordinator binds to `127.0.0.1:8090`.
- CPU fallback is disabled for the target workflow.
- No terminal bootstrap is required; no repeated extraction, rebuild, or re-download happens after the one-time restore.
- Optional Cloudflare Quick Tunnel may expose only the authenticated coordinator temporarily.
- A saved Kaggle notebook version is a public execution surface; it does not replace the source `.ipynb` in Git.
- Structural notebook validation is complete; live `Run All` and saved-version verification remain future publication steps.
