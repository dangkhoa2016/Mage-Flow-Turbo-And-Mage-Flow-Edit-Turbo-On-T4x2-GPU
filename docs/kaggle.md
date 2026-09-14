# Kaggle runtime contract

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](kaggle.vi.md)

- Accelerator: NVIDIA T4 x2.
- T2I model is attached as a read-only Kaggle Model input.
- Edit model is attached as a read-only Kaggle Model input.
- Validated Mage runtime is restored under `/kaggle/working`.
- T2I worker binds to `127.0.0.1:8101` and uses `cuda:0`.
- Edit worker binds to `127.0.0.1:8102` and uses `cuda:1`.
- REST coordinator binds to `127.0.0.1:8090`.
- CPU fallback is disabled for the target workflow.
- Optional Cloudflare Quick Tunnel may expose only the authenticated coordinator temporarily.
- A saved Kaggle notebook version is a public execution surface; it does not replace the source `.ipynb` in Git.
- Structural notebook validation is already complete, while public live `Run All` and saved-version verification remain publication steps.
