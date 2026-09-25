# Kaggle runtime contract

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](kaggle.vi.md)

- Accelerator: NVIDIA T4 x2.
- Before Save & Run All, open **Settings → Input → Add Input** in the Notebook editor and attach exactly one Dataset plus three Models.
- Dataset: `dangkhoa2016/mage-flow-t4x2-runtime-support`.
- Models: `dangkhoa2016/mage-flow-community-mage-flow-turbo`, `dangkhoa2016/mage-flow-community-mage-flow-edit-turbo`, and `dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf`.
- Search exact names and choose resources owned by `dangkhoa2016`; do not manually copy them into `/kaggle/working`.
- Internet is enabled for the public repository checkout.
- Runtime/support Dataset dangkhoa2016/mage-flow-t4x2-runtime-support is attached read-only.
- T2I model is attached as a read-only Kaggle Model input.
- Edit model is attached as a read-only Kaggle Model input.
- Qwen3-VL GGUF safety model is attached as a read-only Kaggle Model input.
- The public notebook clones the repository into /kaggle/working, records the exact Git commit used, verifies and applies the pinned optimization patch, and does not require a pre-created project directory.
- The validated runtime archive is SHA-256 verified and restored from the attached support Dataset into /kaggle/working on a clean run; later invocations in the same session reuse it.
- T2I worker binds to 127.0.0.1:8101 and uses cuda:0.
- Edit worker binds to 127.0.0.1:8102 and uses cuda:1.
- REST coordinator binds to 127.0.0.1:8090.
- CPU fallback is disabled for the target workflow.
- Optional Cloudflare Quick Tunnel may expose only the authenticated coordinator temporarily.
- A saved Kaggle notebook version is a public execution surface; it does not replace the source .ipynb in Git.
- Structural notebook validation is complete; public Save & Run All and saved-version verification are the remaining publication checks.
