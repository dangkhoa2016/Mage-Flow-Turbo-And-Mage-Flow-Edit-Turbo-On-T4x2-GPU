# Kaggle runtime contract

> 🌐 Language / Ngôn ngữ: [English](kaggle.md) | **Tiếng Việt**

- Accelerator: NVIDIA T4 x2.
- T2I model được attach dưới dạng Kaggle Model input read-only.
- Edit model được attach dưới dạng Kaggle Model input read-only.
- Mage runtime đã xác minh được restore dưới `/kaggle/working`.
- T2I worker bind `127.0.0.1:8101` và dùng `cuda:0`.
- Edit worker bind `127.0.0.1:8102` và dùng `cuda:1`.
- REST coordinator bind `127.0.0.1:8090`.
- CPU fallback bị vô hiệu hóa trong workflow mục tiêu.
- Cloudflare Quick Tunnel tùy chọn chỉ được expose coordinator đã có authentication và chỉ mang tính tạm thời.
- Saved Kaggle notebook version là một public execution surface; nó không thay thế source `.ipynb` trong Git.
- Structural notebook validation đã hoàn tất, còn public live `Run All` và saved-version verification vẫn là các bước publication.
