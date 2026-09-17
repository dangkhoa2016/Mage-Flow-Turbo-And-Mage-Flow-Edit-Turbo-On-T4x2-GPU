# Kaggle runtime contract

> 🌐 Language / Ngôn ngữ: [English](kaggle.md) | **Tiếng Việt**

- Accelerator: NVIDIA T4 x2.
- Internet: ON — Stage 01 lấy đúng mã nguồn public từ GitHub tại commit đã chốt.
- Ba Kaggle input read-only được attach:
  - `dangkhoa2016/mage-flow-community-mage-flow-turbo` (model T2I).
  - `dangkhoa2016/mage-flow-community-mage-flow-edit-turbo` (model Edit).
  - `dangkhoa2016/mage-flow-t4x2-runtime-cache` (archive runtime Mage chuẩn).
- Stage 01 xác minh archive runtime chuẩn đã attach (đúng size + SHA-256), khôi phục hoặc tái sử dụng runtime cache, đưa đúng mã nguồn public từ GitHub tại SHA đã chốt vào `PROJECT_ROOT`, xác minh provenance runtime, và chỉ sau đó mới import các module của dự án.
- Sự tồn tại của `RUNTIME_ROOT` là guard tái sử dụng runtime: `BOOTSTRAP_RUNTIME_RESTORE_COUNT=0` nghĩa là runtime đã được xác minh và có sẵn; `=1` nghĩa là runtime được khôi phục trong session notebook này.
- T2I worker bind `127.0.0.1:8101` và dùng `cuda:0`.
- Edit worker bind `127.0.0.1:8102` và dùng `cuda:1`.
- REST coordinator bind `127.0.0.1:8090`.
- CPU fallback bị vô hiệu hóa trong workflow mục tiêu.
- Không cần chuẩn bị terminal; không có lần giải nén, rebuild hay tải lại nào sau khi khôi phục một lần.
- Cloudflare Quick Tunnel tùy chọn chỉ được expose coordinator đã có authentication và chỉ mang tính tạm thời.
- Saved Kaggle notebook version là một public execution surface; nó không thay thế source `.ipynb` trong Git.
- Structural notebook validation đã hoàn tất; live `Run All` và saved-version verification vẫn là các bước publication.
