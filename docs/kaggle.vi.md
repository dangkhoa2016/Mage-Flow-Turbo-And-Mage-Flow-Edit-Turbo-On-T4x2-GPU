# Kaggle runtime contract

> 🌐 Language / Ngôn ngữ: [English](kaggle.md) | **Tiếng Việt**

- Accelerator: NVIDIA T4 x2.
- Trước Save & Run All, mở **Settings → Input → Add Input** trong Notebook editor và attach đúng một Dataset cùng ba Model.
- Dataset: `dangkhoa2016/mage-flow-t4x2-runtime-support`.
- Models: `dangkhoa2016/mage-flow-community-mage-flow-turbo`, `dangkhoa2016/mage-flow-community-mage-flow-edit-turbo` và `dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf`.
- Tìm đúng tên và chọn resource có owner `dangkhoa2016`; không copy thủ công chúng vào `/kaggle/working`.
- Internet được bật để checkout public repository.
- Runtime/support Dataset dangkhoa2016/mage-flow-t4x2-runtime-support được attach read-only.
- T2I model được attach dưới dạng Kaggle Model input read-only.
- Edit model được attach dưới dạng Kaggle Model input read-only.
- Qwen3-VL GGUF safety model được attach dưới dạng Kaggle Model input read-only.
- Public notebook clone repository vào /kaggle/working, ghi lại chính xác Git commit được dùng, xác minh và áp dụng pinned optimization patch, và không yêu cầu project directory được tạo sẵn.
- Validated runtime archive được xác minh SHA-256 và restore từ support Dataset đã attach vào /kaggle/working trong clean run; các lần chạy sau trong cùng session sẽ tái sử dụng runtime này.
- T2I worker bind 127.0.0.1:8101 và dùng cuda:0.
- Edit worker bind 127.0.0.1:8102 và dùng cuda:1.
- REST coordinator bind 127.0.0.1:8090.
- CPU fallback bị vô hiệu hóa trong workflow mục tiêu.
- Cloudflare Quick Tunnel tùy chọn chỉ expose coordinator đã có authentication và chỉ mang tính tạm thời.
- Saved Kaggle notebook version là một public execution surface; nó không thay thế source .ipynb trong Git.
- Structural notebook validation đã hoàn tất; public Save & Run All và saved-version verification là các publication check còn lại.
