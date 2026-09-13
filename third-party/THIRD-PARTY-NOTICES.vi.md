# THÔNG BÁO BÊN THỨ BA — MAGE-FLOW T4x2 Public Canonicalization

> 🌐 Language / Ngôn ngữ: [English](THIRD-PARTY-NOTICES.md) | **Tiếng Việt**

Cây mã nguồn này phân phối lại hoặc tham chiếu các thành phần của bên thứ ba. Mỗi thành phần được
liệt kê kèm nguồn gốc thượng nguồn, cơ quan cấp phép áp dụng và điều kiện phân phối chính xác.

## 1. llama.cpp (nhị phân llama-server thời gian chạy)
- Nguồn gốc: ggerganov/llama.cpp (MIT — https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)
- Bề mặt thời gian chạy: `runtime/bin/llama-server` trong Dataset hỗ trợ (bản sao đóng băng theo byte của
  bản build llama.cpp DFlash2 đã được đánh giá; SHA trong manifest/SHA256SUMS).
- Điều kiện phân phối: đây là bản sao theo byte của một build được phân phối qua Kaggle public Dataset
  `mage-flow-glimmer-30b-dflash2-llama-runtime`; các byte được phục vụ không đổi. Văn bản giấy phép MIT
  được cung cấp nguyên văn tại `third-party/licenses/MIT-llama.cpp.md`. Chúng tôi không nhận là tác giả của
  llama.cpp; nguồn gốc nhị phân và mã băm bản build được ghi trong runtime-provenance.txt.

## 2. stable-diffusion.cpp (thời gian chạy gguf an toàn CPU; không nằm trong Dataset hợp nhất)
- Nguồn gốc: leejet/stable-diffusion.cpp (MIT — https://github.com/leejet/stable-diffusion.cpp)
- Không gộp vào Dataset thời gian chạy hợp nhất; giữ trên các tài nguyên Kaggle hỗ trợ riêng biệt.

## 3. Trọng số mô hình Cộng đồng Mage-Flow (tái phân phối mở)
- Chủ sở hữu mô hình thượng nguồn (Turbo / Edit Turbo cộng đồng Mage-Flow) cấp phép trọng số của họ cho
  mục đích tái phân phối cộng đồng qua Kaggle Models. Dataset chuẩn không đóng gói trọng số;
  nó tham chiếu tới các bề mặt Kaggle Models công khai.

## 4. Các phụ thuộc Python/Qwen/opensource được sử dụng bởi cây mã nguồn
- Chỉ các nhị phân được tái phân phối thực chất được trích dẫn ở trên. Các phụ thuộc cây mã nguồn được cài
  đặt từ các wheel mirror công khai tại thời điểm chạy và được liệt kê trong requirements; không có tệp thực
  thi bên thứ ba nào được đóng gói cùng kho lưu trữ mã nguồn chuẩn.

Liên hệ: xem LICENSE (dự án) — MIT, các tác giả Mage-Flow.
