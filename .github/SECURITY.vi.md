# Chính sách bảo mật

> 🌐 Language / Ngôn ngữ: [English](SECURITY.md) | **Tiếng Việt**

## Phiên bản được hỗ trợ

Các bản sửa lỗi bảo mật được áp dụng cho nhánh `main` hiện tại và, sau khi có stable release đầu tiên, cho stable release mới nhất còn được hỗ trợ khi phù hợp. Snapshot pre-release và commit lịch sử không được đảm bảo backport bản vá.

## Báo cáo lỗ hổng

Không mở public issue cho lỗ hổng nghi ngờ, credential bị lộ, authentication bypass, xử lý file không an toàn, đường dẫn remote code execution, dependency compromise hoặc phát hiện nhạy cảm về bảo mật khác.

Hãy dùng GitHub private vulnerability reporting khi repository đã bật tính năng này. Nếu kênh đó chưa khả dụng, gửi email tới `i.am@dangkhoa.dev` với:

- mô tả ngắn gọn vấn đề;
- component và revision bị ảnh hưởng;
- bước tái hiện hoặc proof of concept;
- tác động bảo mật dự kiến;
- biện pháp giảm thiểu đề xuất nếu đã biết.

Không đưa credential production thật hoặc dữ liệu riêng tư của bên thứ ba vào báo cáo.

## Kỳ vọng phản hồi

Báo cáo sẽ được xác nhận và phân loại tùy theo khả năng xử lý. Vấn đề được xác nhận sẽ được xử lý theo quy trình sửa lỗi và công bố có phối hợp. Việc công bố công khai nên chờ tới khi có bản sửa hoặc biện pháp giảm thiểu, trừ khi cần công bố ngay để bảo vệ người dùng.

## Ranh giới bảo mật riêng của dự án

REST coordinator công khai được thiết kế với Bearer token tạm thời và model worker chỉ bind localhost. Đây là production-style demo, không phải cam kết về multi-tenant isolation, high availability hoặc SLA bảo mật cho dịch vụ internet-facing.
