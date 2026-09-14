# Giới hạn

> 🌐 Language / Ngôn ngữ: [English](limitations.md) | **Tiếng Việt**

Project này nhắm đến một production-style demo, không phải production service có bảo đảm vận hành.

Project không tuyên bố:

- SLA-backed availability
- High availability
- Multi-tenant isolation
- Public endpoint vĩnh viễn
- Khả năng CPU fallback cho workflow T4 x2 mục tiêu
- Public notebook saved-version verification trước khi bước publication đó thực sự hoàn tất

Cloudflare Quick Tunnel, khi được dùng, chỉ mang tính tạm thời và gắn với vòng đời Kaggle session.

GPU acceptance chứng minh candidate path T4 x2 đã được kiểm thử; nó không phải bảo đảm cho phần cứng tùy ý, model revision khác hoặc dependency version trong tương lai.
