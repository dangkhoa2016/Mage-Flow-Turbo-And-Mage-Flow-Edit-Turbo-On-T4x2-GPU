# Limitations

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](limitations.vi.md)

This project targets a reproducible public Kaggle workflow, not a production service with operational guarantees.

It does not claim:

- SLA-backed availability
- High availability
- Multi-tenant isolation
- Permanent public endpoint
- CPU fallback compatibility for the intended T4 x2 workflow
- Public notebook saved-version verification before that publication step is actually completed

Cloudflare Quick Tunnel, when used, is temporary and tied to the Kaggle session lifetime.

GPU acceptance proves the tested T4 x2 acceptance surface; it is not a guarantee for arbitrary hardware, model revisions, or future dependency versions.
