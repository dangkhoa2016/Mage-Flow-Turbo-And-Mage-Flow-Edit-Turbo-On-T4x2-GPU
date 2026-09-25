# Kiến trúc

> 🌐 Language / Ngôn ngữ: [English](architecture.md) | **Tiếng Việt**

```text
                   public/authenticated boundary
                            |
                    REST API coordinator
                    system Python
                    127.0.0.1:8090
                         /      \
                        /        \
           localhost internal    localhost internal
                HTTP :8101            HTTP :8102
                    |                      |
               T2I worker             Edit worker
        validated Mage .venv     validated Mage .venv
               cuda:0                 cuda:1
           Mage-Flow Turbo       Mage-Flow Edit Turbo
```

Coordinator chủ đích không import model runtime Mage/PyTorch nặng. Mỗi model worker chạy trong Mage virtual environment đã được restore và xác minh, còn FastAPI coordinator public chạy trong Python environment bình thường của Kaggle.

Các interface nội bộ của worker chỉ bind localhost và không bao giờ được expose trực tiếp qua Quick Tunnel.

Overall readiness luôn fail-closed: `/ready` chỉ chuyển sang ready khi cả hai worker tự nhận diện đúng và healthy trên device bắt buộc. Không cho phép tự động fallback sang CPU.

## T2I worker

Mage-Flow Turbo chạy trên `cuda:0`:

- dùng Kaggle model mount local; không download model khi mount đã tồn tại,
- dùng runtime Python đã xác minh từ `.venv` được restore,
- strict T4 x2 hardware gate trước khi load model,
- transformer và VAE trên `cuda:0`, text encoder trên CPU theo runtime layout đã được acceptance,
- giữ nguyên upstream prompt screening và Gaussian-Shading behavior,
- generation đồng bộ được serialize bên trong worker,
- PNG sinh ra được trả về dưới dạng `data:image/png;base64,...`.

## Edit worker

Mage-Flow Edit Turbo chạy trên `cuda:1`:

- dùng Kaggle model mount local,
- dùng Mage runtime đã restore và xác minh,
- transformer và VAE trên `cuda:1`, text encoder trên CPU,
- inference setting public đã freeze: `steps=4`, `cfg=1.0`, `prompt_template="mage-flow-edit"`, `vl_cond_long_edge=384`, negative prompt rỗng, kích thước tối đa `1024`,
- ảnh nguồn đi qua coordinator đến localhost worker,
- PNG sinh ra được trả về dưới dạng data URL.

## Mô hình resident service

`scripts/run_public_acceptance.sh` khởi động hoặc tái sử dụng T2I/Edit worker đang healthy và chạy control-plane acceptance mà không generate ảnh. Các stage inference sau đó của notebook thực hiện một request thật cho mỗi model. Evidence reuse ghi nhận worker residency ổn định và không re-extract runtime trên reuse path.

## Authentication boundary

- `GET /health` là liveness endpoint tối giản không cần authentication.
- `GET /ready` và toàn bộ `/v1/*` yêu cầu `Authorization: Bearer <MAGE_FLOW_API_TOKEN>`.
- Workflow tạo hoặc tái sử dụng token tạm theo session.
- Cloudflare Quick Tunnel, nếu bật, chỉ forward đến coordinator đã có authentication.

## Phạm vi production claim

Repository này trình diễn workflow dual-GPU public với fail-closed behavior. Nó không tuyên bố SLA-backed availability, high availability hoặc multi-tenant isolation.
