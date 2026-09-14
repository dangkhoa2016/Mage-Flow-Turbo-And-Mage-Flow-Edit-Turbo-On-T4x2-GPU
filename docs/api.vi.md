# REST API contract

> 🌐 Language / Ngôn ngữ: [English](api.md) | **Tiếng Việt**

Base URL bên trong Kaggle:

```text
http://127.0.0.1:8090
```

Khi bật Cloudflare Quick Tunnel, tunnel chỉ forward đến coordinator đã có authentication. Hai port worker nội bộ `8101` và `8102` luôn chỉ bind localhost và không được expose trực tiếp.

## Authentication

`GET /health` là liveness check tối giản không cần authentication.

Các endpoint sau yêu cầu:

```http
Authorization: Bearer <MAGE_FLOW_API_TOKEN>
```

- `GET /ready`
- `GET /v1/info`
- `POST /v1/images/generations`
- `POST /v1/images/edits`

Workflow Kaggle tạo một token tạm đủ mạnh cho mỗi session và không in giá trị token ra log.

## GET /health

Chỉ trả trạng thái sống của coordinator:

```json
{"status":"ok","coordinator":"healthy"}
```

## GET /ready

Overall readiness hoạt động fail-closed. `ready=true` chỉ khi cả T2I và Edit worker đều báo ready trên đúng device bắt buộc.

## GET /v1/info

Trả runtime contract public và readiness hiện tại của cả hai worker. Routing contract mong đợi:

```text
T2I  -> cuda:0
Edit -> cuda:1
CPU fallback -> false
```

## POST /v1/images/generations

JSON body:

```json
{
  "prompt": "A cinematic mountain lake at sunrise",
  "seed": 42,
  "steps": 4,
  "width": 1024,
  "height": 1024
}
```

`width` và `height` phải là bội số của 16. Public T2I path chạy Mage-Flow Turbo trên physical `cuda:0`. Public API bị giới hạn về profile acceptance đã xác minh bằng GPU: `steps=4`, `width=1024`, `height=1024`.

Dạng response thành công:

```json
{
  "id": "img_...",
  "status": "completed",
  "model": "mage-flow-turbo",
  "device": "cuda:0",
  "seed": 42,
  "width": 1024,
  "height": 1024,
  "elapsed_seconds": 18.42,
  "output": "data:image/png;base64,..."
}
```

Output là PNG data URL để client từ xa không phụ thuộc vào đường dẫn file local của Kaggle.

## POST /v1/images/edits

Các field multipart form:

- `image`: file ảnh nguồn (PNG/JPEG/WebP, tối đa 8 MiB)
- `prompt`: chỉ dẫn chỉnh sửa
- `seed`: integer tùy chọn, mặc định `42`

Public coordinator kiểm tra giới hạn tải lên tối đa **8 MiB** trước khi chuyển tiếp. Upload vượt quá giới hạn trả về `413`, media type không hỗ trợ trả về `415`, upload rỗng trả về `400`.

Dạng response thành công:

```json
{
  "id": "img_...",
  "status": "completed",
  "model": "mage-flow-edit-turbo",
  "device": "cuda:1",
  "seed": 42,
  "elapsed_seconds": 11.25,
  "output": "data:image/png;base64,..."
}
```

Edit runtime dùng các tham số public đã freeze, gồm `steps=4`, `cfg=1.0`, `prompt_template="mage-flow-edit"`, `vl_cond_long_edge=384`, và không tự động fallback sang CPU.

## Chính sách lỗi

- `400/422`: request người dùng không hợp lệ (ví dụ: prompt rỗng, prompt chỉ chứa khoảng trắng, JSON sai định dạng, thiếu các trường bắt buộc)
- `401`: thiếu Bearer token
- `403`: Bearer token không hợp lệ
- `413`: ảnh tải lên vượt quá giới hạn public
- `415`: media type ảnh không được hỗ trợ
- `502`: coordinator gọi được worker nhưng worker request thất bại
- `503`: worker bắt buộc chưa ready
- `5xx`: lỗi runtime/model không mong đợi

Không endpoint public nào âm thầm fallback sang CPU. Prompt chỉ chứa khoảng trắng bị từ chối ở ranh giới public trước khi gọi worker.
