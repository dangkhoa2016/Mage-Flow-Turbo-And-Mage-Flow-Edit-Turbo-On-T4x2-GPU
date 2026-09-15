# Pull Request

> 🌐 Language / Ngôn ngữ: [English](PULL_REQUEST_TEMPLATE.md) | **Tiếng Việt**

## Tóm tắt

<!-- Thay đổi gì và vì sao? Giữ PR tập trung vào một thay đổi có thể review rõ ràng. -->

## Phạm vi

- [ ] Runtime / worker
- [ ] REST coordinator / API
- [ ] Test / CI
- [ ] Notebook / tích hợp Kaggle
- [ ] Tài liệu / publication metadata
- [ ] Khác

## Kiểm tra

- [ ] `python -m pytest -q`
- [ ] `python scripts/validate_bilingual_docs.py`
- [ ] `python scripts/validate_notebook.py`
- [ ] `python -m pip check`
- [ ] Tôi không thêm credential, model weights, runtime cache hoặc secret sinh tự động.

## Tài liệu

- [ ] Không có tài liệu Markdown thay đổi, hoặc mọi file `.md` thay đổi đều có file `.vi.md` tương ứng thay đổi trong cùng commit/PR.
- [ ] Language switcher hợp lệ.

## GPU / release claims

<!-- Nêu rõ GPU qualification không bị ảnh hưởng, cần requalification hay đã có evidence mới. Không suy diễn public Run All hoặc trạng thái release từ CPU-safe CI. -->

## Tác động bảo mật / tương thích

<!-- Ghi rõ ảnh hưởng tới authentication, public API, dependency, backward compatibility hoặc bảo mật. -->
