# Hướng dẫn đóng góp

> 🌐 Language / Ngôn ngữ: [English](CONTRIBUTING.md) | **Tiếng Việt**

Cảm ơn bạn đã đóng góp cho dự án Mage-Flow T4x2 REST API demo. Mọi thay đổi nên dễ review, có thể tái lập và có thể kiểm tra an toàn mà không cần tải model hoặc GPU, trừ khi thay đổi đó nhắm trực tiếp tới runtime Kaggle T4 x2.

## Thiết lập môi trường phát triển

Sử dụng Python 3.10 trở lên:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Kiểm tra bắt buộc

Trước khi mở pull request, hãy chạy:

```bash
python -m pytest -q
python scripts/validate_bilingual_docs.py
python scripts/validate_notebook.py
python scripts/validate_docs_links.py
python scripts/validate_repository.py
python -m pip check
python -m compileall -q server scripts examples
bash -n scripts/*.sh
shellcheck scripts/*.sh
ruff check .
ruff format --check .
mypy server scripts
```

CI chạy cùng các cổng này cộng thêm các bước build-artifact và dependency-audit trên mỗi push/PR.

Các bước kiểm tra an toàn CPU không được load model Mage và không được âm thầm fallback sang CPU đối với workflow tuyên bố chạy GPU T4 x2.

## Chính sách tài liệu song ngữ

Mỗi tài liệu Markdown được Git theo dõi phải có cặp tiếng Anh/tiếng Việt trong cùng thư mục:

- `name.md`
- `name.vi.md`

Commit thay đổi một phía của cặp tài liệu phải thay đổi phía còn lại trong cùng commit. Mỗi cặp phải có language switcher của repository ở gần đầu tài liệu. Chính sách này được `scripts/validate_bilingual_docs.py` và CI kiểm tra tự động.

## Phạm vi thay đổi

Giữ commit tập trung và có thể review độc lập. Không trộn các thay đổi runtime, tài liệu, publication metadata hoặc notebook không liên quan vào cùng một commit. Không đưa file sinh tự động, credential, model weights, runtime cache hoặc artifact môi trường local vào Git.

## Pull request

Một pull request nên nêu rõ:

- thay đổi gì và vì sao;
- thay đổi có ảnh hưởng T2I, Edit, coordinator, notebook, CI hay publication metadata hay không;
- các lệnh validation đã chạy;
- GPU qualification có bị ảnh hưởng hay vẫn giữ nguyên;
- tác động về tương thích, bảo mật hoặc public API nếu có.

Không tuyên bố Kaggle GPU acceptance, public `Run All`, saved-version verification hoặc trạng thái release nếu chưa có evidence tương ứng.
