# Pull Request

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PULL_REQUEST_TEMPLATE.vi.md)

## Summary

<!-- What changed and why? Keep this focused on one reviewable change. -->

## Scope

- [ ] Runtime / worker
- [ ] REST coordinator / API
- [ ] Tests / CI
- [ ] Notebook / Kaggle integration
- [ ] Documentation / publication metadata
- [ ] Other

## Validation

- [ ] `python -m pytest -q`
- [ ] `python scripts/validate_bilingual_docs.py`
- [ ] `python scripts/validate_notebook.py`
- [ ] `python -m pip check`
- [ ] I did not add credentials, model weights, runtime caches, or generated secrets.

## Documentation

- [ ] No Markdown documentation changed, or every changed `.md` file has its `.vi.md` counterpart changed in this same commit/PR.
- [ ] Language switchers are valid.

## GPU / release claims

<!-- State whether GPU qualification is unaffected, requires requalification, or has new evidence. Do not infer public Run All or release status from CPU-safe CI. -->

## Security / compatibility impact

<!-- Note authentication, public API, dependency, backward-compatibility, or security implications. -->
