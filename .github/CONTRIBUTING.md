# Contributing

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Thank you for contributing to the Mage-Flow T4x2 REST API demo. Changes should remain reviewable, reproducible, and safe to validate without requiring model downloads or GPUs unless a change explicitly targets the Kaggle T4 x2 runtime.

## Development setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Required validation

Before opening a pull request, run:

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

CI runs these same gates plus the build-artifact and dependency-audit steps on every push/PR.

CPU-safe validation must not load the Mage models and must not silently fall back to CPU for workflows that claim T4 x2 GPU execution.

## Bilingual documentation policy

Every tracked Markdown document must have an English/Vietnamese pair in the same directory:

- `name.md`
- `name.vi.md`

A commit that changes one side of a documentation pair must change the other side in the same commit. Each pair must include the repository language switcher near the top of the document. The policy is enforced by `scripts/validate_bilingual_docs.py` and CI.

## Change scope

Keep commits focused and independently reviewable. Do not mix unrelated runtime, documentation, publication metadata, or notebook changes in one commit. Avoid generated files, credentials, model weights, runtime caches, and local environment artifacts.

## Pull requests

A pull request should explain:

- what changed and why;
- whether the change affects T2I, Edit, the coordinator, notebook, CI, or publication metadata;
- which validation commands were run;
- whether GPU qualification is affected or remains unchanged;
- any compatibility, security, or public-API impact.

Do not claim Kaggle GPU acceptance, public `Run All`, saved-version verification, or release status without matching evidence.
