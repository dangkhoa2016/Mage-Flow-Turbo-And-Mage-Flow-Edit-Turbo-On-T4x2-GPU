import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "scripts" / "validate_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "mage-flow-turbo-and-mage-flow-edit-turbo-on-t4x2-gpu.ipynb"


def _run_validator(nb_path):
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(nb_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture()
def mutated_notebook(tmp_path):
    nb = json.loads(NOTEBOOK.read_text())
    out = tmp_path / "mutated.ipynb"
    out.write_text(json.dumps(nb))
    return out


def test_validator_passes_on_generated_notebook():
    result = _run_validator(NOTEBOOK)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NOTEBOOK_STRUCTURE_VALID" in result.stdout


def _find_stage_cell(nb, stage: str):
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and f"## {stage}." in "".join(c["source"]):
            return c
    raise AssertionError(f"stage cell {stage} not found")


def test_validator_rejects_missing_stage(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _find_stage_cell(nb, "12")
    cell["source"] = cell["source"][0].replace("## 12.", "## Twenty.", 2)
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "stage sequence mismatch" in result.stdout


def test_validator_rejects_duplicate_stage(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _find_stage_cell(nb, "13")
    cell["source"] = cell["source"][0].replace("## 13.", "## 12.", 2)
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "stage sequence mismatch" in result.stdout


def test_validator_rejects_reordered_stage(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell11 = _find_stage_cell(nb, "11")
    cell12 = _find_stage_cell(nb, "12")
    tmp = cell11["source"]
    cell11["source"] = cell12["source"]
    cell12["source"] = tmp
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "stage sequence mismatch" in result.stdout


def test_validator_rejects_unknown_stage(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _find_stage_cell(nb, "00")
    cell["source"] = cell["source"][0].replace("## 00.", "## 99.", 2)
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "stage sequence mismatch" in result.stdout


def test_validator_rejects_forbidden_label(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _find_stage_cell(nb, "00")
    cell["source"] = [cell["source"][0] + "\n[internal S29 reference]"]
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "forbidden internal label" in result.stdout


def test_validator_rejects_missing_bilingual_markdown(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and "**English**" in "".join(c["source"]):
            c["source"] = [c["source"][0].replace("**Tiếng Việt**", "**Tieng Viet**")]
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "not all markdown cells are bilingual" in result.stdout


def test_notebook_token_file_delegates_to_fail_closed_token_store():
    nb = json.loads(NOTEBOOK.read_text())
    code_text = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
    assert "from scripts.token_store import ensure_token" in code_text
    assert "token_file.write_text(" not in code_text


def _first_code_cell():
    nb = json.loads(NOTEBOOK.read_text())
    return next(c for c in nb["cells"] if c["cell_type"] == "code")


def test_validator_rejects_invalid_python_code_cell():
    nb = json.loads(NOTEBOOK.read_text())
    code_index = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    nb["cells"][code_index]["source"] = ["broken = = not valid python\n"]

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        mutated = Path(td) / "mutated.ipynb"
        mutated.write_text(json.dumps(nb))
        result = _run_validator(mutated)
    assert result.returncode != 0
    assert "does not compile" in result.stdout


def test_notebook_code_cells_are_newline_terminated_fragments():
    nb = json.loads(NOTEBOOK.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for cell in code:
        for fragment in cell.get("source", []):
            assert fragment.endswith("\n"), f"fragment without newline: {fragment[:40]!r}"


def _extract_notebook_functions(cell, names):
    src = "".join(cell.get("source", []))
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            out[node.name] = ast.get_source_segment(src, node)
    for name in names:
        assert name in out, f"function {name} missing from notebook cell"
    return out


def test_notebook_token_mode_new_and_reused_is_enforced(tmp_path):
    cell = _first_code_cell()
    funcs = _extract_notebook_functions(cell, {"resolve_token"})
    project_root = tmp_path / "project"
    ns = {
        "Path": Path,
        "os": os,
        "PROJECT_ROOT": project_root,
    }
    exec("from pathlib import Path\nos = __import__('os')\n", ns)
    exec(funcs["resolve_token"], ns)

    token = project_root / ".runtime" / "api_token"
    os.environ.pop("MAGE_FLOW_API_TOKEN", None)
    value = ns["resolve_token"]()
    assert value and len(value) >= 32
    assert oct(token.stat().st_mode & 0o777) == "0o600"
    assert oct(token.parent.stat().st_mode & 0o777) == "0o700"

    os.chmod(token, 0o644)  # simulate a leaked-permission reused token
    os.chmod(token.parent, 0o755)
    reused = ns["resolve_token"]()
    assert reused == value
    assert oct(token.stat().st_mode & 0o777) == "0o600"
    assert oct(token.parent.stat().st_mode & 0o777) == "0o700"
