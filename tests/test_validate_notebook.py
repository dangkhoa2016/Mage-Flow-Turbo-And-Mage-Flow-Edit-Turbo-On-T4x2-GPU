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


def _intro_cell(nb):
    cell = nb["cells"][0]
    assert cell["cell_type"] == "markdown"
    source = "".join(cell["source"])
    assert source.startswith("# Mage-Flow-Turbo and Mage-Flow-Edit-Turbo on T4x2 GPU - Demo")
    assert "## Introduction + requirements" in source
    return cell


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


def test_validator_rejects_numbered_intro(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _intro_cell(nb)
    for i, line in enumerate(cell["source"]):
        if line.startswith("## Introduction"):
            cell["source"][i] = line.replace("## Introduction", "## 00. Introduction", 1)
            break
    else:
        raise AssertionError("introduction heading not found")
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "stage sequence mismatch" in result.stdout


def test_validator_rejects_forbidden_label(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _intro_cell(nb)
    cell["source"].append("\n[private-workflow C27X reference]\n")
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "private workflow label present" in result.stdout


def test_validator_rejects_missing_bilingual_markdown(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and "**English**" in "".join(c["source"]):
            c["source"] = [line.replace("**Tiếng Việt**", "**Tieng Viet**") for line in c["source"]]
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


def _find_code_cell(nb, needle):
    for c in nb["cells"]:
        if c["cell_type"] == "code" and needle in "".join(c["source"]):
            return c
    raise AssertionError(f"code cell containing {needle!r} not found")


def _drop_cell(nb, index):
    nb["cells"].pop(index)


def test_validator_enforces_exact_cell_topology(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    _drop_cell(nb, 10)  # remove a code cell -> 38 cells
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "stage sequence mismatch" in result.stdout


def test_validator_requires_clean_execution_state(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _find_code_cell(nb, "def run_cmd(")
    cell["execution_count"] = 7
    cell["outputs"] = [{"output_type": "stream", "text": ["stale"]}]
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "execution_count" in result.stdout


def test_validator_requires_metadata_authority(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    nb["metadata"].pop("runtime_target", None)
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "runtime_target" in result.stdout


def test_validator_restricts_torch_to_preflight_cell(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _find_code_cell(nb, "import platform")
    cell["source"] = ["import torch\n"] + cell["source"]
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "torch imported outside" in result.stdout


def test_validator_rejects_operational_assert_gates(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _find_code_cell(nb, "def heartbeat(")
    cell["source"] = ["assert 1 == 1\n"] + cell["source"]
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "operational 'assert'" in result.stdout


def test_validator_requires_heartbeat_helper(mutated_notebook):
    nb = json.loads(mutated_notebook.read_text())
    cell = _find_code_cell(nb, "def heartbeat(")
    cell["source"] = [line.replace("[HEARTBEAT]", "[BEAT]") for line in cell["source"]]
    mutated_notebook.write_text(json.dumps(nb))
    result = _run_validator(mutated_notebook)
    assert result.returncode != 0
    assert "HEARTBEAT" in result.stdout


def test_notebook_rejects_stale_partial_runtime_before_reuse():
    nb = json.loads(NOTEBOOK.read_text())
    code_text = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
    assert "os.access(RUNTIME_PYTHON, os.X_OK)" in code_text
    assert "stale/partial runtime detected; restoring clean runtime" in code_text
    assert "shutil.rmtree(RUNTIME_ROOT)" in code_text


def test_notebook_refreshes_existing_public_checkout():
    nb = json.loads(NOTEBOOK.read_text())
    code_text = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
    assert '"fetch", "--depth", "1", "origin", PUBLIC_BRANCH' in code_text
    assert '"reset", "--hard", "FETCH_HEAD"' in code_text
    assert "refreshing existing public source checkout to latest main" in code_text
