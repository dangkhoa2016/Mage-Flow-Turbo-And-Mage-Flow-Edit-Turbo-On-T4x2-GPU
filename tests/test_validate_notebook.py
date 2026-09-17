import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "scripts" / "validate_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "mage-flow-t4x2-production-rest-api-demo.ipynb"


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
    assert "PROJECT_SOURCE_BOOTSTRAP_CONTRACT_VALID" in result.stdout
    assert "RUNTIME_CACHE_BOOTSTRAP_CONTRACT_VALID" in result.stdout
    assert "FRESH_BOOTSTRAP_ORDER_VALID" in result.stdout
    assert "CREDENTIAL_HYGIENE_VALID" in result.stdout


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
    for _i, c in enumerate(nb["cells"]):
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

    # R1K0-R1 bootstrap contract: exact public-GitHub source + runtime cache.
    assert "https://github.com/dangkhoa2016/Mage-Flow-Turbo-Dual-T4-REST-API.git" in code_text
    assert "db6e3fba417b2f6aa0dc5e1dbaf1ccae68c892e0" in code_text
    assert "/kaggle/working/mage-flow-t4x2-production-rest-api-demo" in code_text
    assert "dangkhoa2016/mage-flow-t4x2-runtime-cache" in code_text
    assert "/kaggle/input/datasets/dangkhoa2016/mage-flow-t4x2-runtime-cache" in code_text
    assert "/kaggle/input/mage-flow-t4x2-runtime-cache" not in code_text
    assert "mage-flow-t4x2-c1-runtime-py312-torch213-cu126.tar.zst" in code_text
    assert "3381276345" in code_text
    assert "f1cd0174c7f8b508feafd1132bf934aeb8f15e5f7832d7dc0b68d7ac788d62d5" in code_text
    assert "/kaggle/working/mage-flow-v5-t4x2-c1-concurrency-source-20260912" in code_text
    assert "76bec2bb3818863f470de7e867c2dc7f1d0bfd83" in code_text

    # Strict ordering: source verification -> sys.path insert -> token_store import,
    # and runtime archive hash verification before extraction.
    lines = code_text.splitlines()

    def _first_line(needle):
        for idx, line in enumerate(lines):
            if needle in line:
                return idx
        raise AssertionError(f"missing {needle!r} in notebook code")

    assert _first_line("verify_exact_sha256(RUNTIME_ARCHIVE") < _first_line(
        "BOOTSTRAP_RUNTIME_RESTORE_COUNT = restore_runtime_once()"
    )
    assert _first_line("PROJECT_SOURCE_BOOTSTRAP_COUNT = restore_project_source_once()") < _first_line(
        "sys.path.insert(0, str(PROJECT_ROOT))"
    )
    assert _first_line("sys.path.insert(0, str(PROJECT_ROOT))") < _first_line(
        "from scripts.token_store import ensure_token"
    )

    # No superseded source-snapshot Dataset design, no Kaggle credentials,
    # no pip install / git pull / model or runtime fetch / unversioned checkout.
    for forbidden in (
        "mage-flow-turbo-dual-t4-rest-api-source-r1a1",
        "project-source-e1825264",
        "SOURCE_COMMIT.txt",
        "source-manifest.json",
        "KAGGLE_TOKEN",
        "KAGGLE_KEY",
        "kaggle.json",
        "pip install",
        "git pull",
        "wget",
        "curl",
        "model download",
        "runtime archive download",
        "origin/main",
    ):
        assert forbidden not in code_text, f"forbidden notebook string present: {forbidden}"


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
