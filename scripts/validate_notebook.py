#!/usr/bin/env python
"""CPU-only structural validation of the public bilingual notebook.

Checks (CPU-safe, no model load, no network):
  1. Exact cell topology (P1-15): 39 cells, stage 00 markdown-only intro, then
     for stages 01-19 exactly one markdown cell followed by exactly one code
     cell, all in ascending order with no missing/extra/reordered duplicates.
  2. Framework import location (P1-16): torch is imported ONLY inside the stage
     03 (Verify NVIDIA T4 x2) GPU hardware-preflight code cell.
  3. Heartbeat enforcement (P1-17): the long-running helper emits [HEARTBEAT].
  4. Execution-clean source state (P1-18): execution_count is null and outputs
     are empty for every code cell.
  5. Metadata authority (P1-19): Python kernel/language, runtime_target and
     production_claim must be present and non-empty.
  6. No operational ``assert`` safety gates in cells (gates must be explicit
     raises so they survive ``python -O``).
  7. Stages 00-19 appear in order in markdown; a bilingual (EN + VI) markdown
     cell precedes every code cell.
  8. No forbidden internal labels, no CPU fallback, no model imports at cell
     scope, no heavy framework imports at import time.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "mage-flow-t4x2-production-rest-api-demo.ipynb"

FORBIDDEN_LABELS = ("C3", "r4", "C2D1", "R5", "S29", "authority-source")
EXPECTED_STAGES = tuple(f"{n:02d}" for n in range(20))
EXPECTED_TOTAL_CELLS = 1 + 2 * (len(EXPECTED_STAGES) - 1)
GPU_PREFLIGHT_STAGE = "03"
HEARTBEAT_MARKER = "[HEARTBEAT]"
FORBIDDEN_IMPORTS = (
    "import transformers",
    "from transformers",
    "import diffusers",
    "from diffusers",
    "import keras",
    "from keras",
    "import mage_flow",
    "from mage_flow",
    "import accelerate",
    "from accelerate",
    "import safetensors",
    "from safetensors",
)
FORBIDDEN_CLASSES = ("MageFlowPipeline", "MageFlow", "DiffusionPipeline", "AutoPipeline", "StableDiffusionPipeline")

# --- R1K0-R1: exact public-GitHub source + existing runtime-cache bootstrap ---
PROJECT_REPO_URL = "https://github.com/dangkhoa2016/Mage-Flow-Turbo-Dual-T4-REST-API.git"
EXPECTED_PROJECT_SOURCE_SHA = "db6e3fba417b2f6aa0dc5e1dbaf1ccae68c892e0"
PROJECT_ROOT_PATH = "/kaggle/working/mage-flow-t4x2-production-rest-api-demo"
RUNTIME_INPUT_DATASET = "dangkhoa2016/mage-flow-t4x2-runtime-cache"
RUNTIME_INPUT_ROOT_PATH = "/kaggle/input/datasets/dangkhoa2016/mage-flow-t4x2-runtime-cache"
RUNTIME_ARCHIVE_FILENAME = "mage-flow-t4x2-c1-runtime-py312-torch213-cu126.tar.zst"
EXPECTED_RUNTIME_ARCHIVE_SIZE = "3381276345"
EXPECTED_RUNTIME_ARCHIVE_SHA256 = "f1cd0174c7f8b508feafd1132bf934aeb8f15e5f7832d7dc0b68d7ac788d62d5"
RUNTIME_ROOT_PATH = "/kaggle/working/mage-flow-v5-t4x2-c1-concurrency-source-20260912"
EXPECTED_MAGE_COMMIT = "76bec2bb3818863f470de7e867c2dc7f1d0bfd83"

SUPERSEDED_SOURCE_SNAPSHOT_STRINGS = (
    "mage-flow-turbo-dual-t4-rest-api-source-r1a1",
    "project-source-e1825264",
    "SOURCE_COMMIT.txt",
    "source-manifest.json",
)
FORBIDDEN_KAGGLE_CREDENTIALS = ("KAGGLE_TOKEN", "KAGGLE_KEY", "kaggle.json")
FORBIDDEN_NETWORK_BOOTSTRAP = ("pip install", "git pull", "model download", "runtime archive download")
BOOTSTRAP_METADATA_FIELDS = (
    "project_source_transport",
    "project_repository",
    "project_source_commit",
    "runtime_cache_dataset",
    "runtime_archive_sha256",
    "bootstrap_contract",
)


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def expected_topology() -> list[tuple[str, str]]:
    sequence: list[tuple[str, str]] = [("markdown", "00")]
    for n in range(1, len(EXPECTED_STAGES)):
        stage = f"{n:02d}"
        sequence.append(("markdown", stage))
        sequence.append(("code", stage))
    return sequence


def check_topology(
    cells: list[dict[str, Any]],
    md_cells: list[dict[str, Any]],
    code_cells: list[dict[str, Any]],
) -> None:
    if len(cells) != EXPECTED_TOTAL_CELLS:
        fail(
            "stage sequence mismatch: expected exactly "
            f"{EXPECTED_TOTAL_CELLS} cells ({len(EXPECTED_STAGES)} markdown + "
            f"{len(EXPECTED_STAGES) - 1} code), got {len(cells)}"
        )
    if len(md_cells) != len(EXPECTED_STAGES):
        fail(f"stage sequence mismatch: expected {len(EXPECTED_STAGES)} markdown cells, got {len(md_cells)}")
    if len(code_cells) != len(EXPECTED_STAGES) - 1:
        fail(f"stage sequence mismatch: expected {len(EXPECTED_STAGES) - 1} code cells, got {len(code_cells)}")
    expected = expected_topology()
    for index, (cell, (cell_type, stage)) in enumerate(zip(cells, expected, strict=False)):
        if cell["cell_type"] != cell_type:
            fail(
                f"stage sequence mismatch: cell @{index} expected {cell_type} (stage {stage}), got {cell['cell_type']}"
            )
        if cell_type != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if not re.match(rf"#+\s+{re.escape(stage)}\b", source.strip()):
            fail(f"stage sequence mismatch: markdown cell @{index} expected heading stage {stage}")


def check_torch_location(cells: list[dict[str, Any]]) -> None:
    expected = expected_topology()
    preflight_index = next(
        (i for i, (_, stage) in enumerate(expected) if stage == GPU_PREFLIGHT_STAGE and i and expected[i][0] == "code"),
        None,
    )
    torch_pattern = re.compile(r"(?m)^\s*(import torch|from torch)\b")
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if torch_pattern.search(source) and index != preflight_index:
            fail(
                "torch imported outside GPU hardware-preflight cell "
                f"(cell @{index}); allowed only in stage {GPU_PREFLIGHT_STAGE}"
            )


def check_heartbeat(code_cells: list[dict[str, Any]], code_text: str) -> None:
    if HEARTBEAT_MARKER not in code_text:
        fail(f"missing {HEARTBEAT_MARKER} marker; long-running cells must be heartbeat-enabled")
    helper_cells = [c for c in code_cells if "def run_cmd(" in "".join(c.get("source", []))]
    if not helper_cells:
        fail("missing long-command heartbeat helper (def run_cmd(...))")
    helper_ok = any(
        HEARTBEAT_MARKER in "".join(c.get("source", [])) and "heartbeat(" in "".join(c.get("source", []))
        for c in helper_cells
    )
    if not helper_ok:
        fail("run_cmd helper must emit heartbeats through the [HEARTBEAT] heartbeat function")


def check_clean_state(code_cells: list[dict[str, Any]]) -> None:
    for index, cell in enumerate(code_cells):
        if cell.get("execution_count") is not None:
            fail(f"code cell @{index} must have execution_count=null in the source notebook")
        if cell.get("outputs"):
            fail(f"code cell @{index} must have outputs=[] in the source notebook")


def check_no_asserts(cells: list[dict[str, Any]]) -> None:
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        if re.search(r"(?m)^\s*assert\b", source):
            fail(
                f"code cell @{index} uses an operational 'assert' safety gate; "
                "convert to an explicit raise so it survives python -O"
            )


def code_text(cells: list[dict[str, Any]]) -> str:
    return "\n".join("".join(c.get("source", [])) for c in cells if c["cell_type"] == "code")


def check_bootstrap_constants(cells: list[dict[str, Any]]) -> None:
    text = code_text(cells)
    missing = [v for v in (PROJECT_REPO_URL, EXPECTED_PROJECT_SOURCE_SHA, PROJECT_ROOT_PATH) if v not in text]
    if missing:
        fail(f"project source bootstrap constants missing: {missing}")
    missing = [
        v
        for v in (
            RUNTIME_INPUT_DATASET,
            RUNTIME_INPUT_ROOT_PATH,
            RUNTIME_ARCHIVE_FILENAME,
            EXPECTED_RUNTIME_ARCHIVE_SIZE,
            EXPECTED_RUNTIME_ARCHIVE_SHA256,
            RUNTIME_ROOT_PATH,
            EXPECTED_MAGE_COMMIT,
        )
        if v not in text
    ]
    if missing:
        fail(f"runtime bootstrap constants missing: {missing}")
    for needle in SUPERSEDED_SOURCE_SNAPSHOT_STRINGS:
        if needle in text:
            fail(f"superseded source-snapshot design string present: {needle}")
    if "rev-parse HEAD" not in text:
        fail("source bootstrap must verify git rev-parse HEAD")
    if "--detach" not in text:
        fail("source bootstrap must use git checkout --detach to the exact SHA")
    for bare in ("origin/main", "latest"):
        if re.search(rf"\b{re.escape(bare)}\b", text):
            fail(f"source authority must not be a bare branch/ref: {bare}")
    print("[PASS] PROJECT_SOURCE_BOOTSTRAP_CONTRACT_VALID")
    print("[PASS] RUNTIME_CACHE_BOOTSTRAP_CONTRACT_VALID")


def check_bootstrap_order(cells: list[dict[str, Any]]) -> None:
    stage01: str | None = None
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        if index >= 1 and cells[index - 1]["cell_type"] == "markdown":
            previous = "".join(cells[index - 1].get("source", []))
            if re.search(r"#+\s+01\b", previous):
                stage01 = "".join(cell.get("source", []))
                break
    if stage01 is None:
        fail("could not locate Stage 01 code cell")
        return
    lines = stage01.splitlines()

    def line_no(needle: str) -> int:
        for idx, line in enumerate(lines):
            if needle in line:
                return idx
        return -1

    def after(needle_a: str, needle_b: str, label: str) -> None:
        ia, ib = line_no(needle_a), line_no(needle_b)
        if ia == -1:
            fail(f"bootstrap order: missing {needle_a!r} in Stage 01")
        if ib == -1:
            fail(f"bootstrap order: missing {needle_b!r} in Stage 01")
        if not (ia < ib):
            fail(f"bootstrap order violated ({label}): {needle_a} must appear before {needle_b}")

    after(
        "verify_exact_sha256(RUNTIME_ARCHIVE",
        "BOOTSTRAP_RUNTIME_RESTORE_COUNT = restore_runtime_once()",
        "runtime archive hash verification before runtime extraction",
    )
    after(
        "PROJECT_SOURCE_BOOTSTRAP_COUNT = restore_project_source_once()",
        "sys.path.insert(0, str(PROJECT_ROOT))",
        "source materialization/verification before sys.path insertion",
    )
    after(
        "sys.path.insert(0, str(PROJECT_ROOT))",
        "from scripts.token_store import ensure_token",
        "sys.path insertion before token_store import",
    )
    print("[PASS] FRESH_BOOTSTRAP_ORDER_VALID")


def check_credential_and_network_hygiene(cells: list[dict[str, Any]]) -> None:
    text = code_text(cells)
    for needle in FORBIDDEN_KAGGLE_CREDENTIALS:
        if re.search(rf"\b{re.escape(needle)}\b|\bKAGGLE_TOKEN\b|\bKAGGLE_KEY\b|kaggle\.jsons?", text):
            fail(f"forbidden Kaggle credential access pattern: {needle}")
    for needle in FORBIDDEN_NETWORK_BOOTSTRAP:
        if needle in text:
            fail(f"forbidden network/bootstrap pattern: {needle}")
    for needle in ("wget", "curl"):
        if re.search(rf"\b{re.escape(needle)}\b", text):
            fail(f"forbidden model/runtime fetch tool in notebook source: {needle}")
    print("[PASS] CREDENTIAL_HYGIENE_VALID")


def check_bootstrap_metadata(metadata: dict[str, Any]) -> None:
    missing = [field for field in BOOTSTRAP_METADATA_FIELDS if not str(metadata.get(field) or "").strip()]
    if missing:
        fail(f"metadata bootstrap fields missing or empty: {missing}")
    print("[PASS] BOOTSTRAP_METADATA_VALID")


def main():
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else NOTEBOOK
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]

    md_cells = [c for c in cells if c["cell_type"] == "markdown"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    if not code_cells:
        fail("no code cells found")
    for index, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            compile(source, f"notebook-cell-{index}", "exec")
        except SyntaxError as exc:
            fail(f"code cell @{index} does not compile: {exc}")
        if index == 0 or cells[index - 1]["cell_type"] != "markdown":
            fail(f"code cell @{index} has no preceding markdown")
        if not source.strip():
            fail(f"code cell @{index} is empty")

    check_topology(cells, md_cells, code_cells)
    check_torch_location(cells)
    check_no_asserts(cells)

    full_md = [("".join(c.get("source", [])) or "") for c in md_cells]
    combined_md = "\n".join(full_md)
    combined_all = combined_md + "\n" + "\n".join("".join(c.get("source", [])) for c in code_cells)

    for label in FORBIDDEN_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", combined_all):
            fail(f"forbidden internal label present: {label}")

    found_stages: list[str] = []
    for m in full_md:
        if not m.strip():
            continue
        first_heading = re.match(r"#+\s+(\d{2})\b", m.strip())
        if first_heading:
            found_stages.append(first_heading.group(1))
    if found_stages != list(EXPECTED_STAGES):
        fail(f"stage sequence mismatch: expected {list(EXPECTED_STAGES)}, found {found_stages}")

    code_text = "\n".join("".join(c.get("source", [])) for c in code_cells)
    for imp in FORBIDDEN_IMPORTS:
        pat = rf"^\s*{re.escape(imp)}"
        if re.search(pat, code_text, flags=re.M):
            fail(f"forbidden model import at cell scope: {imp}")
    for cls in FORBIDDEN_CLASSES:
        pat = rf"\b{re.escape(cls)}\b"
        if re.search(pat, combined_all):
            fail(f"forbidden model class reference: {cls}")

    en_ok = all(re.search(r"\*\*English\*\*", m) for m in full_md if m.strip())
    vi_ok = all(re.search(r"\*\*Tiếng Việt\*\*", m) for m in full_md if m.strip())
    if not en_ok or not vi_ok:
        fail("not all markdown cells are bilingual (EN/VI)")

    if "[HOLD]" in combined_all:
        fail("public notebook must not contain HOLD text")
    lower = combined_all.lower()
    if re.search(r"cpu_fallback\s*[:=\s]{0,4}true", lower):
        fail("public notebook must not enable CPU fallback")
    if re.search(r"fall[\s-]*back to cpu", lower) or re.search(r"enabl\w+ cpu fallback", lower):
        fail("public notebook must not enable CPU fallback")

    if "[PASS]" not in combined_all:
        fail("no [PASS] markers found")

    check_heartbeat(code_cells, code_text)
    check_clean_state(code_cells)

    if nb.get("nbformat") != 4:
        fail(f"nbformat must be 4, got {nb.get('nbformat')}")
    metadata = nb.get("metadata", {})
    kernelspec = metadata.get("kernelspec", {})
    language_info = metadata.get("language_info", {})
    if kernelspec.get("language") != "python" or kernelspec.get("name") != "python3":
        fail(f"kernel must be Python 3 (python3), got {kernelspec}")
    if language_info.get("name") != "python":
        fail(f"language_info must be python, got {language_info}")
    runtime_target = metadata.get("runtime_target")
    production_claim = metadata.get("production_claim")
    if not runtime_target or not str(runtime_target).strip():
        fail("metadata runtime_target missing or empty")
    if not production_claim or not str(production_claim).strip():
        fail("metadata production_claim missing or empty")

    check_bootstrap_constants(cells)
    check_bootstrap_order(cells)
    check_credential_and_network_hygiene(cells)
    check_bootstrap_metadata(metadata)

    print(f"[INFO] valid notebook cells: {len(code_cells)} code, {len(md_cells)} markdown")
    print("[INFO] exact topology (stage 00 intro + 01-19 markdown/code pairs) enforced")
    print("[INFO] torch restricted to the GPU hardware-preflight cell")
    print(f"[INFO] {HEARTBEAT_MARKER} heartbeat helper enforced")
    print("[INFO] execution-clean state and no operational assert gates enforced")
    print("[INFO] metadata authority enforced (python3 kernel, runtime_target, production_claim)")
    print("[INFO] every code cell compiles (CPU-safe compile, no execution)")
    print("[INFO] stages 00-19 present in order, bilingual markdown before every code cell")
    print("[INFO] no forbidden labels, no CPU fallback, no model imports at cell scope")
    print("[INFO] exact public-GitHub SHA source bootstrap and runtime-cache authority enforced")
    print("[INFO] fresh-bootstrap ordering, credential and network hygiene enforced")
    print("[PASS] NOTEBOOK_STRUCTURE_VALID")


if __name__ == "__main__":
    main()
