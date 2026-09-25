#!/usr/bin/env python
"""CPU-only structural validation of the public bilingual notebook.

Checks (CPU-safe, no model load, no network):
  1. Exact cell topology: 39 cells, one unnumbered bilingual introduction,
     then for stages 01-19 exactly one markdown cell followed by exactly one
     code cell, all in ascending order with no missing/extra/reordered duplicates.
  2. Framework import location: torch is imported ONLY inside the stage
     03 (Verify NVIDIA T4 x2) GPU hardware-preflight code cell.
  3. Heartbeat enforcement: the long-running helper emits [HEARTBEAT].
  4. Execution-clean source state: execution_count is null and outputs
     are empty for every code cell.
  5. Metadata authority: Python kernel/language, runtime_target and
     production_claim must be present and non-empty.
  6. No operational ``assert`` safety gates in cells (gates must be explicit
     raises so they survive ``python -O``).
  7. Numbered stages 01-19 appear in order in markdown; a bilingual (EN + VI)
     markdown cell precedes every code cell.
  8. No forbidden internal labels, no CPU fallback, no model imports at cell
     scope, no heavy framework imports at import time.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "mage-flow-turbo-and-mage-flow-edit-turbo-on-t4x2-gpu.ipynb"

PRIVATE_WORKFLOW_LABEL_PATTERNS = (
    r"\b[crs]\d+[a-z0-9]*\b",
    r"\[[^]\n]*private[-_ ]workflow[^]\n]*\]",
)
EXPECTED_STAGES = tuple(f"{n:02d}" for n in range(1, 20))
EXPECTED_TOTAL_CELLS = 1 + 2 * len(EXPECTED_STAGES)
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


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def expected_topology() -> list[tuple[str, str]]:
    sequence: list[tuple[str, str]] = [("markdown", "intro")]
    for stage in EXPECTED_STAGES:
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
            f"{EXPECTED_TOTAL_CELLS} cells ({len(EXPECTED_STAGES) + 1} markdown + "
            f"{len(EXPECTED_STAGES)} code), got {len(cells)}"
        )
    if len(md_cells) != len(EXPECTED_STAGES) + 1:
        fail(f"stage sequence mismatch: expected {len(EXPECTED_STAGES) + 1} markdown cells, got {len(md_cells)}")
    if len(code_cells) != len(EXPECTED_STAGES):
        fail(f"stage sequence mismatch: expected {len(EXPECTED_STAGES)} code cells, got {len(code_cells)}")
    expected = expected_topology()
    for index, (cell, (cell_type, stage)) in enumerate(zip(cells, expected, strict=False)):
        if cell["cell_type"] != cell_type:
            fail(
                f"stage sequence mismatch: cell @{index} expected {cell_type} (stage {stage}), got {cell['cell_type']}"
            )
        if cell_type != "markdown":
            continue
        source = "".join(cell.get("source", []))
        if stage == "intro":
            expected_title = "# Mage-Flow-Turbo and Mage-Flow-Edit-Turbo on T4x2 GPU - Demo"
            stripped = source.strip()
            if not stripped.startswith(expected_title):
                fail("stage sequence mismatch: markdown cell @0 expected project title")
            if "## Introduction + requirements" not in source:
                fail("stage sequence mismatch: markdown cell @0 expected unnumbered introduction heading")
        elif not re.match(rf"#+\s+{re.escape(stage)}\b", source.strip()):
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

    for pattern in PRIVATE_WORKFLOW_LABEL_PATTERNS:
        if re.search(pattern, combined_all, flags=re.I):
            fail("private workflow label present")

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

    print(f"[INFO] valid notebook cells: {len(code_cells)} code, {len(md_cells)} markdown")
    print("[INFO] exact topology (unnumbered intro + 01-19 markdown/code pairs) enforced")
    print("[INFO] torch restricted to the GPU hardware-preflight cell")
    print(f"[INFO] {HEARTBEAT_MARKER} heartbeat helper enforced")
    print("[INFO] execution-clean state and no operational assert gates enforced")
    print("[INFO] metadata authority enforced (python3 kernel, runtime_target, production_claim)")
    print("[INFO] every code cell compiles (CPU-safe compile, no execution)")
    print(
        "[INFO] stages 01-19 present in order after the unnumbered intro; ",
        "bilingual markdown precedes every code cell",
        sep="",
    )
    print("[INFO] no forbidden labels, no CPU fallback, no model imports at cell scope")
    print("[PASS] NOTEBOOK_STRUCTURE_VALID")


if __name__ == "__main__":
    main()
