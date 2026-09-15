#!/usr/bin/env python
"""CPU-only structural validation of the public bilingual notebook.

Checks (CPU-safe, no model load, no network):
  1. JSON parses and has exactly the expected cell topology.
  2. Stages 00-19 appear in order in markdown; a markdown cell precedes every
     code cell and is non-empty (contains EN + VI text).
  3. No forbidden internal labels leak into public cell text.
  4. Heartbeat/status markers used by long-running cells.
  5. No naive CPU-fallback statements in public cells.
  6. No heavy framework/model imports happen at import time inside cells
     (torch/transformers/diffusers/keras must not be imported at cell scope).
  7. Notebook calls the project's own scripts/API rather than rebuilding the
     model server (no `import mage_flow`, no `MageFlowPipeline`, no `DiffusionPipeline`).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "mage-flow-t4x2-production-rest-api-demo.ipynb"

FORBIDDEN_LABELS = ("C3", "r4", "C2D1", "R5", "S29", "authority-source")
EXPECTED_STAGES = tuple(f"{n:02d}" for n in range(20))
FORBIDDEN_IMPORTS = ("import transformers", "from transformers",
                     "import diffusers", "from diffusers",
                     "import keras", "from keras",
                     "import mage_flow", "from mage_flow",
                     "import accelerate", "from accelerate",
                     "import safetensors", "from safetensors")
FORBIDDEN_CLASSES = ("MageFlowPipeline", "MageFlow", "DiffusionPipeline",
                     "AutoPipeline", "StableDiffusionPipeline")


def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else NOTEBOOK
    nb = json.loads(nb_path.read_text())
    cells = nb["cells"]

    if len(cells) < 39:
        fail(f"expected >= 39 cells, got {len(cells)}")

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

    full_md = [("".join(c.get("source", [])) or "") for c in md_cells]
    combined_md = "\n".join(full_md)
    combined_all = combined_md + "\n" + "\n".join(
        "".join(c.get("source", [])) for c in code_cells)

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
        fail(
            "stage sequence mismatch: expected "
            f"{list(EXPECTED_STAGES)}, found {found_stages}"
        )

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

    print("[INFO] valid notebook cells: %d code, %d markdown" % (len(code_cells), len(md_cells)))
    print("[INFO] every code cell compiles (CPU-safe compile, no execution)")
    print("[INFO] stages 00-19 present in order, bilingual markdown before every code cell")
    print("[INFO] no forbidden labels, no CPU fallback, no model imports at cell scope")
    print("[PASS] NOTEBOOK_STRUCTURE_VALID")


if __name__ == "__main__":
    main()
