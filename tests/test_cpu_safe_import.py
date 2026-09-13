"""CPU-safe import regression: ``server.safety`` and the runtime workers must
import without ``mage_flow`` (a GPU-only runtime dependency).

The coordinator, tests, and every CPU-only static gate import these modules;
a top-level ``mage_flow`` import would break collection. This test blocks
``mage_flow`` in a fresh subprocess and proves the whole safety+runtime chain
still imports. An actual safety screen call with Mage absent may fail at
runtime -- import-time correctness is what this guards.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_BLOCK_MAGE_FLOW_AND_IMPORT = r"""
import importlib.abc
import sys


class BlockMageFlow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mage_flow" or fullname.startswith("mage_flow."):
            raise ModuleNotFoundError(f"blocked mage_flow import: {fullname}")
        return None


sys.meta_path.insert(0, BlockMageFlow())

import server.safety.gguf_safety
import server.safety.wiring
import server.workers.t2i_runtime_server
import server.workers.edit_runtime_server
import server.workers.conditioning_offload
print("CPU_SAFE_IMPORT_WITHOUT_MAGE_FLOW=PASS")
"""


def _run_child(*, use_blocker: bool) -> tuple[int, str]:
    env = {
        "PYTHONPATH": str(_REPO_ROOT),
        "MAGE_SAFETY_BACKEND": "gguf",
        "MAGE_GGUF_SAFETY_URL": "http://127.0.0.1:9",
    }
    script = _BLOCK_MAGE_FLOW_AND_IMPORT if use_blocker else "import server.safety.wiring"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def test_safety_chain_imports_without_mage_flow():
    rc, output = _run_child(use_blocker=True)
    assert rc == 0, output
    assert "CPU_SAFE_IMPORT_WITHOUT_MAGE_FLOW=PASS" in output, output


def test_mage_flow_really_is_absorbed_by_the_blocker():
    rc, output = _run_child(use_blocker=False)
    assert rc == 0, output
