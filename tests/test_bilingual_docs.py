import subprocess
import sys


def test_bilingual_documentation_pairing_and_history():
    result = subprocess.run(
        [sys.executable, "scripts/validate_bilingual_docs.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS] BILINGUAL_DOCUMENTATION_PAIRING_VALID" in result.stdout
