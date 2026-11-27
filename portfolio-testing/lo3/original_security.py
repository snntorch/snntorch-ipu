"""Run the corrected dataset checks against the downloader from the starting commit.

Import that version from a temporary directory so the checkout stays intact.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

HERE = Path(__file__).resolve().parent
OUT = HERE / "evidence" / "original-security"


def main():
    baseline = "eebfaa883f44c735b3e42cd881973ba42adf8fac"
    original = subprocess.check_output(
        ["git", "show", f"{baseline}:snntorch/spikevision/neuromorphic_dataset.py"],
        cwd=HERE.parents[1],
    )
    OUT.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="lo3-original-") as td:
        path = Path(td) / "original.py"
        path.write_bytes(original)
        spec = importlib.util.spec_from_file_location(
            "snntorch.spikevision.lo3_original", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class OriginalPlugin:
            def pytest_collection_modifyitems(self, items):
                for item in items:
                    if item.module.__name__ == "test_security":
                        item.module.ds = module

        os.environ["LO3_RESULTS"] = str(OUT / "results.json")
        return pytest.main(
            [
                str(HERE / "tests" / "test_security.py"),
                "-q",
                "--tb=short",
                "--junitxml=" + str(OUT / "results.xml"),
            ],
            plugins=[OriginalPlugin()],
        )


if __name__ == "__main__":
    sys.exit(main())
