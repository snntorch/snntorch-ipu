"""Check whether the tests catch three changes made deliberately.

Each change runs in memory in a fresh process. The defect count excludes these changes.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def child(fault):
    import torch
    import snntorch as snn
    from snntorch import surrogate, utils
    import pytest

    if fault == "threshold_inclusive":
        snn.SpikingNeuron.fire = lambda self, mem: (mem >= self.threshold).float()
        node = "test_functional.py::test_R1_threshold_neighbours"
    elif fault == "gradient_sign":
        original = surrogate.FastSigmoid.backward

        def backward(ctx, grad):
            result = original(ctx, grad)
            return -result[0], None

        surrogate.FastSigmoid.backward = staticmethod(backward)
        node = "test_functional.py::test_R2_surrogate"
    else:
        utils.reset = lambda net: None
        node = "test_state.py::test_R4_reset_zero_one_many"
    args = [
        str(HERE / "tests" / node),
        "-q",
        "--tb=short",
        "--junitxml=" + str(HERE / "evidence" / "mutations" / (fault + ".xml")),
    ]
    if fault == "gradient_sign":
        args.extend(["-k", "fast"])
    return pytest.main(args)


def main():
    if len(sys.argv) > 1:
        return child(sys.argv[1])
    out = HERE / "evidence" / "mutations"
    out.mkdir(parents=True, exist_ok=False)
    files = [ROOT / p for p in json.loads((HERE / "design.json").read_text())["scope"]]
    before = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    results = []
    for fault in ["threshold_inclusive", "gradient_sign", "omitted_reset"]:
        command = [sys.executable, str(Path(__file__).resolve()), fault]
        p = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        (out / (fault + ".txt")).write_text(p.stdout + p.stderr, encoding="utf-8")
        results.append(
            dict(
                fault=fault,
                command=command,
                exitcode=p.returncode,
                detected=p.returncode == 1,
            )
        )
    after = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    assert before == after, "Deliberate faults must never remain on disk"
    (out / "results.json").write_text(
        json.dumps(
            dict(
                faults=results,
                production_unchanged=True,
                excluded=0,
                equivalent=0,
                source_hashes=after,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    return 0 if all(r["detected"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
