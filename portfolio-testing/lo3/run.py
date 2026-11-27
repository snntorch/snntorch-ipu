"""Run the repository suite, portfolio suite and smoke checks from the repository root.

Use baseline or final as the stage argument. The script saves each command, output,
exit code and duration. The tests use controlled network fixtures.
"""

import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCOPE = json.loads((HERE / "design.json").read_text())["scope"]


def execute(args, path, env=None):
    start = time.monotonic()
    result = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    path.write_text(result.stdout + result.stderr, encoding="utf-8")
    return dict(
        command=args,
        returncode=result.returncode,
        seconds=time.monotonic() - start,
        log=str(path.relative_to(ROOT)),
    )


def main():
    stage = sys.argv[1]
    out = HERE / "evidence" / stage
    out.mkdir(parents=True, exist_ok=False)
    import torch

    metadata = dict(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        client_date="2026-11-02",
        python=sys.version,
        executable=sys.executable,
        os=platform.platform(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        devices=dict(
            cpu=True,
            cuda=torch.cuda.is_available(),
            cuda_count=torch.cuda.device_count(),
            mps=torch.backends.mps.is_available(),
        ),
        packages={
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        },
        hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *(ROOT / s for s in SCOPE),
                *HERE.glob("*.json"),
                *HERE.glob("*.py"),
                *HERE.glob("tests/*.py"),
            ]
        },
    )
    (out / "environment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    commands = []
    # Keep the existing suite separate from portfolio counts and coverage.
    commands.append(
        execute(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
                "--tb=short",
                f"--junitxml={out / 'upstream.xml'}",
            ],
            out / "upstream.txt",
        )
    )
    env = os.environ.copy()
    env["LO3_RESULTS"] = str(out / "results.json")
    env["COVERAGE_FILE"] = str(out / "coverage-data")
    args = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--branch",
        "--include=" + ",".join(SCOPE),
        "-m",
        "pytest",
        "portfolio-testing/lo3/tests",
        "-q",
        "--tb=short",
        f"--junitxml={out / 'portfolio.xml'}",
    ]
    commands.append(execute(args, out / "portfolio.txt", env))
    commands.append(
        execute(
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                "-o",
                str(out / "coverage.json"),
            ],
            out / "coverage-command.txt",
            env,
        )
    )
    commands.append(
        execute(
            [sys.executable, "-m", "coverage", "report", "-m"],
            out / "coverage-summary.txt",
            env,
        )
    )
    commands.append(
        execute([sys.executable, str(HERE / "smoke.py")], out / "smoke.txt")
    )
    (out / "commands.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")
    print(json.dumps(commands, indent=2))


if __name__ == "__main__":
    main()
