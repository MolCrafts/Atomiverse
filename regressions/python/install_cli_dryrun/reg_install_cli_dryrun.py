#!/usr/bin/env python3
"""Public CLI dry-run regression — staged cmake config/compile/install."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "python" / "src"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    import os

    env = os.environ.copy()
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + prev if prev else "")
    return subprocess.run(
        [sys.executable, "-m", "atomiverse", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
        env=env,
    )


def main() -> int:
    if not (SRC / "atomiverse").is_dir():
        print(f"FAIL: pure sources not found at {SRC / 'atomiverse'}")
        return 1

    print("== install_cli_dryrun (staged cmake) ==")
    print(f"  python: {sys.executable}")
    print(f"  repo:   {REPO}")

    proc = _run([])
    if proc.returncode != 0:
        print("FAIL: bare invocation")
        print(proc.stdout, proc.stderr)
        return 1
    blob = (proc.stdout + proc.stderr).lower()
    if "config" not in blob or "compile" not in blob:
        print("FAIL: bare missing config/compile guidance")
        print(proc.stdout)
        return 1
    print("ok: bare banner")

    proc = _run(["--help"])
    if proc.returncode != 0:
        print("FAIL: --help")
        return 1
    print("ok: --help")

    # mutual exclusion
    proc = _run(["config", "--config", "x.toml", "md", "--src", str(REPO)])
    if proc.returncode == 0:
        print("FAIL: expected mutual exclusion")
        return 1
    if "mutual" not in (proc.stdout + proc.stderr).lower():
        print("FAIL: mutual exclusion message")
        print(proc.stderr)
        return 1
    print("ok: mutual exclusion")

    # empty config
    proc = _run(["config", "--src", str(REPO)])
    if proc.returncode == 0:
        print("FAIL: empty config should fail")
        return 1
    print("ok: empty config rejected")

    # --hf is a normal module (spec py-scf-single-point): Gaussian SCF
    # configures like any other product module, no rejection.
    proc = _run(["config", "md", "--hf", "--src", str(REPO), "--dry-run"])
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print("FAIL: config md --hf")
        print(out)
        return 1
    if "ATV_BUILD_HF=ON" not in out:
        print("FAIL: missing ATV_BUILD_HF=ON")
        print(out)
        return 1
    print("ok: --hf configures ATV_BUILD_HF=ON")

    # dry-run config md --mpi
    proc = _run(
        ["config", "md", "--mpi", "--src", str(REPO), "--dry-run"]
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print("FAIL: config dry-run")
        print(out)
        return 1
    for needle in (
        "cmake",
        "ATV_BUILD_MD=ON",
        "ATV_BUILD_POTENTIALS=ON",
        "ATV_BUILD_NEIGHBOR=ON",
        "ATV_BUILD_PYTHON=ON",
        "ATV_USE_MPI=ON",
    ):
        if needle not in out:
            print(f"FAIL: missing {needle}")
            print(out)
            return 1
    print("ok: config dry-run (cmake + defines + global mpi)")

    # compile dry-run without state may fail — acceptable; just check subcommand exists
    proc = _run(["compile", "--help"])
    if proc.returncode != 0:
        print("FAIL: compile --help")
        return 1
    print("ok: compile --help")

    proc = _run(["install", "--help"])
    if proc.returncode != 0:
        print("FAIL: install --help")
        return 1
    print("ok: install --help")

    proc = _run(["status"])
    if proc.returncode != 0:
        print("FAIL: status")
        return 1
    print("ok: status")

    print("REGRESSION PASSED: python/install_cli_dryrun")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
