#!/usr/bin/env bash
# Shared prerequisite of the python/ regressions: the interpreter CMake found
# must carry atomiverse's runtime dependencies (python/pyproject.toml
# [project.dependencies]); the extension itself is deliberately absent.
#
# Usage: require_runtime_deps.sh <python>
set -euo pipefail
"$1" - <<'PY'
import importlib.util
import sys

missing = [m for m in ("numpy", "typer", "molpy")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"FAIL: {sys.executable} lacks atomiverse runtime dependencies: "
          f"{', '.join(missing)}. Reconfigure with "
          "-DPython3_EXECUTABLE=<python with numpy, typer, molcrafts-molpy> "
          "(python/pyproject.toml [project.dependencies]).")
    sys.exit(1)
PY
