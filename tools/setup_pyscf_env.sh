#!/usr/bin/env bash
# tools/setup_pyscf_env.sh — bootstrap the PySCF comparison (oracle) venv.
#
# Usage:   bash tools/setup_pyscf_env.sh
# Target:  ~/work/.x86_64 (the x86_64 login-node venv; ships without pip)
#
# Installs, idempotently (each step probes before acting):
#   1. pip           via `python -m ensurepip --upgrade`
#   2. pyscf         the external oracle engine (NEVER a build/test dependency
#                    of Atomiverse — numbers are run externally and pinned;
#                    see docs/tutorial/appendix-pyscf-oracle.md)
#   3. molvis        editable install of the local sibling repo's Python
#                    package (~/work/molcrafts/molvis/python)
#
# Note on molplot: as of 2026-07-20 molplot is a TypeScript sub-package of the
# molvis repo (~/work/molcrafts/molvis/molplot) with no Python distribution.
# The final self-check therefore requires `import pyscf, molvis` and only
# reports molplot availability informationally.
#
# Safe under both `bash -lc` and plain bash; no Lmod modules required.
set -euo pipefail

PY="$HOME/work/.x86_64/bin/python"
MOLVIS_PY="$HOME/work/molcrafts/molvis/python"

[ -x "$PY" ] || { echo "error: venv python not found at $PY" >&2; exit 1; }

echo "== venv python: $("$PY" --version)"

# 1. pip (ensurepip is idempotent by design, but probe first to stay quiet)
if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "== bootstrapping pip via ensurepip"
    "$PY" -m ensurepip --upgrade
else
    echo "== pip present: $("$PY" -m pip --version)"
fi

# 2. pyscf
if ! "$PY" -c 'import pyscf' >/dev/null 2>&1; then
    echo "== installing pyscf"
    "$PY" -m pip install pyscf
else
    echo "== pyscf present: $("$PY" -c 'import pyscf; print(pyscf.__version__)')"
fi

# 3. molvis (editable, local sibling repo)
if ! "$PY" -c 'import molvis' >/dev/null 2>&1; then
    echo "== installing molvis (editable) from $MOLVIS_PY"
    [ -f "$MOLVIS_PY/pyproject.toml" ] || { echo "error: $MOLVIS_PY/pyproject.toml missing" >&2; exit 1; }
    "$PY" -m pip install -e "$MOLVIS_PY"
else
    echo "== molvis present"
fi

# Self-check (hard requirement: pyscf + molvis; molplot informational)
echo "== self-check"
"$PY" - <<'EOF'
import pyscf, molvis
print("pyscf ", pyscf.__version__)
print("molvis", getattr(molvis, "__version__", "(no __version__)"))
try:
    import molplot  # noqa: F401
    print("molplot", getattr(molplot, "__version__", "(no __version__)"))
except ImportError:
    print("molplot: not installed (TypeScript-only as of 2026-07;"
          " see tools/setup_pyscf_env.sh header)")
EOF
echo "== setup_pyscf_env.sh OK"
