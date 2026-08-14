#!/usr/bin/env bash
# Public CLI contract — staged cmake config/compile/install.
set -euo pipefail

PYTHON="${1:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SRC="$REPO/python/src"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

echo "== install_cli regression (staged) =="
echo "  python: $PYTHON"
echo "  repo:   $REPO"

OUT="$("$PYTHON" -m atomiverse config --help 2>&1)" || {
    echo "FAIL: config --help"; echo "$OUT"; exit 1
}
echo "$OUT" | grep -q -- '--config' || { echo "FAIL: missing --config"; exit 1; }
echo "$OUT" | grep -q -- '--mpi' || { echo "FAIL: missing --mpi"; exit 1; }
echo "$OUT" | grep -q -- '--src' || { echo "FAIL: missing --src"; exit 1; }
echo "ok: config --help"

OUT="$("$PYTHON" -m atomiverse compile --help 2>&1)" || true
echo "$OUT" | grep -qi compile || { echo "FAIL: compile help"; exit 1; }
echo "ok: compile --help"

OUT="$("$PYTHON" -m atomiverse install --help 2>&1)" || true
echo "$OUT" | grep -qi install || { echo "FAIL: install help"; exit 1; }
echo "ok: install --help"

# mutual exclusion
if "$PYTHON" -m atomiverse config --config "$HERE/fixtures/minimal_ok.toml" md --src "$REPO" >/dev/null 2>&1; then
    echo "FAIL: expected mutual exclusion"
    exit 1
fi
echo "ok: --config + md rejected"

# --hf is a normal module (spec py-scf-single-point): it resolves
# ATV_BUILD_HF=ON like every other module instead of being rejected.
if [[ -f "$REPO/CMakeLists.txt" ]]; then
    OUT="$("$PYTHON" -m atomiverse config md --hf --src "$REPO" --dry-run 2>&1)" || {
        echo "FAIL: config md --hf"; echo "$OUT"; exit 1
    }
    echo "$OUT" | grep -q 'ATV_BUILD_HF=ON' || { echo "FAIL: HF define"; echo "$OUT"; exit 1; }
    echo "ok: --hf configures ATV_BUILD_HF=ON"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/python"
echo 'project(x)' >"$TMP/CMakeLists.txt"
cat > "$TMP/python/pyproject.toml" <<'EOF'
[project]
name = "atomiverse"
version = "0.0.0-mismatch"
EOF
if "$PYTHON" -m atomiverse config md --src "$TMP" --dry-run >/dev/null 2>&1; then
    echo "FAIL: expected version mismatch"
    exit 1
fi
echo "ok: version mismatch rejected"

# dry-run config on real tree
if [[ -f "$REPO/CMakeLists.txt" ]]; then
    OUT="$("$PYTHON" -m atomiverse config md --mpi --src "$REPO" --dry-run 2>&1)" || {
        echo "FAIL: config dry-run"; echo "$OUT"; exit 1
    }
    echo "$OUT" | grep -q 'ATV_BUILD_MD=ON' || { echo "FAIL: MD define"; echo "$OUT"; exit 1; }
    echo "$OUT" | grep -q 'ATV_USE_MPI=ON' || { echo "FAIL: MPI global"; echo "$OUT"; exit 1; }
    echo "$OUT" | grep -q 'cmake' || { echo "FAIL: cmake command"; echo "$OUT"; exit 1; }
    echo "ok: config dry-run md --mpi"
fi

echo "REGRESSION PASSED: python/install_cli"
