#!/usr/bin/env bash
# Launcher parity regression (mpi-launcher-embedded-python-04): the SAME
# scripts serial-python-hosted vs 2-rank launcher-hosted; four legs + compare.
#
# Environment (spec Domain basis): the ~/work/.x86_64 venv (atomiverse +
# molrs importable), buildenv-gcc/2026.03-mpich, LD_LIBRARY_PATH mirrored.
# Arguments: $1 = python executable, $2 = atomiverse launcher binary.
set -euo pipefail

PYTHON="${1:?usage: reg_lj_parity.sh <python> <atomiverse-launcher>}"
LAUNCHER="${2:?usage: reg_lj_parity.sh <python> <atomiverse-launcher>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# Runtime interpreter discovery (05): pin the launcher to the SAME venv
# interpreter the serial legs use.
export ATV_PYTHON="$PYTHON"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== LJ serial leg =="
"$PYTHON" "$HERE/atv_lj.py" "$WORK/lj_serial.npz"
echo "== LJ 2-rank launcher leg =="
mpirun -np 2 "$LAUNCHER" "$HERE/atv_lj.py" "$WORK/lj_mpi.npz"
echo "== AM1 serial leg =="
"$PYTHON" "$HERE/atv_am1.py" "$WORK/am1_serial.npz"
echo "== AM1 2-rank launcher leg =="
mpirun -np 2 "$LAUNCHER" "$HERE/atv_am1.py" "$WORK/am1_mpi.npz"

echo "== compare =="
"$PYTHON" "$HERE/compare.py" \
    "$WORK/lj_serial.npz" "$WORK/lj_mpi.npz" \
    "$WORK/am1_serial.npz" "$WORK/am1_mpi.npz"

echo "== failure-path assertions (2-rank) =="
# Script exception: traceback printed, nonzero exit, no hang.
echo "raise RuntimeError('boom')" > "$WORK/boom.py"
if mpirun -np 2 "$LAUNCHER" "$WORK/boom.py" 2>/dev/null; then
    echo "FAIL: script exception did not propagate nonzero"; exit 1
fi
# Usage errors: all ranks exit nonzero cleanly.
if mpirun -np 2 "$LAUNCHER" 2>/dev/null; then
    echo "FAIL: missing-script usage did not exit nonzero"; exit 1
fi
if mpirun -np 2 "$LAUNCHER" /no/such/file.py 2>/dev/null; then
    echo "FAIL: unreadable script did not exit nonzero"; exit 1
fi

echo "REGRESSION PASSED: launcher lj_parity"
