#!/usr/bin/env bash
#
# run.sh — drive the Atomiverse amorphous-PE LAMMPS comparison.
#
# Builds the runner if the binary is missing, then launches it under MPI.
# Pass `--short` to use 1k+2k step cuts for a sanity check; otherwise the
# full 10k+50k LAMMPS protocol runs (the latter takes a while on 8 ranks).
#
# Usage:
#   tools/debug_amorphous/run.sh                        # 8 ranks, full protocol
#   tools/debug_amorphous/run.sh --np 4 --short         # 4 ranks, 1k+2k steps
#   tools/debug_amorphous/run.sh --np 1                 # serial reference

set -euo pipefail

# ── locate paths --------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BUILD="${ATV_BUILD_DIR:-$ROOT/build}"
BIN="$BUILD/tools/debug_amorphous/run_atomiverse_amorphous"
DATA="$HERE/PE_nc10_cl1000.dat"
LOG="$HERE/log.atomiverse.txt"

# ── parse args ----------------------------------------------------------
NP=8
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)
            NP="$2"; shift 2 ;;
        --short)
            EXTRA+=("--short"); shift ;;
        --steps-stage1|--steps-stage2|--data|--log)
            EXTRA+=("$1" "$2"); shift 2 ;;
        *)
            echo "unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ── build if missing ----------------------------------------------------
if [[ ! -x "$BIN" ]]; then
    echo "[run.sh] building runner ..."
    cmake -B "$BUILD" \
        -DATV_USE_MPI=ON \
        -DATV_BUILD_TESTS=OFF \
        -DATV_BUILD_DEBUG_AMORPHOUS=ON >/dev/null
    cmake --build "$BUILD" --target run_atomiverse_amorphous \
        -j"$(getconf _NPROCESSORS_ONLN || echo 4)"
fi

[[ -f "$DATA" ]] || { echo "missing $DATA" >&2; exit 1; }

echo "[run.sh] launching $NP ranks"
exec mpiexec -n "$NP" --oversubscribe "$BIN" \
    --data "$DATA" --log "$LOG" ${EXTRA[@]+"${EXTRA[@]}"}
