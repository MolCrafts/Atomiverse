#!/bin/bash
# Speed and memory scaling at fixed density. Engine setup is included in the
# ten-second per-engine deadline.
#
#   bash run_scaling.sh BUILD    (or RUN_SCRIPT=.../run_scaling.sh in gpu_job.sh)
#
# Required: DATA (equilibrated GAFF PEO data), FF, OPENMM_ENV.
# Optional: PRECISION [double], REPLICAS ["1 2 3 4"].
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${1:?build directory required}"
: "${OPENMM_ENV:?set OPENMM_ENV to a Python env with openmm}"
: "${DATA:?set DATA to equilibrated GAFF PEO data}"
: "${FF:?set FF to GAFF coefficients}"
for rep in ${REPLICAS:-1 2 3 4}; do
    "$OPENMM_ENV/bin/python" "$HERE/compare.py" \
        --data "$DATA" --ff "$FF" --atv "$BUILD/bin/bench_gaff" \
        --out "logs/gaff-scaling/${PRECISION:-double}-r$rep" \
        --replicate "$rep" --precision "${PRECISION:-double}" \
        --steps 30 --warmup 10 --samples 3 --budget 10 --measure-memory
done
