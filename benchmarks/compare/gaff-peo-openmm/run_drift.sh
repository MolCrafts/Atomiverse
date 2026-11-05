#!/bin/bash
# NVE energy drift, ATV against OpenMM at the same precision, at DT and DT/2
# for several fixed velocity seeds.
#
#   bash run_drift.sh BUILD      (or RUN_SCRIPT=.../run_drift.sh in gpu_job.sh)
#
# Required: DATA, FF, OPENMM_ENV.
# Optional: PRECISION [double], REPLICAS [1], DT [0.25], DRIFT_TIME [4000],
#           DRIFT_EVERY [10], SEEDS [1,2,3,4,5,6], BUDGET [600].
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${1:?build directory required}"
: "${OPENMM_ENV:?set OPENMM_ENV to a Python env with openmm}"
: "${DATA:?set DATA to GAFF LAMMPS data}"
: "${FF:?set FF to GAFF coefficients}"
for rep in ${REPLICAS:-1}; do
    "$OPENMM_ENV/bin/python" "$HERE/compare.py" \
        --data "$DATA" --ff "$FF" --atv "$BUILD/bin/bench_gaff" \
        --out "logs/gaff-drift/${PRECISION:-double}-r$rep" --replicate "$rep" \
        --precision "${PRECISION:-double}" --drift --dt "${DT:-0.25}" \
        --drift-time "${DRIFT_TIME:-4000}" --drift-every "${DRIFT_EVERY:-10}" \
        --seeds "${SEEDS:-1,2,3,4,5,6}" --budget "${BUDGET:-600}"
done
