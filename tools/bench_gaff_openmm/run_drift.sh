#!/bin/bash
# NVE energy drift, ATV against OpenMM at the same precision, at DT and DT/2
# for several fixed velocity seeds. Run inside the GPU job after building
# (RUN_SCRIPT=tools/bench_gaff_openmm/run_drift.sh tools/bench_gaff_gpu_job.sh).
set -euo pipefail
BUILD="${1:?build directory required}"
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
ENV_DIR="${OPENMM_ENV:-$PWD/.openmm-env}"
: "${DATA:?set DATA to GAFF LAMMPS data}"
: "${FF:?set FF to GAFF coefficients}"
for rep in ${REPLICAS:-1}; do
    "$ENV_DIR/bin/python" tools/bench_gaff_openmm/compare.py \
        --data "$DATA" --ff "$FF" --atv "$BUILD/bin/bench_gaff" \
        --out "logs/gaff-drift/${PRECISION:-double}-r$rep" --replicate "$rep" \
        --precision "${PRECISION:-double}" --drift --dt "${DT:-0.25}" \
        --drift-time "${DRIFT_TIME:-4000}" --drift-every "${DRIFT_EVERY:-10}" \
        --seeds "${SEEDS:-1,2,3,4,5,6}" --budget "${BUDGET:-600}"
done
