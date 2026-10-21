#!/bin/bash
# Profile previously prepared cases; use uninstrumented compare.py for throughput.
set -euo pipefail
BUILD="${1:?build directory}"
for rep in ${PROFILE_REPLICAS:-1 3}; do
    PROFILE_INPUT="${PROFILE_INPUT_ROOT:?prepared case root}/${PRECISION:-mixed}-r$rep/input.txt" \
    PROFILE_OUT="logs/gaff-profile/${SLURM_JOB_ID:?run under Slurm}/r$rep" \
        bash tools/bench_gaff_openmm/profile.sh "$BUILD"
done
