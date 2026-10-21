#!/bin/bash
# Force-term comparisons within a GPU batch allocation. Any failed slice is
# retained in its summary; the aggregate exits nonzero after all slices.
set -uo pipefail
BUILD="${1:?build directory required}"
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
ENV_DIR="${OPENMM_ENV:-$PWD/.openmm-env}"
RC=0
for term in nonbonded bond angle dihedral; do
    "$ENV_DIR/bin/python" tools/bench_gaff_openmm/compare.py \
        --data "$DATA" --ff "$FF" --atv "$BUILD/bin/bench_gaff" \
        --out "logs/gaff-openmm/diagnose-$term" --term "$term" \
        --steps 5 --warmup 1 --budget 10 || RC=1
done
exit "$RC"
