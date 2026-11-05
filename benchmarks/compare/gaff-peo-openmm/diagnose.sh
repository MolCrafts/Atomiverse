#!/bin/bash
# Per-term force comparisons (nonbonded, bond, angle, dihedral). Any failed
# slice is retained in its summary; the aggregate exits nonzero after all.
#
#   bash diagnose.sh BUILD       (or RUN_SCRIPT=.../diagnose.sh in gpu_job.sh)
#
# Required: DATA, FF, OPENMM_ENV.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${1:?build directory required}"
: "${OPENMM_ENV:?set OPENMM_ENV to a Python env with openmm}"
: "${DATA:?set DATA to GAFF LAMMPS data}"
: "${FF:?set FF to GAFF coefficients}"
RC=0
for term in nonbonded bond angle dihedral; do
    "$OPENMM_ENV/bin/python" "$HERE/compare.py" \
        --data "$DATA" --ff "$FF" --atv "$BUILD/bin/bench_gaff" \
        --out "logs/gaff-openmm/diagnose-$term" --term "$term" \
        --steps 5 --warmup 1 --budget 10 || RC=1
done
exit "$RC"
