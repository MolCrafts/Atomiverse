#!/bin/bash
# Fragment check, optional OpenMM Reference-platform check, then the timed
# ATV-vs-OpenMM comparison per replication. gpu_job.sh runs it after the
# build; it also runs standalone on any CUDA node.
#
#   bash run.sh BUILD
#
# Required: DATA, FF (LAMMPS real data + coefficients), OPENMM_ENV.
# Optional: PRECISION [double], STEPS [100], REPLICAS ["1 3"] (n^3
#           orthorhombic), SKEW_REPLICAS [""] (same system, tilted supercell),
#           REFERENCE_CHECK_DATA (adds a nonbonded check on OpenMM Reference).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${1:?build directory required}"
: "${OPENMM_ENV:?set OPENMM_ENV to a Python env with openmm}"
: "${DATA:?set DATA to GAFF LAMMPS data}"
: "${FF:?set FF to GAFF coefficients}"
PY="$OPENMM_ENV/bin/python"
ATV="$BUILD/bin/bench_gaff"
[[ -x "$ATV" ]] || { echo "missing $ATV (build target bench_gaff)" >&2; exit 1; }
mkdir -p logs/gaff-openmm
"$PY" "$HERE/compare.py" \
    --data "$HERE/fragment.data" --ff "$HERE/fragment.ff" --atv "$ATV" \
    --out "logs/gaff-openmm/fragment-${PRECISION:-double}" \
    --precision "${PRECISION:-double}" --steps 5 --warmup 1 --budget 10
if [[ -n "${REFERENCE_CHECK_DATA:-}" ]]; then
    timeout 10s "$PY" "$HERE/compare.py" \
        --data "$REFERENCE_CHECK_DATA" --ff "$FF" --atv "$ATV" \
        --out logs/gaff-openmm/raw-reference --engine openmm \
        --platform Reference --term nonbonded --steps 1 --warmup 1
fi
cases=()
for rep in ${REPLICAS:-1 3}; do cases+=("$rep:"); done
for rep in ${SKEW_REPLICAS:-}; do cases+=("$rep:skew"); done
for case in "${cases[@]}"; do
    rep=${case%%:*}
    skew=${case#*:}
    out="logs/gaff-openmm/${PRECISION:-double}-r$rep${skew:+-skew}"
    "$PY" "$HERE/compare.py" \
        --data "$DATA" --ff "$FF" --atv "$ATV" \
        --out "$out" --replicate "$rep" ${skew:+--skew} \
        --precision "${PRECISION:-double}" \
        --steps "${STEPS:-100}" --warmup 10 --budget 10
    ATV_BENCH_TIMING=1 timeout 10s "$ATV" \
        "$out/input.txt" "$out/profile-forces.txt" 50 10 all 1 \
        > "$out/profile.log"
done
