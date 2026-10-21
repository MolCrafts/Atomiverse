#!/bin/bash
# Arrhenius allocation only. Engine setup is included in the ten-second deadline.
set -euo pipefail
BUILD="${1:?build directory required}"
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
ENV_DIR="${OPENMM_ENV:-$PWD/.openmm-env}"
: "${DATA:?set DATA to equilibrated GAFF PEO data}"
: "${FF:?set FF to GAFF coefficients}"
resource_args=()
if [[ "${MEASURE_MEMORY:-0}" == "1" ]]; then
    resource_args+=(--measure-memory)
fi
for rep in ${REPLICAS:-1 2 3 4}; do
    "$ENV_DIR/bin/python" tools/bench_gaff_openmm/compare.py \
        --data "$DATA" --ff "$FF" --atv "$BUILD/bin/bench_gaff" \
        --out "${OUT_ROOT:-logs/gaff-throughput}/${PRECISION:-double}-r$rep" \
        --replicate "$rep" --precision "${PRECISION:-double}" \
        --steps "${STEPS:-100}" --warmup 10 --samples "${SAMPLES:-3}" --budget 10 "${resource_args[@]}"
done
