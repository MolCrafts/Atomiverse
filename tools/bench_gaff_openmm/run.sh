#!/bin/bash
# Run inside the Arrhenius GPU job allocation after building.
set -euo pipefail
BUILD="${1:?build directory required}"
if [[ "${SKIP_CHECKS:-0}" != "1" ]]; then
    timeout 10s "$BUILD/tests/test_cuda_cluster_mask"
    timeout 10s "$BUILD/tests/test_cuda_multi_gpu" --gtest_filter=MultiGpu.SingleGpuClusterMatchesBruteForce
    timeout 10s "$BUILD/tests/test_cuda_harmonic_coefficients"
    timeout 10s "$BUILD/tests/test_cuda_packed_tiles"
    timeout 10s "$BUILD/tests/test_cuda_hilbert"
    timeout 10s "$BUILD/tests/test_cuda_timer"
    timeout 10s "$BUILD/regressions/reg_cuda-velocity-verlet-initial-force"
    timeout 10s "$BUILD/regressions/reg_cuda-gaff-fragment"
    timeout 10s "$BUILD/regressions/reg_cuda-pme-nacl"
    timeout 10s "$BUILD/regressions/reg_cuda-pme-half-spectrum"
    timeout 10s "$BUILD/regressions/reg_cuda-ewald-nacl"
    timeout 10s "$BUILD/bin/bench_nonbonded_style" 10000 100 10
fi
module load GPU/Python/3.13.5-bare-gcc-2025b-eb
ENV_DIR="${OPENMM_ENV:-$PWD/.openmm-env}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    python3 -m venv "$ENV_DIR"
    "$ENV_DIR/bin/python" -m pip install 'openmm[cuda13]' numpy
fi
"$ENV_DIR/bin/python" -m pip freeze
: "${DATA:?set DATA to GAFF LAMMPS data}"
: "${FF:?set FF to GAFF coefficients}"
mkdir -p logs/gaff-openmm
"$ENV_DIR/bin/python" tools/bench_gaff_openmm/compare.py \
    --data tools/bench_gaff_openmm/fragment.data \
    --ff tools/bench_gaff_openmm/fragment.ff --atv "$BUILD/bin/bench_gaff" \
    --out "logs/gaff-openmm/fragment-${PRECISION:-double}" \
    --precision "${PRECISION:-double}" --steps 5 --warmup 1 --budget 10
if [[ -n "${REFERENCE_CHECK_DATA:-}" ]]; then
    timeout 10s "$ENV_DIR/bin/python" tools/bench_gaff_openmm/compare.py \
        --data "$REFERENCE_CHECK_DATA" --ff "$FF" --atv "$BUILD/bin/bench_gaff" \
        --out logs/gaff-openmm/raw-reference --engine openmm \
        --platform Reference --term nonbonded --steps 1 --warmup 1
fi
for rep in ${REPLICAS:-1 3}; do
    "$ENV_DIR/bin/python" tools/bench_gaff_openmm/compare.py \
        --data "$DATA" --ff "$FF" --atv "$BUILD/bin/bench_gaff" \
        --out "logs/gaff-openmm/${PRECISION:-double}-r$rep" \
        --replicate "$rep" --precision "${PRECISION:-double}" \
        --steps "${STEPS:-100}" --warmup 10 --budget 10
    ATV_BENCH_TIMING=1 timeout 10s "$BUILD/bin/bench_gaff" \
        "logs/gaff-openmm/${PRECISION:-double}-r$rep/input.txt" \
        "logs/gaff-openmm/${PRECISION:-double}-r$rep/profile-forces.txt" 50 10 all 1 \
        > "logs/gaff-openmm/${PRECISION:-double}-r$rep/profile.log"
done
