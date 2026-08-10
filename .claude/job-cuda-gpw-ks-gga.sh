#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J cuda-gpw-ks-gga
#SBATCH -o .claude/cuda-gpw-ks-gga-%j.out
#SBATCH -e .claude/cuda-gpw-ks-gga-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
# aarch64 GH200: do NOT override RUSTUP_TOOLCHAIN
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1

# Isolate build name so clearup wipe-shield does not race siblings.
BUILD=build_cuda_gpw_ks_gga
rm -rf "${BUILD}" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

echo "=== configure CUDA GPW KS-GGA @ $(hostname) $(date -Is) ==="
cmake -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
  -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS

echo "=== scoped build ==="
cmake --build "${BUILD}" -j"$(nproc)" --target atomiverse \
  test_cuda_gpw_ks_gga reg_cuda-gpw-ks-gga

echo "=== unit test ==="
ctest --test-dir "${BUILD}" -R '^test_cuda_gpw_ks_gga$' --output-on-failure

echo "=== dual-gate regression ==="
cmake --build "${BUILD}" --target run_reg_cuda-gpw-ks-gga

echo "=== wipe build ==="
rm -rf "${BUILD}" CMakeCache.txt CMakeFiles Testing .cache
find . -maxdepth 3 -type d \( -name build -o -name 'build-*' -o -name 'build_*' \) 2>/dev/null || true
echo "DONE cuda-gpw-ks-gga $(date -Is)"
