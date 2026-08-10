#!/bin/bash -l
#SBATCH -J atv-cuda-gpw-h2o-gate
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -o .claude/cuda-gpw-product-h2o-gate-%j.out
#SBATCH -e .claude/cuda-gpw-product-h2o-gate-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="${OPENBLASDIR:-}"
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
# aarch64 GH200: do NOT override RUSTUP_TOOLCHAIN

echo "=== node $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ==="
which nvcc || true
nvidia-smi -L || true

rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
  -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS

cmake --build build -j"$(nproc)" --target atomiverse \
  test_cuda_gpw_full_product reg_cuda-gpw-product-h2o-gate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Trace only the product-h2o reg (full suite is noisy).
unset ATV_GPW_CUDA_TRACE || true

ctest --test-dir build -R '^test_cuda_gpw_full_product$' --output-on-failure

export ATV_GPW_CUDA_TRACE=1
cmake --build build --target run_reg_cuda-gpw-product-h2o-gate

echo "=== PASS cuda-gpw-product-h2o-gate GH200 job ${SLURM_JOB_ID:-local} ==="

# disk law
rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
find . .claude -maxdepth 3 -type d \( -name build -o -name 'build-*' \) 2>/dev/null || true
