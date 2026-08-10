#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J cuda-gpw-uks-gamma
#SBATCH -o job_cuda-gpw-uks-gamma.%j.out
#SBATCH -e job_cuda-gpw-uks-gamma.%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
# GH200 aarch64: do NOT override RUSTUP_TOOLCHAIN
export OPENBLAS_ROOT="${OPENBLASDIR:-}"
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

git submodule update --init --recursive molrs || true

rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
  -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS

cmake --build build -j"$(nproc)" --target atomiverse \
  test_cuda_gpw_full_product reg_cuda-gpw-uks-gamma

echo "=== ctest unit ==="
ctest --test-dir build -R '^test_cuda_gpw_full_product$' --output-on-failure

echo "=== regression dual-gate ==="
cmake --build build --target run_reg_cuda-gpw-uks-gamma

echo "=== DONE green ==="
# Disk law: wipe build after green evidence is printed
rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
find . -maxdepth 3 -type d \( -name build -o -name 'build-*' \) 2>/dev/null || true
echo "builds wiped"
