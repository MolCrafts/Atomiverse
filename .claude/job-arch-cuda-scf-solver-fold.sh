#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J atv-arch-cuda-scf-fold
#SBATCH -o .claude/arch-cuda-scf-solver-fold-%j.out
#SBATCH -e .claude/arch-cuda-scf-solver-fold-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu

echo "host=$(hostname) arch=$(uname -m)"
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
echo "HEAD=$(git rev-parse HEAD)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
nvidia-smi -L || true

rm -rf build-cuda-scf-fold CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B build-cuda-scf-fold \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON \
  -DATV_BUILD_CUDA=ON \
  -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_PW=OFF \
  -DATV_BUILD_SEMIEMPIRICAL=OFF \
  -DATV_BUILD_TORCH=OFF \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}"

cmake --build build-cuda-scf-fold -j"$(nproc)" --target \
  atomiverse \
  test_cuda_scf \
  test_cuda_scf_rhf \
  reg_arch-cuda-scf-solver-fold

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
ctest --test-dir build-cuda-scf-fold --output-on-failure \
  -R '^(test_cuda_scf|test_cuda_scf_rhf)$' \
  || TEST_RC=$?

REG_RC=0
cmake --build build-cuda-scf-fold --target run_reg_arch-cuda-scf-solver-fold \
  || REG_RC=$?

echo "CTEST_RC=${TEST_RC} REG_RC=${REG_RC}"
if [ "${TEST_RC}" -ne 0 ] || [ "${REG_RC}" -ne 0 ]; then
  echo "GPU_JOB_TESTS_FAILED ctest=${TEST_RC} reg=${REG_RC}"
  exit 1
fi
echo "GPU_JOB_OK"
rm -rf build-cuda-scf-fold build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
echo "BUILDS_WIPED"
