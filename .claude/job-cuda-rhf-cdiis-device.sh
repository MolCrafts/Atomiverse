#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J atv-cdiis
#SBATCH -o .claude/job-cuda-rhf-cdiis-device-%j.out
#SBATCH -e .claude/job-cuda-rhf-cdiis-device-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
# Drop x86_64 shadow path that breaks aarch64 cmake/python wrappers.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
# Do NOT override RUSTUP on aarch64 GH200 — default toolchain is correct.
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"

echo "host=$(hostname) arch=$(uname -m)"
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
nvidia-smi -L || true

# Clean any leftover partial build
rm -rf build build-cuda CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B build \
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

# Build only the CUDA SCF twin targets (full `tests/` with SE=OFF trips
# pre-existing test_cpu_qm_engine link failures that are out of this slice).
cmake --build build -j"$(nproc)" --target \
  atomiverse \
  test_cuda_scf_rhf \
  test_cuda_scf \
  test_cuda_scf_device_integrals

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
ctest --test-dir build --output-on-failure \
  -R 'test_cuda_scf_rhf|test_cuda_scf$|test_cuda_scf_device_integrals' \
  || TEST_RC=$?

echo "CTEST_RC=${TEST_RC}"
if [ "${TEST_RC}" -ne 0 ]; then
  echo "GPU_JOB_TESTS_FAILED rc=${TEST_RC}"
  # Keep build for log diagnosis on failure; worker will wipe.
  exit "${TEST_RC}"
fi
echo "GPU_JOB_OK"
# Wipe multi-GB build on the node after green
rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true
echo "BUILDS_WIPED"
