#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J atv-cuda-rhf-stream
#SBATCH -o .claude/cuda-rhf-nondefault-stream-%j.out
#SBATCH -e .claude/cuda-rhf-nondefault-stream-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

git submodule update --init --recursive molrs

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"

echo "host=$(hostname) arch=$(uname -m)"
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
echo "HEAD=$(git rev-parse HEAD)"
nvidia-smi -L || true

BUILD=build-cuda-rhf-nondefault-stream
rm -rf "${BUILD}" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_REGRESSIONS=OFF \
  -DATV_BUILD_HF=ON \
  -DATV_BUILD_CUDA=ON \
  -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_PW=OFF \
  -DATV_BUILD_SEMIEMPIRICAL=OFF \
  \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}"

cmake --build "${BUILD}" -j"$(nproc)" --target atomiverse test_cuda_scf_rhf test_cuda_scf_device_integrals

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
ctest --test-dir "${BUILD}" --output-on-failure \
  -R 'test_cuda_scf_rhf|test_cuda_scf_device_integrals' \
  || TEST_RC=$?

rm -rf "${BUILD}" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true

echo "TEST_RC=${TEST_RC}"
if [[ "${TEST_RC}" -ne 0 ]]; then
  exit 1
fi
echo "PASS: cuda-rhf-nondefault-stream GPU job"
echo "BUILDS_WIPED"
