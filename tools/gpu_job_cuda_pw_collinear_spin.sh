#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J atv-cuda-pw-spin
#SBATCH -o .claude/cuda-pw-collinear-spin-%j.out
#SBATCH -e .claude/cuda-pw-collinear-spin-%j.err

# GH200 gate for cuda-pw-collinear-spin: device nspin=2 LSDA + dual-gate O2.

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

BUILD=build-cuda-pw-collinear-spin
rm -rf "${BUILD}" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON \
  -DATV_BUILD_CUDA=ON \
  -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_PW=ON \
  -DATV_BUILD_SEMIEMPIRICAL=OFF \
  -DATV_BUILD_TORCH=OFF \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}"

echo "=== STAGE 1: library ==="
cmake --build "${BUILD}" -j"$(nproc)" --target atomiverse

TARGETS="${ATV_PW_TARGETS:-test_cuda_pw_fft test_cuda_pw_poisson test_cuda_pw_grid_map test_cuda_pw_xc_lda test_cuda_pw_xc_pbe test_cuda_pw_veff test_cuda_pw_guards test_cuda_pw_density test_cuda_scf_rhf reg_cuda-pw-collinear-spin reg_pw-cuda-lda-si}"
echo "=== STAGE 2: tests: ${TARGETS} ==="
# shellcheck disable=SC2086
cmake --build "${BUILD}" -j"$(nproc)" --target ${TARGETS}

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
echo "=== STAGE 3: ctest (only built PW slice + RHF) ==="
# Explicit names: a broad test_cuda_pw_ regex also matches force_local /
# cell_stress which are not in TARGETS above and would report Not Run.
ctest --test-dir "${BUILD}" --output-on-failure \
  -R 'test_cuda_pw_(fft|poisson|grid_map|xc_lda|xc_pbe|veff|guards|density)$|test_cuda_scf_rhf' \
  || TEST_RC=$?

echo "=== STAGE 4: collinear spin dual-gate ==="
REG_RC=0
"${BUILD}/regressions/reg_cuda-pw-collinear-spin" --device=both || REG_RC=$?

echo "=== STAGE 5: wipe build ==="
rm -rf "${BUILD}" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

echo "TEST_RC=${TEST_RC} REG_RC=${REG_RC}"
if [[ "${TEST_RC}" -ne 0 || "${REG_RC}" -ne 0 ]]; then
  exit 1
fi
echo "PASS: cuda-pw-collinear-spin GPU job"
