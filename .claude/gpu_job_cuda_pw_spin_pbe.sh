#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J cuda-pw-spin-pbe
#SBATCH -o .claude/cuda-pw-spin-pbe-%j.out
#SBATCH -e .claude/cuda-pw-spin-pbe-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="${OPENBLASDIR:-}"
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
# aarch64 GH200: do NOT override RUSTUP_TOOLCHAIN

BUILD=build-gpu-spin-pbe
rm -rf "${BUILD}" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=OFF \

  -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS

cmake --build "${BUILD}" -j"$(nproc)" --target \
  atomiverse \
  test_cuda_pw_xc_pbe_spin \
  test_cuda_pw_guards \
  reg_cuda-pw-spin-pbe
BUILD_RC=$?

UNIT_SPIN_RC=1
UNIT_GUARD_RC=1
REG_RC=1
if [ "$BUILD_RC" -eq 0 ]; then
  echo "=== unit: test_cuda_pw_xc_pbe_spin ==="
  ctest --test-dir "${BUILD}" -R '^test_cuda_pw_xc_pbe_spin$' --output-on-failure
  UNIT_SPIN_RC=$?

  echo "=== unit: test_cuda_pw_guards ==="
  ctest --test-dir "${BUILD}" -R '^test_cuda_pw_guards$' --output-on-failure
  UNIT_GUARD_RC=$?

  echo "=== regression: reg_cuda-pw-spin-pbe --device=both ==="
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  "${BUILD}/regressions/reg_cuda-pw-spin-pbe" --device=both
  REG_RC=$?
fi

echo "BUILD_RC=$BUILD_RC UNIT_SPIN_RC=$UNIT_SPIN_RC UNIT_GUARD_RC=$UNIT_GUARD_RC REG_RC=$REG_RC"

# Disk wipe (binding)
rm -rf "${BUILD}" build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
find . -maxdepth 3 -type d \( -name build -o -name 'build-*' \) 2>/dev/null || true
echo "DONE wipe"
exit $(( BUILD_RC != 0 || UNIT_SPIN_RC != 0 || UNIT_GUARD_RC != 0 || REG_RC != 0 ))
