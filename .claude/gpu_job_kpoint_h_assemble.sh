#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J cuda-gpw-kpt-Hass
#SBATCH -o .claude/cuda-gpw-kpoint-h-assemble-%j.out
#SBATCH -e .claude/cuda-gpw-kpoint-h-assemble-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="${OPENBLASDIR:-}"
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
# aarch64 GH200: do NOT override RUSTUP_TOOLCHAIN

if [ ! -f molrs/Cargo.toml ]; then
  git submodule update --init --recursive molrs
fi

BUILD=build-gpu-kpt-hass
rm -rf "${BUILD}" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
  -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS

cmake --build "${BUILD}" -j"$(nproc)" --target \
  atomiverse \
  test_cuda_gpw_kpoint_h_assemble \
  test_cuda_gpw_kpoint_integrate \
  test_cuda_gpw_kpoints \
  reg_cuda-gpw-kpoint-h-assemble \
  reg_cuda-gpw-kpoints
BUILD_RC=$?

UNIT_H_RC=1
UNIT_INT_RC=1
UNIT_KPTS_RC=1
REG_RC=1
REG_KPTS_RC=1
if [ "$BUILD_RC" -eq 0 ]; then
  echo "=== unit: test_cuda_gpw_kpoint_h_assemble ==="
  ctest --test-dir "${BUILD}" -R '^test_cuda_gpw_kpoint_h_assemble$' --output-on-failure
  UNIT_H_RC=$?

  echo "=== unit: test_cuda_gpw_kpoint_integrate (hold) ==="
  ctest --test-dir "${BUILD}" -R '^test_cuda_gpw_kpoint_integrate$' --output-on-failure
  UNIT_INT_RC=$?

  echo "=== unit: test_cuda_gpw_kpoints (hold) ==="
  ctest --test-dir "${BUILD}" -R '^test_cuda_gpw_kpoints$' --output-on-failure
  UNIT_KPTS_RC=$?

  echo "=== regression: reg_cuda-gpw-kpoint-h-assemble --device=both ==="
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  "${BUILD}/regressions/reg_cuda-gpw-kpoint-h-assemble" --device=both
  REG_RC=$?

  echo "=== regression: reg_cuda-gpw-kpoints --device=both (parent dual-gate hold) ==="
  "${BUILD}/regressions/reg_cuda-gpw-kpoints" --device=both
  REG_KPTS_RC=$?
fi

echo "BUILD_RC=$BUILD_RC UNIT_H_RC=$UNIT_H_RC UNIT_INT_RC=$UNIT_INT_RC UNIT_KPTS_RC=$UNIT_KPTS_RC REG_RC=$REG_RC REG_KPTS_RC=$REG_KPTS_RC"

if [ "$BUILD_RC" -eq 0 ] && [ "$UNIT_H_RC" -eq 0 ] && [ "$UNIT_INT_RC" -eq 0 ] \
   && [ "$UNIT_KPTS_RC" -eq 0 ] && [ "$REG_RC" -eq 0 ] && [ "$REG_KPTS_RC" -eq 0 ]; then
  echo "GPU_JOB_OK"
  OK=0
else
  echo "GPU_JOB_FAIL"
  OK=1
fi

# Disk wipe (binding)
rm -rf "${BUILD}" build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
find . -maxdepth 3 -type d \( -name build -o -name 'build-*' \) 2>/dev/null || true
echo "DONE wipe"
exit $OK
