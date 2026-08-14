#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 02:00:00
#SBATCH -c 16
#SBATCH -J atv-cuda-pw
#SBATCH -o .claude/cuda-pw-slice-%j.out
#SBATCH -e .claude/cuda-pw-slice-%j.err

# GH200 verification for the CUDA plane-wave slice (cuda::pw::PW method class +
# device-resident density-side SCF pipeline). CUDA compiles only on the
# aarch64 GH200 node, so this batch job is the compile + parity gate for any
# change under src/cuda/planewave.
#
#   sbatch tools/gpu_job_cuda_pw.sh
#
# First green run: job 948055 on n438 — 8/8 ctest, MAX_EIG_DIFF 6.051e-15 Ha,
# |E_cuda - E_cpu| = 2.132e-14 Ry. Replaces the retired tools/gpu_job.sh,
# which still referenced the deleted test_cuda_pw_si target and benchmarks/.
#
# W2 additions to watch: test_cuda_pw_density (DeviceDensity stage parity vs
# the CPU stage helpers), the rewritten test_cuda_pw_xc_lda (VWN5, not PW92)
# and test_cuda_pw_veff (complex V_loc). diag_band_parity now binds device
# grids instead of uploading V_eff per iteration.
#
# ATV_TEST_HEAVY is NOT needed: W1 deleted tests/cuda/planewave/test_pw_si.cu
# (two full SCF solves registered under ctest — iron law 1), so no CUDA PW
# target matches ATV_TEST_HEAVY_REGEX any more. CPU/CUDA parity now lives in
# regressions/pw/pw-cuda-lda-si (--device=both), hence REGRESSIONS=ON below.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"

echo "host=$(hostname) arch=$(uname -m)"
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
echo "HEAD=$(git rev-parse HEAD)"
nvidia-smi -L || true

BUILD=build-cuda-pw-slice
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
  \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}"

# Stage 1: the library alone — this is where the new .cu sources must compile.
echo "=== STAGE 1: library ==="
cmake --build "${BUILD}" -j"$(nproc)" --target atomiverse

# Stage 2: the CUDA PW test binaries. TARGETS is edited per wave to include any
# newly registered test; a missing target here is a wiring bug, not a skip.
TARGETS="${ATV_PW_TARGETS:-test_cuda_pw_fft test_cuda_pw_poisson test_cuda_pw_grid_map test_cuda_pw_xc_lda test_cuda_pw_veff test_cuda_pw_guards test_cuda_pw_density test_cuda_scf_rhf diag_band_parity reg_pw-cuda-lda-si}"
echo "=== STAGE 2: tests: ${TARGETS} ==="
# shellcheck disable=SC2086
cmake --build "${BUILD}" -j"$(nproc)" --target ${TARGETS}

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
echo "=== STAGE 3: ctest ==="
ctest --test-dir "${BUILD}" --output-on-failure \
  -R 'test_cuda_pw_|test_cuda_scf_rhf' \
  || TEST_RC=$?

echo "CTEST_RC=${TEST_RC}"
if [ "${TEST_RC}" -ne 0 ]; then
  echo "GPU_JOB_TESTS_FAILED rc=${TEST_RC}"
  exit "${TEST_RC}"
fi

# Stage 4: band-level arbiter (diagnostic, not ctest) — the only check that
# sees the Hermitian subspace eigensolve at non-TRIM k-points. Prints
# MAX_EIG_DIFF and exits non-zero above 1e-5 Ha.
echo "=== STAGE 4: diag_band_parity ==="
DIAG_RC=0
# No CMAKE_RUNTIME_OUTPUT_DIRECTORY is set for this target — locate it.
DIAG_BIN=$(find "${BUILD}" -type f -name diag_band_parity -perm -u+x | head -1)
"${DIAG_BIN}" tests/data/pw/Si_lda_nc.upf 12.0 || DIAG_RC=$?
echo "DIAG_BAND_PARITY_RC=${DIAG_RC}"

# Stage 5: end-to-end CPU/CUDA parity + ABINIT-class literature window.
# Timed: this slug is the only in-tree exercise of the full CUDA SCF loop, so
# its CUDA-leg wall time is the before/after signal for band-side work. It is
# a 2-atom cell — a same-backend before/after only, never a CPU-vs-CUDA claim.
echo "=== STAGE 5: reg_pw-cuda-lda-si --device=both ==="
REG_RC=0
REG_BIN=$(find "${BUILD}" -type f -name reg_pw-cuda-lda-si -perm -u+x | head -1)
for dev in cpu cuda; do
  echo "--- timing ${dev} leg ---"
  /usr/bin/time -f "WALL_${dev}=%e s  MAXRSS=%M KB" \
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "${REG_BIN}" --device=${dev} || REG_RC=$?
done
cmake --build "${BUILD}" --target run_reg_pw-cuda-lda-si || REG_RC=$?
echo "REG_PW_CUDA_RC=${REG_RC}"

if [ "${DIAG_RC}" -ne 0 ] || [ "${REG_RC}" -ne 0 ]; then
  echo "GPU_JOB_PARITY_FAILED diag=${DIAG_RC} reg=${REG_RC}"
  exit 1
fi
echo "GPU_JOB_OK"
# Wipe only THIS job's build dir — a concurrent login-node CPU verification
# build (build-cpu-pw-verify) must survive; `build-*` would have eaten it.
rm -rf "${BUILD}" 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true
echo "BUILDS_WIPED"
