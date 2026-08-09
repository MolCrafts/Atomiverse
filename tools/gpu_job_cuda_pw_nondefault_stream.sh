#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:30:00
#SBATCH -c 16
#SBATCH -J atv-pw-stream
#SBATCH -o /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/cuda-pw-nondefault-stream-%j.out
#SBATCH -e /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/cuda-pw-nondefault-stream-%j.err

# GH200 gate for slug cuda-pw-nondefault-stream.
#   sbatch tools/gpu_job_cuda_pw_nondefault_stream.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Worktrees do not always carry a populated molrs submodule.
if [ ! -f molrs/CMakeLists.txt ]; then
  git submodule update --init --recursive molrs
fi

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
# Pin compilers so a mid-build cmake reconfigure cannot lose them.
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
# GH200 is aarch64 — never keep a login-node x86 RUSTUP override.
unset RUSTUP_TOOLCHAIN || true
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
export PATH="${HOME}/.rustup/toolchains/${RUSTUP_TOOLCHAIN}/bin:${PATH}"

echo "host=$(hostname) arch=$(uname -m)"
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
echo "CC=$CC CXX=$CXX"
echo "rustc=$(command -v rustc) $(rustc --version 2>/dev/null || true)"
echo "HEAD=$(git rev-parse HEAD)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
nvidia-smi -L || true

# Name deliberately avoids `build` / `build-*` so concurrent disk-law
# sweeps cannot unmap the tree mid-compile (jobs 1049633/1049641).
BUILD=_pw_stream_work
# Keep a failed tree for diagnosis unless ATV_FRESH=1 or success wipe.
if [ "${ATV_FRESH:-1}" = "1" ]; then
  rm -rf "${BUILD}"
fi
rm -rf CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

for attempt in 1 2 3; do
  if cmake -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_CUDA_HOST_COMPILER="${CXX}" \
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
    -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}"; then
    break
  fi
  echo "cmake configure attempt ${attempt} failed — wipe and retry"
  rm -rf "${BUILD}"
  if [ "${attempt}" -eq 3 ]; then
    echo "CMAKE_CONFIGURE_FAILED"
    exit 1
  fi
  sleep 5
done

echo "=== STAGE 1: library ==="
cmake --build "${BUILD}" -j"$(nproc)" --target atomiverse

# Build test binaries one-by-one to avoid concurrent cmake regenerations
# racing FindMPI (seen as CMAKE_CXX_COMPILER not set mid-stage-2).
TARGETS=(
  test_cuda_pw_fft
  test_cuda_pw_poisson
  test_cuda_pw_grid_map
  test_cuda_pw_xc_lda
  test_cuda_pw_xc_pbe
  test_cuda_pw_veff
  test_cuda_pw_guards
  test_cuda_pw_density
  test_cuda_scf_rhf
  diag_band_parity
  reg_pw-cuda-lda-si
  reg_cuda-pw-nondefault-stream
)
echo "=== STAGE 2: tests (sequential targets) ==="
for t in "${TARGETS[@]}"; do
  echo "--- build ${t} ---"
  cmake --build "${BUILD}" -j"$(nproc)" --target "${t}"
done

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
echo "=== STAGE 3: ctest ==="
# Exact names only — do not use a broad test_cuda_pw_ regex (picks up
# unbuilt targets if TARGETS and CMakeLists drift).
ctest --test-dir "${BUILD}" --output-on-failure \
  -R '^(test_cuda_pw_fft|test_cuda_pw_poisson|test_cuda_pw_grid_map|test_cuda_pw_xc_lda|test_cuda_pw_xc_pbe|test_cuda_pw_veff|test_cuda_pw_guards|test_cuda_pw_density|test_cuda_scf_rhf)$' \
  || TEST_RC=$?

echo "CTEST_RC=${TEST_RC}"
if [ "${TEST_RC}" -ne 0 ]; then
  echo "GPU_JOB_TESTS_FAILED rc=${TEST_RC}"
  exit "${TEST_RC}"
fi

echo "=== STAGE 4: diag_band_parity ==="
DIAG_RC=0
DIAG_BIN=$(find "${BUILD}" -type f -name diag_band_parity -perm -u+x | head -1)
"${DIAG_BIN}" tests/data/pw/Si_lda_nc.upf 12.0 || DIAG_RC=$?
echo "DIAG_BAND_PARITY_RC=${DIAG_RC}"

echo "=== STAGE 5: regressions ==="
REG_RC=0
cmake --build "${BUILD}" --target run_reg_pw-cuda-lda-si || REG_RC=$?
cmake --build "${BUILD}" --target run_reg_cuda-pw-nondefault-stream || REG_RC=$?
echo "REG_RC=${REG_RC}"

if [ "${DIAG_RC}" -ne 0 ] || [ "${REG_RC}" -ne 0 ]; then
  echo "GPU_JOB_PARITY_FAILED diag=${DIAG_RC} reg=${REG_RC}"
  exit 1
fi
echo "GPU_JOB_OK"
rm -rf "${BUILD}" 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true
echo "BUILDS_WIPED"
