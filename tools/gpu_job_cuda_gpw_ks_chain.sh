#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -J atv-cuda-gpw-ks-chain
#SBATCH -o /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/worktrees/wf_escuda_cuda_gpw_ks_chain/.claude/cuda-gpw-ks-chain-%j.out
#SBATCH -e /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/worktrees/wf_escuda_cuda_gpw_ks_chain/.claude/cuda-gpw-ks-chain-%j.err

# GH200 gate for slug cuda-gpw-ks-chain (composed ρ→V_eff stage unit).
#   sbatch tools/gpu_job_cuda_gpw_ks_chain.sh
#
# Build dir deliberately avoids `build` / `build-*` so concurrent disk-law
# sweeps cannot unmap the tree mid-compile (seen on jobs 1049633/1049757).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

if [ ! -f molrs/CMakeLists.txt ]; then
  git submodule update --init --recursive molrs
fi

module load GPU/buildenv-gcccuda/2026.03-cu13.0
# Drop x86 login-node bins that cannot execute on aarch64 GH200.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | grep -v '/\.cargo/bin' | paste -sd:)
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
hash -r

echo "=== HOST $(hostname) arch $(uname -m) ==="
echo "cmake=$(command -v cmake) ($($(command -v cmake) --version | head -1))"
echo "CC=$CC CXX=$CXX"
echo "rustc=$(command -v rustc) ($(rustc --version 2>/dev/null || echo fail))"
echo "cxxbridge=$(command -v cxxbridge || true)"
nvcc --version | head -4 || true
nvidia-smi -L || true
echo "=== GIT ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
git log -1 --oneline

FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"
# Name avoids build/build-* (disk-law sweeps) and lives under the worktree
# like the green cuda-pw-nondefault-stream job (not /tmp — rust/molrs cache
# paths are large; NFS stale was from wiping build-* mid-job, not NFS itself).
BUILD=_ks_chain_work
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
    -DATV_BUILD_PW=ON \
    -DATV_BUILD_GPW=ON \
    -DATV_BUILD_SEMIEMPIRICAL=OFF \
    -DATV_BUILD_MD=OFF \
    -DATV_BUILD_MC=OFF \
    -DATV_BUILD_MINIMIZER=OFF \
    -DATV_BUILD_VIB=OFF \
    -DATV_BUILD_PATH=OFF \
    -DATV_BUILD_TORCH=OFF \
    -DATV_USE_MPI=OFF \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}"; then
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

NJ=$(( $(nproc) > 32 ? 32 : $(nproc) ))
echo "=== STAGE 1: atomiverse (-j${NJ}) ==="
cmake --build "${BUILD}" -j"${NJ}" --target atomiverse

TARGETS=(
  test_cuda_gpw_ks_chain
  test_cuda_gpw_device_poisson
  test_cuda_gpw_xc_lda
  test_cuda_gpw_product_collocate
  test_cuda_gpw_product_integrate
  reg_cuda-gpw-ks-chain
)
echo "=== STAGE 2: scoped tests (sequential) ==="
for t in "${TARGETS[@]}"; do
  echo "--- build ${t} ---"
  cmake --build "${BUILD}" -j"${NJ}" --target "${t}"
done

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
echo "=== STAGE 3: ctest KS chain + footholds ==="
ctest --test-dir "${BUILD}" --output-on-failure \
  -R '^(test_cuda_gpw_ks_chain|test_cuda_gpw_device_poisson|test_cuda_gpw_xc_lda|test_cuda_gpw_product_collocate|test_cuda_gpw_product_integrate)$' \
  || TEST_RC=$?
echo "CTEST_RC=${TEST_RC}"

REG_RC=0
echo "=== STAGE 4: run_reg_cuda-gpw-ks-chain ==="
cmake --build "${BUILD}" --target run_reg_cuda-gpw-ks-chain || REG_RC=$?
echo "REG_RC=${REG_RC}"

# Disk law — wipe this work dir + any accidental build trees.
rm -rf "${BUILD}" 2>/dev/null || true
rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true
LEFTOVER=$(find . -maxdepth 3 -type d \( -name build -o -name 'build-*' -o -name '_ks_chain_work' \) 2>/dev/null || true)
if [ -n "${LEFTOVER:-}" ]; then
  echo "WARNING leftover builds: ${LEFTOVER}"
  # shellcheck disable=SC2086
  rm -rf ${LEFTOVER} 2>/dev/null || true
fi
echo "builds_wiped=true"
echo "BUILDS_WIPED"

if [ "${TEST_RC}" -ne 0 ] || [ "${REG_RC}" -ne 0 ]; then
  echo "GPU_JOB_FAILED ctest=${TEST_RC} reg=${REG_RC}"
  exit 1
fi
echo "GPU_JOB_OK"
