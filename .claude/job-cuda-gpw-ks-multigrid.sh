#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -J cuda-gpw-ks-mg
#SBATCH -o .claude/cuda-gpw-ks-multigrid-%j.out
#SBATCH -e .claude/cuda-gpw-ks-multigrid-%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

BUILD="${TMPDIR:-/tmp}/atv_ks_mg_${SLURM_JOB_ID:-$$}"

wipe_builds() {
  rm -rf "${BUILD}" 2>/dev/null || true
  rm -rf build build-* build_* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
  find . -maxdepth 2 -type d \( -name build -o -name 'build-*' -o -name 'build_*' \) \
    -exec rm -rf {} + 2>/dev/null || true
  echo "builds_wiped=true"
  echo "BUILDS_WIPED"
}
trap wipe_builds EXIT

if [ ! -f molrs/CMakeLists.txt ]; then
  git submodule update --init --recursive molrs
fi

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | grep -v '/\.cargo/bin' | paste -sd:)
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
unset RUSTUP_TOOLCHAIN || true
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
export PATH="${HOME}/.rustup/toolchains/${RUSTUP_TOOLCHAIN}/bin:${PATH}"
hash -r

echo "=== HOST $(hostname) arch $(uname -m) ==="
echo "CC=$CC CXX=$CXX"
echo "rustc=$(command -v rustc) ($(rustc --version 2>/dev/null || echo fail))"
nvcc --version | head -3 || true
nvidia-smi -L || true
echo "HEAD=$(git rev-parse HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"
git log -1 --oneline

echo "BUILD_DIR=${BUILD}"
rm -rf "${BUILD}"
mkdir -p "${BUILD}"

if ! cmake -S "${SLURM_SUBMIT_DIR}" -B "${BUILD}" \
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
  -DBLA_VENDOR=OpenBLAS
then
  echo "CMAKE_FAILED"; exit 2
fi

NJ=$(( $(nproc) > 32 ? 32 : $(nproc) ))
echo "=== STAGE 1: atomiverse (-j${NJ}) ==="
if ! cmake --build "${BUILD}" -j"${NJ}" --target atomiverse; then
  echo "ATOMIVERSE_BUILD_FAILED"; exit 3
fi

TARGETS=(
  test_cuda_gpw_ks_multigrid
  test_cuda_gpw_ks_chain
  test_cuda_gpw_ks_gga
  reg_cuda-gpw-ks-multigrid
)
echo "=== STAGE 2: scoped tests ==="
for t in "${TARGETS[@]}"; do
  echo "--- build ${t} ---"
  if ! cmake --build "${BUILD}" -j"${NJ}" --target "${t}"; then
    echo "TARGET_BUILD_FAILED ${t}"; exit 3
  fi
done

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
echo "=== STAGE 3: ctest multigrid + chain + gga ==="
ctest --test-dir "${BUILD}" --output-on-failure \
  -R 'test_cuda_gpw_ks_multigrid|test_cuda_gpw_ks_chain|test_cuda_gpw_ks_gga' \
  || TEST_RC=$?
echo "CTEST_RC=${TEST_RC}"

REG_RC=0
echo "=== STAGE 4: reg_cuda-gpw-ks-multigrid ==="
REG=$(find "${BUILD}" -name 'reg_cuda-gpw-ks-multigrid' -type f -executable | head -1)
echo "REG=${REG}"
if [ -z "${REG}" ]; then
  echo "REG_MISSING"; REG_RC=1
else
  "${REG}" --device=both || REG_RC=$?
fi
echo "REG_RC=${REG_RC}"

if [ "${TEST_RC}" -ne 0 ] || [ "${REG_RC}" -ne 0 ]; then
  echo "GPU_JOB_FAILED ctest=${TEST_RC} reg=${REG_RC}"
  exit 4
fi
echo "ALL_GREEN"
echo "GPU_JOB_OK"
exit 0
