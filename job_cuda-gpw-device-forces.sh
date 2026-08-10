#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J cuda-gpw-dfor
#SBATCH -o job_cuda-gpw-device-forces.%j.out
#SBATCH -e job_cuda-gpw-device-forces.%j.err

# GH200 dual-gate for slug cuda-gpw-device-forces
# (host AO Pulay + device grid V_eff / ionic HF / local-PP HF free-fn forces).
#   sbatch job_cuda-gpw-device-forces.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

BUILD="${TMPDIR:-/tmp}/atv_gpw_dfor_${SLURM_JOB_ID:-$$}"

wipe_builds() {
  rm -rf "${BUILD}" 2>/dev/null || true
  rm -rf build build-* _gpw_dfor_work CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
  find . -maxdepth 2 -type d \( -name build -o -name 'build-*' -o -name '_gpw_dfor_work' \) \
    -exec rm -rf {} + 2>/dev/null || true
  echo "builds_wiped=true"
  echo "BUILDS_WIPED"
}
trap wipe_builds EXIT

if [ ! -f molrs/Cargo.toml ]; then
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
echo "cmake=$(command -v cmake) ($($(command -v cmake) --version | head -1))"
echo "CC=$CC CXX=$CXX"
echo "rustc=$(command -v rustc) ($(rustc --version 2>/dev/null || echo fail))"
nvcc --version | head -4 || true
nvidia-smi -L || true
echo "=== GIT ==="
echo "HEAD=$(git rev-parse HEAD)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
git log -1 --oneline

FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"
echo "BUILD_DIR=${BUILD}"
rm -rf "${BUILD}"
mkdir -p "${BUILD}"
rm -rf CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

CFG_RC=1
for attempt in 1 2 3; do
  if cmake -S "${SLURM_SUBMIT_DIR}" -B "${BUILD}" \
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
    CFG_RC=0
    break
  fi
  echo "cmake configure attempt ${attempt} failed — wipe and retry"
  rm -rf "${BUILD}"
  sleep 5
done
if [ "${CFG_RC}" -ne 0 ]; then
  echo "CMAKE_CONFIGURE_FAILED"
  exit 1
fi

NJ=$(( $(nproc) > 32 ? 32 : $(nproc) ))
echo "=== STAGE 1: atomiverse (-j${NJ}) ==="
cmake --build "${BUILD}" -j"${NJ}" --target atomiverse || {
  echo "ATOMIVERSE_BUILD_FAILED"
  exit 1
}

TARGETS=(
  test_cuda_gpw_forces
  test_cuda_gpw_full_product
  reg_cuda-gpw-device-forces
)
echo "=== STAGE 2: scoped tests ==="
for t in "${TARGETS[@]}"; do
  echo "--- build ${t} ---"
  cmake --build "${BUILD}" -j"${NJ}" --target "${t}" || {
    echo "TARGET_BUILD_FAILED ${t}"
    exit 1
  }
done

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
echo "=== STAGE 3: ctest forces + full_product ==="
ctest --test-dir "${BUILD}" --output-on-failure \
  -R '^(test_cuda_gpw_forces|test_cuda_gpw_full_product)$' \
  || TEST_RC=$?
echo "CTEST_RC=${TEST_RC}"

REG_RC=0
echo "=== STAGE 4: run_reg_cuda-gpw-device-forces ==="
cmake --build "${BUILD}" --target run_reg_cuda-gpw-device-forces || REG_RC=$?
echo "REG_RC=${REG_RC}"

if [ "${TEST_RC}" -ne 0 ] || [ "${REG_RC}" -ne 0 ]; then
  echo "GPU_JOB_FAILED ctest=${TEST_RC} reg=${REG_RC}"
  exit 1
fi
echo "GPU_JOB_OK"
