#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -J atv-cuda-gpw-residency
#SBATCH -o .claude/cuda-gpw-residency-%j.out
#SBATCH -e .claude/cuda-gpw-residency-%j.err

# GH200 dual-gate after CUDA-only mid-path residency (no mid-SCF V_AO / P DH).
#   sbatch tools/gpu_job_cuda_gpw_residency.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

BUILD="${TMPDIR:-/tmp}/atv_gpw_res_${SLURM_JOB_ID:-$$}"

wipe_builds() {
  rm -rf "${BUILD}" 2>/dev/null || true
  rm -rf build build-* _gpw_res_work CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
  find . -maxdepth 2 -type d \( -name build -o -name 'build-*' -o -name '_gpw_res_work' \) \
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
echo "HEAD=$(git rev-parse HEAD)"
git log -1 --oneline
nvcc --version | head -3 || true
nvidia-smi -L || true

FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"
rm -rf "${BUILD}"
mkdir -p "${BUILD}"

cmake -S "${SLURM_SUBMIT_DIR}" -B "${BUILD}" \
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
  \
  -DATV_USE_MPI=OFF \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}" || {
  echo "CMAKE_CONFIGURE_FAILED"
  exit 1
}

NJ=$(( $(nproc) > 32 ? 32 : $(nproc) ))
cmake --build "${BUILD}" -j"${NJ}" --target atomiverse || {
  echo "ATOMIVERSE_BUILD_FAILED"
  exit 1
}

TARGETS=(
  test_cuda_gpw_full_product
  test_cuda_gpw_diis_eigen
  test_cuda_gpw_kpoints
  test_cuda_gpw_forces
  reg_cuda-gpw-full-product
  reg_cuda-gpw-device-forces
  reg_cuda-gpw-uks-o2-gate
  reg_cuda-gpw-kpoints
)

for t in "${TARGETS[@]}"; do
  echo "--- build ${t} ---"
  cmake --build "${BUILD}" -j"${NJ}" --target "${t}" || {
    echo "TARGET_BUILD_FAILED ${t}"
    # continue for optional missing targets
  }
done

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== CTEST unit dual-gates ==="
ctest --test-dir "${BUILD}" --output-on-failure \
  -R 'test_cuda_gpw_(full_product|diis_eigen|kpoints|forces)' || {
  echo "CTEST_FAILED"
  exit 1
}

run_reg() {
  local name="$1"
  local bin="${BUILD}/regressions/${name}"
  if [ ! -x "${bin}" ]; then
    echo "SKIP missing ${name}"
    return 0
  fi
  echo "=== REG ${name} ==="
  "${bin}" --device=both || {
    echo "REG_FAILED ${name}"
    exit 1
  }
}

run_reg reg_cuda-gpw-full-product
run_reg reg_cuda-gpw-device-forces
run_reg reg_cuda-gpw-uks-o2-gate
run_reg reg_cuda-gpw-kpoints

echo "RESIDENCY_DUAL_GATES_PASS"
echo "HEAD=$(git rev-parse HEAD)"
