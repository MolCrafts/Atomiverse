#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -J atv-cuda-gpw-product-si
#SBATCH -o .claude/cuda-gpw-product-si-%j.out
#SBATCH -e .claude/cuda-gpw-product-si-%j.err

# GH200 dual-gate: Si Γ RKS PADE ngrids=1 after KB Vnl fix (product-si-gate).
#   sbatch tools/gpu_job_cuda_gpw_product_si_gate.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

BUILD="${TMPDIR:-/tmp}/atv_gpw_si_${SLURM_JOB_ID:-$$}"

wipe_builds() {
  rm -rf "${BUILD}" 2>/dev/null || true
  rm -rf build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
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

FFTW_HINT=""
for d in /software/*/fftw* /usr; do
  if [ -f "${d}/include/fftw3.h" ] || [ -f "${d}/include/fftw3.f" ]; then
    FFTW_HINT="$d"
    break
  fi
done

cmake -S . -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
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
  -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}" || {
  echo "CMAKE_CONFIGURE_FAILED"
  exit 1
}

NJ=$(( $(nproc) > 32 ? 32 : $(nproc) ))
cmake --build "${BUILD}" -j"${NJ}" --target atomiverse || {
  echo "ATOMIVERSE_BUILD_FAILED"
  exit 1
}

for t in test_cuda_gpw_full_product reg_cuda-gpw-product-si-gate reg_cuda-gpw-full-product reg_cuda-gpw-product-h2o-gate; do
  echo "--- build ${t} ---"
  cmake --build "${BUILD}" -j"${NJ}" --target "${t}" || {
    echo "TARGET_BUILD_FAILED ${t}"
    exit 1
  }
done

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "=== CTEST unit suite (no Si e2e; iron-law 10s) ==="
ctest --test-dir "${BUILD}" --output-on-failure -R 'test_cuda_gpw_full_product' || {
  echo "CTEST_FAILED"
  exit 1
}

run_reg() {
  local name="$1"
  local bin="${BUILD}/regressions/${name}"
  if [ ! -x "${bin}" ]; then
    echo "SKIP missing ${name}"
    return 1
  fi
  echo "=== REG ${name} ==="
  "${bin}" --device=both || {
    echo "REG_FAILED ${name}"
    exit 1
  }
}

# Primary gate for this slug (Si diamond e2e lives here, not in tests/)
run_reg reg_cuda-gpw-product-si-gate
run_reg reg_cuda-gpw-full-product
run_reg reg_cuda-gpw-product-h2o-gate

echo "PRODUCT_SI_GATE_PASS"
echo "GPU_JOB_OK"
echo "HEAD=$(git rev-parse HEAD)"
