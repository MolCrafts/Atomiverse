#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J atv-cuda-gpw-product
#SBATCH -o .claude/cuda-gpw-product-%j.out
#SBATCH -e .claude/cuda-gpw-product-%j.err

# GH200 dual-gate for cuda-gpw-product matrix:
#   water σ, UKS F/σ, multi-k F/σ, plus force/stress unit guards.
#   sbatch tools/gpu_job_cuda_gpw_product.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

BUILD="${TMPDIR:-/tmp}/atv_gpw_product_${SLURM_JOB_ID:-$$}"

wipe_builds() {
  rm -rf "${BUILD}" 2>/dev/null || true
  echo "builds_wiped=true"
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

echo "=== HOST $(hostname) arch $(uname -m) $(date -Is) ==="
echo "HEAD=$(git rev-parse HEAD)"
git log -1 --oneline
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
  -DATV_USE_MPI=OFF \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}"

NJ=$(( $(nproc) > 32 ? 32 : $(nproc) ))
echo "=== STAGE 1: atomiverse ==="
cmake --build "${BUILD}" -j"${NJ}" --target atomiverse

echo "=== STAGE 2: units + regs ==="
for t in test_cuda_gpw_forces test_cuda_gpw_stress \
         reg_cuda-gpw-product-h2o-stress \
         reg_cuda-gpw-uks-o2-forces \
         reg_cuda-gpw-kpoint-forces; do
  echo "--- build ${t} ---"
  cmake --build "${BUILD}" -j"${NJ}" --target "${t}"
done

ctest --test-dir "${BUILD}" --output-on-failure --timeout 180 \
  -R '^test_cuda_gpw_forces$|^test_cuda_gpw_stress$'

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
echo "=== water σ ==="
"${BUILD}/regressions/reg_cuda-gpw-product-h2o-stress" --device=both
echo "=== UKS F/σ ==="
"${BUILD}/regressions/reg_cuda-gpw-uks-o2-forces" --device=both
echo "=== multi-k F/σ ==="
"${BUILD}/regressions/reg_cuda-gpw-kpoint-forces" --device=both

echo "GPU_JOB_OK"
echo "=== DONE green $(date -Is) ==="
