#!/bin/bash -l
#SBATCH -J atv-gpw-xc-lda
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -o job_cuda_gpw_xc_lda.%j.out
#SBATCH -e job_cuda_gpw_xc_lda.%j.err

# Build + unit-test CUDA GPW Padé LDA XC foothold on GH200.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"

module load GPU/buildenv-gcccuda/2026.03-cu13.0

# Drop x86 venv bins that shadow cmake on aarch64 GPU nodes.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE '/(\.x86|/molcrafts/\.(x86|aarch64))/bin/?$' \
    | grep -v '/\.x86/' | paste -sd:)"
hash -r

export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
unset RUSTUP_TOOLCHAIN || true
# aarch64 GH200: default toolchain is correct — do not force x86.
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=4
export ATV_TEST_HEAVY=OFF

CMAKE="$(command -v cmake || true)"
if [ -z "$CMAKE" ] || ! "$CMAKE" --version >/dev/null 2>&1; then
    echo "ERROR: no working cmake on PATH" >&2
    echo "PATH=$PATH" >&2
    exit 1
fi
CTEST="$(dirname "$CMAKE")/ctest"
[ -x "$CTEST" ] || CTEST="$(command -v ctest)"

echo "=== HOST $(hostname) arch $(uname -m) date $(date -Is) ==="
echo "cmake: $CMAKE ($("$CMAKE" --version | head -1))"
nvcc --version | head -4 || true
nvidia-smi -L || true
echo "=== GIT ==="
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git log -1 --oneline

if [ ! -f molrs/Cargo.toml ] && [ -f .gitmodules ]; then
    git submodule update --init --recursive molrs || true
fi

FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"
BD=build-cuda-gpw-xc

# Disk law: wipe stale builds before configure.
rm -rf build build-* "$BD" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

"$CMAKE" -B "$BD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DATV_BUILD_CUDA=ON \
    -DATV_BUILD_HF=ON \
    -DATV_BUILD_PW=ON \
    -DATV_BUILD_GPW=ON \
    -DATV_BUILD_TESTS=ON \
    -DATV_BUILD_REGRESSIONS=OFF \
    -DATV_BUILD_SEMIEMPIRICAL=OFF \
    -DATV_BUILD_MD=ON \
    -DATV_BUILD_MC=OFF \
    -DATV_BUILD_MINIMIZER=OFF \
    -DATV_BUILD_VIB=OFF \
    -DATV_BUILD_PATH=OFF \
    -DATV_BUILD_KSPACE=OFF \
    -DATV_BUILD_TORCH=OFF \
    -DATV_USE_MPI=OFF \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}"

# Foothold unit + sibling Poisson/PW XC + RHF smoke.
"$CMAKE" --build "$BD" -j"$(nproc)" \
    --target \
    atomiverse \
    test_cuda_gpw_xc_lda \
    test_cuda_gpw_device_poisson \
    test_cuda_pw_xc_lda \
    test_cuda_scf_rhf \
    test_cuda_scf_device_integrals

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
echo "=== ctest ==="
"$CTEST" --test-dir "$BD" --output-on-failure \
  -R 'test_cuda_gpw_xc_lda|test_cuda_gpw_device_poisson|test_cuda_pw_xc_lda|test_cuda_scf_rhf|test_cuda_scf_device_integrals'

echo "=== PASS marker ==="
echo "GPU_JOB_OK job=${SLURM_JOB_ID:-local} host=$(hostname)"

# Disk law: never leave multi-GB builds.
rm -rf build build-* "$BD" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true
echo "builds_wiped=true"
