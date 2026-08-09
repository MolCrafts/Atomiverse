#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J atv-cuda-gpw-ks-chain
#SBATCH -o .claude/cuda-gpw-ks-chain-%j.out
#SBATCH -e .claude/cuda-gpw-ks-chain-%j.err

# GH200: build + dual-gate for composed CUDA GPW KS chain stage.
#   sbatch tools/gpu_job_cuda_gpw_ks_chain.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0

# Drop x86 venv bins that shadow cmake on aarch64 GPU nodes.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE '/(\.x86|/molcrafts/\.(x86|aarch64))/bin/?$' \
    | grep -v '/\.x86/' | paste -sd:)"
hash -r

export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.cargo/bin' | paste -sd:):$HOME/.rustup/toolchains/stable-aarch64-unknown-linux-gnu/bin"
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

CMAKE="$(command -v cmake || true)"
if [ -z "$CMAKE" ] || ! "$CMAKE" --version >/dev/null 2>&1; then
    echo "ERROR: no working cmake on PATH" >&2
    exit 1
fi
CTEST="$(dirname "$CMAKE")/ctest"
[ -x "$CTEST" ] || CTEST="$(command -v ctest)"

echo "=== HOST $(hostname) arch $(uname -m) ==="
echo "cmake: $CMAKE ($("$CMAKE" --version | head -1))"
nvcc --version | head -4 || true
nvidia-smi -L || true
echo "=== GIT ==="
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
echo "rustc: $(command -v rustc) ($(rustc --version 2>/dev/null || echo fail))"

if [ ! -f molrs/Cargo.toml ] && [ -f .gitmodules ]; then
    git submodule update --init --recursive molrs || true
fi

FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"
BD=build-cuda-gpw-ks-chain

rm -rf "$BD" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

"$CMAKE" -B "$BD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DATV_BUILD_CUDA=ON \
    -DATV_BUILD_HF=ON \
    -DATV_BUILD_PW=ON \
    -DATV_BUILD_GPW=ON \
    -DATV_BUILD_TESTS=ON \
    -DATV_BUILD_REGRESSIONS=ON \
    -DATV_BUILD_SEMIEMPIRICAL=OFF \
    -DATV_BUILD_MD=ON \
    -DATV_BUILD_MC=OFF \
    -DATV_BUILD_MINIMIZER=OFF \
    -DATV_BUILD_VIB=OFF \
    -DATV_BUILD_PATH=OFF \
    -DATV_USE_MPI=OFF \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}"

echo "=== STAGE 1: atomiverse ==="
"$CMAKE" --build "$BD" -j"$(nproc)" --target atomiverse

echo "=== STAGE 2: scoped tests + regression ==="
"$CMAKE" --build "$BD" -j"$(nproc)" --target \
    test_cuda_gpw_ks_chain \
    test_cuda_gpw_device_poisson \
    test_cuda_gpw_xc_lda \
    test_cuda_gpw_product_collocate \
    test_cuda_gpw_product_integrate \
    reg_cuda-gpw-ks-chain

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
echo "=== STAGE 3: ctest KS chain + footholds ==="
"$CTEST" --test-dir "$BD" --output-on-failure \
    -R '^(test_cuda_gpw_ks_chain|test_cuda_gpw_device_poisson|test_cuda_gpw_xc_lda|test_cuda_gpw_product_collocate|test_cuda_gpw_product_integrate)$' \
    || TEST_RC=$?
echo "CTEST_RC=$TEST_RC"

REG_RC=0
echo "=== STAGE 4: run_reg_cuda-gpw-ks-chain ==="
"$CMAKE" --build "$BD" --target run_reg_cuda-gpw-ks-chain || REG_RC=$?
echo "REG_RC=$REG_RC"

# Disk law
rm -rf "$BD" build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
LEFTOVER=$(find . -maxdepth 3 -type d \( -name build -o -name 'build-*' \) 2>/dev/null || true)
if [ -n "$LEFTOVER" ]; then
    echo "WARNING leftover builds: $LEFTOVER"
    rm -rf $LEFTOVER 2>/dev/null || true
fi
echo "builds_wiped=true"

if [ "$TEST_RC" -ne 0 ] || [ "$REG_RC" -ne 0 ]; then
    echo "GPU_JOB_FAILED ctest=$TEST_RC reg=$REG_RC"
    exit 1
fi
echo "GPU_JOB_OK"
