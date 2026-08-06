#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J atv-cuda-gpw-foothold
#SBATCH -o .claude/cuda-gpw-foothold-%j.out
#SBATCH -e .claude/cuda-gpw-foothold-%j.err

# Build + unit-test the CUDA GPW Γ Poisson foothold on GH200.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0

# Drop x86 venv bins that shadow cmake on aarch64 GPU nodes.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE '/(\.x86|/molcrafts/\.(x86|aarch64))/bin/?$' \
    | grep -v '/\.x86/' | paste -sd:)"
hash -r

export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
# Prefer aarch64 rustc/cxxbridge on GH200; login often defaults to x86_64.
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
# Drop x86 cargo bin so cxxbridge is not the non-runnable x86 wrapper.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.cargo/bin' | paste -sd:):$HOME/.rustup/toolchains/stable-aarch64-unknown-linux-gnu/bin"
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
echo "rustc: $(command -v rustc) ($(rustc --version 2>/dev/null || echo fail))"
echo "cxxbridge: $(command -v cxxbridge || true)"

CMAKE="$(command -v cmake || true)"
if [ -z "$CMAKE" ] || ! "$CMAKE" --version >/dev/null 2>&1; then
    echo "ERROR: no working cmake on PATH" >&2
    echo "PATH=$PATH" >&2
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

# Ensure molrs submodule present (hard dependency of GPW/HF products).
if [ ! -f molrs/Cargo.toml ] && [ -f .gitmodules ]; then
    git submodule update --init --recursive molrs || true
fi

FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"
BD=build-cuda-gpw

rm -rf "$BD" CMakeCache.txt CMakeFiles 2>/dev/null || true

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
    -DATV_USE_MPI=OFF \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}"

# Foothold unit test + related RHF smoke (optional) + PW poisson sibling.
"$CMAKE" --build "$BD" -j"$(nproc)" \
    --target test_cuda_gpw_device_poisson test_cuda_pw_poisson

export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
TEST_RC=0
"$CTEST" --test-dir "$BD" --output-on-failure \
    -R 'test_cuda_gpw_device_poisson|test_cuda_pw_poisson' \
    || TEST_RC=$?

echo "=== CTEST_RC=$TEST_RC ==="
if [ "$TEST_RC" -ne 0 ]; then
    echo "GPU_JOB_TESTS_FAILED rc=$TEST_RC"
    # Keep build for log inspection only until wipe at end.
fi

# Disk law: never leave multi-GB builds on the node after the job.
rm -rf "$BD" build build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true
echo "builds_wiped=true"

if [ "$TEST_RC" -ne 0 ]; then
    exit "$TEST_RC"
fi
echo "GPU_JOB_OK"
