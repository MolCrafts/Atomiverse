#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J atv-path-select
#SBATCH -o .claude/cuda-rhf-path-select-%j.out
#SBATCH -e .claude/cuda-rhf-path-select-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0

export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE '/(\.x86|/molcrafts/\.(x86|aarch64))/bin/?$' \
    | grep -v '/\.x86/' | paste -sd:)"
hash -r

export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
# aarch64 GH200: do not override to x86 Rust.
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

if [ ! -f molrs/Cargo.toml ] && [ -f .gitmodules ]; then
    git submodule update --init --recursive molrs || true
fi

BD=build-cuda-path-select
rm -rf "$BD" CMakeCache.txt CMakeFiles 2>/dev/null || true

"$CMAKE" -B "$BD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DATV_BUILD_CUDA=ON \
    -DATV_BUILD_HF=ON \
    -DATV_BUILD_PW=OFF \
    -DATV_BUILD_GPW=OFF \
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
    -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}"

echo "=== BUILD tests ==="
"$CMAKE" --build "$BD" -j"${SLURM_CPUS_PER_TASK:-8}" \
    --target test_cuda_scf_rhf test_cuda_scf_device_integrals test_cuda_scf

echo "=== CTEST ==="
"$CTEST" --test-dir "$BD" --output-on-failure \
    -R 'test_cuda_scf_rhf|test_cuda_scf_device_integrals|test_cuda_scf$'
echo "CTEST_RC=$?"

echo "=== WIPE BUILD ==="
rm -rf "$BD" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true
echo "builds_wiped=true"
echo "=== DONE ==="
