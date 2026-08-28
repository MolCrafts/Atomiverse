#!/bin/bash -l
#SBATCH -J atv-cuda-tree-check
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -o logs/atv-cuda-tree-check-%j.out
#SBATCH -e logs/atv-cuda-tree-check-%j.err

# Full CUDA-ON configure + build + ctest of the current working tree on the
# GH200 node: gates that the comm-brick chain + MD-perf changes stay
# CUDA-clean before any CUDA MD work starts.
set -uo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
mkdir -p logs

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
unset CARGO_BUILD_TARGET || true

echo "=== host $(hostname) $(date -Is) ==="
nvidia-smi -L || true

BUILD=build-cuda-check
if [[ -f "$BUILD/CMakeCache.txt" ]] && grep -q 'x86_64-unknown-linux-gnu' "$BUILD/CMakeCache.txt" 2>/dev/null; then
  rm -rf "$BUILD"
fi
if [[ ! -f "$BUILD/CMakeCache.txt" ]]; then
  git submodule update --init --recursive molrs || true
  cmake -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DATV_BUILD_CUDA=ON \
    -DATV_BUILD_TESTS=ON \
    -DATV_BUILD_HF=OFF \
    -DATV_BUILD_PW=OFF \
    -DATV_BUILD_GPW=OFF \
    -DBLA_VENDOR=OpenBLAS || exit 1
fi

echo "=== build all ==="
cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tail -5
RC=$?
echo "BUILD_RC=$RC"
[[ $RC -ne 0 ]] && cmake --build "$BUILD" -j1 2>&1 | grep -B3 -A8 "error" | head -60

echo "=== ctest (serial: one GH200, GPU tests contend at -j>1) ==="
ctest --test-dir "$BUILD" --output-on-failure -j 1 --timeout 120 2>&1 | tee /tmp/ctest-cuda.log | tail -15
echo "=== failure details ==="
grep -B 3 -A 25 "\[  FAILED  \]" /tmp/ctest-cuda.log | head -120
echo "=== done $(date -Is) ==="
