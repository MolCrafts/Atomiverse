#!/bin/bash -l
#SBATCH -J atv-cuda-fails
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -o logs/atv-cuda-fails-%j.out
#SBATCH -e logs/atv-cuda-fails-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
module load GPU/buildenv-gcccuda/2026.03-cu13.0
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
BUILD=build-cuda-check
cmake --build "$BUILD" -j"$(nproc)" --target test_cuda_multi_gpu 2>&1 | tail -2
for T in test_cuda_multi_gpu; do
  echo "=== $T ==="
  BIN=$(find "$BUILD" -name "$T" -type f -executable | head -1)
  timeout 300 "$BIN" 2>&1 | grep -B2 -A12 "Failure" | head -80
  echo "RC=$?"
done
echo DONE
