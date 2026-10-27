#!/bin/bash -l
#SBATCH --job-name=dft-cuda-negative
#SBATCH --account=naiss2026-1-53-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/campaigns/dft-gap-20261026/cuda/negative-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load GPU/buildenv-gcccuda/2026.03-cu13.0
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
export PATH
hash -r
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
build_dir="$PWD/.claude/campaigns/dft-gap-20261026/cuda/build"
receipt_dir="$PWD/.claude/campaigns/dft-gap-20261026/cuda"
hostname
nvidia-smi -L
cmake -S . -B "$build_dir" -DATV_PW_QUADRATURE_BASELINE="$receipt_dir/baseline"
cmake --build "$build_dir" -j8 --target cuda-pw-density-before
set +e
"$build_dir/bin/cuda-pw-density-before" --gtest_filter=CudaPWDensity.PaddedXcResolvesHighFrequencyDensity > "$receipt_dir/before-high-frequency.log" 2>&1
before_rc=$?
set -e
printf 'before_high_frequency_exit=%s\n' "$before_rc"
if [[ "$before_rc" != 1 ]]; then
    echo "Expected original density implementation to fail the strict new stage test"
    exit 1
fi
compute-sanitizer --tool memcheck --error-exitcode 86 "$build_dir/tests/test_cuda_pw_density" --gtest_filter='CudaPWDensity.Padded*' > "$receipt_dir/memcheck.log" 2>&1
sha256sum include/atomiverse/cpu/planewave/internal/xc_quadrature.h \
  src/cpu/planewave/internal/xc_builder.cpp \
  include/atomiverse/cuda/planewave/internal/device_density.cuh \
  src/cuda/planewave/internal/device_density.cu \
  include/atomiverse/cuda/planewave/internal/pw_stress.cuh \
  src/cuda/planewave/internal/pw_stress.cu \
  include/atomiverse/cuda/planewave/pw_solver.cuh src/cuda/planewave/pw_solver.cu \
  tests/cuda/planewave/test_density.cu tests/cuda/planewave/test_guards.cu \
  tools/diagnostics/cuda-pw-public/main.cpp > "$receipt_dir/source-sha256.txt"
