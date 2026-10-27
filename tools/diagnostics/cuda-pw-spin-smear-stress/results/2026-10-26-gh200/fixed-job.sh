#!/bin/bash -l
#SBATCH --job-name=dft-spin-fixed-basis
#SBATCH --account=naiss2026-1-53-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/campaigns/dft-gap-wave2-20261026/cuda/fixed-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load GPU/buildenv-gcccuda/2026.03-cu13.0
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
export PATH
hash -r
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE
receipt="$PWD/.claude/campaigns/dft-gap-wave2-20261026/cuda"
build="$receipt/build"
cmake -S . -B "$build" -DATV_PW_SPIN_BASELINE="$receipt/baseline"
cmake --build "$build" -j8 --target cuda-pw-spin-smear-stress test_cuda_pw_density test_cuda_pw_guards test_cuda_pw_cell_stress cuda-pw-spin-residual-before
set +e
"$build/bin/cuda-pw-spin-residual-before" --gtest_filter=CudaPWDensity.OppositeSpinUpdatesHaveJointUnmixedResidual > "$receipt/before-joint-residual-fixed.log" 2>&1
baseline_rc=$?
set -e
printf 'before_joint_residual_exit=%s\n' "$baseline_rc"
[[ "$baseline_rc" == 1 ]]
for target in test_cuda_pw_density test_cuda_pw_guards test_cuda_pw_cell_stress; do
  "$build/tests/$target" | tee "$receipt/fixed-$target.log"
done
"$build/bin/cuda-pw-spin-smear-stress" "$receipt/fixed-results.json" .02 | tee "$receipt/fixed-public.log"
sha256sum "$build/bin/cuda-pw-spin-smear-stress" "$build/libatomiverse.a" > "$receipt/binary-sha256.txt"
