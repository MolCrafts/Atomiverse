#!/bin/bash -l
#SBATCH --job-name=dft-cuda-quadrature
#SBATCH --account=naiss2026-1-53-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/campaigns/dft-gap-20261026/cuda/quadrature-%j.log
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
# This executable was compiled before the quadrature fix. Preserve its genuine
# strict-stage failures before rebuilding; do not count this as an after pass.
set +e
"$build_dir/tests/test_cuda_pw_density" > "$receipt_dir/before-quadrature-density.log" 2>&1
before_rc=$?
set -e
printf 'before_density_exit=%s\n' "$before_rc"
cmake --build "$build_dir" -j8 --target cuda-pw-public test_cuda_pw_guards test_cuda_pw_xc_pbe test_cuda_pw_xc_pbe_spin test_cuda_pw_force_local test_cuda_pw_cell_stress test_cuda_pw_density
# Run the actual dedicated executables: the CTest names are individual Google
# Test names, not the CMake target names. Empty selections must not pass.
for test_name in guards xc_pbe xc_pbe_spin force_local cell_stress density; do
  "$build_dir/tests/test_cuda_pw_$test_name" | tee "$receipt_dir/after-$test_name.log"
done
"$build_dir/bin/cuda-pw-public" | tee "$receipt_dir/after-public.log"
sha256sum "$build_dir/bin/cuda-pw-public" "$build_dir/libatomiverse.a" > "$receipt_dir/after-binary-sha256.txt"
