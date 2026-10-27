#!/bin/bash -l
#SBATCH --job-name=dft-pw-cuda-gaps
#SBATCH --account=naiss2026-1-53-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/campaigns/dft-gap-20261026/cuda/job-%j.log
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
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_CUDA=ON -DATV_BUILD_PW=ON -DATV_BUILD_HF=OFF -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
  -DATV_BUILD_VIB=OFF -DATV_BUILD_PATH=OFF -DATV_BUILD_SEMIEMPIRICAL=OFF \
  -DATV_BUILD_POTENTIALS=OFF -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_ANALYSIS=OFF \
  -DATV_BUILD_IO=OFF -DATV_BUILD_EVALUATE=ON -DATV_USE_MPI=OFF \
  -DATV_MOLPY_PATH="$PWD/../molpy" -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLASDIR:-}" \
  -DCMAKE_PROJECT_Atomiverse_INCLUDE="$PWD/tools/diagnostics/cuda-pw-public/register.cmake"
cmake --build "$build_dir" -j8 --target cuda-pw-public test_cuda_pw_guards test_cuda_pw_xc_pbe test_cuda_pw_xc_pbe_spin test_cuda_pw_force_local test_cuda_pw_cell_stress test_cuda_pw_density
ctest --test-dir "$build_dir" --output-on-failure -R '^test_cuda_pw_(guards|xc_pbe|xc_pbe_spin|force_local|cell_stress|density)' | tee "$receipt_dir/unit.log"
"$build_dir/bin/cuda-pw-public" | tee "$receipt_dir/public.log"
sha256sum "$build_dir/bin/cuda-pw-public" "$build_dir/libatomiverse.a" > "$receipt_dir/binary-sha256.txt"
