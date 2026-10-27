#!/bin/bash -l
#SBATCH --job-name=dft-spin-smear-stress
#SBATCH --account=naiss2026-1-53-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/campaigns/dft-gap-wave2-20261026/cuda/job-%j.log
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
mkdir -p "$build"
# Reuse the prior GPU Cargo cache through Cargo's normal fingerprints. No ABI
# check is disabled; the new CMake/CPU/CUDA build and evidence are independent.
if [[ ! -e "$build/cargo" ]]; then
    ln -s "$PWD/.claude/campaigns/dft-gap-20261026/cuda/build/cargo" "$build/cargo"
fi
hostname
nvidia-smi
cmake -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_CUDA=ON -DATV_BUILD_PW=ON -DATV_BUILD_HF=OFF -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_MD=OFF -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF \
  -DATV_BUILD_VIB=OFF -DATV_BUILD_PATH=OFF -DATV_BUILD_SEMIEMPIRICAL=OFF \
  -DATV_BUILD_POTENTIALS=OFF -DATV_BUILD_NEIGHBOR=OFF -DATV_BUILD_ANALYSIS=OFF \
  -DATV_BUILD_IO=OFF -DATV_BUILD_EVALUATE=ON -DATV_USE_MPI=OFF \
  -DATV_MOLPY_PATH="$PWD/../molpy" -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLASDIR:-}" \
  -DCMAKE_PROJECT_Atomiverse_INCLUDE="$PWD/tools/diagnostics/cuda-pw-spin-smear-stress/register.cmake"
cmake --build "$build" -j8 --target cuda-pw-spin-smear-stress test_cuda_pw_guards test_cuda_pw_cell_stress test_cuda_pw_density
for target in test_cuda_pw_guards test_cuda_pw_cell_stress test_cuda_pw_density; do
  "$build/tests/$target" | tee "$receipt/$target.log"
done
"$build/bin/cuda-pw-spin-smear-stress" "$receipt/results.json" | tee "$receipt/public.log"
sha256sum "$build/bin/cuda-pw-spin-smear-stress" "$build/libatomiverse.a" > "$receipt/binary-sha256.txt"
