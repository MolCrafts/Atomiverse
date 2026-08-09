#!/bin/bash -l
# GH200 CUDA leg for parity-gap slug reg-cuda-grad-honesty.
# Builds the CUDA config in this worktree and runs the two multi-device
# gradient regressions with --device=both:
#   scf/h2o_rhf_sto3g_grad    (fail-closed device s/p gradient path)
#   scf/h2o_rhf_def2svp_grad  (documented host-fallback leg, flags asserted)
set -euo pipefail

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/$(uname -m)/cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

cd /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/worktrees/wf_f384f427-0b3-11

rm -rf build-cuda
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DATV_BUILD_CUDA=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
      -DATV_BUILD_SEMIEMPIRICAL=OFF \
      -DBLA_VENDOR=OpenBLAS \
      -DRust_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"
cmake --build build-cuda -j8 --target atomiverse reg_h2o_rhf_sto3g_grad reg_h2o_rhf_def2svp_grad

echo "=== run: reg_h2o_rhf_sto3g_grad --device=both ==="
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./build-cuda/regressions/reg_h2o_rhf_sto3g_grad --device=both
echo "=== run: reg_h2o_rhf_def2svp_grad --device=both ==="
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./build-cuda/regressions/reg_h2o_rhf_def2svp_grad --device=both
echo "=== ALL CUDA LEGS PASS ==="
