#!/bin/bash -l
# Validate regressions/md/lj_argon multi-device (.cu) on the GH200 node:
# scoped build of atomiverse + reg_lj_argon, then the registered
# run_reg_lj_argon leg (--device=both, OMP_NUM_THREADS=1).
set -e
module load GPU/buildenv-gcccuda/2026.03-cu13.0

cd "$SLURM_SUBMIT_DIR"

# The molcrafts per-arch venvs ship a pip 'cmake' stub whose shebang points at
# an x86 python3; drop those bins so cmake resolves to a real aarch64 binary.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE '/molcrafts/\.(x86|aarch64)/bin/?$' | paste -sd:)"

export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
# aarch64 cargo/rustc live in the arch-split rustup home; the shared-HOME
# default toolchain is x86_64 and cannot exec here.
export PATH="$HOME/.local/$(uname -m)/cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

echo "== node: $(hostname)  arch: $(uname -m)"
nvidia-smi -L || true
echo "== cmake: $(command -v cmake) ($(cmake --version | head -1))"

rm -rf build-cuda-mdreg
cmake -B build-cuda-mdreg -DCMAKE_BUILD_TYPE=Release \
    -DATV_BUILD_CUDA=ON -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_TESTS=OFF \
    -DBLA_VENDOR=OpenBLAS \
    -DRust_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

time cmake --build build-cuda-mdreg --target atomiverse reg_lj_argon -j"$(nproc)"

echo "== running regression leg (devices=both, 1t)"
time cmake --build build-cuda-mdreg --target run_reg_lj_argon
echo "== DONE rc=$?"
