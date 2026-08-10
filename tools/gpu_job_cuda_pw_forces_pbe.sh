#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 01:30:00
#SBATCH -c 16
#SBATCH -J atv-pw-forces-pbe
#SBATCH -o .claude/cuda-pw-forces-pbe-%j.out
#SBATCH -e .claude/cuda-pw-forces-pbe-%j.err

# GH200 gate for slug cuda-pw-forces-pbe (PBE dual-gate forces).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
# Drop x86-only cargo/rustc shims that break aarch64 GH200.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/$(uname -m)/cargo/bin:$HOME/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"
if [ -x "$HOME/.local/$(uname -m)/cargo/bin/cxxbridge" ]; then
  export PATH="$HOME/.local/$(uname -m)/cargo/bin:$PATH"
fi

if [ ! -f molrs/Cargo.toml ]; then
  git submodule update --init --recursive molrs
fi

echo "host=$(hostname) arch=$(uname -m)"
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
echo "CC=${CC:-} CXX=${CXX:-}"
echo "rustc=$(command -v rustc) $(rustc --version 2>/dev/null || true)"
echo "HEAD=$(git rev-parse HEAD)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
nvidia-smi -L || true

BUILD=_pw_forces_pbe_work
rm -rf "$BUILD" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DATV_BUILD_CUDA=ON -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
      -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=OFF \
      -DATV_BUILD_SEMIEMPIRICAL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DBLA_VENDOR=OpenBLAS \
      -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}" \
      -DRust_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

echo "=== STAGE 1: library + targets (BUILD=$BUILD) ==="
cmake --build "$BUILD" -j"$(nproc)" --target atomiverse \
      test_cuda_pw_force_local reg_cuda-pw-forces-pbe 2>&1

echo "=== STAGE 2: unit ==="
ctest --test-dir "$BUILD" --output-on-failure \
      -R '^test_cuda_pw_force_local$' --timeout 300
echo "CUDA_PW_FORCES_PBE_UNITS_OK"

echo "=== STAGE 3: regression dual-gate (PBE) ==="
cmake --build "$BUILD" -j"$(nproc)" --target run_reg_cuda-pw-forces-pbe 2>&1
echo "CUDA_PW_FORCES_PBE_REG_OK"
echo "=== DONE green ==="

# Wipe only on success (set -e exits before this on failure, leaving build for
# debug). Always wipe on success to honor the disk law.
rm -rf "$BUILD" _pw_forces_pbe_work _pw_forces_pbe_work_* build build-* \
       CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
leftover=$(find . -maxdepth 2 -type d \( -name build -o -name 'build-*' -o -name '_pw_forces_pbe_work*' \) 2>/dev/null || true)
if [ -n "${leftover}" ]; then
  echo "WARN leftover builds: $leftover"
  rm -rf $leftover 2>/dev/null || true
fi
echo "builds_wiped=true"
