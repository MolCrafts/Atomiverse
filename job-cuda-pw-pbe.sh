#!/bin/bash -l
# CUDA compile + unit-test + e2e dual-gate for wf/cuda-pw-pbe-device (GH200).
# Device PBE-GGA for cuda::pw::PW — pointwise, DeviceDensity stage, full SCF.
set -euo pipefail
cd /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/worktrees/wf_f384f427-0b3-14

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/$(uname -m)/cargo/bin:$PATH"
# On aarch64 GH200 the default Rust toolchain is correct — do not override to x86.
export RUSTUP_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

# molrs submodule (HF+PW force ATV_BUILD_MOLRS=ON)
if [ ! -f molrs/Cargo.toml ]; then
  git submodule update --init --recursive molrs
fi

# Wipe any leftover build before configure (disk law).
rm -rf build build-cuda build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DATV_BUILD_CUDA=ON -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
      -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=OFF \
      -DATV_BUILD_SEMIEMPIRICAL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DBLA_VENDOR=OpenBLAS \
      -DRust_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

# Scoped build: library + unit suite + e2e dual-gate regression.
cmake --build build-cuda -j"$(nproc)" --target atomiverse \
      test_cuda_pw_xc_pbe test_cuda_pw_xc_lda test_cuda_pw_density \
      test_cuda_pw_guards reg_cuda-pw-pbe-device 2>&1

# Unit suite first (pointwise + stage parity + guards).
ctest --test-dir build-cuda --output-on-failure \
      -R 'cuda_pw_(xc_pbe|xc_lda|density|guards)' \
      --timeout 120
echo "CUDA_PW_PBE_UNITS_OK"

# E2E dual-gate (CPU twin oracle).
cmake --build build-cuda -j"$(nproc)" --target run_reg_cuda-pw-pbe-device 2>&1
echo "CUDA_PW_PBE_REG_OK"
echo "CUDA_PW_PBE_JOB_OK"

# Disk law: wipe builds before job exit.
rm -rf build build-cuda build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
echo "BUILDS_WIPED_OK"
