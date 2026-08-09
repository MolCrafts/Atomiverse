#!/bin/bash -l
# CUDA compile + unit-test + e2e dual-gate for wf/cuda-pw-pbe-device (GH200).
# Device PBE-GGA for cuda::pw::PW — pointwise, DeviceDensity stage, full SCF.
set -euo pipefail
ROOT=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/worktrees/wf_f384f427-0b3-14
cd "$ROOT"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
# Prefer aarch64 cargo/bin and aarch64 rustc; never x86 defaults on GH200.
export PATH="$HOME/.local/$(uname -m)/cargo/bin:$HOME/.cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"
# Prefer local aarch64 cxxbridge over a stale x86 one in ~/.cargo/bin
if [ -x "$HOME/.local/$(uname -m)/cargo/bin/cxxbridge" ]; then
  export PATH="$HOME/.local/$(uname -m)/cargo/bin:$PATH"
fi

if [ ! -f molrs/Cargo.toml ]; then
  git submodule update --init --recursive molrs
fi

# Unique build dir to avoid cross-job wipe races on build-cuda.
BUILD="build-cuda-pw-pbe-$$"
rm -rf "$BUILD" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true

cmake -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
      -DATV_BUILD_CUDA=ON -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
      -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=OFF \
      -DATV_BUILD_SEMIEMPIRICAL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DBLA_VENDOR=OpenBLAS \
      -DRust_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

cmake --build "$BUILD" -j"$(nproc)" --target atomiverse \
      test_cuda_pw_xc_pbe test_cuda_pw_xc_lda test_cuda_pw_density \
      test_cuda_pw_guards reg_cuda-pw-pbe-device 2>&1

# Strict unit filter — do NOT match unrelated test_scf_* via loose OR.
ctest --test-dir "$BUILD" --output-on-failure \
      -R '^(test_cuda_pw_xc_pbe|test_cuda_pw_xc_lda|test_cuda_pw_density|test_cuda_pw_guards)$' \
      --timeout 120
echo "CUDA_PW_PBE_UNITS_OK"

cmake --build "$BUILD" -j"$(nproc)" --target run_reg_cuda-pw-pbe-device 2>&1
echo "CUDA_PW_PBE_REG_OK"
echo "CUDA_PW_PBE_JOB_OK"

# Disk law: wipe this job's build only.
rm -rf "$BUILD" build build-cuda build-* CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
# Verify empty
leftover=$(find . -maxdepth 2 -type d \( -name build -o -name 'build-*' \) 2>/dev/null || true)
if [ -n "${leftover}" ]; then
  echo "WARN leftover builds: $leftover"
  rm -rf $leftover 2>/dev/null || true
fi
echo "BUILDS_WIPED_OK"
