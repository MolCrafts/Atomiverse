#!/bin/bash -l
# CUDA compile + unit-test job for wf/cuda-pw-pbe-device (GH200, aarch64).
# Scoped: atomiverse + the four cuda pw test binaries touched by the slug.
set -e
cd /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.claude/worktrees/wf_f384f427-0b3-14

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
export PATH="$HOME/.local/$(uname -m)/cargo/bin:$PATH"
export RUSTUP_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

rm -rf build-cuda
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DATV_BUILD_CUDA=ON -DATV_BUILD_TESTS=ON \
      -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=OFF \
      -DATV_BUILD_SEMIEMPIRICAL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DBLA_VENDOR=OpenBLAS \
      -DRust_TOOLCHAIN="stable-$(uname -m)-unknown-linux-gnu"

cmake --build build-cuda -j"$(nproc)" --target atomiverse \
      test_cuda_pw_xc_pbe test_cuda_pw_xc_lda test_cuda_pw_density \
      test_cuda_pw_guards

ctest --test-dir build-cuda --output-on-failure \
      -R "cuda_pw_(xc_pbe|xc_lda|density|guards)"
echo "CUDA_PW_PBE_JOB_OK"
