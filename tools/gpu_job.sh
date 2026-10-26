#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH -c 8
#SBATCH -J atv-gpu

# Parameterised GH200 batch job: configure a CUDA build, build the requested
# targets, run the requested ctest selection, then wipe the tree.
#
# CUDA builds and runs only on the aarch64 GPU node -- login nodes have no
# toolkit -- and `module load` must not be piped (a subshell loses the env),
# which is why this is a batch script rather than an inline command.
#
# Submit from the repository root:
#
#   sbatch -A <slurm-account> tools/gpu_job.sh
#   sbatch -A <slurm-account> \
#     --export=ALL,ATV_CMAKE_FLAGS="-DATV_BUILD_PW=ON -DATV_BUILD_GPW=OFF",\
# ATV_TARGETS="atv_tests_cuda",ATV_CTEST_RE='PW' \
#     tools/gpu_job.sh
#
# Environment knobs (all optional):
#   ATV_CMAKE_FLAGS  extra -D flags appended to the configure line
#   ATV_TARGETS      space-separated build targets   (default: the ctest deps)
#   ATV_CTEST_RE     ctest -R regex                  (default: CUDA unit tests,
#                    label cuda)
#   ATV_BUILD_DIR    build tree name                 (default: build-gpu-job)
#   ATV_KEEP_BUILD   set to 1 to keep the tree for follow-up runs
#   ATV_RUN          shell command after ctest (GPU module env; ${BUILD}
#                    expands in that shell).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${ATV_MOLPY_PATH:=$PWD/../molrs}"
export ATV_MOLPY_PATH

module load GPU/buildenv-gcccuda/2026.03-cu13.0
# The login-node x86 shims shadow the aarch64 toolchain on the GPU node.
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
export PATH
hash -r
# rustup's user default is often the login-node x86 toolchain. On the
# GH200 the installed toolchain is aarch64; do not keep the x86 one.
if [[ "$(uname -m)" == aarch64 ]]; then
    export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
fi
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"

echo "host=$(hostname) arch=$(uname -m)"
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
echo "HEAD=$(git rev-parse HEAD)"
nvidia-smi -L || true

BUILD="${ATV_BUILD_DIR:-build-gpu-job}"

# shellcheck disable=SC2086
cmake -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_CUDA=ON \
  -DATV_BUILD_TESTS=ON \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}" \
  ${ATV_CMAKE_FLAGS:-}

echo "=== build: library ==="
cmake --build "${BUILD}" -j"$(nproc)" --target atomiverse

if [[ -n "${ATV_TARGETS:-}" ]]; then
    echo "=== build: ${ATV_TARGETS} ==="
    # shellcheck disable=SC2086
    cmake --build "${BUILD}" -j"$(nproc)" --target ${ATV_TARGETS}
else
    echo "=== build: all ==="
    cmake --build "${BUILD}" -j"$(nproc)"
fi

RC=0
if [[ -n "${ATV_CTEST_RE:-}" ]]; then
    echo "=== ctest -R '${ATV_CTEST_RE}' ==="
    ctest --test-dir "${BUILD}" --output-on-failure -R "${ATV_CTEST_RE}" || RC=$?
else
    echo "=== ctest -L cuda -LE regression ==="
    ctest --test-dir "${BUILD}" --output-on-failure -L cuda -LE regression || RC=$?
fi

if [[ -n "${ATV_RUN:-}" ]]; then
    echo "=== ATV_RUN: ${ATV_RUN} ==="
    # Stay in this shell's module env. A login shell would drop it.
    # BUILD is a script local; export it so ${BUILD} inside ATV_RUN expands.
    export BUILD
    bash -c "${ATV_RUN}" || RC=$?
fi

if [[ "${ATV_KEEP_BUILD:-0}" != "1" ]]; then
    echo "=== wipe build ==="
    rm -rf "${BUILD}"
fi

echo "RC=${RC}"
exit "${RC}"
