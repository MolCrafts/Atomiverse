#!/bin/bash -l
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 00:40:00
#SBATCH -J atv-cpu

# Parameterised CPU batch job: configure the full CPU product set, build, and
# run the unit-test suite. Mirrors the login-shell recipe in CLAUDE.md, but as
# a batch job so a full build does not run on a shared login node.
#
# Submit from the repository root:
#
#   sbatch -A <slurm-account> tools/cpu_job.sh
#   sbatch -A <slurm-account> \
#     --export=ALL,ATV_CMAKE_FLAGS="-DATV_BUILD_GPW=OFF",ATV_CTEST_RE='gpw' \
#     tools/cpu_job.sh
#
# Environment knobs match tools/gpu_job.sh: ATV_CMAKE_FLAGS, ATV_TARGETS,
# ATV_CTEST_RE, ATV_BUILD_DIR, ATV_KEEP_BUILD.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

git submodule update --init --recursive molrs

module load buildenv-gcc/2026.03-mpich
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
# The module sets the *link* path only; the runtime path must mirror it or
# libopenblas.so.0 / GLIBCXX_3.4.32 fail at run time.
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname) arch=$(uname -m)"
echo "HEAD=$(git rev-parse HEAD)"

BUILD="${ATV_BUILD_DIR:-build-cpu-job}"

# shellcheck disable=SC2086
cmake -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_CUDA=OFF \
  -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_HF=ON \
  -DATV_BUILD_PW=ON \
  -DATV_BUILD_GPW=ON \
  -DATV_BUILD_KSPACE=ON \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}" \
  ${ATV_CMAKE_FLAGS:-}

if [[ -n "${ATV_TARGETS:-}" ]]; then
    # shellcheck disable=SC2086
    cmake --build "${BUILD}" -j"$(nproc)" --target ${ATV_TARGETS}
else
    cmake --build "${BUILD}" -j"$(nproc)"
fi

RC=0
ctest --test-dir "${BUILD}" --output-on-failure -j8 \
      ${ATV_CTEST_RE:+-R "${ATV_CTEST_RE}"} || RC=$?

if [[ "${ATV_KEEP_BUILD:-0}" != "1" ]]; then
    rm -rf "${BUILD}"
fi

echo "RC=${RC}"
exit "${RC}"
