#!/bin/bash -l
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 00:40:00
#SBATCH -J atv-cpu

# Parameterised CPU batch job: configure the full CPU product set, build, and
# run the unit-test suite. Mirrors the login-shell recipe in docs/developer/project-contract.md, but as
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
# ATV_CTEST_RE (default: every unit test; a pattern matching nothing, e.g.
# '^$', skips ctest),
# ATV_BUILD_DIR, ATV_KEEP_BUILD, ATV_RUN (shell command after ctest in this
# module env; ${BUILD} expands in it). A benchmark needs the whole node:
#   sbatch -A <account> -n 256 --exclusive -t 02:00:00 \
#     --export=ALL,ATV_RUN='benchmarks/gaff/run_cpu_matrix.sh' tools/cpu_job.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

: "${ATV_MOLPY_PATH:=$PWD/../molrs}"
export ATV_MOLPY_PATH

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
# Default: the unit tests. ATV_CTEST_RE selects by name instead (regression
# cases are named <domain>/<case>).
if [[ -n "${ATV_CTEST_RE:-}" ]]; then
    ctest --test-dir "${BUILD}" --output-on-failure -j8 -R "${ATV_CTEST_RE}" || RC=$?
else
    ctest --test-dir "${BUILD}" --output-on-failure -j8 -LE regression || RC=$?
fi

if [[ -n "${ATV_RUN:-}" ]]; then
    echo "=== ATV_RUN: ${ATV_RUN} ==="
    # Stay in this shell's module env. A login shell would drop it.
    export BUILD
    bash -c "${ATV_RUN}" || RC=$?
fi

if [[ "${ATV_KEEP_BUILD:-0}" != "1" ]]; then
    rm -rf "${BUILD}"
fi

echo "RC=${RC}"
exit "${RC}"
