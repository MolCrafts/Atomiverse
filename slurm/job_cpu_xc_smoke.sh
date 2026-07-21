#!/bin/bash -l
#SBATCH -A naiss2026-3-381-cpu
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 01:00:00
#SBATCH -J atv_xc_smoke
#SBATCH -o slurm/%x-%j.out
#
# Smoke / baseline job for the gpw-xc-parity branch (CPU backend).
# Configures + builds libatomiverse with the libxc test-oracle, then runs the
# native-XC-vs-libxc parity gate. This validates the whole CPU toolchain
# (FetchContent, molrs/cargo, FFTW/OpenBLAS, libxc oracle, ctest) in one shot
# and reports the current native-vs-libxc baseline (expected: LSDA + PBE green).
#
# Submit from the worktree root:
#   sbatch slurm/job_cpu_xc_smoke.sh
set -euo pipefail

# CPU build environment: gcc + MPICH + OpenBLAS + FFTW + ScaLAPACK.
module load buildenv-gcc/2026.03-mpich

cd "$SLURM_SUBMIT_DIR"

# Same gotcha as the GPU job: the molcrafts venvs ship a pip 'cmake' stub whose
# shebang points at an x86 python; drop both per-arch venv bins so cmake/ctest
# resolve to real binaries. Keep ~/.local cargo bin (needed to build molrs).
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE '/molcrafts/\.(x86|x86_64|aarch64)/bin/?$' | paste -sd:)"
export PATH="$HOME/.local/x86_64/cargo/bin:$PATH"   # cargo for the molrs crate

CMAKE="$(command -v cmake)"
CTEST="$(dirname "$CMAKE")/ctest"; [ -x "$CTEST" ] || CTEST="$(command -v ctest)"
echo "cmake : $CMAKE ($("$CMAKE" --version | head -1))"
echo "cargo : $(command -v cargo) ($(cargo --version 2>/dev/null || echo MISSING))"

# Dependency roots. libxc (test-only oracle) comes from the CP2K Spack view,
# which ships lib64/cmake/Libxc/LibxcConfig.cmake.
SPACK_VIEW=/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/spack/spack/opt/spack/view
FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"
MOLRS=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molrs

BUILD=build-cpu-xc

"$CMAKE" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DATV_BUILD_CUDA=OFF \
    -DATV_BUILD_TESTS=ON \
    -DATV_BUILD_HF=ON \
    -DATV_BUILD_XC_ORACLE=ON \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT:-};${SPACK_VIEW}" \
    -DATV_MOLRS_PATH="$MOLRS"

# Build the XC parity gate (pulls in libatomiverse + the libxc oracle).
"$CMAKE" --build "$BUILD" -j"$(nproc)" \
    --target test_cpu_scf_xc_vs_libxc test_xc_libxc_oracle

# Runtime: module sets LIBRARY_PATH (link) but not LD_LIBRARY_PATH (run);
# add it plus the Spack view so libxc.so resolves.
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${SPACK_VIEW}/lib64:${SPACK_VIEW}/lib:${LD_LIBRARY_PATH:-}"

"$CTEST" --test-dir "$BUILD" \
    -R "test_(cpu_scf_xc_vs_libxc|xc_libxc_oracle)" --output-on-failure
echo "=== JOB DONE rc=$? ==="
