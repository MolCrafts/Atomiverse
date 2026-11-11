#!/bin/bash -l
# Build the LAMMPS reference binary for benchmarks/compare/gaff-liquid.
#
# Packages the deck needs: MOLECULE (bond/angle harmonic, improper cvff),
# EXTRA-MOLECULE (dihedral fourier), KSPACE (pppm, ewald), RIGID (fix
# rattle), OPENMP (the threaded "-sf omp" arm).
#
#   LAMMPS_SRC=/path/to/lammps [LAMMPS_MODULE=<lmod module>] bash build_lammps.sh
#
# Required: LAMMPS_SRC, a full checkout (the cmake/ tree is needed); record
#           its SHA in the benchmark manifest. Output: ${LAMMPS_SRC}/build-gaff/lmp
# Optional: LAMMPS_MODULE, an Lmod toolchain module (MPI + FFTW3) to load
#           first; use the one the Atomiverse build uses on that cluster
#           (tools/cpu_job.sh). When unset, the toolchain must be on PATH.
set -euo pipefail
: "${LAMMPS_SRC:?set LAMMPS_SRC to a full LAMMPS checkout}"

if [[ -n "${LAMMPS_MODULE:-}" ]]; then
    # `bash script.sh` is not a login shell, so Lmod's `module` may be
    # undefined: load the site profile explicitly.
    if ! type module >/dev/null 2>&1 && [[ -f /etc/profile ]]; then
        set +u
        source /etc/profile
        set -u
    fi
    module load "${LAMMPS_MODULE}"
    # The module sets the link path only; mirror it into the runtime path.
    export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
fi

BUILD="${LAMMPS_SRC}/build-gaff"
cmake -S "${LAMMPS_SRC}/cmake" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native" \
    -DBUILD_MPI=ON -DBUILD_OMP=ON \
    -DPKG_MOLECULE=ON -DPKG_EXTRA-MOLECULE=ON -DPKG_KSPACE=ON \
    -DPKG_RIGID=ON -DPKG_OPENMP=ON \
    -DFFT=FFTW3
cmake --build "${BUILD}" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo "LAMMPS SHA $(git -C "${LAMMPS_SRC}" rev-parse HEAD)"
"${BUILD}/lmp" -h | sed -n '/Installed packages/,/^$/p'
for pkg in MOLECULE EXTRA-MOLECULE KSPACE RIGID OPENMP; do
    "${BUILD}/lmp" -h | grep -q -- "${pkg}" \
        || { echo "missing package ${pkg}" >&2; exit 1; }
done
