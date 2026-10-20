#!/bin/bash -l
# Build the LAMMPS reference binary for benchmarks/gaff with the same module
# environment Atomiverse's CPU build uses on Arrhenius (tools/cpu_job.sh).
#
# Packages the deck needs: MOLECULE (bond/angle harmonic, improper cvff),
# EXTRA-MOLECULE (dihedral fourier), KSPACE (pppm, ewald), RIGID (fix
# rattle), OPENMP (the threaded "-sf omp" arm).
#
#   LAMMPS_SRC=/path/to/lammps  bash benchmarks/gaff/build_lammps.sh
#
# LAMMPS_SRC must be a full checkout (the cmake/ tree is needed); record its
# SHA in the benchmark manifest. Output: ${LAMMPS_SRC}/build-gaff/lmp
set -euo pipefail
# `bash script.sh` is not a login shell, so the shebang's -l does not apply
# and Lmod's `module` may be undefined: load the site profile explicitly.
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile ]]; then
    set +u
    source /etc/profile
    set -u
fi
: "${LAMMPS_SRC:?set LAMMPS_SRC to a full LAMMPS checkout}"

if type module >/dev/null 2>&1; then
    module load buildenv-gcc/2026.03-mpich
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
