#!/usr/bin/env bash
# benchmarks/semiempirical/dftb_peg_liquid — fair wall-clock of one SCC-DFTB2
# single point (energy + forces), ATV vs DFTB+, on the same 1088-atom PEG
# liquid cell at matched thread counts and matched SCC tolerance.
#
# NOT part of ctest. Methodology in README.md. Usage (login shell with the
# build environment loaded and the repo already built):
#
#   bash -lc 'module load buildenv-gcc/2026.03-mpich
#             export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
#             benchmarks/semiempirical/dftb_peg_liquid/run.sh'
#
# Env overrides: BUILD_DIR (default <repo>/build), GEO, THREADS, SCC_TOL.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
THREADS="${THREADS:-4 16 32 64}"
SCC_TOL="${SCC_TOL:-1e-5}"
GEO="${GEO:-$HOME/work/lab/projects/peo-polar-length/experiments/n-series/dftb_bench/n2/omp16_sp/geo.gen}"

# DO NOT set OMP_PROC_BIND / OMP_PLACES here.
#
# Pinning helps ATV's own OpenMP loops but an OpenMP-flavour OpenBLAS drops to
# a SINGLE thread the moment any OpenMP affinity is requested — measured on
# OpenBLAS 0.3.32 (ZEN, USE_OPENMP): dgemm 807 -> 106 GFLOP/s and a 2432^2
# dsygvd 0.52 s -> 9.7 s. A DFTB single point is one dense generalized
# eigenproblem per SCC iteration, so binding would hand in a number ~10x worse
# than the code can do. See src/cpu/device/device_runtime.cpp.
unset OMP_PROC_BIND OMP_PLACES || true

# Compile the bench main by reusing an existing regression target's
# compile/link recipe — no hardcoded toolchain paths (same trick as
# benchmarks/scf/validation/ri_speed/run.sh).
REF="$BUILD/regressions/CMakeFiles/reg_dftb2-ewald-gamma.dir"
[ -f "$REF/link.txt" ] || {
    echo "error: build $BUILD lacks the reg_dftb2-ewald-gamma target" \
         "(configure with -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_SEMIEMPIRICAL=ON" \
         "and build)" >&2
    exit 1
}
CXX="$(awk '{print $1}' "$REF/link.txt")"
FLAGS="$(sed -n 's/^CXX_FLAGS = //p'   "$REF/flags.make")"
DEFS="$(sed -n 's/^CXX_DEFINES = //p'  "$REF/flags.make")"
INCS="$(sed -n 's/^CXX_INCLUDES = //p' "$REF/flags.make")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# shellcheck disable=SC2086
"$CXX" $FLAGS $DEFS $INCS -c "$HERE/bench_dftb_sp.cpp" -o "$TMP/bench.o"
LINK="$(cat "$REF/link.txt")"
LINK="${LINK/CMakeFiles\/reg_dftb2-ewald-gamma.dir\/semiempirical\/dftb2-ewald-gamma\/reg_dftb2-ewald-gamma.cpp.o/$TMP/bench.o}"
LINK="${LINK/-o reg_dftb2-ewald-gamma/-o $TMP/bench_dftb_sp}"
(cd "$BUILD/regressions" && eval "$LINK")
BIN="$TMP/bench_dftb_sp"

if [[ ! -f "$GEO" ]]; then
  echo "error: geometry $GEO not found; set GEO=<path to a .gen>" >&2
  exit 1
fi

echo "geometry : $GEO"
echo "scc_tol  : $SCC_TOL  (DFTB+ default SCCTolerance is 1e-5)"
echo
printf '%-8s %14s %10s %16s\n' threads wall_s_median scc_iter energy_eV
for nt in $THREADS; do
  out="$(ATV_TIMER=0 OMP_NUM_THREADS="$nt" "$BIN" "$GEO" "$SCC_TOL" 3 2>/dev/null)"
  wall=$(sed -E 's/.*"wall_s_median": ([0-9.]+).*/\1/' <<<"$out")
  it=$(sed -E 's/.*"scc_iterations": ([0-9]+).*/\1/' <<<"$out")
  e=$(sed -E 's/.*"energy_eV": (-?[0-9.]+).*/\1/' <<<"$out")
  printf '%-8s %14s %10s %16s\n' "$nt" "$wall" "$it" "$e"
done
