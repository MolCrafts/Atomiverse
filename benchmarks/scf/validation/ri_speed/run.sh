#!/usr/bin/env bash
# benchmarks/scf/validation/ri_speed — glycine/def2-SVP wall-clock:
#   ours-RI-JK vs ours-exact-ERI vs PySCF-DF.
#
# NOT part of ctest. Methodology in README.md. Usage (login shell with
# the build environment loaded and the repo already built):
#
#   bash -lc 'module load buildenv-gcc/2026.03-mpich
#             export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
#             benchmarks/scf/validation/ri_speed/run.sh'
#
# Env overrides: BUILD_DIR (default <repo>/build),
#                PYTHON (default ~/work/.x86_64/bin/python, pyscf 2.14.0),
#                NTHREADS (default 8).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"
PY="${PYTHON:-$HOME/work/.x86_64/bin/python}"
NT="${NTHREADS:-8}"

export OMP_NUM_THREADS="$NT"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# ── Compile the engine benchmark main, reusing the compile/link recipe
#    of an existing regression target (no hardcoded toolchain paths).
REF="$BUILD/regressions/CMakeFiles/reg_h2o_ri_jk.dir"
[ -f "$REF/link.txt" ] || {
    echo "error: build $BUILD lacks the reg_h2o_ri_jk target" \
         "(configure with -DATV_BUILD_REGRESSIONS=ON and build)" >&2
    exit 1
}
CXX="$(awk '{print $1}' "$REF/link.txt")"
FLAGS="$(sed -n 's/^CXX_FLAGS = //p'    "$REF/flags.make")"
DEFS="$(sed -n 's/^CXX_DEFINES = //p'   "$REF/flags.make")"
INCS="$(sed -n 's/^CXX_INCLUDES = //p'  "$REF/flags.make")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# shellcheck disable=SC2086
"$CXX" $FLAGS $DEFS $INCS -c "$HERE/bench_ri_speed.cpp" -o "$TMP/bench.o"
# Reuse the regression link line with our object and output substituted.
LINK="$(cat "$REF/link.txt")"
LINK="${LINK/CMakeFiles\/reg_h2o_ri_jk.dir\/scf\/h2o_ri_jk\/reg_h2o_ri_jk.cpp.o/$TMP/bench.o}"
LINK="${LINK/-o reg_h2o_ri_jk/-o $TMP/bench_ri_speed}"
(cd "$BUILD/regressions" && eval "$LINK")

echo "== engine (Atomiverse), OMP_NUM_THREADS=$NT =="
"$TMP/bench_ri_speed"

echo
echo "== PySCF density_fit ($PY), same geometry/basis/threads =="
OMP_NUM_THREADS="$NT" "$PY" - << 'EOF'
import time
from pyscf import gto, scf, lib

atom = '''N  1.9310  0.0900 -0.0340
C  0.5610 -0.4000  0.0080
C -0.4400  0.7360 -0.0100
O -0.1560  1.9160 -0.0330
O -1.7100  0.3210  0.0180
H  2.0350  0.8420  0.6390
H  2.5810 -0.6500  0.2130
H  0.4110 -1.0180  0.9010
H  0.3480 -1.0480 -0.8500
H -2.2670  1.1120  0.0090'''

mol = gto.M(atom=atom, basis='def2-svp')  # spherical
for r in range(3):
    mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol = 1e-9
    t0 = time.perf_counter()
    e = mf.kernel()
    t1 = time.perf_counter()
    print('pyscf-df run %d: %.3f s  E = %.9f Ha' % (r, t1 - t0, e))
print('threads:', lib.num_threads())
EOF
