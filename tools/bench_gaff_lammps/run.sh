#!/usr/bin/env bash
# Head-to-head GAFF timing: Atomiverse (MPI) vs LAMMPS (MPI).
# Default system: packmol_n100_r2 melt (17700 atoms) — paper flexible arm.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
MELT_DIR="${MELT_DIR:-/home/jicli594/work/molpack-paper/figures/relaxation/ref_lammps/runs/packmol_n100_r2/melt}"
NP="${NP:-4}"
STEPS="${STEPS:-500}"
WARMUP="${WARMUP:-50}"
LMP="${LMP:-/home/jicli594/.local/x86_64/bin/lmp}"
OUT="${OUT:-/tmp/atv_gaff_vs_lammps}"
mkdir -p "$OUT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export LD_LIBRARY_PATH="/software/sse2/el9_epyc9005/manual/FFTW/3.3.11/g14/mp43/hpc1/lib:/software/sse2/el9_epyc9005/manual/OpenBLAS/0.3.32/g14/hpc1/openmp/lp64/lib:/software/sse2/el9_epyc9005/manual/MPICH/4.3.2/g14/hpc2/lib:/software/sse2/el9_epyc9005/manual/GCC/14.3.0/hpc1/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ATV_BIN="$BUILD/tools/bench_gaff_lammps/run_atv_gaff"
if [[ ! -x "$ATV_BIN" ]]; then
  ATV_BIN="$BUILD/run_atv_gaff"
fi
if [[ ! -x "$ATV_BIN" ]]; then
  echo "missing $ATV_BIN — build with -DATV_BUILD_KSPACE=ON first" >&2
  exit 1
fi

echo "=== Atomiverse  np=$NP steps=$STEPS ==="
mpirun -np "$NP" "$ATV_BIN" \
  --data "$MELT_DIR/melt.data" \
  --steps "$STEPS" --warmup "$WARMUP" \
  2>&1 | tee "$OUT/log.atomiverse.np${NP}.txt"

echo "=== LAMMPS  np=$NP steps=$STEPS ==="
WORKDIR="$OUT/lmp_np${NP}"
mkdir -p "$WORKDIR"
cp -f "$MELT_DIR/melt.data" "$MELT_DIR/melt.ff" "$WORKDIR/"
cp -f "$(dirname "$0")/in.lammps" "$WORKDIR/in.lammps"
(
  cd "$WORKDIR"
  mpirun -np "$NP" "$LMP" -in in.lammps \
    -var data melt.data -var nsteps "$STEPS" \
    -log log.lammps 2>&1 | tee "$OUT/log.lammps.np${NP}.txt"
)

echo "=== Extracted performance lines ==="
grep -E 'PERFORMANCE|Loop time|Performance:|Section|Pair|Kspace|Bond|Neigh' \
  "$OUT/log.atomiverse.np${NP}.txt" "$OUT/log.lammps.np${NP}.txt" | head -80
echo "logs in $OUT"
