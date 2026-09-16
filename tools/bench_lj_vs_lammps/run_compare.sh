#!/bin/bash
# Head-to-head LJ NVE: Atomiverse OpenMM CUDA vs LAMMPS Kokkos CUDA.
# density=0.8, cut=2.5, skin=0.3, dt=0.005, eps=sig=1, NVE.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ATV_BENCH="${ATV_BENCH:-$ROOT/build-cuda-bench/bin/bench_nonbonded_style}"
LMP_BIN="${LMP_BIN:-$HOME/.local/$(uname -m)/bin/lmp}"
SIZES="${SIZES:-100000 250000 500000 1000000 2000000 5000000 10000000}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"

echo "=== LJ NVE A/B: Atomiverse OpenMM CUDA vs LAMMPS Kokkos CUDA ==="
echo "host=$(hostname)  date=$(date -Is)"
echo "ATV_BENCH=$ATV_BENCH"
echo "LMP_BIN=$LMP_BIN"
nvidia-smi -L 2>/dev/null || true
echo
printf '%-12s %-10s %-14s %-14s %-8s %-8s\n' N engine ms_per_step Matom_step_s warm steps
printf '%-12s %-10s %-14s %-14s %-8s %-8s\n' ------------ ---------- -------------- -------------- -------- --------

for N in $SIZES; do
  if (( N <= 100000 )); then warm=20; steps=100
  elif (( N <= 500000 )); then warm=15; steps=50
  elif (( N <= 1000000 )); then warm=10; steps=40
  elif (( N <= 2000000 )); then warm=8; steps=30
  elif (( N <= 5000000 )); then warm=5; steps=20
  else warm=5; steps=15
  fi

  # ── Atomiverse ──────────────────────────────────────────────
  atv_log=$(mktemp)
  if "$ATV_BENCH" "$N" "$steps" "$warm" >"$atv_log" 2>&1; then
    atv_out=$(grep '^path=' "$atv_log" | tail -1)
    atv_N=$(echo "$atv_out" | sed -n 's/.* N=\([0-9]*\).*/\1/p')
    atv_ms=$(echo "$atv_out" | sed -n 's/.* ms\/step=\([0-9.]*\).*/\1/p')
    atv_m=$(echo "$atv_out" | sed -n 's/.* Matom\*step\/s=\([0-9.]*\).*/\1/p')
    printf '%-12s %-10s %-14s %-14s %-8s %-8s\n' \
      "${atv_N:-$N}" "atv" "${atv_ms:--}" "${atv_m:--}" "$warm" "$steps"
  else
    printf '%-12s %-10s %-14s %-14s %-8s %-8s\n' "$N" "atv" "FAIL" "-" "$warm" "$steps"
    grep -E 'error|Error|ERROR' "$atv_log" | head -5 >&2 || true
  fi
  rm -f "$atv_log"

  # ── LAMMPS Kokkos CUDA ──────────────────────────────────────
  lmp_log=$(mktemp)
  set +e
  "$LMP_BIN" -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -in "$DIR/in.lj_nve.kk" \
    -var N "$N" -var WARM "$warm" -var STEPS "$steps" \
    -log none -screen "$lmp_log"
  lmp_rc=$?
  set -e
  if (( lmp_rc != 0 )); then
    printf '%-12s %-10s %-14s %-14s %-8s %-8s\n' "$N" "lmp_kk" "FAIL" "-" "$warm" "$steps"
    grep ERROR "$lmp_log" | head -3 >&2 || true
    rm -f "$lmp_log"
    continue
  fi
  loop_line=$(grep -E 'Loop time' "$lmp_log" | tail -1 || true)
  # Loop time of 1.234 on 1 procs for 50 steps with 100000 atoms
  loop_s=$(echo "$loop_line" | sed -n 's/.*Loop time of \([0-9.eE+-]*\).*/\1/p')
  lmp_atoms=$(echo "$loop_line" | sed -n 's/.*with \([0-9]*\) atoms.*/\1/p')
  steps_l=$(echo "$loop_line" | sed -n 's/.*for \([0-9]*\) steps.*/\1/p')
  if [[ -z "$loop_s" || -z "$lmp_atoms" || -z "$steps_l" ]]; then
    printf '%-12s %-10s %-14s %-14s %-8s %-8s\n' "$N" "lmp_kk" "PARSE" "-" "$warm" "$steps"
    rm -f "$lmp_log"
    continue
  fi
  lmp_ms=$(python3 -c "print(f'{1000.0*float(\"$loop_s\")/float(\"$steps_l\"):.4f}')")
  lmp_m=$(python3 -c "print(f'{float(\"$lmp_atoms\")/(1000.0*float(\"$loop_s\")/float(\"$steps_l\"))*1e-3:.3f}')")
  printf '%-12s %-10s %-14s %-14s %-8s %-8s\n' \
    "$lmp_atoms" "lmp_kk" "$lmp_ms" "$lmp_m" "$warm" "$steps_l"
  rm -f "$lmp_log"
done

echo
echo "=== done $(date -Is) ==="
