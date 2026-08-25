#!/usr/bin/env bash
# widebox_parity.sh — cross-rank single-point parity gate for a wide-box,
# multi-hop processor grid (spec comm-topology-03-widebox-validation).
#
# Runs `run_atv_gaff --single-point` TWICE with byte-identical physics flags:
#
#   (1) np=1                          — the reference: no decomposition at all
#   (2) np=$NP, ATV_FORCE_DIMS=$DIMS  — the processor grid under test
#
# and compares every published quantity: bond / angle / dihedral /
# pair+kspace / total E / fmax / fnorm. Potential energy and forces are
# functions of the system Hamiltonian alone — they do not depend on the rank
# count or on the processor grid — so the two runs may differ only by
# double-precision summation order. That is what the windows below encode;
# they are NOT a freeze of ATV's current output.
#
# The reference is ATV's own np=1 result: this is cross-rank self-consistency,
# not an external oracle. It must never be recorded as an `oracle:` pin.
#
# Windows (spec Design item 2; denominator always max(1, |reference|)):
#
#   bond / angle / dihedral    rel < 1e-9                  (--tol-bonded)
#   pair+kspace                rel < 1e-8                  (--tol-pair)
#   total E                    rel < 1e-8                  (--tol-total)
#                              AND |dE|/N < 1e-6 eV/atom   (spec-fixed)
#   fmax / fnorm               rel < 1e-6                  (--tol-force)
#
# Comparisons are STRICT (`dev < tol`). A window of zero is therefore an
# impossible window, which is what makes `--tol-total 0` a usable proof that
# this gate actually bites instead of always printing PASS.
#
# The script also asserts that the np=$NP run really is a MULTI-HOP geometry:
# at least one `hops` component of its `[atv] GRID` line must be >= 2, and the
# grid it reports must be the grid that was requested. A single-hop run that
# reports PASS is the most dangerous false green on this chain. Both geometry
# assertions are skipped when NP == 1 (a one-rank grid has no hops).
#
# Environment (same conventions as run.sh in this directory):
#
#   BUILD     build dir holding tools/bench_gaff_lammps/run_atv_gaff
#             (default: <repo>/build-gaff)
#   DATA      LAMMPS data file          — REQUIRED, no fallback
#   FF        LAMMPS coefficient file   — REQUIRED, no fallback
#   NP        rank count under test     (default 256)
#   DIMS      ATV_FORCE_DIMS grid       (default 8,8,4)
#   LAUNCHER  MPI launcher              (default "srun --mpi=pmi2
#             --cpu-bind=cores" when srun exists, else "mpirun")
#   OUT       log directory             (default /tmp/atv_widebox_parity)
#
# Flags: --tol-total / --tol-pair / --tol-bonded / --tol-force override the
# defaults above. EVERY other argument is passed through to BOTH runs
# unchanged — that is how --alpha / --grid-spacing / --cutoff / --tail /
# --shift stay identical between the two runs.
#
# Usage:
#   DATA=/path/eq.data FF=/path/bench.ff NP=256 DIMS=8,8,4 \
#     tools/bench_gaff_lammps/widebox_parity.sh --alpha 0.25232811 \
#     --grid-spacing 1.98
#
# Exit status: 0 only when every component PASSes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${BUILD:-$ROOT/build-gaff}"
DATA="${DATA:-}"
FF="${FF:-}"
NP="${NP:-256}"
DIMS="${DIMS:-8,8,4}"
OUT="${OUT:-/tmp/atv_widebox_parity}"

if [[ -z "${LAUNCHER:-}" ]]; then
  if command -v srun >/dev/null 2>&1; then
    LAUNCHER="srun --mpi=pmi2 --cpu-bind=cores"
  else
    LAUNCHER="mpirun"
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# Spec-table defaults.
TOL_TOTAL="1e-8"
TOL_PAIR="1e-8"
TOL_BONDED="1e-9"
TOL_FORCE="1e-6"
TOL_PER_ATOM="1e-6" # eV/atom companion of the total-energy window

PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tol-total)  TOL_TOTAL="$2";  shift 2 ;;
    --tol-pair)   TOL_PAIR="$2";   shift 2 ;;
    --tol-bonded) TOL_BONDED="$2"; shift 2 ;;
    --tol-force)  TOL_FORCE="$2";  shift 2 ;;
    *)            PASSTHRU+=("$1"); shift ;;
  esac
done

# ── Inputs are mandatory: a parity run on the wrong system is worthless ──
if [[ -z "$DATA" || -z "$FF" ]]; then
  echo "FAIL: DATA and FF are required (no fallback system)." >&2
  echo "  DATA=/path/eq.data FF=/path/bench.ff $0 [--alpha ...]" >&2
  exit 2
fi
for f in "$DATA" "$FF"; do
  if [[ ! -r "$f" ]]; then
    echo "FAIL: missing input '$f'" >&2
    exit 2
  fi
done

ATV_BIN="$BUILD/tools/bench_gaff_lammps/run_atv_gaff"
if [[ ! -x "$ATV_BIN" ]]; then
  ATV_BIN="$BUILD/run_atv_gaff"
fi
if [[ ! -x "$ATV_BIN" ]]; then
  echo "FAIL: missing run_atv_gaff under '$BUILD' — build with" \
       "-DATV_USE_MPI=ON -DATV_BUILD_KSPACE=ON first" >&2
  exit 2
fi

mkdir -p "$OUT"
REF_LOG="$OUT/log.ref.np1.txt"
TEST_LOG="$OUT/log.test.np${NP}.txt"

read -r -a LAUNCH <<<"$LAUNCHER"

echo "=== widebox parity ==============================================="
echo "  binary   : $ATV_BIN"
echo "  data     : $DATA"
echo "  ff       : $FF"
echo "  launcher : $LAUNCHER"
echo "  reference: np=1"
echo "  test     : np=$NP  ATV_FORCE_DIMS=$DIMS"
echo "  args     : ${PASSTHRU[*]:-<none>}"
echo "  logs     : $OUT"
echo

# @brief Run one single-point evaluation and tee its log.
# $1 rank count, $2 log path, $3 ATV_FORCE_DIMS value ("" = leave unset).
run_case() {
  local np="$1" log="$2" dims="${3:-}"
  echo "--- np=$np${dims:+  ATV_FORCE_DIMS=$dims} ---"
  if ! env ${dims:+ATV_FORCE_DIMS="$dims"} \
      "${LAUNCH[@]}" -n "$np" "$ATV_BIN" \
      --data "$DATA" --ff "$FF" --single-point \
      "${PASSTHRU[@]}" 2>&1 | tee "$log"; then
    echo "FAIL: np=$np run did not complete (see $log)" >&2
    exit 1
  fi
  echo
}

run_case 1 "$REF_LOG" ""
run_case "$NP" "$TEST_LOG" "$DIMS"

echo "=== comparison (reference = np=1) ================================"

set +e
gawk -v np_test="$NP" -v want_dims="$DIMS" \
     -v tol_total="$TOL_TOTAL" -v tol_pair="$TOL_PAIR" \
     -v tol_bonded="$TOL_BONDED" -v tol_force="$TOL_FORCE" \
     -v tol_per_atom="$TOL_PER_ATOM" '
function absv(x) { return x < 0 ? -x : x }
function clean(s) { gsub(/[<>]/, "", s); return s }
function verdict(name, ref, got, dev, tol,    v) {
    v = (dev < tol) ? "PASS" : "FAIL"
    if (v == "FAIL") nfail++
    printf "  %-12s want=% .10e  got=% .10e  dev=%.3e  tol=%.3e  %s\n",
           name, ref, got, dev, tol, v
}
function have(key,   ok) {
    ok = ((1, key) in val) && ((2, key) in val)
    if (!ok) {
        printf "  %-12s MISSING from the np=1 and/or np=%d log%s  FAIL\n",
               key, np_test, ""
        nfail++
    }
    return ok
}
function relcheck(key, tol,   r, g, d) {
    if (!have(key)) return
    r = val[1, key]; g = val[2, key]
    d = absv(g - r) / (absv(r) > 1 ? absv(r) : 1)
    verdict(key, r, g, d, tol)
}
function kvline(which,   i, n, kv) {
    for (i = 1; i <= NF; ++i) {
        n = split($i, kv, "=")
        if (n == 2) { val[which, kv[1]] = clean(kv[2]) + 0 }
    }
}
FNR == 1 { which++ }
/^\[atv\] ranks=/ {
    for (i = 1; i <= NF; ++i) {
        n = split($i, kv, "=")
        if (n == 2 && kv[1] == "atoms") natoms[which] = kv[2] + 0
    }
}
/^\[atv\] SINGLE-POINT \(eV\)/     { kvline(which) }
/^\[atv\] SINGLE-POINT FORCES/     { kvline(which) }
/^\[atv\] GRID /                   {
    grid[which] = $0
    for (i = 1; i <= NF; ++i) {
        n = split($i, kv, "=")
        if (n != 2) continue
        if (kv[1] == "dims") dims[which] = clean(kv[2])
        if (kv[1] == "hops") hops[which] = clean(kv[2])
        if (kv[1] == "halo") halo[which] = clean(kv[2]) + 0
    }
}
END {
    # ── energies ────────────────────────────────────────────────────────
    relcheck("bond",        tol_bonded)
    relcheck("angle",       tol_bonded)
    relcheck("dihedral",    tol_bonded)
    relcheck("pair+kspace", tol_pair)
    relcheck("E",           tol_total)
    if ((1, "E") in val && (2, "E") in val) {
        if (natoms[1] > 0) {
            d = absv(val[2, "E"] - val[1, "E"]) / natoms[1]
            verdict("E/atom", val[1, "E"] / natoms[1],
                    val[2, "E"] / natoms[1], d, tol_per_atom)
        } else {
            printf "  %-12s atom count not found in the np=1 log  FAIL\n",
                   "E/atom"
            nfail++
        }
    }
    if (natoms[1] > 0 && natoms[2] > 0 && natoms[1] != natoms[2]) {
        printf "  %-12s np=1 has %d atoms, np=%d has %d  FAIL\n",
               "atoms", natoms[1], np_test, natoms[2]
        nfail++
    }

    # ── forces ──────────────────────────────────────────────────────────
    relcheck("fmax",  tol_force)
    relcheck("fnorm", tol_force)

    # ── geometry: this must be the multi-hop grid we asked for ──────────
    printf "\n"
    printf "  grid np=%-5d %s\n", 1,
           (1 in grid) ? grid[1] : "<no [atv] GRID line>"
    printf "  grid np=%-5d %s\n", np_test,
           (2 in grid) ? grid[2] : "<no [atv] GRID line>"
    if (np_test > 1) {
        if (!(2 in grid)) {
            printf "  %-12s the np=%d log has no [atv] GRID line  FAIL\n",
                   "hops", np_test
            nfail++
        } else {
            if (dims[2] != want_dims) {
                printf "  %-12s want=%s got=%s (ATV_FORCE_DIMS ignored?)  FAIL\n",
                       "dims", want_dims, dims[2]
                nfail++
            } else {
                printf "  %-12s want=%s got=%s  PASS\n",
                       "dims", want_dims, dims[2]
            }
            nh = split(hops[2], hv, ",")
            hmax = 0
            for (i = 1; i <= nh; ++i) if (hv[i] + 0 > hmax) hmax = hv[i] + 0
            if (hmax >= 2) {
                printf "  %-12s max=%d (>= 2, halo %.3f A spans %d subdomains)  PASS\n",
                       "hops", hmax, halo[2], hmax
            } else {
                printf "  %-12s max=%d — not a multi-hop geometry  FAIL\n",
                       "hops", hmax
                nfail++
            }
        }
    } else {
        printf "  %-12s skipped (NP=1: a one-rank grid has no hops)\n", "hops"
    }

    printf "\n"
    if (nfail > 0) {
        printf "RESULT: FAIL (%d component(s) outside the window)\n", nfail
        exit 1
    }
    printf "RESULT: PASS (every component inside the window)\n"
    exit 0
}
' "$REF_LOG" "$TEST_LOG"
rc=$?
set -e

echo "logs in $OUT"
exit $rc
