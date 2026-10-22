#!/usr/bin/env bash
# GPU arm of the general-GAFF benchmark on one GH200: OpenMM reference runs
# per system size and precision. The Atomiverse CUDA arm joins this matrix
# once CUDA GAFF passes gate G1 (multi-type LJ, Amber 1-4 exceptions,
# periodic torsions/impropers and X-H constraints on the GPU).
#
# Required:
#   GAFF_SIZES   space-separated system directories, each with
#                system.prmtop/system.inpcrd (+ system.json for the mesh)
# Optional (defaults in brackets):
#   PYTHON [python3]  interpreter with openmm (CUDA platform)
#   PRECISIONS ["double mixed"]  REPS [3]  STEPS [10000]  WARMUP [1000]
#   THERMO [1000]  CUT [9.0]  GEWALD [0.35]  SPACING [1.0166] (Å, mesh)
#   OUT [gaff-gpu-<date>]
#
# The PME mesh per size is the FFT-friendly K nearest box/SPACING, the same
# rule the CPU arm uses (64 for the 65 Å base cell), so every size keeps the
# base cell's mesh spacing and force accuracy.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
: "${GAFF_SIZES:?set GAFF_SIZES to the system directories}"
PYTHON="${PYTHON:-python3}"
PRECISIONS="${PRECISIONS:-double mixed}"
REPS="${REPS:-3}"
STEPS="${STEPS:-10000}"
WARMUP="${WARMUP:-1000}"
THERMO="${THERMO:-1000}"
CUT="${CUT:-9.0}"
GEWALD="${GEWALD:-0.35}"
SPACING="${SPACING:-1.0166}"
OUT="${OUT:-gaff-gpu-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${OUT}"
{
    echo "host=$(hostname) date=$(date +%Y-%m-%dT%H:%M:%S)"
    nvidia-smi -L 2>/dev/null || true
    "${PYTHON}" -c "import openmm; print('openmm', openmm.__version__)"
    echo "sizes=${GAFF_SIZES} precisions=${PRECISIONS} reps=${REPS}"
    echo "steps=${STEPS} warmup=${WARMUP} cut=${CUT} gewald=${GEWALD}"
} > "${OUT}/manifest.txt"

for dir in ${GAFF_SIZES}; do
    mesh=$("${PYTHON}" - "${dir}" "${SPACING}" <<'EOF'
import json, sys
box = json.load(open(f"{sys.argv[1]}/system.json"))["box_A"][0]
target = box / float(sys.argv[2])
def smooth(k):  # only factors 2, 3, 5 (FFT-friendly)
    for p in (2, 3, 5):
        while k % p == 0:
            k //= p
    return k == 1
good = [k for k in range(16, 1025) if smooth(k)]
print(min(good, key=lambda k: abs(k - target)))
EOF
)
    name="$(basename "${dir}")"
    for prec in ${PRECISIONS}; do
        for rep in $(seq 1 "${REPS}"); do
            tag="${name}_${prec}_r${rep}"
            echo "=== ${tag} mesh=${mesh} ==="
            "${PYTHON}" "${HERE}/run_openmm.py" --dir "${dir}" --mode md \
                --platform CUDA --precision "${prec}" --cut "${CUT}" \
                --gewald "${GEWALD}" --mesh "${mesh}" "${mesh}" "${mesh}" \
                --warmup "${WARMUP}" --steps "${STEPS}" --thermo "${THERMO}" \
                --out "${OUT}/omm_${tag}.json" > "${OUT}/omm_${tag}.log" 2>&1
        done
    done
done
"${PYTHON}" - "${OUT}" <<'EOF' | tee "${OUT}/summary.txt"
import json, statistics, sys, pathlib, collections
runs = collections.defaultdict(list)
for f in sorted(pathlib.Path(sys.argv[1]).glob("omm_*.json")):
    r = json.loads(f.read_text())
    runs[(r["atoms"], r["properties"]["Precision"])].append(r["ns_per_day"])
print(f"{'atoms':>10} {'precision':>9} {'ns/day (median)':>16} {'reps':>5}")
for (atoms, prec), v in sorted(runs.items()):
    print(f"{atoms:>10} {prec:>9} {statistics.median(v):16.2f} {len(v):>5}")
EOF
