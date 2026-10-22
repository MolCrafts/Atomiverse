#!/usr/bin/env bash
# CPU arm of the general-GAFF benchmark: Atomiverse vs LAMMPS, same node,
# same input, paired runs (ATV then LAMMPS, per configuration and repeat).
#
# Required:
#   GAFF_DATA_DIR  directory with system.data + system.ff (prmtop_to_lammps.py)
#   BUILD          Atomiverse build tree with run_atv_gaff (tools/cpu_job.sh
#                  exports it; -DATV_BUILD_BENCH_GAFF=ON -DATV_USE_MPI=ON)
# Optional (defaults in brackets):
#   LMP       LAMMPS binary [lmp]  (MOLECULE KSPACE RIGID EXTRA-MOLECULE OPENMP)
#   CONFIGS   "ranks x threads" list ["1x1 16x1 64x1 128x1 256x1 64x4 128x2"]
#   REPS [3]  STEPS [5000]  WARMUP [200]  THERMO [1000]
#   CUT [9.0] GEWALD [0.35] MESH [64] DT [2.0] SHAKE [yes] TEMP [300]
#   LAUNCH    launcher prefix, {np} is substituted
#             [srun -n {np} --cpus-per-task={nt} --cpu-bind=cores]
#   OUT       output directory [gaff-cpu-<date>]
#
# Threads arm: ATV gets OMP_NUM_THREADS=nt, LAMMPS gets `-sf omp -pk omp nt`
# (its fastest threaded path). Ranks-only configurations run plain styles on
# both sides. Summarise with summarize_cpu.py OUT.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
: "${GAFF_DATA_DIR:?set GAFF_DATA_DIR to the system.data/system.ff directory}"
: "${BUILD:?set BUILD to the Atomiverse build tree}"
LMP="${LMP:-lmp}"
CONFIGS="${CONFIGS:-1x1 16x1 64x1 128x1 256x1 64x4 128x2}"
REPS="${REPS:-3}"
STEPS="${STEPS:-5000}"
WARMUP="${WARMUP:-200}"
THERMO="${THERMO:-1000}"
CUT="${CUT:-9.0}"
GEWALD="${GEWALD:-0.35}"
MESH="${MESH:-64}"
DT="${DT:-2.0}"
SHAKE="${SHAKE:-yes}"
TEMP="${TEMP:-300}"
# The default lives in its own variable: a `}` inside `${LAUNCH:-...}` would
# end the expansion early.
DEFAULT_LAUNCH='srun -n {np} --cpus-per-task={nt} --cpu-bind=cores'
LAUNCH="${LAUNCH:-${DEFAULT_LAUNCH}}"
OUT="${OUT:-gaff-cpu-$(date +%Y%m%d-%H%M%S)}"

ATV_BIN="${BUILD}/tools/bench_gaff_lammps/run_atv_gaff"
[[ -x "${ATV_BIN}" ]] || { echo "missing ${ATV_BIN}" >&2; exit 1; }
command -v "${LMP}" >/dev/null || [[ -x "${LMP}" ]] \
    || { echo "missing LAMMPS binary ${LMP}" >&2; exit 1; }

mkdir -p "${OUT}"
cp "${GAFF_DATA_DIR}/system.data" "${GAFF_DATA_DIR}/system.ff" "${OUT}/"
{
    echo "host=$(hostname) date=$(date +%Y-%m-%dT%H:%M:%S)"
    echo "atv_head=$(git -C "${HERE}" rev-parse HEAD 2>/dev/null || echo ?)"
    echo "lammps=$("${LMP}" -h | sed -n '2p')"
    echo "configs=${CONFIGS} reps=${REPS} steps=${STEPS} warmup=${WARMUP}"
    echo "cut=${CUT} gewald=${GEWALD} mesh=${MESH} dt=${DT} shake=${SHAKE}"
} > "${OUT}/manifest.txt"

shake_flag=()
[[ "${SHAKE}" == yes ]] && shake_flag=(--shake)

for cfg in ${CONFIGS}; do
    np="${cfg%x*}"
    nt="${cfg#*x}"
    launch="${LAUNCH//\{np\}/${np}}"
    launch="${launch//\{nt\}/${nt}}"
    lmp_omp=()
    [[ "${nt}" -gt 1 ]] && lmp_omp=(-sf omp -pk omp "${nt}")
    for rep in $(seq 1 "${REPS}"); do
        tag="np${np}_nt${nt}_r${rep}"
        echo "=== ${tag} ==="
        # shellcheck disable=SC2086
        OMP_NUM_THREADS="${nt}" ${launch} "${ATV_BIN}" \
            --data "${OUT}/system.data" --ff "${OUT}/system.ff" \
            --cutoff "${CUT}" --alpha "${GEWALD}" --mesh "${MESH}" \
            --dt "${DT}" --T "${TEMP}" --tau-T 100 ${shake_flag[@]+"${shake_flag[@]}"} \
            --warmup "${WARMUP}" --steps "${STEPS}" \
            > "${OUT}/atv_${tag}.log" 2>&1
        (
            cd "${OUT}"
            # shellcheck disable=SC2086
            OMP_NUM_THREADS="${nt}" ${launch} "${LMP}" ${lmp_omp[@]+"${lmp_omp[@]}"} \
                -in "${HERE}/in.gaff" -var mode md -var cut "${CUT}" \
                -var gewald "${GEWALD}" -var meshx "${MESH}" \
                -var meshy "${MESH}" -var meshz "${MESH}" -var dt "${DT}" \
                -var shake "${SHAKE}" -var temp "${TEMP}" \
                -var warmup "${WARMUP}" -var steps "${STEPS}" \
                -var thermo "${THERMO}" -log "lmp_${tag}.log" -screen none
        )
    done
done
python3 "${HERE}/summarize_cpu.py" "${OUT}" | tee "${OUT}/summary.txt"
