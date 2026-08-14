#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -t 04:30:00
#SBATCH -c 16
#SBATCH -J atv-pw-cell
#SBATCH -o .claude/cuda-pw-large-cell-%j.out
#SBATCH -e .claude/cuda-pw-large-cell-%j.err

# GH200 large-cell CPU-vs-CUDA plane-wave wall measurement (es-cpu-cuda slug
# `cuda-pw-large-cell-measure`, ES order step 4). Same-settings LSDA Si
# supercell ladder (8 / 64 / 216 atoms, Γ-point, ecut 15 Ry, UPF
# Si_lda_nc.upf) on cpu::pw::PW vs cuda::pw::PW, scaled until the CPU wall
# exceeds 60 s. Results feed .claude/notes/cuda-pw-large-cell.md; no perf
# claim is made from the small sizes. n=2 CPU is O(nb^3) dense and can
# exceed 2 h — budget prefers finishing that wall over n=3 once the
# >60 s gate is already met (or lower-bounded by a leg timeout).
#
#   sbatch tools/gpu_job_cuda_pw_large_cell.sh
#
# The harness tools/bench_cuda_pw_cell.cu is compiled with the exact
# compile / device-link / link commands CMake generates for the existing
# reg_pw-cuda-lda-si target (compile line harvested via `make -n`; dlink.txt
# and link.txt reused with path substitution), so no tracked CMake file is
# modified — this branch must not overlap the in-flight variable-cell tree
# (regressions/CMakeLists.txt and tests/CMakeLists.txt are dirty there).

set -uo pipefail # no -e: the ladder must always reach the summary + wipe
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"

echo "host=$(hostname) arch=$(uname -m)"
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
echo "HEAD=$(git rev-parse HEAD)"
nvidia-smi -L || true
lscpu | grep -E 'Model name|^CPU\(s\)' || true

WALL_BUDGET=16200 # seconds, mirrors -t 04:30:00 above
# Name deliberately avoids `build` / `build-*` so concurrent disk-law
# sweeps (`rm -rf build build-*`, find worktrees -name 'build-*') cannot
# unmap the tree mid-ladder (job 1048264 lost leg files that way).
BUILD=_pw_cell_work
LEGDIR="${SLURM_SUBMIT_DIR}/.claude/pw-cell-legs"
UPF="${SLURM_SUBMIT_DIR}/tests/data/pw/Si_lda_nc.upf"
# Reuse a surviving work tree from a failed run (ATV_FRESH=1 forces a wipe).
[ "${ATV_FRESH:-0}" = "1" ] && rm -rf "${BUILD}"
rm -rf CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
mkdir -p "${LEGDIR}"

cmake -B "${BUILD}" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_TESTS=OFF \
  -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON \
  -DATV_BUILD_CUDA=ON \
  -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_PW=ON \
  -DATV_BUILD_SEMIEMPIRICAL=OFF \
  \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}" || exit 2

echo "=== STAGE 1: library + reg_pw-cuda-lda-si ==="
cmake --build "${BUILD}" -j"$(nproc)" --target atomiverse || exit 2
cmake --build "${BUILD}" -j"$(nproc)" --target reg_pw-cuda-lda-si || exit 2
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

echo "=== STAGE 2: harvest reg compile/link commands, build harness ==="
REGDIR="${BUILD}/regressions"
DIR="${REGDIR}/CMakeFiles/reg_pw-cuda-lda-si.dir"
SRC_REG="${SLURM_SUBMIT_DIR}/regressions/pw/pw-cuda-lda-si/reg_pw-cuda-lda-si.cu"
SRC_BENCH="${SLURM_SUBMIT_DIR}/tools/bench_cuda_pw_cell.cu"
OBJ_REG='CMakeFiles/reg_pw-cuda-lda-si\.dir/pw/pw-cuda-lda-si/reg_pw-cuda-lda-si\.cu\.o'
OBJ_BENCH='CMakeFiles/reg_pw-cuda-lda-si.dir/bench/bench_cuda_pw_cell.cu.o'
mkdir -p "${DIR}/bench"

touch "${SRC_REG}"
cmake --build "${BUILD}" --target reg_pw-cuda-lda-si -- -n \
  > "${BUILD}/regcmds.txt" 2>/dev/null
COMPILE_LINE=$(grep -F 'reg_pw-cuda-lda-si.cu' "${BUILD}/regcmds.txt" \
  | grep -- ' -c ' | grep -v cmake_echo_color | head -1)
if [ -z "${COMPILE_LINE}" ]; then
  echo "HARVEST_FAILED: no compile line in make -n output; head follows"
  head -80 "${BUILD}/regcmds.txt"
  exit 3
fi
echo "--- harvested compile line ---"; echo "${COMPILE_LINE}"

printf '%s\n' "${COMPILE_LINE}" \
  | sed -e "s|${SRC_REG}|${SRC_BENCH}|g" \
        -e "s|${OBJ_REG}|${OBJ_BENCH}|g" \
  > "${BUILD}/bench_compile.sh"
# The dlink object list lives in a response file — substitute it too, or the
# device link binds the regression's object (job 1035238: HARNESS_LINK_FAILED,
# mismatched __cudaRegisterLinkedBinary/__fatbinwrap symbols).
sed -e "s|${OBJ_REG}|${OBJ_BENCH}|g" \
    -e "s|pw/pw-cuda-lda-si/reg_pw-cuda-lda-si\.cu\.o|bench/bench_cuda_pw_cell.cu.o|g" \
    "${DIR}/deviceObjects1.rsp" > "${DIR}/bench_deviceObjects.rsp"
sed -e "s|${OBJ_REG}|${OBJ_BENCH}|g" \
    -e "s|deviceObjects1\.rsp|bench_deviceObjects.rsp|g" \
    -e "s|cmake_device_link\.o|bench_device_link.o|g" \
    "${DIR}/dlink.txt" > "${BUILD}/bench_dlink.sh"
sed -e "s|${OBJ_REG}|${OBJ_BENCH}|g" \
    -e "s|cmake_device_link\.o|bench_device_link.o|g" \
    -e "s|-o reg_pw-cuda-lda-si|-o bench_cuda_pw_cell|g" \
    "${DIR}/link.txt" > "${BUILD}/bench_link.sh"
echo "--- bench compile/dlink/link ---"
cat "${BUILD}/bench_compile.sh" "${BUILD}/bench_dlink.sh" "${BUILD}/bench_link.sh"

bash "${BUILD}/bench_compile.sh" || { echo HARNESS_COMPILE_FAILED; exit 3; }
( cd "${REGDIR}" && bash "${SLURM_SUBMIT_DIR}/${BUILD}/bench_dlink.sh" ) \
  || { echo HARNESS_DLINK_FAILED; exit 3; }
( cd "${REGDIR}" && bash "${SLURM_SUBMIT_DIR}/${BUILD}/bench_link.sh" ) \
  || { echo HARNESS_LINK_FAILED; exit 3; }
BENCH="${REGDIR}/bench_cuda_pw_cell"
test -x "${BENCH}" || { echo HARNESS_MISSING; exit 3; }
echo "HARNESS_OK ${BENCH}"

echo "=== STAGE 3: slice sanity — reg_pw-cuda-lda-si --device=both (1t) ==="
# Twin correctness for this slug: CUDA↔CPU PW LSDA energy parity (2-atom Si).
REG_BIN=$(find "${REGDIR}" -maxdepth 1 -type f -name reg_pw-cuda-lda-si -perm -u+x | head -1)
timeout 900 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${REG_BIN}" --device=both
SLICE_RC=$?
echo "SLICE_SANITY_RC=${SLICE_RC}"
[ "${SLICE_RC}" -eq 0 ] || { echo "SLICE_SANITY_FAILED"; exit 4; }

echo "=== STAGE 4: supercell ladder (LSDA, Γ, ecut 15 Ry, tol 1e-6) ==="
LAST_WALL=0
LAST_RC=0
leg() { # dev threads n_rep timeout_s
  local dev=$1 thr=$2 n=$3 tmo=$4
  local tag="n${n}_${dev}_t${thr}"
  echo "--- LEG ${tag} (timeout ${tmo}s, elapsed $((SECONDS))s) ---"
  # Legs live under .claude/ (not BUILD) so a concurrent disk wipe cannot
  # drop the BENCH_RESULT lines the note is written from. Also stream to
  # the job .out via tee so the SLURM log is self-contained.
  local out="${LEGDIR}/leg_${tag}.txt"
  # line-buffer the harness so long CPU legs show progress in the log;
  # pipe+tee so the SLURM .out always carries BENCH_* even if LEGDIR is
  # later disturbed. PIPESTATUS[0] is timeout/bench (pipefail is on via -o).
  set +o pipefail
  timeout "${tmo}" env OMP_NUM_THREADS="${thr}" OPENBLAS_NUM_THREADS="${thr}" \
    stdbuf -oL -eL \
    "${BENCH}" "${UPF}" "${n}" "${dev}" 15.0 1 1e-6 100 2>&1 \
    | tee "${out}"
  local rc=${PIPESTATUS[0]}
  set -o pipefail
  LAST_RC=${rc}
  if [ "${rc}" -eq 124 ]; then
    # timeout: wall lower-bounded by the timeout itself (gate + note)
    echo "LEG_TIMEOUT ${tag} after ${tmo}s"
    echo "WALL_LOWER_BOUND wall_s>${tmo} tag=${tag}"
    LAST_WALL=${tmo}
  else
    LAST_WALL=$(grep -o 'wall_s=[0-9.]*' "${out}" 2>/dev/null | tail -1 | cut -d= -f2)
    LAST_WALL=${LAST_WALL:-0}
  fi
  echo "LEG_RC=${rc} ${tag} LAST_WALL=${LAST_WALL}"
  return 0
}
gt60() { awk "BEGIN{exit !(${1:-0} > 60)}"; }

# n=1 — 8 atoms: warm-up + same-process energy parity (correctness only).
leg both 16 1 2400
# Serial CPU reference at n=1 (OpenBLAS-threaded vs serial sensitivity).
leg cpu 1 1 3600

# n=2 — 64 atoms: expected >60 s CPU crossing (dense O(nb^3) on host).
leg cuda 16 2 2400
# Prefer finishing this wall over climbing to n=3: ~2–3 h expected from
# n=1 ~18 s × (128/16)^3 dense-diag scaling.
REMAIN=$((WALL_BUDGET - SECONDS))
CPU2_TMO=$((REMAIN - 900))
[ "${CPU2_TMO}" -gt 12000 ] && CPU2_TMO=12000
[ "${CPU2_TMO}" -lt 3600 ] && CPU2_TMO=3600
leg cpu 16 2 "${CPU2_TMO}"
CPU2_WALL=${LAST_WALL}
CPU2_RC=${LAST_RC}
GATE_MET=0
if gt60 "${CPU2_WALL}"; then
  GATE_MET=1
  if [ "${CPU2_RC}" -eq 124 ]; then
    echo "GATE n=2: CPU(16t) wall >${CPU2_WALL}s (timeout) — gate met at 64 atoms"
  else
    echo "GATE n=2: CPU(16t) wall ${CPU2_WALL}s > 60 s — gate met at 64 atoms"
  fi
else
  echo "GATE n=2: CPU(16t) wall ${CPU2_WALL}s <= 60 s — must scale to n=3"
fi

# n=3 — 216 atoms only if the >60 s gate is still open, or residual budget
# is large enough for a CUDA-only large-cell point.
REMAIN=$((WALL_BUDGET - SECONDS))
if [ "${GATE_MET}" -eq 1 ]; then
  if [ "${REMAIN}" -gt 1200 ]; then
    echo "GATE already met — optional n=3 CUDA-only if budget remains (${REMAIN}s)"
    leg cuda 16 3 1000
  else
    echo "SKIP n=3: gate met at n=2; only ${REMAIN}s walltime left"
  fi
elif [ "${REMAIN}" -gt 3000 ]; then
  leg cuda 16 3 2400
  REMAIN=$((WALL_BUDGET - SECONDS))
  CPU3_TMO=$((REMAIN - 600))
  [ "${CPU3_TMO}" -gt 7200 ] && CPU3_TMO=7200
  if [ "${CPU3_TMO}" -gt 900 ]; then
    leg cpu 16 3 "${CPU3_TMO}"
  else
    echo "SKIP n=3 cpu leg: only ${REMAIN}s walltime left"
  fi
else
  echo "SKIP n=3: only ${REMAIN}s walltime left and gate not yet met"
fi

echo "=== SUMMARY (all BENCH_RESULT / BENCH_PARITY lines) ==="
grep -hE 'BENCH_RESULT|BENCH_PARITY' "${LEGDIR}"/leg_*.txt 2>/dev/null || true
echo "GATE_MET=${GATE_MET} SLICE_SANITY_RC=${SLICE_RC}"
echo "GPU_JOB_DONE elapsed=$((SECONDS))s"

# Wipe only THIS job's work dir (not .claude/ legs — note is written from them).
rm -rf "${BUILD}" 2>/dev/null || true
rm -rf /tmp/atv-* 2>/dev/null || true
echo "BUILDS_WIPED"
