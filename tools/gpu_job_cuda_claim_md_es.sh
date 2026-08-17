#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 16
#SBATCH -t 02:30:00
#SBATCH -J atv-cuda-claim-md-es
#SBATCH -o .claude/cuda-claim-md-es-%j.out
#SBATCH -e .claude/cuda-claim-md-es-%j.err

# GH200 dual-gate for the live CUDA claim-completion slugs that are
# compile+run on device (not GPW product — that is a sibling job):
#   cuda-md-lj-bench   — OpenMM tile LJ wall (cuda_nonbonded_style_bench)
#   cuda-scf-gradient  — RHF forces + H2 AIMD smoke
#   cuda-pw-spin-pbe   — existing energy dual-gate (forces still reject)
#
#   sbatch tools/gpu_job_cuda_claim_md_es.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
unset RUSTUP_TOOLCHAIN || true
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
unset CARGO_BUILD_TARGET || true

if [ ! -f molrs/Cargo.toml ]; then
  git submodule update --init --recursive molrs
fi

echo "=== HOST $(hostname) arch $(uname -m) $(date -Is) ==="
echo "cmake=$(command -v cmake) nvcc=$(command -v nvcc || true)"
echo "CC=${CC:-} CXX=${CXX:-}"
echo "rustc=$(command -v rustc) $(rustc --version 2>/dev/null || true)"
echo "HEAD=$(git rev-parse HEAD)"
echo "branch=$(git rev-parse --abbrev-ref HEAD)"
git log -1 --oneline
nvidia-smi -L || true

BUILD="${TMPDIR:-/tmp}/atv_claim_md_es_${SLURM_JOB_ID:-$$}"
rm -rf "${BUILD}"
mkdir -p "${BUILD}"

cmake -S "${SLURM_SUBMIT_DIR}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_CUDA=ON \
  -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=ON \
  -DATV_BUILD_PW=ON \
  -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_MD=ON \
  -DATV_BUILD_SEMIEMPIRICAL=OFF \
  -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_PREFIX_PATH="${OPENBLAS_ROOT}" \
  -DRust_TOOLCHAIN="${RUSTUP_TOOLCHAIN}"

echo "=== STAGE 1: library ==="
cmake --build "${BUILD}" -j"$(nproc)" --target atomiverse

echo "=== STAGE 2: CUDA MD LJ bench (cuda-md-lj-bench) ==="
cmake --build "${BUILD}" -j"$(nproc)" --target cuda_nonbonded_style_bench
BENCH=$(find "${BUILD}" -name cuda_nonbonded_style_bench -type f -executable | head -1)
echo "BENCH=${BENCH}"
"${BENCH}" 10000 80 15
echo "CUDA_MD_LJ_BENCH_OK"

echo "=== STAGE 3: CUDA SCF gradient + AIMD (cuda-scf-gradient) ==="
cmake --build "${BUILD}" -j"$(nproc)" --target \
  test_cuda_scf_rhf reg_h2o_rhf_sto3g_grad reg_h2_rhf_sto3g_aimd
ctest --test-dir "${BUILD}" --output-on-failure --timeout 300 \
  -R '^test_cuda_scf_rhf$'
"${BUILD}/regressions/reg_h2o_rhf_sto3g_grad" --device=both
"${BUILD}/regressions/reg_h2_rhf_sto3g_aimd" --device=cuda
echo "CUDA_SCF_GRADIENT_OK"

echo "=== STAGE 4: CUDA PW spin-PBE energy (cuda-pw-spin-pbe) ==="
cmake --build "${BUILD}" -j"$(nproc)" --target \
  test_cuda_pw_xc_pbe test_cuda_pw_force_local reg_cuda-pw-spin-pbe
ctest --test-dir "${BUILD}" --output-on-failure --timeout 300 \
  -R '^test_cuda_pw_xc_pbe$|^test_cuda_pw_force_local$'
"${BUILD}/regressions/reg_cuda-pw-spin-pbe" --device=both
echo "CUDA_PW_SPIN_PBE_ENERGY_OK"

echo "=== STAGE 5: wipe ==="
rm -rf "${BUILD}"
echo "builds_wiped=true"
echo "=== DONE green $(date -Is) ==="
