#!/bin/bash -l
#SBATCH -A naiss2026-3-381-cpu
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 00:40:00
#SBATCH -J atv_full
#SBATCH -o slurm/%x-%j.out
set -euo pipefail
module load buildenv-gcc/2026.03-mpich
cd "$SLURM_SUBMIT_DIR"
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -vE '/molcrafts/\.(x86|x86_64|aarch64)/bin/?$' | paste -sd:)"
export PATH="$HOME/.local/x86_64/cargo/bin:$PATH"
MOLRS=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molrs
SPACK_VIEW=/nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/spack/spack/opt/spack/view
B=build-full
cmake -B "$B" -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_CUDA=OFF -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON -DATV_BUILD_XC_ORACLE=ON \
  -DBLA_VENDOR=OpenBLAS -DCMAKE_PREFIX_PATH="${FFTW_ROOT:-};${OPENBLAS_ROOT:-};${SPACK_VIEW}" \
  -DATV_MOLRS_PATH="$MOLRS" >/dev/null
cmake --build "$B" -j"$(nproc)" 2>&1 | grep -iE 'error:' | head -20 || true
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${SPACK_VIEW}/lib64:${SPACK_VIEW}/lib:${LD_LIBRARY_PATH:-}"
"$(command -v ctest)" --test-dir "$B" --output-on-failure -E 'mpi' --timeout 400 | tail -25
echo "=== FULL DONE rc=$? ==="
