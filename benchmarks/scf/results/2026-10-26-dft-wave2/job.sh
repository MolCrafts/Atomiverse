#!/bin/bash -l
#SBATCH --job-name=dft-wave2-public
#SBATCH --account=naiss2026-1-53-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=.claude/campaigns/dft-gap-wave2-20261026/public/job-%j.log
set -euo pipefail
cd /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse
module load buildenv-gcc/2026.03-mpich
DFT_W2_BLAS=/software/sse2/el9_epyc9005/manual/OpenBLAS/0.3.32/g14/hpc1/pthread/lp64/lib
export LD_LIBRARY_PATH="$DFT_W2_BLAS:$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
DFT_W2_OUT="$PWD/.claude/campaigns/dft-gap-wave2-20261026/public"
mkdir -p "$DFT_W2_OUT/bin"
cp build-hf-auto/_atomiverse.cpython-313-x86_64-linux-gnu.so "$DFT_W2_OUT/bin/"
sha256sum "$DFT_W2_OUT/bin/"*.so > "$DFT_W2_OUT/binary-before.sha256"
trap 'sha256sum "$DFT_W2_OUT/bin/"*.so > "$DFT_W2_OUT/binary-after.sha256"' EXIT
export PYTHONPATH="$PWD/python/src:$DFT_W2_OUT/bin:$PWD"
.claude/scratch/hf-auto/env/bin/python -m pytest -q \
  python/tests/test_ks_grid.py tests/benchmarks/test_dft_benchmarks.py \
  tests/benchmarks/test_qe_stress_capture.py tests/benchmarks/test_scientific_benchmarks.py \
  > "$DFT_W2_OUT/contracts.log" 2>&1
.claude/scratch/hf-auto/env/bin/python -m benchmarks run --method rks --method uks \
  --tier acceptance --threads 1 --repeats 1 --warmup 0 --timeout 300 \
  --no-snapshot --out "$DFT_W2_OUT/acceptance" > "$DFT_W2_OUT/acceptance.log" 2>&1
cmp "$DFT_W2_OUT/binary-before.sha256" <(sha256sum "$DFT_W2_OUT/bin/"*.so)
