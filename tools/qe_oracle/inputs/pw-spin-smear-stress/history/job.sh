#!/bin/bash -l
#SBATCH --job-name=dft-qe-spin-stress
#SBATCH --account=naiss2026-1-53-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=.claude/campaigns/dft-gap-wave2-20261026/qe/job-%j.log
set -euo pipefail
cd /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse
module load buildenv-gcc/2026.03-mpich
module load QuantumESPRESSO/7.6-gcc-2026.03-mpich-eb
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.claude/scratch/hf-auto/env/bin/python \
  tools/qe_oracle/inputs/pw-spin-smear-stress/capture.py \
  "$(command -v pw.x)" \
  --out "$PWD/.claude/campaigns/dft-gap-wave2-20261026/qe/probes"
