#!/bin/bash -l
set -euo pipefail
cd /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse
module load buildenv-gcc/2026.03-mpich
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
pw_blas=/software/sse2/el9_epyc9005/manual/OpenBLAS/0.3.32/g14/hpc1/pthread/lp64/lib
export LD_LIBRARY_PATH="$pw_blas:$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
.claude/scratch/hf-auto/env/bin/python .claude/campaigns/dft-gap-wave2-20261026/cpu-pw/guard-unit.py
