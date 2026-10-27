#!/bin/bash -l
set -euo pipefail
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.claude/scratch/hf-auto/env/bin/python tools/diagnostics/pw-smearing/capture_qe.py --source-dir .claude/campaigns/dft-gap-wave2-20261026/smearing/qe-sources --out .claude/campaigns/dft-gap-wave2-20261026/smearing/final-oracle-v2 --header .claude/campaigns/dft-gap-wave2-20261026/smearing/final-oracle-v2/pins.h
cmp tests/cpu/planewave/smearing_qe_pins.h .claude/campaigns/dft-gap-wave2-20261026/smearing/final-oracle-v2/pins.h
.claude/scratch/hf-auto/env/bin/python tools/diagnostics/pw-smearing/compare.py --qe-reference .claude/campaigns/dft-gap-wave2-20261026/smearing/final-oracle-v2/reference.json --before-source .claude/campaigns/dft-gap-wave2-20261026/smearing/before.cpp --out .claude/campaigns/dft-gap-wave2-20261026/smearing/comparison
