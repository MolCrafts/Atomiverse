#!/bin/bash -l
set -euo pipefail
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SMEAR_CAMPAIGN="$PWD/.claude/campaigns/dft-gap-wave2-20261026/smearing"
c++ -O2 -std=c++20 -Iinclude -isystem build/_deps/googletest-src/googletest/include tests/cpu/planewave/test_smearing.cpp src/cpu/planewave/internal/smearing.cpp build-hf-auto/lib/libgtest_main.a build-hf-auto/lib/libgtest.a -lpthread -o "$SMEAR_CAMPAIGN/smearing-units"
"$SMEAR_CAMPAIGN/smearing-units" > "$SMEAR_CAMPAIGN/unit.log" 2>&1
