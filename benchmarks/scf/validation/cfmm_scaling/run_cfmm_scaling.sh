#!/usr/bin/env bash
# Time Direct vs CFMM J-build on an alkane series (not part of ctest).
# Fill README.md table from the printed medians.
set -euo pipefail

echo "cfmm_scaling: methodology skeleton — wire a dedicated bench binary or"
echo "reuse RHF runs over C_nH_{2n+2}/STO-3G with coulomb=Direct|CFMM,"
echo "cfmm_order=10, cfmm_threshold=1e-9, OMP_NUM_THREADS fixed."
echo "Record median wall times in README.md and ch19."
exit 0
