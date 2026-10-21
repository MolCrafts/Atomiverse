#!/bin/bash
# Two visible GPUs in one Arrhenius batch allocation; each case stays short.
set -euo pipefail
BUILD="${1:?build directory required}"
timeout 10s "$BUILD/tests/test_cuda_timer"
timeout 10s "$BUILD/tests/test_cuda_multi_gpu" \
    --gtest_filter=MultiGpu.LjForceParityNlistVsSingleGpu
for case_name in NveTrajectoryParity NveTrajectoryParityNlist NveMigrationParity MigratePreservesBondedTopology; do
    timeout 10s "$BUILD/regressions/reg_cuda-multi-gpu-trajectory" "$case_name"
done
timeout 10s "$BUILD/bin/bench_multi_gpu_force" 100 10
