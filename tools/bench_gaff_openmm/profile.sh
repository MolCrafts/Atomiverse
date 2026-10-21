#!/bin/bash
# Nsight diagnosis only: run inside an Arrhenius GPU allocation.
# Profile timings are never accepted as throughput benchmark results.
set -euo pipefail
BUILD="${1:?build directory}"
INPUT="${PROFILE_INPUT:?set PROFILE_INPUT to a validated prepared input}"
OUT="${PROFILE_OUT:-logs/gaff-profile/${SLURM_JOB_ID:?run under Slurm}}"
mkdir -p "$OUT"
unset ATV_BENCH_TIMING
sha256sum "$BUILD/bin/bench_gaff" "$INPUT" > "$OUT/input-build.sha256"
nsys --version > "$OUT/tool-version.txt"
hostname > "$OUT/host.txt"
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --force-overwrite=true --stats=false \
    --output="$OUT/trace" \
    timeout 10s "$BUILD/bin/bench_gaff" "$INPUT" "$OUT/forces.txt" \
    100 10 all 1 > "$OUT/profile.log" 2>&1
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    "$OUT/trace.nsys-rep" > "$OUT/kernel-summary.csv" 2> "$OUT/kernel-stats.log"
