#!/bin/bash
# MPS packing throughput: K concurrent CUDA MD processes on one GPU under
# NVIDIA MPS. Run inside a GPU allocation after building md-residency
# (tools/diagnostics/md-residency) in each tree to compare.
#
#   run.sh OUT BUILD_DIR [BUILD_DIR ...]
#
# For every build, case (lj-nve, ion-pme), size (side 13/20/32 ≈ 2k/8k/32k
# atoms) and K in {1,2,4,8}: launch K identical processes at once and report
# the aggregate steady rate, the sum of each process's steps/s over its own
# timed interval (process start-up and graph capture excluded), next to the
# whole wall time. MPS shares the SMs.
set -uo pipefail
OUT=$1; shift
mkdir -p "$OUT"
export CUDA_MPS_PIPE_DIRECTORY=$(mktemp -d) CUDA_MPS_LOG_DIRECTORY=$(mktemp -d)
nvidia-cuda-mps-control -d || { echo "MPS daemon failed to start"; exit 1; }
trap 'echo quit | nvidia-cuda-mps-control' EXIT
STEPS=${STEPS:-2000} WARM=${WARM:-200}
printf "build case side K steps wall_s agg_steps_per_s\n" | tee "$OUT/mps.txt"
for build in "$@"; do
  for c in lj-nve ion-pme; do
    for side in 13 20 32; do
      for K in 1 2 4 8; do
        start=$(date +%s.%N)
        for ((p = 0; p < K; ++p)); do
          "$build/bin/md-residency" --case $c --side $side --steps $STEPS \
            --warm $WARM --out /dev/null >"$OUT/log-$(basename $(dirname $build))-$c-$side-$K-$p.txt" 2>&1 &
        done
        wait
        end=$(date +%s.%N)
        wall=$(echo "$end - $start" | bc -l)
        agg=$(cat "$OUT"/log-$(basename $(dirname $build))-$c-$side-$K-*.txt \
              | sed -n 's/.*us_per_step=\([0-9.e+-]*\).*/\1/p' \
              | awk '{s += 1e6 / $1} END {print s}')
        rm -f "$OUT"/log-$(basename $(dirname $build))-$c-$side-$K-*.txt
        printf "%s %s %s %s %s %.3f %.1f\n" "$(basename $(dirname $build))" $c $side $K $STEPS $wall $agg | tee -a "$OUT/mps.txt"
      done
    done
  done
done
