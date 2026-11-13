#!/bin/bash
# Packing throughput of the 512-ion molten-NaCl NEMD model on one GPU:
# K independent Python processes (nacl_pack.py) sharing the card under
# NVIDIA MPS. Run inside a GPU allocation with one core per process, with a
# Python that imports atomiverse (CUDA build) and molpy:
#
#   nacl_pack.sh OUT [PYTHON]
#
# For every state (liquid, fluid), thermostat (gauss, config) and K in $KS:
# one PACK line (aggregate steps/s over the common timed interval) appended
# to OUT/pack.txt. MPS=0 skips the daemon (the processes are time-sliced).
# The daemon lives in the job's own pipe directory and is stopped on exit.
#
# A production pack is the same recipe: start the daemon as below, launch
# one process per simulation in the background (`taskset -c CORE python
# script.py &`), `wait`.
set -uo pipefail
OUT=$1
PYTHON=${2:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
KS=${KS:-"1 2 4 8 16"}
STATES=${STATES:-"liquid fluid"}
THERMOSTATS=${THERMOSTATS:-"gauss config"}
STEPS=${STEPS:-20000} WARM=${WARM:-500} MELT=${MELT:-2000}
mkdir -p "$OUT"
if [[ "${MPS:-1}" != 0 ]]; then
  export CUDA_MPS_PIPE_DIRECTORY=$(mktemp -d)
  export CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY/log
  mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d || { echo "MPS daemon failed to start"; exit 1; }
  trap 'echo quit | nvidia-cuda-mps-control; rm -rf "$CUDA_MPS_PIPE_DIRECTORY"' EXIT
  sleep 2
fi
for state in $STATES; do
  for thermostat in $THERMOSTATS; do
    for K in $KS; do
      "$PYTHON" "$HERE/nacl_pack.py" "$state" "$thermostat" --pack "$K" \
        --steps "$STEPS" --warm "$WARM" --melt "$MELT" \
        | sed "s/^PACK /PACK mps=${MPS:-1} /" | tee -a "$OUT/pack.txt"
    done
  done
done
