#!/bin/bash -l
set -euo pipefail
cd "/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/Atomiverse/.molq/next/gaussian-common-promotion/attempt-1/qe/si2-rks-pbesol"
module load QuantumESPRESSO/7.6-gcc-2026.03-mpich-eb
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command -v pw.x >binary-path.txt
sha256sum "$(command -v pw.x)" >binary.sha256
module list >modules.txt 2>&1
hostname >host.txt
date -u +%FT%TZ >started.txt
set +e
/usr/bin/time -p pw.x -in input.in >qe.out 2>qe.stderr
task_exit=$?
printf '%s\n' "$task_exit" >exit-code.txt
date -u +%FT%TZ >finished.txt
exit "$task_exit"
