#!/usr/bin/env bash
# Run a command inside the current reusable, user-authorized DFT allocation.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
role=${1:?usage: run-allocation.sh cpu|cuda [--tasks N] [--cpus N] -- command ...}
shift
case "$role" in
    cpu) tasks=1; cpus=16 ;;
    cuda) tasks=1; cpus=8 ;;
    *) printf 'unknown allocation role: %s\n' "$role" >&2; exit 2 ;;
esac
while (( $# )); do
    case "$1" in
        --tasks) tasks=$2; shift 2 ;;
        --cpus) cpus=$2; shift 2 ;;
        --) shift; break ;;
        *) break ;;
    esac
done
(( $# )) || { printf 'a command is required\n' >&2; exit 2; }
job=$(python3 - "$repo/.molq/gpw-driver/allocations.json" "$role" <<'PY'
import json,sys
data=json.load(open(sys.argv[1]))[sys.argv[2]]
assert isinstance(data['job_id'],int) and data['job_id']>0
print(data['job_id'])
PY
)
state=$(squeue --noheader --jobs="$job" --format='%T')
[[ "$state" == RUNNING ]] || {
    printf 'allocation %s is %s; command was not started\n' "$job" "${state:-unavailable}" >&2
    exit 75
}
gpu=()
step_binding=(--cpu-bind=threads)
if [[ "$role" == cuda ]]; then
    gpu=(--gpus=1)
    step_binding=(--overlap)
fi
exec srun --jobid="$job" "${step_binding[@]}" --exact --nodes=1 --ntasks="$tasks" \
    --cpus-per-task="$cpus" --mpi=pmi2 --export=ALL --chdir="$repo" "${gpu[@]}" \
    bash -lc '
set -euo pipefail
role=$1
shift
if [[ "$role" == cuda ]]; then
    module load GPU/buildenv-gcccuda/2026.03-cu13.0
    PATH=$(printf "%s" "$PATH" | tr ":" "\n" | grep -v "/\.x86/" | paste -sd:)
    export PATH RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
    hash -r
else
    module load buildenv-gcc/2026.03-mpich
    export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
fi
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
exec "$@"
' dft-allocation-step "$role" "$@"
