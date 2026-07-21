#!/usr/bin/env bash
# Migration gate (arch-refactor-07-molrs): cuda::RDF must consume the molrs
# streaming accumulator (cpu::compute::RDFAccumulator, exposed CUDA-safe via
# cpu/observer/trajectory_metrics.h), NOT the deleted legacy host histogram
# helpers. rdf_write_curve — the one g(r) file emitter — stays allowed.
# The forbidden tokens are assembled by concatenation below so the repo-wide
# deletion grep (acceptance ac-004) does not match this gate's own source.
# Pure grep, runs on any node (no CUDA needed).
set -u

root="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
rdf_cuh="$root/include/atomiverse/cuda/observer/rdf.cuh"
fail=0

if [ ! -f "$rdf_cuh" ]; then
    echo "FAIL: $rdf_cuh not found"
    exit 1
fi

forbid() {
    local pattern="$1"
    local label="$2"
    local hits
    hits=$(grep -n "$pattern" "$rdf_cuh")
    if [ -n "$hits" ]; then
        echo "FAIL: $label still present in cuda/observer/rdf.cuh:"
        echo "$hits"
        fail=1
    fi
}

# (a) legacy header must be gone
forbid "rdf_""histogram\.h" "legacy histogram header include"
# (b) legacy histogram/normalization calls must be gone
#     (the '(' suffix excludes the kept rdf_write_curve emitter by word shape)
forbid "rdf_""accumulate" "legacy host accumulate call"
forbid "rdf_""write(" "legacy host normalize-and-write call"

# (c) the molrs streaming-accumulator header must be the replacement
if ! grep -q "atomiverse/cpu/observer/trajectory_metrics\.h" "$rdf_cuh"; then
    echo "FAIL: cuda/observer/rdf.cuh does not include" \
         "atomiverse/cpu/observer/trajectory_metrics.h" \
         "(cpu::compute::RDFAccumulator)"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi
echo "OK: cuda::RDF is on the molrs streaming accumulator (legacy histogram gone)."
