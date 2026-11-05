#!/bin/bash -l
# Create the OpenMM reference environment for benchmarks/compare/gaff-liquid
# (conda-forge openmm with the CUDA platform).
#
#   OMM_ENV=/path/to/env [OMM_CONDA_MODULE=<lmod module>] bash setup_openmm.sh
#
# Run it on the GPU node itself: the env is architecture-specific. Prints the
# OpenMM version and platforms, and fails unless CUDA is one of them.
#
# Required: OMM_ENV (environment prefix to create or reuse).
# Optional: OMM_CONDA_MODULE, an Lmod module that provides `conda` (cluster
#           specific; when unset, `conda` must already be on PATH).
set -euo pipefail
: "${OMM_ENV:?set OMM_ENV to the environment prefix}"
if [[ -n "${OMM_CONDA_MODULE:-}" ]]; then
    # `bash script.sh` is not a login shell, so Lmod's `module` may be
    # undefined: load the site profile explicitly.
    if ! type module >/dev/null 2>&1 && [[ -f /etc/profile ]]; then
        set +u
        source /etc/profile
        set -u
    fi
    module load "${OMM_CONDA_MODULE}"
fi
if [[ ! -x "${OMM_ENV}/bin/python" ]]; then
    command -v conda >/dev/null \
        || { echo "conda not found: put it on PATH or set OMM_CONDA_MODULE" >&2; exit 1; }
    conda create -y -p "${OMM_ENV}" -c conda-forge \
        python=3.12 "openmm>=8.2" "cuda-version=12.*" numpy
fi
"${OMM_ENV}/bin/python" - <<'PY'
import openmm as mm
names = [mm.Platform.getPlatform(i).getName()
         for i in range(mm.Platform.getNumPlatforms())]
print("openmm", mm.__version__, "platforms", names)
if "CUDA" not in names:
    raise SystemExit("CUDA platform missing")
PY
