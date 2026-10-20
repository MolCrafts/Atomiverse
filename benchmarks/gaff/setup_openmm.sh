#!/bin/bash -l
# Create the OpenMM reference environment for benchmarks/gaff on an Arrhenius
# GH200 node (aarch64; conda-forge has CUDA builds of openmm for it).
#
#   OMM_ENV=/path/to/env bash benchmarks/gaff/setup_openmm.sh
#
# Run it on the GPU node itself (inside tools/gpu_job.sh ATV_RUN or an
# interactive GPU allocation): the env is architecture-specific. Prints the
# OpenMM version and platforms, and fails unless CUDA is one of them.
set -euo pipefail
# `bash script.sh` is not a login shell, so the shebang's -l does not apply
# and Lmod's `module` may be undefined: load the site profile explicitly.
if ! type module >/dev/null 2>&1 && [[ -f /etc/profile ]]; then
    set +u
    source /etc/profile
    set -u
fi
: "${OMM_ENV:?set OMM_ENV to the environment prefix}"
# Arrhenius defines no default version; name one explicitly.
module load "${OMM_CONDA_MODULE:-GPU/Miniforge/26.3.2-2-eb}"
if [[ ! -x "${OMM_ENV}/bin/python" ]]; then
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
