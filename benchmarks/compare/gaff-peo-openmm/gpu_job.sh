#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH -t 00:15:00
#SBATCH -J atv-gaff-cuda
#SBATCH -o gaff-gpu-%j.log

# Cluster-specific Slurm wrapper for the GAFF PEO CUDA-vs-OpenMM comparison:
# load the GPU toolchain, build bench_gaff, prepare the OpenMM env, then run
# RUN_SCRIPT. Submit from the repository root:
#
#   OPENMM_ENV=/path/to/env DATA=... FF=... \
#     sbatch -A <gpu-account> benchmarks/compare/gaff-peo-openmm/gpu_job.sh
#
# Required: OPENMM_ENV  Python venv with openmm (created here if missing)
# Optional: RUN_SCRIPT  [benchmarks/compare/gaff-peo-openmm/run.sh]
#           BUILD       [build-gaff-gpu]   PRECISION [double] (double|mixed)
#           TARGETS     [bench_gaff]
#           ATV_GPU_MODULE     Lmod CUDA toolchain [GPU/buildenv-gcccuda/2026.03-cu13.0]
#           ATV_PYTHON_MODULE  Lmod Python         [GPU/Python/3.13.5-bare-gcc-2025b-eb]
# The module defaults name the Arrhenius GH200 stack; override them elsewhere.
set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${OPENMM_ENV:?set OPENMM_ENV to the OpenMM Python env (created if missing)}"
module load "${ATV_GPU_MODULE:-GPU/buildenv-gcccuda/2026.03-cu13.0}"
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export OPENBLAS_ROOT="${OPENBLASDIR:-}"
export OMP_NUM_THREADS=1
# Some batch shells expose every node GPU. Honor Slurm's actual allocation
# before either engine initializes CUDA; indices follow nvidia-smi PCI order.
if [[ -n "${SLURM_JOB_GPUS:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi
BUILD="${BUILD:-build-gaff-gpu}"
echo "host=$(hostname) arch=$(uname -m)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 -DATV_BUILD_CUDA=ON \
    -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON -DATV_BUILD_EVALUATE=ON \
    -DATV_BUILD_POTENTIALS=ON -DATV_BUILD_NEIGHBOR=ON \
    -DATV_BUILD_BENCHMARKS=ON \
    -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
    -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF -DATV_BUILD_VIB=OFF \
    -DATV_BUILD_PATH=OFF -DATV_BUILD_SEMIEMPIRICAL=OFF \
    -DATV_BUILD_ANALYSIS=OFF -DATV_BUILD_IO=OFF -DBLA_VENDOR=OpenBLAS \
    -DATV_CUDA_PREC_MODE="${PRECISION:-double}"
read -r -a targets <<< "${TARGETS:-bench_gaff}"
cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-8}" --target "${targets[@]}"

module load "${ATV_PYTHON_MODULE:-GPU/Python/3.13.5-bare-gcc-2025b-eb}"
if [[ ! -x "$OPENMM_ENV/bin/python" ]]; then
    python3 -m venv "$OPENMM_ENV"
    "$OPENMM_ENV/bin/python" -m pip install 'openmm[cuda13]' numpy
fi
export OPENMM_ENV
"$OPENMM_ENV/bin/python" -m pip freeze

mkdir -p logs/gaff-openmm
sha256sum benchmarks/perf/md/bench_gaff.cu \
    src/cuda/potentials/pair_lj_cut_coul_long.cu \
    src/cuda/potentials/internal/packed_tiles.cuh \
    src/cuda/neighbor/hilbert.cuh src/cuda/neighbor/sort_and_cluster.cu \
    src/cuda/longrange/pme.cu src/cuda/neighbor/neighbor_list.cu \
    src/cuda/neighbor/fused_search.cu src/cuda/neighbor/pair_search.cu \
    src/cuda/dynamics/init.cu \
    include/atomiverse/cuda/dynamics/md.cuh \
    benchmarks/compare/gaff-peo-openmm/compare.py \
    benchmarks/compare/gaff-peo-openmm/measure_resources.py \
    > "logs/gaff-openmm/sources-${SLURM_JOB_ID:-manual}.sha256"
bash "${RUN_SCRIPT:-benchmarks/compare/gaff-peo-openmm/run.sh}" "$BUILD"
