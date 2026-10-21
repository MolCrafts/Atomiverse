#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH -t 00:15:00
#SBATCH -J atv-gaff-cuda
#SBATCH -o gaff-gpu-%j.log

# Short CUDA correctness/performance experiments. Submit from the checkout:
# sbatch -A <gpu-account> tools/bench_gaff_gpu_job.sh
# BUILD, TARGETS and RUN_SCRIPT allow reuse without modifying the batch job.
set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load GPU/buildenv-gcccuda/2026.03-cu13.0
# IO/analysis builds also need the aarch64 Rust toolchain used by molrs.
# Select site modules explicitly when rustup is not installed in this account.
if [[ -n "${RUST_MODULES:-}" ]]; then
    read -r -a rust_modules <<< "$RUST_MODULES"
    module load "${rust_modules[@]}"
fi
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
read -r -a extra_cmake_flags <<< "${ATV_CMAKE_FLAGS:-}"
echo "host=$(hostname) arch=$(uname -m)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 -DATV_BUILD_CUDA=ON \
    -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON -DATV_BUILD_EVALUATE=ON \
    -DATV_BUILD_POTENTIALS=ON -DATV_BUILD_NEIGHBOR=ON -DATV_BUILD_TESTS=ON \
    -DATV_BUILD_BENCHMARKS=ON -DATV_BUILD_REGRESSIONS=ON \
    -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
    -DATV_BUILD_MC=OFF -DATV_BUILD_MINIMIZER=OFF -DATV_BUILD_VIB=OFF \
    -DATV_BUILD_PATH=OFF -DATV_BUILD_SEMIEMPIRICAL=OFF \
    -DATV_BUILD_ANALYSIS=OFF -DATV_BUILD_IO=OFF -DBLA_VENDOR=OpenBLAS \
    -DATV_CUDA_PREC_MODE="${PRECISION:-double}" "${extra_cmake_flags[@]}"
read -r -a targets <<< "${TARGETS:-bench_gaff reg_cuda-velocity-verlet-initial-force reg_cuda-gaff-fragment test_cuda_multi_gpu test_cuda_cluster_mask test_cuda_harmonic_coefficients test_cuda_packed_tiles test_cuda_hilbert test_cuda_timer reg_cuda-pme-nacl reg_cuda-pme-half-spectrum reg_cuda-ewald-nacl bench_nonbonded_style}"
cmake --build "$BUILD" -j "${SLURM_CPUS_PER_TASK:-8}" --target "${targets[@]}"
mkdir -p logs/gaff-openmm
# Record the complete C++/CUDA source cohort as well as the focused hot path.
# In particular, precision-state headers determine the device ABI.
find include src benchmarks tests regressions cmake -type f \
    \( -name '*.h' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.cu' \
       -o -name '*.cmake' -o -name 'CMakeLists.txt' \) -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum \
    > "logs/gaff-openmm/source-tree-${SLURM_JOB_ID:-manual}.sha256"
sha256sum CMakeLists.txt \
    >> "logs/gaff-openmm/source-tree-${SLURM_JOB_ID:-manual}.sha256"
sha256sum benchmarks/cuda/bench_gaff.cu \
    src/cuda/potentials/pair_lj_cut_coul_long.cu \
    src/cuda/potentials/internal/packed_tiles.cuh \
    src/cuda/neighbor/hilbert.cuh src/cuda/neighbor/sort_and_cluster.cu \
    src/cuda/longrange/pme.cu src/cuda/neighbor/neighbor_list.cu \
    src/cuda/neighbor/fused_search.cu src/cuda/neighbor/pair_search.cu \
    src/cuda/dynamics/init.cu \
    include/atomiverse/cuda/dynamics/md.cuh \
    tools/bench_gaff_openmm/compare.py \
    tools/bench_gaff_openmm/measure_resources.py \
    regressions/forcefield/cuda-pme-half-spectrum/reg_cuda-pme-half-spectrum.cu \
    > "logs/gaff-openmm/sources-${SLURM_JOB_ID:-manual}.sha256"
if [[ -n "${RUN_SCRIPT:-}" ]]; then
    bash "$RUN_SCRIPT" "$BUILD"
else
    timeout 10s "$BUILD/tests/test_cuda_multi_gpu" \
        --gtest_filter=MultiGpu.SingleGpuClusterMatchesBruteForce
    timeout 10s "$BUILD/regressions/reg_cuda-gaff-fragment"
    timeout 10s "$BUILD/regressions/reg_cuda-pme-nacl"
    timeout 10s "$BUILD/regressions/reg_cuda-ewald-nacl"
    timeout 10s "$BUILD/bin/bench_nonbonded_style" 10000 100 10
fi
