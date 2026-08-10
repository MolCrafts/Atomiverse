#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -A naiss2026-4-715-gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J atv-cuda-gpw-kpoints
#SBATCH -o .claude/cuda-gpw-kpoints-%j.out
#SBATCH -e .claude/cuda-gpw-kpoints-%j.err

# Scoped build+test for cuda-gpw-kpoints on GH200.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

module load GPU/buildenv-gcccuda/2026.03-cu13.0

export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE '/(\.x86|/molcrafts/\.(x86|aarch64))/bin/?$' \
    | grep -v '/\.x86/' | paste -sd:)"
hash -r

export OPENBLAS_ROOT="${OPENBLASDIR:-${OPENBLAS_ROOT:-}}"
export RUSTUP_TOOLCHAIN=stable-aarch64-unknown-linux-gnu
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v '/\.cargo/bin' | paste -sd:):$HOME/.rustup/toolchains/stable-aarch64-unknown-linux-gnu/bin"
hash -r
export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

CMAKE="$(command -v cmake || true)"
if [ -z "$CMAKE" ] || ! "$CMAKE" --version >/dev/null 2>&1; then
    echo "ERROR: no working cmake on PATH" >&2
    exit 1
fi
CTEST="$(dirname "$CMAKE")/ctest"
[ -x "$CTEST" ] || CTEST="$(command -v ctest)"

echo "=== HOST $(hostname) arch $(uname -m) ==="
echo "cmake: $CMAKE ($("$CMAKE" --version | head -1))"
nvcc --version | head -4 || true
nvidia-smi -L || true
echo "=== GIT ==="
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git log -1 --oneline

if [ ! -f molrs/Cargo.toml ] && [ -f .gitmodules ]; then
    git submodule update --init --recursive molrs || true
fi

FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"
BD=build-cuda-gpw-kpoints

rm -rf "$BD" CMakeCache.txt CMakeFiles 2>/dev/null || true

"$CMAKE" -B "$BD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DATV_BUILD_CUDA=ON \
    -DATV_BUILD_HF=ON \
    -DATV_BUILD_PW=ON \
    -DATV_BUILD_GPW=ON \
    -DATV_BUILD_TESTS=ON \
    -DATV_BUILD_REGRESSIONS=ON \
    -DATV_BUILD_SEMIEMPIRICAL=OFF \
    -DATV_BUILD_MD=ON \
    -DATV_BUILD_MC=OFF \
    -DATV_BUILD_MINIMIZER=OFF \
    -DATV_BUILD_VIB=OFF \
    -DATV_BUILD_PATH=OFF \
    -DATV_USE_MPI=OFF \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}"

echo "=== SCOPED BUILD ==="
"$CMAKE" --build "$BD" -j"$(nproc)" --target atomiverse \
    test_cuda_gpw_kpoints reg_cuda-gpw-kpoints

echo "=== UNIT TEST ==="
"$CTEST" --test-dir "$BD" -R '^test_cuda_gpw_kpoints$' --output-on-failure

echo "=== REGRESSION ==="
"$CMAKE" --build "$BD" -j"$(nproc)" --target run_reg_cuda-gpw-kpoints

echo "=== PASS cuda-gpw-kpoints ==="

# Disk law: wipe build before exit
rm -rf "$BD" CMakeCache.txt CMakeFiles Testing .cache 2>/dev/null || true
echo "builds wiped"
