#!/bin/bash -l
# Build + test the CUDA plane-wave FFT layer on the GH200 GPU node.
set -e
module load GPU/buildenv-gcccuda/2026.03-cu13.0

cd "$SLURM_SUBMIT_DIR"

# The molcrafts .x86 venv ships a pip 'cmake' Python stub whose shebang points
# at an x86 python3; on the aarch64 GPU node it cannot exec and gets parsed as a
# shell script ("syntax error near -script.pyw"). When this node's login PATH
# puts that venv bin ahead of the module/system cmake, the build breaks. Drop
# both per-arch venv bins from PATH so cmake/ctest resolve to real aarch64
# binaries; the C++/CUDA/Rust build does not need the venv.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' \
    | grep -vE '/molcrafts/\.(x86|aarch64)/bin/?$' | paste -sd:)"

CMAKE="$(command -v cmake || true)"
if [ -z "$CMAKE" ] || ! "$CMAKE" --version >/dev/null 2>&1; then
    echo "ERROR: no working cmake on PATH after sanitization" >&2
    echo "PATH=$PATH" >&2
    exit 1
fi
CTEST="$(dirname "$CMAKE")/ctest"
[ -x "$CTEST" ] || CTEST="$(command -v ctest)"
echo "Using cmake: $CMAKE ($("$CMAKE" --version | head -1))"

# Help find_library locate FFTW3 from the module (it exports *_ROOT/LIBRARY_PATH
# but not CMAKE search paths). Collect plausible roots without hardcoding.
FFTW_HINT="${FFTW_ROOT:-${FFTWDIR:-${FFTW_HOME:-}}}"

"$CMAKE" -B build-cuda-pw \
    -DATV_BUILD_CUDA=ON \
    -DATV_BUILD_TESTS=ON \
    -DATV_BUILD_PW=ON \
    -DBLA_VENDOR=OpenBLAS \
    -DCMAKE_PREFIX_PATH="${FFTW_HINT};${OPENBLAS_ROOT}" \
    -DATV_MOLRS_PATH="$HOME/work/molcrafts/molrs"

# Build the CUDA pw tests (pull in libatomiverse).
"$CMAKE" --build build-cuda-pw -j"$(nproc)" \
    --target test_cuda_pw_fft test_cuda_pw_poisson test_cuda_pw_grid_map \
            test_cuda_pw_xc_lda test_cuda_pw_veff

# Runtime: module sets LIBRARY_PATH (link) but not LD_LIBRARY_PATH (run).
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
"$CTEST" --test-dir build-cuda-pw -R "test_cuda_pw_(fft|poisson|grid_map|xc_lda|veff)" --output-on-failure
echo "=== JOB DONE rc=$? ==="
