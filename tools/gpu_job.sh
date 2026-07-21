#!/bin/bash -l
# GPU build + test + benchmark for the CUDA plane-wave SCF backend.
module load GPU/buildenv-gcccuda/2026.03-cu13.0
set -e
cd "$SLURM_SUBMIT_DIR"

# The login profile prepends an x86_64 tool dir (…/.x86/bin) that shadows
# cmake/python with non-runnable wrappers on this aarch64 node.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/\.x86/' | paste -sd:)
hash -r
echo "cmake: $(command -v cmake)"

env | grep -iE "OPENBLAS|FFTW|BLAS" || true

# molrs: git submodule at ./molrs (default ATV_MOLRS_PATH). Override only if needed.
cmake -B build-cuda \
      -DATV_BUILD_CUDA=ON \
      -DATV_BUILD_PW=ON \
      -DATV_BUILD_HF=ON \
      -DATV_BUILD_TESTS=ON \
      -DATV_BUILD_BENCHMARKS=ON \
      -DBLA_VENDOR=OpenBLAS \
      -DCMAKE_PREFIX_PATH="$OPENBLAS_ROOT;$FFTW3_ROOT;$FFTW_ROOT"
cmake --build build-cuda -j$(nproc)

export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
TEST_RC=0
ctest --test-dir build-cuda --output-on-failure \
      -R "test_cuda_pw_si|test_pw_si|test_pw_components" \
      || TEST_RC=$?

# ── CUDA benchmarks: Si EOS (science) + SCF timing (performance) ──
# ecut 60 Ry: the ONCV pseudo needs ~60 Ry; lower cutoffs leave a
# basis-incompleteness volume dependence that tilts E(V).
cd build-cuda/benchmarks
rm -f cuda_si_eos.csv pw_scf_perf_cuda.csv
./bench_pw_si_cp2k_eos --ecut 60 --kmesh 4 --backend cuda --out cuda_si_eos.csv \
    > eos_cuda.log 2>&1 || echo "EOS_CUDA_FAILED (see eos_cuda.log)"
tail -12 eos_cuda.log
./bench_pw_scf_perf --backend cuda --ecut-list 20,30,40,60 --kmesh 2 \
    --out pw_scf_perf_cuda.csv > perf_cuda.log 2>&1
grep -E "^cuda" pw_scf_perf_cuda.csv
if [ "$TEST_RC" -ne 0 ]; then
    echo "GPU_JOB_TESTS_FAILED rc=$TEST_RC"
    exit "$TEST_RC"
fi
echo "GPU_JOB_OK"
