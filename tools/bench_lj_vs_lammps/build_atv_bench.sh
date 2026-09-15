#!/bin/bash
# Build ATV bench_nonbonded_style with a LAMMPS-aligned CUDA precision mode.
#
# Modes match LAMMPS Kokkos `-D KOKKOS_PREC=…` (one knob):
#   double  → real=work=prec=double   (LMP default)
#   mixed   → real=work=float, prec=double
#   single  → real=work=prec=float
#
# Usage (on GH200, modules already loaded):
#   ./build_atv_bench.sh double
#   ./build_atv_bench.sh mixed
#   ./build_atv_bench.sh single
#
# Output binary:
#   build-cuda-bench-<mode>/bin/bench_nonbonded_style
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-double}"
case "$MODE" in
  double|mixed|single) ;;
  float)
    echo "note: 'float' is not a LMP mode; use 'mixed' (float math + double acc)" >&2
    MODE=mixed
    ;;
  *)
    echo "usage: $0 double|mixed|single" >&2
    exit 2
    ;;
esac

BUILD="build-cuda-bench-${MODE}"

echo "=== configure ATV CUDA prec_mode=$MODE arch=90 → $BUILD ==="
cmake -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DATV_BUILD_CUDA=ON \
  -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_HF=OFF \
  -DATV_BUILD_PW=OFF \
  -DATV_BUILD_GPW=OFF \
  -DATV_CUDA_PREC_MODE="$MODE" \
  -DBLA_VENDOR=OpenBLAS

cmake --build "$BUILD" -j"$(nproc)" --target bench_nonbonded_style
ls -la "$BUILD/bin/bench_nonbonded_style"
echo "export ATV_BENCH=$ROOT/$BUILD/bin/bench_nonbonded_style"
