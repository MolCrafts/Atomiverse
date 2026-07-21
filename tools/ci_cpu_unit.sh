#!/usr/bin/env bash
# Single source of truth for the CPU unit gate.
# Used by:
#   - .pre-commit-config.yaml  (stage: pre-push)  — THE gate; blocks a red push
#   - .github/workflows/ci.yml (workflow_dispatch only) — optional remote re-run
#
# Not run on every GitHub push. Multi-backend (CUDA) is machine-local, not GHA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

jobs="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
BUILD_DIR="${ATV_CI_BUILD_DIR:-build}"

if [[ -n "${ATV_MOLRS_PATH:-}" && -d "${ATV_MOLRS_PATH}" ]]; then
  MOLRS="$(cd "${ATV_MOLRS_PATH}" && pwd)"
elif [[ -d "${ROOT}/molrs" && -f "${ROOT}/molrs/Cargo.toml" ]]; then
  MOLRS="${ROOT}/molrs"
else
  echo "ci_cpu_unit: molrs missing. Init the submodule:" >&2
  echo "  git submodule update --init --recursive molrs" >&2
  echo "Or set ATV_MOLRS_PATH to a compatible molrs tree." >&2
  exit 1
fi

echo "ci_cpu_unit: molrs=${MOLRS}"
echo "ci_cpu_unit: build=${BUILD_DIR}"

cmake -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_CUDA=OFF \
  -DATV_BUILD_TORCH=OFF \
  -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_EXAMPLES=OFF \
  -DATV_BUILD_HF=ON \
  -DATV_REAL_STORAGE=double \
  -DATV_MOLRS_PATH="${MOLRS}"

cmake --build "${BUILD_DIR}" -j"${jobs}"

# tests/ only. no_alloc is runtime-dependent (glibc/libgomp) — not a portable gate.
ctest --test-dir "${BUILD_DIR}" -j"${jobs}" --output-on-failure -E 'no_alloc'
