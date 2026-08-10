# cuda-gpw-ks-gga

## What

Composed Γ-point CUDA GPW KS **stage** with closed-shell **PBE-GGA**:
product collocate → Poisson → spectral ∇ρ → device PBE (local + flux) →
−∇·flux → V_eff → product integrate. Dual-gate vs same-settings CPU
`ks_grid_build_rks(..., XcKind::PBE)`.

**Not** a full `GpwDft` method product. No multigrid, no spin, no public method.

## Oracle

| Field | Value |
| --- | --- |
| Oracle | same-settings CPU twin (`ks_grid_build_rks` + continuous product shapes) |
| Units | Hartree |
| Stage tols | e_rel 1e-10; V_eff 1e-9·peak + 1e-12; acc 1e-9·peak + 1e-12 |

## Build / run (GH200 only for CUDA)

```bash
#!/bin/bash -l
module load GPU/buildenv-gcccuda/2026.03-cu13.0
# do NOT override RUSTUP on aarch64
export OPENBLAS_ROOT="${OPENBLASDIR:-$OPENBLAS_ROOT}"
export LD_LIBRARY_PATH="${LIBRARY_PATH}:${LD_LIBRARY_PATH}"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
  -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target atomiverse test_cuda_gpw_ks_gga \
  reg_cuda-gpw-ks-gga
ctest --test-dir build -R '^test_cuda_gpw_ks_gga$' --output-on-failure
cmake --build build --target run_reg_cuda-gpw-ks-gga
```

## Evidence (fill after GH200)

| Field | Value |
| --- | --- |
| HEAD | (worker) |
| Dual-gate | (PASS line) |
| Log | (path) |
