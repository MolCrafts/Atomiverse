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
sbatch .claude/job-cuda-gpw-ks-gga.sh
# or tools-style: SEMIEMPIRICAL=OFF, CMAKE_CUDA_ARCHITECTURES=90, aarch64 rustc
```

## Evidence

| Field | Value |
| --- | --- |
| HEAD | `4a5f5926` on `wf/escuda-cuda-gpw-ks-gga` |
| Dual-gate | `PASS cuda-gpw-ks-gga dual-gate` — e_H/e_xc/e_loc ~1e-15; max\|ΔV_eff\|=8.0e-16; max\|Δacc\|=3.9e-14 |
| Unit | `test_cuda_gpw_ks_gga` **Passed 1.17 s** (1/1, 100%) |
| Log | `.claude/cuda-gpw-ks-gga-1050278.out` (GH200 job **1050278**) |
| Perf | **perf_exempt** — stage-unit foothold on small Γ grid; not a fair CP2K twin |
