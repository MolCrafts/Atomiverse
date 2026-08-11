# `cuda-gpw-product-h2o-gate`

## What

Production-class dual-gate for public `ATV::cuda::gpw::GpwDft` on **water**
(Γ closed-shell RKS Padé LDA). Extends the H₂-only first product
(`cuda-gpw-full-product`) to multi-species AO / grid depth without a new
kernel.

## Twin

Same-settings CPU `cpu::gpw::GpwDft`:

| Setting | Value |
| --- | --- |
| System | H₂O, 3 atoms |
| Basis / PP | O: `SZV-GTH-PADE` + `GTH-PADE-q6`; H: `DZV-GTH-PADE` + `GTH-PADE-q1` |
| Cell | periodic cubic 12 Bohr |
| CUTOFF | 100 Ry |
| XC | Padé LDA (`XcKind::PADE`) |
| Occupations | `n_occ = 4` (8 valence e⁻) |
| SCF | DIIS depth 6, `tol=1e-8`, `max_iter=80` |
| Grid | single-grid (`ngrids=1`), `local_pp_on_grid=true` |

Oracle: **CPU twin energy (and free-fn forces)** — not CP2K (CUDA rejects MT /
nonperiodic). Geometry: experimental-ish water centered in the cell
(CP2K tutorial bond lengths in Bohr).

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁵ Ha** |
| Forces max\|ΔF\| | **5×10⁻⁴ Ha/Bohr** |
| Both SCF converged | required |

Tols match the O₂ p-shell production class (`cuda-gpw-uks-o2-gate` 5e-5 Ha),
looser than pure-s H₂ (5e-6) for multi-center product residual (cuFFT vs FFTW).

## Perf

**perf_exempt** — small-molecule correctness-only (CUDA ES open-gate class).

## Non-goals

Multi-k, UKS water, OT, multigrid product, nonperiodic MT, PBE water (→ later
after this PADE gate / `product-h2o-pbe-gate`), AO-Pulay device path. Device
RKS DIIS/eigen is on base (`cuda-gpw-device-diis-eigen`); this gate uses it.

## Build / run (GH200)

```bash
#!/bin/bash -l
module load GPU/buildenv-gcccuda/2026.03-cu13.0
export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_TESTS=ON \
  -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
  -DATV_BUILD_GPW=ON -DATV_BUILD_CUDA=ON -DBLA_VENDOR=OpenBLAS
cmake --build build -j$(nproc) --target atomiverse \
  test_cuda_gpw_full_product reg_cuda-gpw-product-h2o-gate
ctest --test-dir build -R '^test_cuda_gpw_full_product$' --output-on-failure
cmake --build build --target run_reg_cuda-gpw-product-h2o-gate
```

## GH200 evidence (authority: SPEC)

| Job | Node | \|dE\| Ha | max\|dF\| Ha/Bohr | Note |
| --- | --- | ---: | ---: | --- |
| **1130195** | n46 | **7.526e−07** | **3.213e−06** | Assign wave-5 base `7f8da1f0` |
| 1130140 | n193 | 7.526e−07 | 3.213e−06 | Assign wave-4 base |

`acceptance_met=true` — both SCF converge; tols held; **perf_exempt**.
