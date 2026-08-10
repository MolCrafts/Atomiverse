# cuda-gpw-uks-o2-gate

method=cuda | track: es-cpu-cuda | parent: `cuda-gpw-uks-gamma` (H₂ foothold)

## Goal

Production-class **Γ UKS PADE energy** dual-gate on public `cuda::gpw::GpwDft`
for **O₂ triplet** (n_α=7, n_β=5). H₂ high-spin UKS is a foothold ≠ closed
capability vs the CPU fair twin class (`gpw-uks-o2-gamma`).

## Lever

| Item | Value |
| --- | --- |
| File / API | `src/cuda/gpw/gpw_solver.cu` public `GpwDft::run` UKS branch; `KsChainDevice::run_uks` |
| Function | reuses landed `run_uks` — **no new kernel** |
| Metric | \|E_cuda − E_cpu\| [Ha]; ⟨S²⟩ high-spin identity |
| Acceptance | both SCF converged; energy dual-gate within stated tol; GH200 log |

## Twin

Same-settings CPU `cpu::gpw::GpwDft` on O₂:

- Geometry: R(O–O)=1.2075 Å, centered in periodic cubic cell (CUDA rejects MT)
- Basis / PP: `SZV-GTH-PADE` + `GTH-PADE-q6` (table PADE pair for O; p-shell AO depth)
- CUTOFF 100 Ry, ngrids=1, XcKind::PADE, DIIS, fixed occ n_α=7 n_β=5

Oracle is **CPU twin**, not live CP2K (device surface is periodic; CP2K
`gpw-uks-o2-gamma` pin is MT isolated — different Poisson class).

## Non-goals

UKS forces/stress, UKS-PBE, multigrid UKS, multi-k UKS, OT, MT, new kernels.

## Perf

**perf_exempt** — small-molecule correctness dual-gate (not fair-wall claim).
