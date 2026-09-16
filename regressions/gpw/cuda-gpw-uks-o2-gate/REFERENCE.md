# `cuda-gpw-uks-o2-gate`

## What

Production-class Γ collinear UKS **energy** dual-gate on public
`ATV::cuda::gpw::GpwDft` after the H₂ UKS foothold (`cuda-gpw-uks-gamma`).
O₂ triplet (n_α=7, n_β=5), Padé LSD, DIIS, single-grid. Device mid-SCF
reuses `KsChainDevice::run_uks` — **no new kernel**.

## Twin

Same-settings CPU `cpu::gpw::GpwDft`:

| Item | Value |
| --- | --- |
| Molecule | O₂, R(O–O)=**1.2075 Å** along *z*, centered |
| Cell | **12 Bohr** cubic, **periodic** (CUDA rejects MT) |
| Basis | `SZV-GTH-PADE` |
| PP | `GTH-PADE-q6` (6e / O → 12 valence e⁻) |
| XC | Padé LSD (`XcKind::PADE`) |
| Cutoff | 100 Ry, `ngrids=1` |
| Spin | UKS, multiplicity 3 → **Nα=7, Nβ=5** |

Oracle: **CPU twin** (not CP2K). The CPU fair twin `reg_gpw-uks-o2-gamma`
uses MT / DZVP-MOLOPT-GTH / GTH-PBE-q6 — a different Poisson + basis class;
CUDA product is periodic-only for this slice.

## Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁵ Ha** |
| Both SCF converged | required |
| CUDA ⟨S²⟩ | > 1.5 |
| Force dual-gate | **not** this slug (`cuda-gpw-uks-forces`) |

## Perf

**perf_exempt** — small molecule correctness-only.

## Non-goals

UKS forces/stress, UKS-PBE, multi-k / multigrid UKS, OT, MT, device DIIS/eigen.

## GH200 evidence

Job **1127173** (2026-08-10, node n87 aarch64 GH200):

- `test_cuda_gpw_full_product` PASS (2.92 s; includes `O2UksPadeEnergyDualGate`)
- dual-gate:
  - CPU  E=−31.422608201904 Ha  ⟨S²⟩=2.000448  iters=24
  - CUDA E=−31.422604681912 Ha  ⟨S²⟩=2.000448  iters=70
  - \|ΔE\|=**3.520e-06** Ha (tol 5e-5)
- `GPU_JOB_OK`; `builds_wiped=true`
