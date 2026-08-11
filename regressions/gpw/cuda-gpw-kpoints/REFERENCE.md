# cuda-gpw-kpoints — multi-k RKS DIIS energy dual-gate

## What

Public `cuda::gpw::GpwDft` multi-k closed-shell RKS DIIS (Padé LDA) vs
same-settings CPU `GpwDft`. H₂, DZV-GTH-PADE / GTH-PADE-q1, cell 12 Bohr,
CUTOFF 80 Ry, `k_mesh={2,1,1}` with **`k_time_reversal=false`** (nk=2 at
κ_x=±1/4; TR-on would collapse to one off-Γ k), single-grid,
`local_pp_on_grid=true`.

Oracle is **same-settings CPU**, not CP2K (CUDA product twin). Absolute energy
is not pinned — only CUDA↔CPU parity.

## Dual-gate

| Quantity | Tol |
| --- | ---: |
| \(\lvert E_{\mathrm{cuda}}-E_{\mathrm{cpu}}\rvert\) | \(5\times10^{-6}\) Ha |

## Perf

**perf_exempt** — small-molecule multi-k energy foothold; not a fair twin wall
cut vs CP2K. Deeper multi-k MG wall lives on the CPU track / later product depth.

## Lever

Host phased \(S_k/T_k/V_{\mathrm{nl}}^k\) + χ^k collocate/integrate
(`cpu::gpw::internal` kpoints); device Poisson+XC via
`KsChainDevice::run_from_rho`. Forces / stress / UKS×k / MG×k remain
fail-closed deeper slugs.
