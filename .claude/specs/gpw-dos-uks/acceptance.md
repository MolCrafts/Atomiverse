# Acceptance — gpw-dos-uks

## Capability

Density of states on the GPW multi-k mesh with **collinear UKS** (dual-spin
\(D_\alpha\), \(D_\beta\), total \(D = D_\alpha + D_\beta\)). Product surface
is `cpu::gpw::Dos` / `DosResult` when `GpwDft` is UKS (`n_alpha`/`n_beta`).

Twin class: CP2K 2026.1 Quickstep GPW + LSD + MONKHORST-PACK + `&PRINT &DOS`
(GTH-GPW, same representation as ATV).

## System (pin)

Diamond Si BCT 4-atom, DZVP-MOLOPT-GTH + GTH-PBE-q4, PBE LSD multiplicity 1
(\(N_\alpha = N_\beta = 8\)), CUTOFF 150 Ry, NGRIDS 1, MP 2×2×2.

## Dual gate

1. **Energy** vs offline CP2K twin: \(|E_{\mathrm{ATV}} - E_{\mathrm{CP2K}}|
   \le 10^{-3}\) Ha.
2. **DOS character:** `nspin == 2`, multi-k (`n_kpts ≥ 2`), dual channels
   sized to total, occupied integral \(\approx 16\), open gap ≥ 0.1 eV,
   \(D(E) \ge 0\) with a positive peak, \(\int D_\alpha + \int D_\beta \approx
   \int D\).

## Perf

Fair wall on the **same** twin SCF+DOS job (1 MPI × fixed OMP):
\(T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \le 10\). No `perf_exempt` for missing
binary / model mismatch.

## Out of scope

- Metallic smearing DOS (`gpw-dos-metallic`)
- NSCF band path under UKS (`gpw-bands-uks`)
- Absolute histogram \(D(E)\) bin-by-bin vs CP2K (broadening differs)

## Status

- [x] CP2K twin run + pin recorded (E = −15.802416412655198 Ha; wall 5.164 s)
- [x] Regression dual-gate green (`reg_gpw-dos-uks`; \|ΔE\| 2.23e-5 Ha)
- [x] Fair wall ratio ≤ 10 (13.648 / 5.164 ≈ 2.64)
- [x] acceptance_met = true
