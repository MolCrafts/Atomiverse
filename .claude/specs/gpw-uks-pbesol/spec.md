---
slug: gpw-uks-pbesol
status: approved
created: 2026-08-01
domain: gpw
queue: cp2k-method-loop
depends_on: [gpw-pbesol-si, gpw-uks-pbe]
---

# Spec: `gpw-uks-pbesol` — Γ-UKS PBEsol spin-GGA

**Slug:** `gpw-uks-pbesol`  
**Domain:** GPW / unrestricted KS  
**Queue:** CP2K method loop  
**Oracle:** offline CP2K Quickstep (never live in ctest)

## Summary

Dedicated **Γ-point collinear UKS** energy pin for **PBEsol** (Perdew et al.
2008 solid-state GGA: revised \(\mu\) for exchange; PBEsol correlation —
libxc `GGA_X_PBE_SOL` + `GGA_C_PBE_SOL` / CP2K `PARAMETRIZATION PBESOL`).

**Capability is already product:** `XcKind::PBEsol` is on the Γ/k UKS
allow-list; energy/forces/stress dispatch through `xc_grid_pbesol_spin`. RKS
is pinned by `gpw-pbesol-si`; Γ-UKS PBE by `gpw-uks-pbe`. This slug is
**pin-only** — close the missing dedicated UKS-PBEsol energy regression
against a matched CP2K Quickstep twin (no force/stress scope).

## Acceptance

1. `XcKind::PBEsol` accepted for Γ-UKS (`GPWEvaluator` with `n_alpha`/`n_beta`).
2. Dual KS SCF converges on centered O₂ (MT, DZVP-MOLOPT-GTH, GTH-PBE-q6,
   multiplicity 3, 400 Ry).
3. Offline CP2K pin for \(E_{\mathrm{tot}}\) and \(\langle S^2\rangle\) in
   `regressions/gpw/gpw-uks-pbesol/REFERENCE.md` (binary + version + input +
   output line; actually run).
4. Dual-gate: \(|E_{\mathrm{PBEsol}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha on the
   same geometry.
5. Public Frame energy remains eV; internal Ha cited in REFERENCE.
6. Perf: fair wall on the **same** twin SCF job;
   \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\) (`within_10x`); no
   `perf_exempt` for missing binary / mismatch.
7. No public `*Config` / factories; public API = `GPWEvaluator` +
   `single_point` / members only.

## Non-goals

- Analytic forces / stress regression for UKS-PBEsol (product path already
  wires spin XC; not this slug's pin).
- Multi-k UKS-PBEsol dedicated pin (product allow-list already includes it).
- Hybrid / meta-GGA, OT-specific pin.

## Implementation plan

1. Confirm product path (`xc_grid_pbesol_spin` + UKS allow-list) — no algorithm
   gap expected.
2. CP2K twin `.inp` under `tools/cp2k_oracle/inputs/gpw-uks-pbesol/`.
3. Run CP2K offline; record pin + wall in REFERENCE.md.
4. Regression `reg_gpw-uks-pbesol` dual-gate + energy / ⟨S²⟩ vs pin.
5. CMake `atv_add_regression(gpw gpw-uks-pbesol …)`.

## System (oracle twin)

| Item | Value |
|------|--------|
| Molecule | O₂, centered, R = 1.2075 Å |
| Cell | 12³ Å, `PERIODIC NONE`, MT |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q6 |
| Spin | LSD, multiplicity 3 → Nα=7, Nβ=5 |
| XC | PBEsol (`PARAMETRIZATION PBESOL` or `GGA_X_PBE_SOL` + `GGA_C_PBE_SOL`) |
| Cutoff | 400 Ry |

## Citations

- J. P. Perdew et al., *Phys. Rev. Lett.* **100**, 136406 (2008) — PBEsol.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT.
