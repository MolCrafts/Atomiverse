---
slug: gpw-ts-kpoint-forces
status: approved
created: 2026-08-01
revised: 2026-08-01
domain: gpw / dispersion
queue: cp2k-method-loop
depends_on: [gpw-ts-forces, gpw-ts-stress, gpw-kpoint-forces]
unblocks: []
grilled: false
sibling: gpw-xdm-kpoint-forces
---

# Spec: `gpw-ts-kpoint-forces` — multi-k free-atom TS analytic ionic forces

**Slug:** `gpw-ts-kpoint-forces`  
**Domain:** GPW + Tkatchenko–Scheffler (free-atom) multi-k forces (CPU)  
**Queue:** CP2K capability loop  
**Status:** approved, implement now  
**Deps:** `gpw-ts-forces` ✓, `gpw-ts-stress` ✓, multi-k force assembly ✓  
**Sibling:** `gpw-xdm-kpoint-forces` (independent free-atom XDM multi-k force pin)

## Summary

Deliver a **dedicated multi-k product pin** that multi-k analytic ionic force
assembly folds **free-atom Tkatchenko–Scheffler (TS)** dispersion forces on top
of GPW (PBE class / PADE product mesh twin). Γ already folds TS
(`gpw-ts-forces`); multi-k stress is a sibling (`gpw-ts-kpoint-stress`). This
slug asserts the multi-k **force** path matches the geometry-only TS kernel.

Product path (no new public symbols):

1. Multi-k SCF (`GpwDft::k_mesh`) supplies electronic \(P_k\), \(W_k\).
2. Analytic k-summed Hellmann–Feynman / Pulay assembly (`gpw-kpoint-forces`).
3. Geometry-only \(F_{\mathrm{TS}} = -\nabla E_{\mathrm{TS}}\) from `ts_pbe`
   (free-atom volumes; pure-RS periodic lattice sum, rcut 50 Å) is **added**
   when `GpwDft::ts = true` on `assemble_gpw_forces_kpoints`.

## Oracle policy

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no stock `TYPE TS`**. Hard pin is **libMBD**
free-atom TS energy + **force dual-gate** + **multi-k wiring**, same policy as
`gpw-ts-forces` / `gpw-ts-ice` / `gpw-mbd-kpoint-forces`. Never fake
`oracle: cp2k`. **perf_exempt** for wall ratio (no fair CP2K twin).

## Public API

| Surface | Shape |
| ------- | ----- |
| Flags | `GpwDft::ts = true` + electronic `k_mesh` (and `GPWEvaluator` mirrors) |
| Forces | `gpw_forces_from_scf` / `GPWEvaluator(compute_forces=true)` |
| Energy | `GpwResult::e_disp` [Ha] (geometry-only TS) |

No `*Config` bag; no `make_*` factory; no new force-API symbols.

## Acceptance gates (hard)

| Gate | Check |
| ---- | ----- |
| Energy pin | Water-dimer isolated free-atom \(E_{\mathrm{disp}}\) vs libMBD |
| Force dual-gate | product \(F\) vs central FD of \(E_{\mathrm{disp}}\), \(h=10^{-5}\) Bohr |
| Translation | \(\|\sum_A F_A\|\) small |
| Multi-k wiring | \(F^{k+\mathrm{TS}}-F^{k,\mathrm{plain}}\approx F_{\mathrm{TS}}^{\mathrm{geom}}\) on H₂ 1×1×2 |
| Multi-k payload | SCF converged; `n_kpts ≥ 1`, `C_occ_k` non-empty |

## Out of scope

- Hirshfeld-scaled volumes (`gpw-hirshfeld-forces` / kpoint siblings)
- Multi-k free-atom TS **stress** (`gpw-ts-kpoint-stress`)
- Free-atom **XDM** multi-k forces (`gpw-xdm-kpoint-forces`)
- Stock CP2K QS twin (does not exist for TS)
