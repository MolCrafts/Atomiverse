# Spec — gpw-dftu-forces

## Intent

Dedicated **Γ-UKS GPW analytic ionic forces** pin with **Dudarev DFT+U**
(`plus_u` / CP2K `PLUS_U`). Energy + stress + multi-k forces already product;
this slug closes the missing **Γ ENERGY_FORCE + PLUS_U** dual-gate pin.

## Capability claim

- Path: `cpu::gpw::GpwDft` + `gpw_forces_analytic` / `gpw_forces_from_scf`
  (and product `GPWEvaluator` with `compute_forces` + `plus_u`).
- Occupation-matrix Pulay fold already in `gpw_forces.cpp`
  (`subtract_dftu_pulay` on Γ W matrix).
- Twin class: CP2K 2026.1 Quickstep GPW UKS + `PLUS_U` ENERGY_FORCE,
  NiO-class d-shell (same family as `gpw-dftu-nio` / section B of
  `gpw-dftu-kpoint-forces`).

## Gates

| Gate | What | Oracle |
|------|------|--------|
| A | Γ-UKS + U H₂ toy: analytic vs full-6N multi-SCF FD | ATV self dual |
| A | U activation, Newton-3 / \|ΣF\| | internal |
| B | Periodic Γ NiO ENERGY_FORCE: \|E − E_CP2K\| | offline CP2K |
| B | Force twin vs CP2K | **diagnostic only** (projector model divergence — net Mulliken CP2K vs gross ATV; see notes) |
| Perf | T_atv / T_cp2k on NiO twin SCF+forces | ≤ 10 |

## Non-goals

- Multi-k DFT+U forces (done: `gpw-dftu-kpoint-forces`)
- Aligning ATV projector to CP2K net Mulliken
- Solid AFM rocksalt NiO
