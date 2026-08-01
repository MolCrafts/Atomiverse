# Spec — `gpw-d4-kpoint-forces`

## Goal

Dedicated multi-k product pin for **GPW PBE-D4 two-body (BJ)** analytic ionic
forces. Capability already folds geometry-only \(F_{\mathrm{disp}}=-\nabla E_2\)
into `assemble_gpw_forces_kpoints` when `d4=true` and `d4_atm=false` (same
path as Γ `gpw-d4-forces`). Multi-k ATM is already pinned
(`gpw-d4-atm-kpoint-forces`); this slug pins the **two-body** multi-k product.

## Public API

```cpp
cpu::gpw::GPWEvaluator eval({O, H});
eval.xc = cpu::gpw::XcKind::PBE;
eval.k_mesh = {1, 1, 2};
eval.d4 = true;          // two-body BJ, s9 = 0
eval.d4_atm = false;
eval.compute_forces = true;
// Evaluate driver → Frame forces
```

## System (twin)

| Item | Value |
| ---- | ----- |
| System | water dimer, O H H O H H |
| Cell | 12×12×12 Å, `PERIODIC XYZ` |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` / GTH-PBE-q{6,1} |
| XC | PBE |
| k-mesh | Monkhorst–Pack **1×1×2** |
| Dispersion | DFT-D4 BJ two-body only (\(s_9=0\)) |
| ATV cutoff | 120 Ry (dispersion gates geometry-only) |
| CP2K cutoff | 300 Ry |

## Acceptance

1. Standalone `cpu::scf::dftd4` (s9=0) \(E_{\mathrm{disp}}\) + forces match
   offline **dftd4 3.7.0** `--grad` (geometry-only; same digits as Γ
   `gpw-d4-forces`).
2. Multi-k product SCF with `d4=true` converges; carries multi-k payload;
   `e_disp` matches dftd4 pin (k-mesh independent).
3. Wiring: after one shared multi-k SCF,
   \(F(\mathrm{d4})-F(\mathrm{plain})=F_{\mathrm{standalone}}^{\mathrm{D4}}\).
4. Offline **CP2K 2026.1 libdftd4** multi-k `ENERGY_FORCE` twin: FORCE Δ
   \(F(\mathrm{D4})-F(\mathrm{plain})\) matches standalone D4 (cutoff parity
   floor); fair wall \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\) on the
   **same** D4 multi-k ENERGY_FORCE job (1 MPI × 8 OMP).
5. Regression ≤ 60 s; no live oracle in ctest.

## Out of scope

- D4 three-body ATM multi-k forces (`gpw-d4-atm-kpoint-forces` ✓)
- Multi-k D4 stress (`gpw-d4-kpoint-stress` ✓)
- D3 / D3(BJ) multi-k force pins (sibling slugs)
