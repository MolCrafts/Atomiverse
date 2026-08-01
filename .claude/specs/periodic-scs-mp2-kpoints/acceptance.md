# Acceptance — periodic-scs-mp2-kpoints

## Correctness (dual gate)

1. **Literature / product absolute band** on scaled `E_corr` (SCS total):
   freeze first-green ATV digit **−0.0737958960 Ha** with tol **1e-4 Ha**
   (setup integrity; same class as `periodic-mp2-kpoints`).
2. **Algebraic dual** on same multi-k orbitals (Grimme literature):
   - unscaled OS/SS invariant under scale members vs plain MP2 (tol 1e-10)
   - plain OS+SS ≡ plain e_corr (1e-12)
   - SCS e_corr ≡ 1.2·OS + (1/3)·SS (1e-12)
   - e_total ≡ e_scf + e_corr (1e-12)
3. **Multi-k hard gates:** SCF converged; `n_kpts==2`; `C_k.size()==2`;
   `Mp2Result::ri`; E_corr finite and strictly negative; n_occ=8, n_virt=2.

## Oracle honesty

- Offline CP2K 2026.1 **was run** on the twin `.inp`.
- Result: multi-k SCF OK; **MP2_GPW aborts** under DFT KPOINTS.
- Therefore **`oracle: literature`** (Grimme 2003 SCS + Del Ben 2012 RI-GPW);
  never a fake `oracle: cp2k` energy for the correlation.

## Perf

- **perf_exempt:** no fair CP2K twin wall exists (algorithm missing).
- Document twin_wall (SCF+SCS) for ATV; do not invent a CP2K MP2 wall.

## Deliverables

- [x] `tools/cp2k_oracle/inputs/periodic-scs-mp2-kpoints.inp` (+ abort note)
- [x] Real CP2K run evidence + literature pin in `REFERENCE.md`
- [x] `regressions/gpw/periodic-scs-mp2-kpoints/`
- [x] CMake `atv_add_regression(gpw periodic-scs-mp2-kpoints …)`
- [x] Scoped build+test green; `perf_exempt` recorded
