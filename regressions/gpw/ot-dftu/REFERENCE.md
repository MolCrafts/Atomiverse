# Reference — Γ-UKS GPW OT + Dudarev DFT+U (NiO molecule, PADE / SZV)

## Method

Unrestricted Kohn–Sham **GPW** at Γ with Martyna–Tuckerman isolated
electrostatics, PADE LSD, **Orbital Transformation** SCF (`use_ot`), and
**Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken) on Ni \(d\) (\(L=2\),
\(U-J=6.0\) eV).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `use_ot=true`,
`n_alpha`/`n_beta`, `plus_u`, single density grid.

This is the **combo** product pin: OT alone (`reg_ot-uks-o2`) and DFT+U alone
(`reg_gpw-dftu-nio`) were already green; the gap was UKS OT throwing when
`plus_u` was non-empty. CP2K's DFT+U twin already used `&OT`.

## System

| Item | Value |
|------|--------|
| Molecule | NiO, centered in cell |
| Bond length | R(Ni–O) = **1.668 Å** (along *z*) |
| Coordinates (Å) | Ni (5, 5, 4.166), O (5, 5, 5.834) |
| Cell | **10 × 10 × 10 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | Ni `SZV-GTH-PADE-q18`, O `SZV-GTH-PADE` |
| PP | Ni `GTH-PADE-q18` (18e), O `GTH-PADE-q6` (6e) → **24** valence e⁻ |
| XC | PADE (LDA LSD) |
| DFT+U | Ni \(L=2\), \(U-J = 6.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| SCF | OT `FULL_SINGLE_INVERSE` + OUTER_SCF (CP2K); ATV `use_ot=true` |
| Cutoff | 250 Ry (ATV single grid; CP2K NGRIDS 4 / REL_CUTOFF 40) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=13, Nβ=11** |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), local install
  `/nobackup/.../work/cp2k/install` (re-pinned 2026-07-26).
- **Input:** `tools/cp2k_oracle/inputs/ot-dftu/nio_mol_pade_u_ot.inp`
  (twin of `gpw-dftu-nio/nio_mol_pade_u.inp` — same OT+PLUS_U chemistry).
- **Run (offline only; never in ctest):**
  ```bash
  export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
  export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
  OMP_NUM_THREADS=8 cp2k.psmp -i nio_mol_pade_u_ot.inp -o ot_dftu.out
  ```
- **Date:** 2026-07-26
- **Wall (CP2K, OMP=8):** ≈ **13.65 s**

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−184.92466076474310** | **5e-2 Ha** (single-grid vs multigrid) |
| DFT+U energy [Ha] | **0.05230865222344** | diagnostic (\(E_U > 10^{-4}\)) |
| ⟨S²⟩ (single det.) | **2.019198** | **0.25** |
| Mulliken spin m_Ni | **1.177632** | **0.5** (and m_Ni > 0.3) |

Key log lines:

```
  DFT+U energy:                                                 0.05230865222344
  Ideal and single determinant S**2 :                    2.000000       2.019198
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.924660764743095
```

## Regression gates

| Check | Tolerance |
| ----- | --------: |
| \(\|E_{\mathrm{ATV,OT+U}} - E_{\mathrm{CP2K}}\|\) | **5×10⁻² Ha** |
| \(\|E_{\mathrm{ATV,OT+U}} - E_{\mathrm{ATV,DIIS+U}}\|\) | **1×10⁻³ Ha** |
| \(E_U > 0\) | **1×10⁻⁴ Ha** |
| ⟨S²⟩ vs CP2K | **0.25** |

Milli-Ha class is not claimed at 250 Ry single-grid vs CP2K multigrid; the
gate is energy-class + spin + non-trivial \(E_U\) + OT/DIIS fixed-point parity.

## Fair wall (perf gate)

| Twin | Wall [s] | Notes |
| ---- | -------: | ----- |
| CP2K OT+PLUS_U | **13.65** | `cp2k.psmp`, OMP=8 |
| ATV OT+U (2026-07-26) | **3.58** | Release, OMP=8, 23 OT iters |
| ATV DIIS+U (same run) | **3.03** | 65 DIIS iters |

\(r = T_{\mathrm{atv,OT}}/T_{\mathrm{cp2k}} = 3.58/13.65 \approx \mathbf{0.26}\) ≤ 10 ✓.

## Atomiverse (2026-07-26, Release, OMP=8)

| Quantity | Value |
|----------|--------|
| E_tot OT+U [Ha] | ≈ −184.94575 |
| \|ΔE\| vs CP2K | ≈ **2.1e-2 Ha** (within 5e-2) |
| \|E_OT − E_DIIS\| | ≈ **3.8e-4 Ha** |
| ⟨S²⟩ | ≈ 2.015 |
| m_Ni (Mulliken) | ≈ 1.00 |
| OT iters | 23 |

## Citations

- VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT.
- Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified U.
- Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
