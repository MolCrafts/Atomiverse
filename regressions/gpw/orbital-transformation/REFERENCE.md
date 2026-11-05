# gpw/orbital-transformation

One case; every section below is run by `atv_regressions --case gpw/orbital-transformation` and keeps its own oracle record.


## Reference — GPW OT vs DIIS on isolated water (MOLOPT)

### Method

Closed-shell Γ-point **GPW** DFT with **PBE**, **DZVP-MOLOPT-GTH** basis and
**GTH-PBE** pseudopotentials, **Martyna–Tuckerman** isolated electrostatics
(`nonperiodic = true`). Atomiverse product path:

- OT: `cpu::gpw::GPWEvaluator::use_ot = true` (Orbital Transformation SCF)
- DIIS: `use_ot = false` (default diagonalization + Pulay DIIS)

OT formulation (in-tree): Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020)
§VII C; VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003).

Public units on Frame: energy [eV] via `kEnergy`. Method result channel:
`last_energy_hartree()` / `GpwResult::e_total` [Hartree].

### System

CP2K getting-started `h2o` geometry in a **10 Å** cubic cell (coordinates [Å]):

| Atom | x | y | z |
| ---- | ---: | ---: | ---: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

Settings: cutoff **250 Ry** (regression budget; 400 Ry tutorial class is
`examples/cpp/10_gpw.cpp`), `max_iter=200`, `tol=1e-7`, mix `0.3` (DIIS path).

### Offline CP2K oracle

- **Version:** CP2K **2026.1** (`CP2K/2026.1-gmpflf-2026.03` on Arrhenius)
- **Binary:** `cp2k.psmp` (1 MPI rank, `OMP_NUM_THREADS=4`)
- **Data:** `$CP2K_ROOT/share/cp2k/data/{BASIS_MOLOPT,GTH_POTENTIALS}`
- **Inputs (in-tree):**
  - `tools/cp2k_oracle/inputs/ot-minimizer-water-ot.inp` — `&OT` +
    `PRECONDITIONER FULL_SINGLE_INVERSE`, `MINIMIZER DIIS`, outer SCF, **CUTOFF 250**
  - `tools/cp2k_oracle/inputs/ot-minimizer-water-diag.inp` — traditional
    diagonalization + Broyden mixing (no `&OT`)

#### Pinned energies (2026-07-23, CUTOFF 250)

| Path | \(E_{\mathrm{tot}}\) [Ha] | Notes |
| ---- | ------------------------: | ----- |
| **CP2K OT** | **−17.212848291913666** | primary pin |
| Atomiverse OT | −17.2118583480 | 31 OT iters (OMP=1) |
| Atomiverse DIIS | −17.2118583479 | 9 DIIS iters; \|OT−DIIS\| ~4e-11 Ha |

```text
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -17.212848291913666
```

At 400 Ry (tutorial twin, not the regression pin) CP2K OT was
−17.212337205347978 (19 steps); DIAG −17.212337010579070 — retained as
cross-check of the energy class.

#### Exact offline commands

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=.../CP2K/2026.1-gmpflf-2026.03/share/cp2k/data
## patch BASIS/POTENTIAL paths to absolute in the .inp copies, then:
cp2k.psmp -i ot-minimizer-water-ot.inp -o ot.out
```

**ctest never launches CP2K.**

### Regression gates

| Check | Tolerance |
| ----- | --------: |
| \(\|E_{\mathrm{ATV,OT}} - E_{\mathrm{CP2K,OT}}\|\) | **1.5×10⁻³ Ha** |
| \(\|E_{\mathrm{ATV,OT}} - E_{\mathrm{ATV,DIIS}}\|\) | **1×10⁻⁴ Ha** |

Milli-Ha pin tol at 250 Ry absorbs grid/multigrid defaults that still differ
from CP2K (NGRIDS/REL_CUTOFF/collocation). OT vs DIIS in-tree is near machine
parity on the same grids. Product path stays **single-grid** on this twin:
product multigrid is a net loss for 3-atom / 10 Å at 250 Ry.

#### Measured fixed points

Γ OT is Pulay DIIS on H with fixed-H `ot_minimize_fixed` (no Armijo grid
rebuild per micro-step).
\(\|E_{\mathrm{OT}}-E_{\mathrm{CP2K}}\|=9.90\times10^{-4}\) Ha;
in-tree \(\|E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\|\sim1\times10^{-12}\) Ha
(2026-08-05).

### Fair wall (perf gate)

This file carries no wall numbers. The single binding fair-wall source for
this twin — host, CP2K binary provenance, PRINT_LEVEL/timer convention,
reproduction recipe, and the current \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) —
is [`.claude/notes/gpw-perf-baseline.md`](../../../.claude/notes/gpw-perf-baseline.md).
The fair twin is CP2K OT vs ATV OT-only; the full regression binary runs
OT + DIIS and is not a fair wall. Case-local walls were last measured here
2026-08-05 and are superseded by that note.


## Reference — Γ-UKS OT GPW O₂ (PADE / DZVP-MOLOPT-GTH)

### Method

Unrestricted Kohn–Sham **GPW** at Γ with **Orbital Transformation (OT)** SCF
minimizer (`use_ot = true`), Martyna–Tuckerman isolated electrostatics,
spin-resolved PADE LSD, GTH-PBE-q6 / DZVP-MOLOPT-GTH.

Product path: `cpu::gpw::GPWEvaluator` with `n_alpha`/`n_beta` + `use_ot`.
Open-shell OT gradient uses occupation 1 per spin; joint PR+ CG + Armijo on
the rebuilt total energy (same fixed point as UKS DIIS).

OT formulation: VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003);
Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C.

Public Frame energy [eV] via `kEnergy`. Method channel:
`last_energy_hartree()` / `GpwResult::e_total` [Hartree].

### System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q6 |
| XC | PADE (LDA LSD) |
| Cutoff | **250 Ry** (regression budget; 400 Ry is `gpw-uks-o2-gamma` DIIS twin) |
| Spin | UKS, multiplicity 3 → **Nα=7, Nβ=5** |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, git:5e54ba2), module `CP2K/2026.1`
  on Arrhenius (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/ot-uks-o2/o2_uks_ot_pade.inp`
  (`&OT` + `PRECONDITIONER FULL_SINGLE_INVERSE`, `MINIMIZER DIIS`, OUTER_SCF,
  CUTOFF 250).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  export OMP_NUM_THREADS=4
  mpirun -np 1 cp2k.psmp -i o2_uks_ot_pade.inp -o o2_uks_ot_pade.out
  ```
- **Date:** 2026-07-26

| Quantity | CP2K OT pin | Tolerance (regression) |
|----------|------------:|------------------------:|
| E_tot [Ha] | **−31.770029989852723** | **2.5×10⁻³ Ha** |
| ⟨S²⟩ (single det.) | **2.004666** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines:

```
  *** SCF run converged in    30 steps ***
  Total energy:                                               -31.77002998985272
  Ideal and single determinant S**2 :                    2.000000       2.004666
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.770029989852723
```

Cross-check: CP2K DIAG at 400 Ry (centered O₂) was −31.7698035 Ha
(`gpw-uks-o2-gamma`) — same energy class; 250 Ry OT pin is the regression twin.

### Regression gates

| Check | Tolerance |
| ----- | --------: |
| \|E_ATV,OT − E_CP2K,OT\| | **2.5×10⁻³ Ha** |
| \|E_ATV,OT − E_ATV,DIIS\| | **1×10⁻⁴ Ha** |
| \|⟨S²⟩_ATV − ⟨S²⟩_CP2K\| | **0.15** |

Milli-Ha pin tol at 250 Ry absorbs single-grid vs CP2K multigrid (NGRIDS/REL_CUTOFF)
differences; OT/DIIS in-tree is the tight fixed-point gate.

### Fair wall (perf gate)

This file carries no wall numbers. The single binding fair-wall source —
host, CP2K binary provenance, PRINT_LEVEL/timer convention, and the
reproduction recipe — is
[`.claude/notes/gpw-perf-baseline.md`](../../../.claude/notes/gpw-perf-baseline.md).
Case-local walls were last measured here 2026-08-05 and are superseded by
that note.

In-tree OT/DIIS parity: |E_OT − E_DIIS| ≈ **5×10⁻¹⁰ Ha** (same grids).

### Citations

- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT.
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) — CP2K OT.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE/GTH.


## Reference — Γ-UKS GPW OT + Dudarev DFT+U (NiO molecule, PADE / SZV)

### Method

Unrestricted Kohn–Sham **GPW** at Γ with Martyna–Tuckerman isolated
electrostatics, PADE LSD, **Orbital Transformation** SCF (`use_ot`), and
**Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken) on Ni \(d\) (\(L=2\),
\(U-J=6.0\) eV).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `use_ot=true`,
`n_alpha`/`n_beta`, `plus_u`, single density grid.

This is the **combo** product pin: OT alone (`reg_ot-uks-o2`) and DFT+U alone
(`reg_gpw-dftu-nio`) were already green; the gap was UKS OT throwing when
`plus_u` was non-empty. CP2K's DFT+U twin already used `&OT`.

### System

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

### Offline CP2K pin

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

### Regression gates

| Check | Tolerance |
| ----- | --------: |
| \(\|E_{\mathrm{ATV,OT+U}} - E_{\mathrm{CP2K}}\|\) | **5×10⁻² Ha** |
| \(\|E_{\mathrm{ATV,OT+U}} - E_{\mathrm{ATV,DIIS+U}}\|\) | **1×10⁻³ Ha** |
| \(E_U > 0\) | **1×10⁻⁴ Ha** |
| ⟨S²⟩ vs CP2K | **0.25** |

Milli-Ha class is not claimed at 250 Ry single-grid vs CP2K multigrid; the
gate is energy-class + spin + non-trivial \(E_U\) + OT/DIIS fixed-point parity.

### Fair wall (perf gate)

| Twin | Wall [s] | Notes |
| ---- | -------: | ----- |
| CP2K OT+PLUS_U | **13.65** | `cp2k.psmp`, OMP=8 |
| ATV OT+U (2026-07-26) | **3.58** | Release, OMP=8, 23 OT iters |
| ATV DIIS+U (same run) | **3.03** | 65 DIIS iters |

\(r = T_{\mathrm{atv,OT}}/T_{\mathrm{cp2k}} = 3.58/13.65 \approx \mathbf{0.26}\) ≤ 10 ✓.

### Atomiverse (2026-07-26, Release, OMP=8)

| Quantity | Value |
|----------|--------|
| E_tot OT+U [Ha] | ≈ −184.94575 |
| \|ΔE\| vs CP2K | ≈ **2.1e-2 Ha** (within 5e-2) |
| \|E_OT − E_DIIS\| | ≈ **3.8e-4 Ha** |
| ⟨S²⟩ | ≈ 2.015 |
| m_Ni (Mulliken) | ≈ 1.00 |
| OT iters | 23 |

### Citations

- VandeVondele & Hutter, *J. Chem. Phys.* **118**, 4365 (2003) — OT.
- Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified U.
- Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
