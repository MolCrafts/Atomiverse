# gpw/uks

One case; every section below is run by `atv_regressions --case gpw/uks` and keeps its own oracle record.


## Reference — Γ-point UKS GPW O₂ analytic forces (PBE / DZVP-MOLOPT-GTH)

### Method

Unrestricted Kohn–Sham GPW (Quickstep) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved **PBE GGA**, GTH-PBE-q6 pseudopotential,
DZVP-MOLOPT-GTH Gaussian basis. Analytic ionic forces (Hellmann–Feynman + Pulay
with \(W_\alpha+W_\beta\) and spin-resolved grid \(V_\mathrm{eff}^{\alpha/\beta}\)).

Atomiverse path: `cpu::gpw::GpwDft` with `n_alpha=7`, `n_beta=5`,
`XcKind::PBE`, moved into `cpu::gpw::GPWEvaluator` with
`compute_forces=true` → analytic-force UKS branch.

### System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (6 valence e⁻ / O → 12 total) |
| XC | **PBE** (GGA, spin-resolved) |
| Cutoff | 400 Ry |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), EasyBuild module stack on Arrhenius.
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-forces/o2_uks_forces.inp`
- **RUN_TYPE:** `ENERGY_FORCE`
- **SCF:** `EPS_SCF 1e-7`, Broyden α=0.4, DIAGONALIZATION STANDARD
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -n 1 cp2k.psmp -i o2_uks_forces.inp -o o2_uks_forces.out
  ```
- **Date:** 2026-07-25
- **Wall (CP2K, 1 process × 4 OMP):** ≈ **77.3 s** total

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.928797634435675** | **1e-3 Ha** |
| \(F_z\) O1 [Ha/Bohr] | **−3.50081576×10⁻²** | **5e-3** max component |
| \(F_z\) O2 [Ha/Bohr] | **+3.50081358×10⁻²** | same |
| \(F_x,F_y\) | ~1e-12 (numerical) | same |

Key log lines:

```
  Total energy:                                               -31.92879767674141
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.928797634435675
 FORCES| Atomic forces [hartree/bohr]
 FORCES|   Atom     x               y               z               |f|
 FORCES|      1  2.02069184E-12  2.05677955E-12 -3.50081576E-02   3.50081576E-02
 FORCES|      2  2.43439049E-12  2.40869221E-12  3.50081358E-02   3.50081358E-02
 FORCES| Total atomic force                                       2.17692380E-08
```

Hard-coded pins in `reg_gpw-uks-forces.cpp` (Ha/Bohr):

```text
O1:  2.02069184e-12,  2.05677955e-12, -3.50081576e-02
O2:  2.43439049e-12,  2.40869221e-12,  3.50081358e-02
E_tot_Ha: -31.928797634435675
```

### Tolerance

- Component force tol **5×10⁻³ Ha/Bohr** (first green): same-method twin
  (Quickstep GPW UKS); residual is dominated by Atomiverse **single density
  grid** vs CP2K **NGRIDS 4 / REL_CUTOFF 60** multigrid, not a missing force
  term. Unit gate `test_cpu_gpw_forces` (UKS H2 analytic vs FD) is **5e-5**
  Ha/Bohr on the ATV energy functional. Observed twin max|ΔF_z| ≈ **2.7e-3**
  with |F_z| ~ 0.035 Ha/Bohr and energy |ΔE| ≈ **6e-5** Ha.
- Energy |ΔE| **1×10⁻³ Ha** soft check.
- |∑F| residual **5×10⁻⁴ Ha/Bohr** (numerical translational residual).

### Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 77.3 | 1 process, OMP=4, ENERGY_FORCE |
| Atomiverse | **195** | Release, one UKS SCF + force assembly, OMP=4, 2026-07-25 |

`ratio = T_atv / T_cp2k = 195/77.3 ≈ **2.52**` (≤ 10 ✓).

### Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep forces.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.


## Reference — Γ-point UKS GPW O₂ analytic cell stress (PBE / DZVP-MOLOPT-GTH)

### Method

Unrestricted Kohn–Sham GPW (Quickstep) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved **PBE GGA**, GTH-PBE-q6, DZVP-MOLOPT-GTH.

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One UKS SCF at the unstrained cell → \(P_\alpha,P_\beta\) and
   \(W_\sigma=\sum_i^{n_\sigma}\varepsilon_i^\sigma c_i^\sigma(c_i^\sigma)^\top\).
2. Central finite difference of the **non-self-consistent** Lagrangian
   \(E_L[P_\alpha,P_\beta;W_\alpha,W_\beta;\varepsilon]\) under orthorhombic
   diagonal strain at fixed fractional coordinates (rebuild \(T\),
   \(V_{\mathrm{pp}}\), \(S\), grid, Poisson, spin XC; do **not** re-converge SCF).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same UKS method.
Shear (xy/xz/yz) is **0** in v1 (orthorhombic cell).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `n_alpha=7`, `n_beta=5`,
`compute_stress=true`, `stress_use_fd=false` → Frame `kStress` [eV/Å³];
compare via `last_stress_ha_bohr3()` [Ha/Bohr³].

### System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (6 valence e⁻ / O → 12 total) |
| XC | **PBE** (GGA, spin-resolved) |
| Cutoff | 400 Ry |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |
| Stress step | \(h=2\times 10^{-3}\) |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (same twin family as `gpw-uks-forces`).
- **Energy pin** (FORCE_EVAL QS, unperturbed; from `gpw-uks-forces` ENERGY_FORCE):
  \(E = -31.928797634435675\) Hartree
- **Stress input (NUMERICAL):**
  `tools/cp2k_oracle/inputs/gpw-uks-stress/o2_uks_stress_num.inp`
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -n 1 cp2k.psmp -i o2_uks_stress_num.inp -o o2_uks_stress_num.out
  ```
- **Pin host note (2026-07-25):** CP2K module **not available** on the worker
  login node at pin time. Absolute NUMERICAL Voigt digits are **not invented**.
  Decisive gate is **internal dual-gate** analytic vs multi-SCF FD; energy soft
  twin reuses the `gpw-uks-forces` CP2K ENERGY pin. Replace stress absolute pin
  when CP2K is available.

### Tolerances

| Check | Tol |
| ----- | --- |
| max\|\σ_an − σ_fd\| diag | **1×10⁻⁴** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| \|E − E_cp2k\| soft | **1×10⁻³** Ha |
| max\|\σ_diag\| floor | > **1×10⁻⁸** Ha/Bohr³ |

Dual-gate tol is looser than closed-shell water (1e-5) because the O₂ UKS
SCF residual and single-grid vs multi-grid residual are larger; unit gate on
cheap high-spin H₂ uses 5e-5.

### Atomiverse numbers (first green, 2026-07-25)

| Quantity | Value |
| --- | --- |
| **E (Γ-UKS PBE 400 Ry)** | **−31.9288547285 Ha** (SCF 60 iters) |
| \|E − E_cp2k\| | **5.7×10⁻⁵ Ha** |
| σ_xx analytic | 1.34582413e-08 Ha/Bohr³ |
| σ_yy analytic | 1.34582366e-08 Ha/Bohr³ |
| σ_zz analytic | 6.74708540e-06 Ha/Bohr³ |
| max\|σ_an−σ_fd\| | **5.0e-11** Ha/Bohr³ |
| ⟨S²⟩ | 2.0056 |
| T_product (OMP=8, Release) | **140.6 s** |
| T_fd_twin (OMP=8) | **236.5 s** |
| T_dual_gate wall | **377 s** |

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary/module on the pin host for a fair NUMERICAL stress wall twin. Dual-gate wall is dominated by multi-SCF FD (6 SCFs), not product analytic. Fair product wall alone is ~141 s; sibling ENERGY_FORCE CP2K wall on forces twin was ~77 s → product-only ratio ~1.8 if that wall is reused, but NUMERICAL stress is the true stress twin and was not run here. |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |

When CP2K is available, run `o2_uks_stress_num.inp` single-rank, record both
walls for a product-only Evaluate pass, and replace this section (target
r ≤ 10 for analytic product vs fair ENERGY_FORCE twin).

### Provenance

- Spec: `.claude/specs/gpw-uks-stress/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-uks-stress/o2_uks_stress_num.inp`
- Deps: `gpw-uks-forces` ✓, `gpw-analytic-stress` ✓
- Energy digits: `regressions/gpw/uks/REFERENCE.md` (CP2K 2026.1)


## Reference — Γ-point UKS GPW multigrid O₂ (PADE / DZVP-MOLOPT-GTH)

### Method

Unrestricted Kohn–Sham GPW at Γ with Martyna–Tuckerman isolated electrostatics
and **CP2K-style multigrid product collocation** (`ngrids=4`, `rel_cutoff=60`).
Spin-resolved PADE (GTH Padé LDA), GTH-PBE-q6, DZVP-MOLOPT-GTH.

Atomiverse path: `cpu::gpw::GPWEvaluator` / `GpwDft` with `n_alpha`/`n_beta`,
`ngrids`, `rel_cutoff` — dual KS matrices via product collocation +
`prolong_fft` / `restrict_fft` (same machinery as RKS multigrid).

### System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (6 valence e⁻ / O → 12 total) |
| XC | PADE (LDA LSD) |
| Cutoff | **400 Ry** |
| Multigrid | **NGRIDS 4**, **REL_CUTOFF 60** Ry (ATV + CP2K twin) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, git:5e54ba2), module `CP2K/2026.1`
  on Arrhenius (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-multigrid/o2_uks_pade_mg.inp`
  (identical physics twin to `gpw-uks-o2-gamma/o2_uks_pade.inp`: same geometry,
  CUTOFF 400 / NGRIDS 4 / REL_CUTOFF 60 / PADE / LSD / MT).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -np 1 cp2k.psmp -i o2_uks_pade_mg.inp -o o2_uks_pade_mg.out
  ```
- **Date:** 2026-07-23 (centered geometry; pin shared with `gpw-uks-o2-gamma`)
- **Wall (CP2K, 1 MPI × 4 OMP):** ≈ **21.3 s** total (`CP2K` timer MAXIMUM)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.76980349931425** | **1e-3 Ha** |
| ⟨S²⟩ (single det.) | **2.004659** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines (CP2K 2026.1):

```
Total energy:                                               -31.76980349931425
Ideal and single determinant S**2 :                    2.000000       2.004659
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.769803499314250
```

### Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 21.3 | 1 rank, OMP=4, NGRIDS 4 |
| Atomiverse | **32.8** | Release, ngrids=4, rel_cutoff=60, OMP=4 (2026-07-26) |

`ratio = T_atv / T_cp2k = 32.8/21.3 ≈ **1.54**` (≤ 10 ✓).

### Implementation note

ATV multigrid product collocation uses **spectral (FFT) inter-level transfer**
even under Martyna–Tuckerman nonperiodic Poisson: the density lives on a
uniform box grid and is well-contained for centered O₂, so FFT prolong/restrict
(the RKS product path) is the correct adjoint transfer; trilinear fallback on
`periodic=false` left an O(0.4 Ha) Hartree residual.

### Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- CP2K Quickstep `&MGRID` CUTOFF / REL_CUTOFF / NGRIDS multigrid collocation.


## Reference — Γ-point UKS GPW O₂ (PADE / DZVP-MOLOPT-GTH)

### Method

Unrestricted Kohn–Sham GPW (Quickstep LSD) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved PADE (Goedecker–Teter–Hutter Padé LDA),
GTH-PBE-q6 pseudopotential, DZVP-MOLOPT-GTH Gaussian basis.

Atomiverse path: `cpu::gpw::GpwDft` with `n_alpha`/`n_beta`, dual KS matrices,
`xc_grid_pade_lsd`, Löwdin ⟨S²⟩.

### System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell (avoid MT edge truncation) |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (6 valence e⁻ / O → 12 total) |
| XC | PADE (LDA LSD) |
| Cutoff | 400 Ry (Atomiverse single density grid; CP2K twin also NGRIDS 4 / REL_CUTOFF 60) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, git:5e54ba2), module `CP2K/2026.1`
  on Arrhenius (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-o2-gamma/o2_uks_pade.inp`
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  mpirun -np 1 cp2k.psmp -i o2_uks_pade.inp -o o2_uks_pade.out
  ```
- **Date:** 2026-07-23 (centered geometry re-pin)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.76980349931425** | **1e-3 Ha** (first green; single-grid vs CP2K multigrid) |
| ⟨S²⟩ (single det.) | **2.004659** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |
| Integrated spin density | 2.000000 | — |

Key log lines:

```
Total energy:                                               -31.76980349931425
Ideal and single determinant S**2 :                    2.000000       2.004659
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.769803499314250
```

### Perf gate

This file carries no wall numbers. The single binding fair-wall source for
this twin — host, CP2K binary provenance, PRINT_LEVEL/timer convention,
reproduction recipe, and the current \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) —
is [`.claude/notes/gpw-perf-baseline.md`](../../../.claude/notes/gpw-perf-baseline.md).
Case-local walls were last measured here 2026-07-23 and are superseded by
that note.

### Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — ⟨S²⟩ for unrestricted single dets.


## Reference — Γ-point UKS GPW O₂ (PBE / DZVP-MOLOPT-GTH)

### Method

Unrestricted Kohn–Sham GPW (Quickstep LSD) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved **PBE GGA**, GTH-PBE-q6 pseudopotential,
DZVP-MOLOPT-GTH Gaussian basis.

Atomiverse path: `cpu::gpw::GpwDft` / `GPWEvaluator` with `n_alpha`/`n_beta`,
`XcKind::PBE`, dual KS matrices, `xc_grid_pbe_spin`, Löwdin ⟨S²⟩.

### System

| Item | Value |
|------|--------|
| Molecule | O₂, **centered** in cell (avoid MT edge truncation) |
| Bond length | R(O–O) = **1.2075 Å** (along *z*) |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 (6 valence e⁻ / O → 12 total) |
| XC | **PBE** (GGA, spin-resolved) |
| Cutoff | 400 Ry (Atomiverse single density grid; CP2K twin also NGRIDS 4 / REL_CUTOFF 60) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=7, Nβ=5** |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`), module `CP2K/2026.1` on Arrhenius
  (GCC 14.3 / MPICH 4.3.2).
- **Input:** `tools/cp2k_oracle/inputs/gpw-uks-pbe/o2_uks_pbe.inp`
- **SCF:** `EPS_SCF 1e-6`, Broyden α=0.4, DIAGONALIZATION STANDARD (matches
  energy to 1e-4 Ha of a 1e-7 twin; used for fair wall under login load).
- **Run (offline only; never in ctest):**
  ```bash
  module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
  export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
  export OMP_NUM_THREADS=4
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
  cp2k.psmp -i o2_uks_pbe.inp -o o2_uks_pbe.out
  ```
- **Date:** 2026-07-25
- **Wall (CP2K, 1 process × 4 OMP):** ≈ **207 s** total (`CP2K` timer MAXIMUM)

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−31.92879303435954** | **1e-3 Ha** (first green; single-grid vs CP2K multigrid) |
| ⟨S²⟩ (single det.) | **2.005556** | **0.15** |
| Ideal S(S+1) | 2.000000 | — |

Key log lines:

```
  *** SCF run converged in    14 steps ***
  Total energy:                                               -31.92879303435954
  Ideal and single determinant S**2 :                    2.000000       2.005556
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.928793034359536
```

### Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 207 | 1 process, OMP=4, NGRIDS 4 multigrid |
| Atomiverse | **177** | Release, single density grid, OMP=4, 2026-07-25 |

`ratio = T_atv / T_cp2k = 177/207 ≈ **0.85**` (≤ 10 ✓).

### Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — ⟨S²⟩ for unrestricted single dets.


## REFERENCE — gpw-uks-metallic-stress

### Oracle

| Field | Value |
| --- | --- |
| Code | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Twin | `tools/cp2k_oracle/inputs/gpw-uks-metallic-stress/na_bcc_uks_smear_stress.inp` |
| Class | Quickstep LSD UKS + SMEAR FERMI_DIRAC + `STRESS_TENSOR NUMERICAL` |
| System | BCC Na, GTH-PBE-q9 / DZVP-MOLOPT-SR-GTH, PADE, MP 2×2×2 |

### Pins

| Quantity | Value | Tol |
| --- | --- | --- |
| Free energy \(F\) | −92.126330608004 Ha | \(10^{-3}\) Ha |
| Analytic vs FD max\|Δσ_diag\| | product | \(2\times10^{-4}\) Ha/Bohr³ |
| Closed-shell UKS ≡ RKS σ | product identity | \(5\times10^{-5}\) Ha/Bohr³ |
| Mean σ_diag | −1.9177e-2 Ha/Bohr³ (sibling metallic-stress) | \(5\times10^{-4}\) |
| CP2K NUMERICAL class | −1.705e-2 Ha/Bohr³ | \(5\times10^{-3}\) soft |

### Fair wall

| Side | Wall |
| --- | --- |
| ATV SCF (OMP=8, first green 2026-08-04) | 0.78 s |
| CP2K NUMERICAL stress twin | class of metallic-stress ENERGY wall (~0.9 s); gate \(r\le 10\) |


## `cuda-gpw-uks-o2-forces`

### What

Γ collinear UKS O₂ **analytic forces + cell stress** dual-gate. Energy parent
is `cuda-gpw-uks-o2-gate`. CUDA SCF payload + host `gpw_forces_from_scf` /
`gpw_stress` assembly.

### Twin

Same-settings CPU O₂ triplet (n_α=7, n_β=5, SZV-GTH-PADE, 12 Bohr, 100 Ry,
Padé LSD). Oracle: CPU twin.

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|dE\| | **5×10⁻⁵ Ha** |
| max\|ΔF\| | **5×10⁻⁴ Ha/Bohr** |
| max‖Δσ‖ | **5×10⁻⁵ Ha/Bohr³** |

**Claim: ◐** — CPU-twin dual-gate; not an independent CP2K pin.
**perf_exempt**.


## `cuda-gpw-uks-o2-gate`

### What

Production-class Γ collinear UKS **energy** dual-gate on public
`ATV::cuda::gpw::GpwDft` (beyond the H₂ UKS foothold).
O₂ triplet (n_α=7, n_β=5), Padé LSD, DIIS, single-grid. Device mid-SCF
reuses `KsChainDevice::run_uks` — **no new kernel**.

### Twin

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

### Gates

| Gate | Tol |
| --- | ---: |
| Energy \|E_cuda − E_cpu\| | **5×10⁻⁵ Ha** |
| Both SCF converged | required |
| CUDA ⟨S²⟩ | > 1.5 |
| Force dual-gate | **not** this slug (`cuda-gpw-uks-forces`) |

### Perf

**perf_exempt** — small molecule correctness-only.

### Non-goals

UKS forces/stress, UKS-PBE, multi-k / multigrid UKS, OT, MT, device DIIS/eigen.

### GH200 evidence

Job **1127173** (2026-08-10, node n87 aarch64 GH200):

- `test_cuda_gpw_full_product` PASS (2.92 s; includes `O2UksPadeEnergyDualGate`)
- dual-gate:
  - CPU  E=−31.422608201904 Ha  ⟨S²⟩=2.000448  iters=24
  - CUDA E=−31.422604681912 Ha  ⟨S²⟩=2.000448  iters=70
  - \|ΔE\|=**3.520e-06** Ha (tol 5e-5)
- `GPU_JOB_OK`; `builds_wiped=true`
