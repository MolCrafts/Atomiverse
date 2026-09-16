# Reference — Γ-point UKS GPW O₂ analytic forces (PBE / DZVP-MOLOPT-GTH)

## Method

Unrestricted Kohn–Sham GPW (Quickstep) at the Γ point with Martyna–Tuckerman
isolated electrostatics. Spin-resolved **PBE GGA**, GTH-PBE-q6 pseudopotential,
DZVP-MOLOPT-GTH Gaussian basis. Analytic ionic forces (Hellmann–Feynman + Pulay
with \(W_\alpha+W_\beta\) and spin-resolved grid \(V_\mathrm{eff}^{\alpha/\beta}\)).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `n_alpha=7`, `n_beta=5`,
`compute_forces=true`, `XcKind::PBE` → `gpw_forces_analytic` UKS branch.

## System

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

## Offline CP2K pin

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

## Tolerance

- Component force tol **5×10⁻³ Ha/Bohr** (first green): same-method twin
  (Quickstep GPW UKS); residual is dominated by Atomiverse **single density
  grid** vs CP2K **NGRIDS 4 / REL_CUTOFF 60** multigrid, not a missing force
  term. Unit gate `test_cpu_gpw_forces` (UKS H2 analytic vs FD) is **5e-5**
  Ha/Bohr on the ATV energy functional. Observed twin max|ΔF_z| ≈ **2.7e-3**
  with |F_z| ~ 0.035 Ha/Bohr and energy |ΔE| ≈ **6e-5** Ha.
- Energy |ΔE| **1×10⁻³ Ha** soft check.
- |∑F| residual **5×10⁻⁴ Ha/Bohr** (numerical translational residual).

## Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 77.3 | 1 process, OMP=4, ENERGY_FORCE |
| Atomiverse | **195** | Release, one UKS SCF + force assembly, OMP=4, 2026-07-25 |

`ratio = T_atv / T_cp2k = 195/77.3 ≈ **2.52**` (≤ 10 ✓).

## Citations

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep forces.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
