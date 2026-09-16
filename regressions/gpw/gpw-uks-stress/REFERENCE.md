# Reference — Γ-point UKS GPW O₂ analytic cell stress (PBE / DZVP-MOLOPT-GTH)

## Method

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
| Stress step | \(h=2\times 10^{-3}\) |

## Offline CP2K pin

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

## Tolerances

| Check | Tol |
| ----- | --- |
| max\|\σ_an − σ_fd\| diag | **1×10⁻⁴** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| \|E − E_cp2k\| soft | **1×10⁻³** Ha |
| max\|\σ_diag\| floor | > **1×10⁻⁸** Ha/Bohr³ |

Dual-gate tol is looser than closed-shell water (1e-5) because the O₂ UKS
SCF residual and single-grid vs multi-grid residual are larger; unit gate on
cheap high-spin H₂ uses 5e-5.

## Atomiverse numbers (first green, 2026-07-25)

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

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary/module on the pin host for a fair NUMERICAL stress wall twin. Dual-gate wall is dominated by multi-SCF FD (6 SCFs), not product analytic. Fair product wall alone is ~141 s; sibling ENERGY_FORCE CP2K wall on forces twin was ~77 s → product-only ratio ~1.8 if that wall is reused, but NUMERICAL stress is the true stress twin and was not run here. |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |

When CP2K is available, run `o2_uks_stress_num.inp` single-rank, record both
walls for a product-only Evaluate pass, and replace this section (target
r ≤ 10 for analytic product vs fair ENERGY_FORCE twin).

## Provenance

- Spec: `.claude/specs/gpw-uks-stress/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-uks-stress/o2_uks_stress_num.inp`
- Deps: `gpw-uks-forces` ✓, `gpw-analytic-stress` ✓
- Energy digits: `regressions/gpw/gpw-uks-forces/REFERENCE.md` (CP2K 2026.1)
