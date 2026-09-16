# Reference — NiO molecule Γ-UKS GPW + Dudarev DFT+U analytic cell stress

## Method

Unrestricted Kohn–Sham GPW at the Γ point with Martyna–Tuckerman isolated
electrostatics, PADE LSD, and **Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken) on
Ni \(d\) (\(L=2\), \(U-J=6.0\) eV).

**Product cell stress** is the Hellmann–Feynman fixed-density-matrix path:

1. One UKS+DFT+U SCF at the unstrained cell → \(P_\alpha,P_\beta\) and
   \(W_\sigma=\sum_i^{n_\sigma}\varepsilon_i^\sigma c_i^\sigma(c_i^\sigma)^\top\)
   (eigenvalues include \(V_U\)).
2. Central finite difference of the **non-self-consistent** Lagrangian
   \(E_L[P_\alpha,P_\beta;W_\alpha,W_\beta;\varepsilon]\) under orthorhombic
   diagonal strain at fixed fractional coordinates, with
   \(E_U[n(P_\sigma,S)]\) evaluated at fixed spin density and strained
   overlap.
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; E Hartree, \(\Omega\) Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same UKS+U method.
Shear (xy/xz/yz) is **0** in v1 (orthorhombic cell).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `n_alpha=13`, `n_beta=11`,
`plus_u={Ni,L=2,U=6 eV}`, `compute_stress=true`, `stress_use_fd=false` →
Frame `kStress` [eV/Å³]; compare via `last_stress_ha_bohr3()` [Ha/Bohr³].

## System

Same geometry / settings as `gpw-dftu-nio`:

| Item | Value |
|------|--------|
| Molecule | NiO, **centered** in cell |
| Bond length | R(Ni–O) = **1.668 Å** (along *z*) |
| Coordinates (Å) | Ni (5, 5, 4.166), O (5, 5, 5.834) |
| Cell | **10 × 10 × 10 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT`) |
| Basis | Ni `SZV-GTH-PADE-q18`, O `SZV-GTH-PADE` |
| PP | Ni `GTH-PADE-q18` (18e), O `GTH-PADE-q6` (6e) → **24** valence e⁻ |
| XC | PADE (LDA LSD) |
| DFT+U | Ni \(L=2\), \(U-J = 6.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| Cutoff | 250 Ry |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=13, Nβ=11** |
| Stress step | \(h=2\times 10^{-3}\) |

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (same twin family as `gpw-dftu-nio`).
- **Energy pin** (FORCE_EVAL QS, unperturbed):
  \(E = -184.92466076474312\) Hartree
  (from `tools/cp2k_oracle/inputs/gpw-dftu-nio/nio_mol_pade_u.inp`, wall ≈ 15.79 s OMP=8).
- **Stress input (NUMERICAL):**
  `tools/cp2k_oracle/inputs/gpw-dftu-stress/nio_mol_pade_u_stress_num.inp`
- **Run (offline only; never in ctest):**
  ```bash
  export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
  export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
  OMP_NUM_THREADS=8 cp2k.psmp -i nio_mol_pade_u_stress_num.inp -o nio_stress.out
  ```
- **Pin host note (2026-07-26):** CP2K module **not available** on the worker
  login node at pin time. Absolute NUMERICAL Voigt digits are **not invented**.
  Decisive gate is **internal dual-gate** analytic vs multi-SCF FD; energy soft
  twin reuses the `gpw-dftu-nio` CP2K ENERGY pin. Replace stress absolute pin
  when CP2K is available.

## Tolerances

| Check | Tol |
| ----- | --- |
| max\|\σ_an − σ_fd\| diag | **2×10⁻⁴** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| \|E − E_cp2k\| soft | **5×10⁻²** Ha |
| max\|\σ_diag\| floor | > **1×10⁻⁸** Ha/Bohr³ |
| e_dftu | > **1×10⁻⁴** Ha |

## Atomiverse numbers (first green, 2026-07-26)

| Quantity | Value |
| --- | --- |
| **E (Γ-UKS PADE+U 250 Ry)** | **−184.9461084767 Ha** (SCF 78 iters) |
| \|E − E_cp2k\| | **2.15×10⁻² Ha** |
| e_dftu | **0.091249 Ha** |
| σ_xx analytic | 2.10983351e-07 Ha/Bohr³ |
| σ_yy analytic | 2.10983257e-07 Ha/Bohr³ |
| σ_zz analytic | −1.04886653e-05 Ha/Bohr³ |
| max\|σ_an−σ_fd\| | **8.92e-06** Ha/Bohr³ |
| ⟨S²⟩ | 2.0155 |
| T_product (OMP=8, Release) | **12.0 s** |
| T_fd_twin (OMP=8) | **24.7 s** |
| T_dual_gate wall | **~37 s** |

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary/module on the pin host for a fair NUMERICAL stress wall twin. Dual-gate wall is dominated by multi-SCF FD (6 SCFs), not product analytic. Fair product wall alone is ~12 s; sibling ENERGY CP2K wall on `gpw-dftu-nio` was ~15.79 s → product-only ratio ~0.76 if that wall is reused, but NUMERICAL stress is the true stress twin and was not run here. |
| r = T_atv/T_cp2k | n/a (exempt); product-only vs ENERGY twin ≈ 12/15.79 ≈ **0.76** |
| `within_10x` | n/a (exempt); product-only would be within 10× |

When CP2K is available, run `nio_mol_pade_u_stress_num.inp` single-rank, record both
walls for a product-only Evaluate pass, and replace this section (target
r ≤ 10 for analytic product vs fair ENERGY_FORCE twin).

## Provenance

- Spec: `.claude/specs/gpw-dftu-stress/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-dftu-stress/nio_mol_pade_u_stress_num.inp`
- Deps: `gpw-dftu-nio` ✓, `gpw-analytic-stress` ✓ / `gpw-uks-stress` ✓
- Energy digits: `regressions/gpw/gpw-dftu-nio/REFERENCE.md` (CP2K 2026.1)

## Re-pin 2026-07-31 — variational +U scatter

`gpw-dftu-kpoint-forces` fixed the Dudarev Fock contribution to the
variational form ∂E_U/∂P = ½(V̂S+SV̂) (the bare-V̂ scatter left the SCF fixed
point non-stationary for the reported E_U[n(P,S)]; CP2K uses the S-weighted
form). All +U fixed points moved; ATV-self absolute pins below were
re-measured on the new functional (identity/dual gates unchanged and tighter).

- No pin change; the Γ +U stress dual-gate residual improved to **8.4e-8 Ha/Bohr³** (unit) / 2.8e-6 (product) under the variational scatter.
