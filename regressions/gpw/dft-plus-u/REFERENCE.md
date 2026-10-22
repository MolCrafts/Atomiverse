# gpw/dft-plus-u

One case; every section below is run by `atv_regressions --case gpw/dft-plus-u` and keeps its own oracle record.


## Reference — NiO molecule Γ-UKS GPW + Dudarev DFT+U (PADE / SZV-GTH-PADE)

### Method

Unrestricted Kohn–Sham GPW at the Γ point with Martyna–Tuckerman isolated
electrostatics, PADE LSD, and **Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken) on
Ni \(d\) (\(L=2\), \(U-J=6.0\) eV).

Atomiverse path: `cpu::gpw::GPWEvaluator` with `n_alpha`/`n_beta`, `plus_u`,
single density grid.

**Note:** solid rocksalt AFM NiO (2 f.u.) did **not** converge under SZV /
Γ-only in CP2K (diag+Broyden or no-U restart) within 200–300 steps. The
**molecule** is the first-green PLUS_U foothold; solid AFM is follow-on
(k-points / larger basis).

### System

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
| Cutoff | 250 Ry (Atomiverse single grid; CP2K NGRIDS 4 / REL_CUTOFF 40) |
| Spin | UKS / LSD, **multiplicity 3** → **Nα=13, Nβ=11** |

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, Source code revision 85b8a9b49c),
  local install under `/nobackup/.../work/cp2k/install` (GCC 11.5, OpenMP).
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-nio/nio_mol_pade_u.inp`
- **Run (offline only; never in ctest):**
  ```bash
  export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
  export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
  cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
  OMP_NUM_THREADS=8 cp2k.psmp -i nio_mol_pade_u.inp -o nio_mol.out
  ```
- **Date:** 2026-07-24
- **CP2K SCF:** OT `FULL_SINGLE_INVERSE` + OUTER_SCF (diag+Broyden did not
  converge this twin within budget).
- **Wall (CP2K, OMP=8):** ≈ **15.79 s**

| Quantity | CP2K pin | Tolerance (regression) |
|----------|----------|------------------------|
| E_tot [Ha] | **−184.92466076474312** | **5e-2 Ha** (single-grid vs multigrid+OT) |
| DFT+U energy [Ha] | **0.05230865222345** | diagnostic (not asserted) |
| ⟨S²⟩ (single det.) | **2.019198** | **0.25** |
| Ideal S(S+1) | 2.000000 | — |
| Mulliken spin m_Ni | **1.177632** | **0.5** (and m_Ni > 0.3) |
| Mulliken spin m_O | 0.822368 | — |

Key log lines:

```
  DFT+U energy:                                                 0.05230865222345
  Total energy:                                              -184.92466076474312
  Ideal and single determinant S**2 :                    2.000000       2.019198
  Integrated absolute spin density:                                 2.0728941742
       1     Ni       1         9.343192     8.165560     0.491248     1.177632
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.924660764743123
```

### Atomiverse (2026-07-24, Release, OMP=8)

| Quantity | Value |
|----------|--------|
| E_tot [Ha] | ≈ −184.94637 |
| \|ΔE\| vs CP2K | ≈ **2.2e-2 Ha** (within 5e-2) |
| ⟨S²⟩ | ≈ 2.016 |
| m_Ni (Mulliken) | ≈ 1.07 |
| SCF iters | ≈ 56 |
| Wall | ≈ **3.3 s** |

### Perf gate

| Side | Wall [s] | Notes |
|------|----------|--------|
| CP2K 2026.1 | 15.79 | OMP=8, OT + OUTER_SCF, multigrid |
| Atomiverse | 3.29 | Release, single density grid, OMP=8 |

`ratio = T_atv / T_cp2k = 3.29/15.79 ≈ **0.21**` (≤ 10 ✓).

### Citations

- V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991) — DFT+U.
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified rotationally invariant U.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.


## Reference — Γ-UKS GPW analytic ionic forces with Dudarev DFT+U

### Method

Dudarev occupation \(n^\sigma=\tfrac12\mathrm{Herm}[(P^\sigma S)+(S P^\sigma)]\)
restricted to the correlated (atom, ℓ) block depends on geometry through \(S\),
so DFT+U contributes an **occupation-matrix force term**

\[
dE_U/dR=\sum_\sigma\mathrm{Tr}\big[M^\sigma\,dS/dR\big],
\qquad M^\sigma=\tfrac12\big(\hat V^\sigma P^\sigma+P^\sigma\hat V^\sigma\big),
\]

folded as \(W\leftarrow W-M\) into the existing Pulay \(W\!\cdot\!dS\)
contraction (`internal/dft_plus_u.h`, `subtract_dftu_pulay` on the Γ path).

Product path already packs `cfg.plus_u` in `gpw_forces.cpp` (Γ and multi-k).
This slug is the **dedicated Γ ENERGY_FORCE + PLUS_U pin** (multi-k dual gate
lives in `gpw-dftu-kpoint-forces`).

### Regression system A — Γ dual gate (decisive force)

| Item | Value |
|------|--------|
| Molecule | H₂ high-spin (Nα=2, Nβ=0), compressed bond 1.0 Bohr along x |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic 12 Bohr, periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1, PADE LSD, 100 Ry, `local_pp_on_grid` |
| k | Γ only |
| DFT+U | H L=0, **U−J = 2.0 eV** |

U−J = 2 eV deliberately (same as multi-k force pin): at 4 eV this DZV s-shell
toy is multi-branch under gross-Mulliken; see
`.claude/notes/gpw-dftu-projector-divergence.md`.

#### Measured (first green 2026-08-01, Release, OMP=8)

| Gate | Value | Tol |
|------|-------|-----|
| A1 E(Γ-UKS+U, 100 Ry) | −0.611955472155 Ha (e_dftu −0.10209039) | payload sanity |
| A2 max\|F_an − F_fd\| (full 6N `gpw_forces_fd`, h=1e-3) | **6.9×10⁻⁶ Ha/Bohr** | 5×10⁻⁵ |
| A3 \|ΣF\| / Newton-3 | 2.2×10⁻¹³ | 1×10⁻⁶ |
| A4 max\|F(U)−F(U=0)\| | 5.4×10⁻² Ha/Bohr | ≥ 1×10⁻⁴ |

### Regression system B — CP2K ENERGY_FORCE pin (NiO d shell)

Periodic Γ NiO stretched molecule: 10 Å cubic box (PBC), Ni (5,5,4.1) Å,
O (5,5,5.9) Å (R = 1.8 Å), SZV-GTH-PADE(-q18) / GTH-PADE-q18/q6, PADE LSD,
Nα=13 / Nβ=11, PLUS_U Mulliken Ni d U−J = 6 eV.

#### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, git:**5e54ba2**), local install
  `/nobackup/.../work/cp2k/install`, OMP=8, `mpirun -np 1`.
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-forces/nio_pbc_gamma_ot_force.inp`
  (OT FULL_SINGLE_INVERSE + OUTER_SCF, multigrid 250 Ry / REL 40 / NGRIDS 4,
  `RUN_TYPE ENERGY_FORCE`).
- **Date:** 2026-08-01. Wall **13.24 s** (`/usr/bin/time`, arrhenius1).
- **Host run dir:** `/tmp/atv-gpw-dftu-forces` (symlink BASIS_SET +
  GTH_POTENTIALS from `CP2K_DATA_DIR`).
- **Output lines:**

```
  *** SCF run converged in    14 steps ***
  DFT+U energy:                                                 0.04952309241681
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.929424563910430
 FORCES|      1  3.17871301E-07 -1.91561767E-06 -8.35837089E-03   8.35837111E-03
 FORCES|      2 -3.55176558E-08  2.19071246E-06 -7.27092413E-03   7.27092447E-03
 FORCES| Sum     2.82353646E-07  2.75094789E-07 -1.56292950E-02
```

#### Asserted vs diagnostic

| Check | Status | Value / tol |
|-------|--------|-------------|
| \|E_ATV − E_CP2K\| | **asserted** ≤ 5×10⁻² Ha | **1.18×10⁻²** (single grid vs multigrid+OT) |
| transverse F ≈ 0 | **asserted** < 5×10⁻⁴ | ~1×10⁻⁸ |
| ΔF_z = F_z(Ni)−F_z(O) | **diagnostic** | ATV +8.97×10⁻², CP2K −1.09×10⁻³ |

**Projector-model divergence (not a bug):** CP2K `PLUS_U_METHOD MULLIKEN`
projects on the on-site **net** AO block \(q_{ab}=P_{ab}S_{ab}\)
(geometry-independent ⇒ no occupation force), while ATV projects on the
symmetrized **gross** product \(\tfrac12(PS+SP)|_\mathrm{block}\). Detail:
`.claude/notes/gpw-dftu-projector-divergence.md`. Decisive force correctness
is the A2 full-6N FD dual gate.

### Performance

| Item | Value |
|------|-------|
| CP2K ENERGY_FORCE twin (NiO, OMP=8, 1 MPI) | **13.24 s** |
| ATV NiO SCF + analytic forces (section B, OMP=8, Release) | **32.88 s** |
| ratio T_atv / T_cp2k | **2.483** (≤ 10 ✓) |
| ATV regression total (A1–A4 + B, OMP=8) | **37.6 s** |

Fair wall is the **NiO SCF + force assembly only** (same twin job class as
CP2K ENERGY_FORCE), not the multi-gate regression total (which includes
full-6N FD).

### Citations

- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — U−J functional.
- VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter, *Comput. Phys.
  Commun.* **167**, 103 (2005) — Quickstep GPW forces.
- CP2K 2026.1 `src/dft_plus_u.F` — Mulliken / Mulliken-charges +U forms.


## Reference — NiO molecule Γ-UKS GPW + Dudarev DFT+U analytic cell stress

### Method

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

### System

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

### Offline CP2K pin

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

### Tolerances

| Check | Tol |
| ----- | --- |
| max\|\σ_an − σ_fd\| diag | **2×10⁻⁴** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| \|E − E_cp2k\| soft | **5×10⁻²** Ha |
| max\|\σ_diag\| floor | > **1×10⁻⁸** Ha/Bohr³ |
| e_dftu | > **1×10⁻⁴** Ha |

### Atomiverse numbers (first green, 2026-07-26)

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

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary/module on the pin host for a fair NUMERICAL stress wall twin. Dual-gate wall is dominated by multi-SCF FD (6 SCFs), not product analytic. Fair product wall alone is ~12 s; sibling ENERGY CP2K wall on `gpw-dftu-nio` was ~15.79 s → product-only ratio ~0.76 if that wall is reused, but NUMERICAL stress is the true stress twin and was not run here. |
| r = T_atv/T_cp2k | n/a (exempt); product-only vs ENERGY twin ≈ 12/15.79 ≈ **0.76** |
| `within_10x` | n/a (exempt); product-only would be within 10× |

When CP2K is available, run `nio_mol_pade_u_stress_num.inp` single-rank, record both
walls for a product-only Evaluate pass, and replace this section (target
r ≤ 10 for analytic product vs fair ENERGY_FORCE twin).

### Provenance

- Spec: `.claude/specs/gpw-dftu-stress/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-dftu-stress/nio_mol_pade_u_stress_num.inp`
- Deps: `gpw-dftu-nio` ✓, `gpw-analytic-stress` ✓ / `gpw-uks-stress` ✓
- Energy digits: `regressions/gpw/dft-plus-u/REFERENCE.md` (CP2K 2026.1)

### Re-pin 2026-07-31 — variational +U scatter

`gpw-dftu-kpoint-forces` fixed the Dudarev Fock contribution to the
variational form ∂E_U/∂P = ½(V̂S+SV̂) (the bare-V̂ scatter left the SCF fixed
point non-stationary for the reported E_U[n(P,S)]; CP2K uses the S-weighted
form). All +U fixed points moved; ATV-self absolute pins below were
re-measured on the new functional (identity/dual gates unchanged and tighter).

- No pin change; the Γ +U stress dual-gate residual improved to **8.4e-8 Ha/Bohr³** (unit) / 2.8e-6 (product) under the variational scatter.


## Reference — multi-k GPW analytic ionic forces with Dudarev DFT+U

### Method

The Dudarev occupation \(n^\sigma=\sum_k w_k\,\tfrac12\mathrm{Herm}[(P_k^\sigma
S_k)+(S_k P_k^\sigma)]\) restricted to the correlated (atom, ℓ) block depends
on geometry through \(S_k\), so DFT+U contributes an **occupation-matrix force
term**

\[
dE_U/dR=\sum_\sigma\sum_k w_k\,\mathrm{Re\,Tr}\big[M_k^\sigma\,dS_k/dR\big],
\qquad M_k^\sigma=\tfrac12\big(\hat V^\sigma P_k^\sigma+P_k^\sigma\hat V^\sigma\big),
\]

folded as \(W_k\leftarrow W_k-M_k\) into the existing Pulay \(W\!\cdot\!dS_k\)
contraction (Γ and multi-k alike; `internal/dft_plus_u.h`,
`subtract_dftu_pulay[_k]`). This mirrors CP2K's own device for its
geometry-dependent Mulliken-charges +U (`dft_plus_u.F`: `matrix_w`
correction \(w\mathrel{+}= -\tfrac12(dE/dq_i+dE/dq_j)P_{ij}\)).

**Variational scatter fix (2026-07-31, this delivery):** the exact
\(\partial E_U/\partial P\) for this occupation is
\(\tfrac12(\hat VS+S\hat V)\) — the previous bare \(\hat V\) scatter left the
SCF fixed point non-stationary for the reported \(E_U[n(P,S)]\), which made
analytic and multi-SCF FD forces disagree at O(1) Ha/Bohr on the toy (and was
the ~9×10⁻⁴ Ha/Bohr³ residual in the +U stress dual gates). CP2K applies the
same S-weighted form (`h_block += ½(dEdq_i+dEdq_j)·S_ij`). All +U SCF fixed
points moved; the sibling +U regressions were re-pinned in the same change
(each pin cites its pre-fix value inline).

### Regression system A — multi-k dual gate (decisive)

| Item | Value |
|------|--------|
| Molecule | H₂ high-spin (Nα=2, Nβ=0), compressed bond 1.0 Bohr along x |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic 12 Bohr, periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1, PADE LSD, 100 Ry, `local_pp_on_grid` |
| k mesh | MONKHORST-PACK 1×1×2, TR-folded |
| DFT+U | H L=0, **U−J = 2.0 eV** |

U−J = 2 eV deliberately: at 4 eV this DZV s-shell toy sits in the
strong-overlap regime where the gross-Mulliken occupation is unphysical
(CP2K's own `PLUS_U` warns "unphysical Mulliken charges" there) and the displaced multi-SCF FD oracle hops between SCF
branches — no dual gate can hold on a multi-valued surface. At 2 eV the SCF
is single-branch.

#### Measured (first green 2026-07-31, Release, OMP=8)

| Gate | Value | Tol |
|------|-------|-----|
| A1 E(k-UKS+U 1×1×2, 100 Ry) | −0.611940650579 Ha (e_dftu −0.10209345) | payload sanity |
| A2 max\|F_an − F_fd\| (full 6N `gpw_forces_fd`, h=1e-3) | **6.4×10⁻⁶ Ha/Bohr** | 5×10⁻⁵ |
| A3 \|ΣF\| / Newton-3 | 1.1×10⁻¹⁴ | 1×10⁻⁶ |
| A4 max\|F(U)−F(U=0)\| | 5.4×10⁻² Ha/Bohr | ≥ 1×10⁻⁴ (U term live) |
| A5 max\|F_kΓ − F_Γ\| (explicit k=Γ ≡ real Γ fold) | 6.0×10⁻⁸ Ha/Bohr | 5×10⁻⁶ |

Pre-fold control: without the W-fold the same A2 gate reads ~1.6 Ha/Bohr
(wrong sign) — the occupation-matrix term is decisive, four orders above the
gate residual.

### Regression system B — CP2K ENERGY_FORCE pin (NiO d shell)

Periodic Γ NiO stretched molecule: 10 Å cubic box (PBC), Ni (5,5,4.1) Å,
O (5,5,5.9) Å (R = 1.8 Å), SZV-GTH-PADE(-q18) / GTH-PADE-q18/q6, PADE LSD,
Nα=13 / Nβ=11, PLUS_U Mulliken Ni d U−J = 6 eV.

#### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, local install
  `/nobackup/.../work/cp2k/install`), OMP=8.
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-kpoint-forces/nio_pbc_gamma_ot_force.inp`
  (OT FULL_SINGLE_INVERSE + OUTER_SCF, multigrid 250 Ry / REL 40 / NGRIDS 4,
  `RUN_TYPE ENERGY_FORCE`).
- **Date:** 2026-07-31. Wall **12.84 s**.
- **Output lines:**

```
  *** SCF run converged in    14 steps ***
  DFT+U energy:                                                 0.04952309241684
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.929424563910374
 FORCES|      1  3.17871282E-07 -1.91561775E-06 -8.35837089E-03   8.35837111E-03
 FORCES|      2 -3.55176272E-08  2.19071254E-06 -7.27092413E-03   7.27092447E-03
 FORCES| Sum     2.82353655E-07  2.75094796E-07 -1.56292950E-02
```

(ΣF_z = −1.56×10⁻² Ha/Bohr is the multigrid eggbox background common to both
atoms at 250 Ry with the hard Ni q18 PP.)

#### Asserted vs diagnostic

| Check | Status | Value |
|-------|--------|-------|
| \|E_ATV − E_CP2K\| | **asserted** ≤ 5×10⁻² Ha | 1.18×10⁻² (single grid vs multigrid+OT) |
| transverse F ≈ 0 | **asserted** < 5×10⁻⁴ | ~1×10⁻⁸ |
| ΔF_z = F_z(Ni)−F_z(O) vs CP2K | **diagnostic** | ATV +8.97×10⁻², CP2K −1.09×10⁻³ |

#### Offline Γ d-shell FD validation (2026-07-31, not in ctest budget)

NiO +U DIIS needs `tol = 1e-8` (~480 iters/SCF, minutes) before FD noise
drops below the force signal — at the in-budget tol 1e-6 the ± states are
only consistent to ~1e-5 Ha and the 2h=4×10⁻³ divisor amplifies that to
~1e-2 Ha/Bohr of FD noise.

| Quantity (tol 1e-8, h = 2×10⁻³ Bohr, Ni z) | Value [Ha/Bohr] |
|---|---|
| analytic F_z(Ni) | +3.76808342×10⁻² |
| central FD | +3.80477507×10⁻² |
| \|analytic − FD\| | **3.7×10⁻⁴** |

(h = 4×10⁻³ left the −h SCF unconverged at 600 iters — reported for
completeness, not used.) The in-regression decisive force gate is the H₂
full-6N multi-k dual gate A2 (6.4×10⁻⁶ Ha/Bohr).

**Projector-model divergence (surfaced, out of scope here):** CP2K
`PLUS_U_METHOD MULLIKEN` projects on the on-site **net** AO block
\(q_{ab}=P_{ab}S_{ab}\) (elementwise; geometry-independent ⇒ **no**
occupation force term), while ATV projects on the symmetrized **gross**
product \(\tfrac12(PS+SP)\big|_\mathrm{block}\) (geometry-dependent ⇒ the
large ΔF_z above is the model's occupation force, confirmed against ATV's own
FD, not a bug). Aligning ATV's projector with CP2K's (or adding it as a
second method) is a census decision — it rewrites every +U SCF fixed point.

#### Multi-k CP2K twin attempts (documented)

- `nio_pbc_kpoint_force.inp` (MP 1×1×2, diag + Broyden, ATOMIC guess,
  MAX_SCF 300): oscillates, `SCF run NOT converged` (closest approach 1.9×10⁻⁴
  at step 238). Restart from its own `.kp` (ALPHA 0.05, MAX_SCF 600): still
  oscillating — the degenerate minority-spin d manifold of the linear molecule
  defeats fixed-multiplicity diagonalization (OT is not available with
  k-points in CP2K). Same pathology class as documented for
  `gpw-dftu-kpoint-stress`. Multi-k force correctness is
  therefore gated by the internal A2 full-6N FD dual gate + the A5 Γ-limit
  identity, with the Γ d-shell CP2K pin as the external anchor.

### Performance

| Item | Value |
|------|-------|
| ATV regression total (A1–A5 incl. full-6N FD + B, OMP=8, Release) | **38.6 s** |
| ATV B section alone (NiO SCF + analytic forces, 250 Ry single grid) | ≈ 13 s |
| CP2K ENERGY_FORCE twin (same system, multigrid, OMP=8) | 12.84 s |
| ratio T_atv/T_cp2k (comparable single-point + forces) | **≈ 1.0** (≤ 10 ✓) |

### Citations

- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — U−J functional.
- VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter, *Comput. Phys.
  Commun.* **167**, 103 (2005) — Quickstep GPW forces.
- CP2K 2026.1 `src/dft_plus_u.F` — Mulliken / Mulliken-charges +U forms
  (S-weighted KS contribution and `matrix_w` correction).


## Reference — multi-k UKS GPW + Dudarev DFT+U analytic cell stress

### Method

Unrestricted Kohn–Sham GPW with **Monkhorst–Pack k-point sampling**,
**Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken), and **Hellmann–Feynman
fixed-\(\{P_k^\sigma,W_k^\sigma\}\)** analytic cell stress:

1. One multi-k UKS+DFT+U SCF → occupied Bloch MOs and band energies
   (eigenvalues include \(V_U\)) → rebuild \(P_k^\sigma\), \(W_k^\sigma\).
2. Central FD of the non-self-consistent Lagrangian
   \(E_L[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon]\) under orthorhombic
   diagonal strain at fixed fractional coordinates, with BZ-averaged
   \(E_U[n^\sigma(\{P_k^\sigma\},\{S_k\})]\) at strained \(S_k\).
3. \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
   (CP2K `STRESS_TENSOR` sign; Ha/Bohr³).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) on the same method.
Γ-limit identity: complex \(k=\Gamma\) + U stress ≡ real Γ + U stress.

Public API: `GpwDft::stress(frame)` (members `stress_strain_h`, `k_mesh`,
`n_alpha`/`n_beta`, `plus_u`; returns eV/Å³, compared in Ha/Bohr³). No public
`*Config`, no factories.

### System

| Item | Value |
|------|--------|
| Molecule | H₂ high-spin (M = 3), compressed bond along *x* |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic **12 Bohr**, periodic XYZ |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| XC | PADE LSD |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| Cutoff | 100 Ry, `local_pp_on_grid` |
| k mesh | MONKHORST-PACK 1×1×2 (TR-folded, nk=1) |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) |
| Stress step | \(h=2\times 10^{-3}\) |

### Offline CP2K twin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`).
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/h2_uks_dftu_kpoint_stress.inp`
  (NUMERICAL stress + PLUS_U + KPOINTS; 8 Bohr cell variant for wall estimate).
- **Date:** 2026-07-26
- **Result:** same class of pathology as the multi-k H₂+U energy toy — **unphysical
  Mulliken charges**, large negative DFT+U, SCF fails to converge within
  MAX_SCF=100 (abort). **Not** used as absolute stress/energy twin.
- **Decisive gates:** internal dual-gate + Γ-limit identity + U activation.

```
  DFT+U energy: ~ −175 Ha (unphysical)
  Total energy: ~ −174 Ha
  *** SCF run NOT converged ***
```

### Atomiverse first-green (2026-07-26)

| Quantity | Value |
|----------|--------|
| **E (k-UKS+U 1×1×2, 100 Ry)** | **−0.546250995860 Ha** (76 iters, nk=1) |
| e_dftu | **−0.243601 Ha** (multi-function s shell; can be <0) |
| σ_xx analytic | −3.00982070e-04 Ha/Bohr³ |
| σ_yy analytic | −6.59874084e-05 Ha/Bohr³ |
| σ_zz analytic | −6.58998481e-05 Ha/Bohr³ |
| max\|σ_an−σ_fd\| | **9.33×10⁻⁴** Ha/Bohr³ |
| max\|σ_kΓ − σ_Γ\| | **1.3×10⁻⁸** Ha/Bohr³ |
| \|E(U)−E(0)\| | **1.79×10⁻²** Ha |
| Wall (OMP=8, Release) | **~2.4 s** |

### Tolerances

| Check | Tol |
| ----- | --- |
| max\|σ_an − σ_fd\| diag | **1×10⁻³** Ha/Bohr³ |
| off-diagonal product | **1×10⁻⁸** |
| max\|σ_kΓ − σ_Γ\| | **5×10⁻⁴** Ha/Bohr³ |
| \|E − pin\| soft | **1×10⁻³** Ha |
| \|E(U)−E(0)\| floor | **1×10⁻⁴** Ha |
| max\|σ_diag\| floor | > **1×10⁻⁸** Ha/Bohr³ |

Bond-axis residual (xx) dominates the dual-gate; transverse axes agree at
~1e-6. Nonlinear Mulliken \(n(S)\) + large \|e_dftu\| on multi-function s
shells make the multi-SCF FD noisier than bare multi-k UKS (~1e-10). Γ-limit
identity proves the multi-k \(E_U\) Lagrangian path matches the shipped Γ
DFT+U stress.

### Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No fair absolute CP2K NUMERICAL stress twin (unphysical Mulliken + SCF non-convergence on H₂+U multi-k toy). Dual-gate wall ~2.4 s is product+FD, not a fair oracle ratio. |
| r = T_atv/T_cp2k | n/a (exempt) |
| `within_10x` | n/a (exempt) |

### Provenance

- Spec: `.claude/specs/gpw-dftu-kpoint-stress/`
- Twin input: `tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/h2_uks_dftu_kpoint_stress.inp`
- Related: `gpw-dftu-stress`, `gpw-dftu-kpoint-forces`, `gpw-kpoint-stress`
- Citations: Dudarev *et al.*, PRB **57**, 1505 (1998); Nielsen & Martin, PRB **32**, 3780 (1985).

### Re-pin 2026-07-31 — variational +U scatter

`gpw-dftu-kpoint-forces` fixed the Dudarev Fock contribution to the
variational form ∂E_U/∂P = ½(V̂S+SV̂) (the bare-V̂ scatter left the SCF fixed
point non-stationary for the reported E_U[n(P,S)]; CP2K uses the S-weighted
form). All +U fixed points moved; ATV-self absolute pins below were
re-measured on the new functional (identity/dual gates unchanged and tighter).

- E(k-UKS+U 1×1×2, 100 Ry): −0.546250995860 → **−0.734377348082 Ha**; max|σ_an−σ_fd| improved 9.3e-4 → **3.0e-9 Ha/Bohr³** (gate tightened to 1e-4).
