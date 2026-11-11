# GPW Dudarev DFT+U

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Dudarev DFT+U (CP2K `PLUS_U` / Mulliken occupation) on collinear unrestricted
GPW corrects self-interaction on partially filled correlated shells by adding an
effective Hubbard \(U_{\mathrm{eff}}=U-J\) energy and potential. It is a member
of the concrete method: `cpu::gpw::GpwDft::plus_u` is a
`std::vector<PlusU>` with `PlusU{element, l, u_minus_j_ev}` (\(U-J\) in **eV**,
converted to Hartree inside), next to the integer spin fillings `n_alpha` /
`n_beta` and, for Bloch sampling, `k_mesh` / `k_points` + `k_point_weights` /
`k_time_reversal`. No `*Config` bag, no factories. The method runs directly
(`build(frame)` → `run(frame)` / `forces(frame)` / `stress(frame)`, with the
multi-SCF finite-difference oracles `gpw_forces_fd` / `gpw_stress_fd`), or is
bound into an outer `cpu::Evaluate` / `cpu::single_point` through
`cpu::gpw::GPWEvaluator` (`compute_forces`, `compute_stress`,
`stress_use_fd`). Total energy on the Frame is **eV** (`kEnergy`); Hartree
totals, the DFT+U contribution `GpwResult::e_dftu`, \(\langle S^2\rangle\) and
Mulliken spins are on the result objects.

The case covers Γ-point energy (`gpw-dftu-nio`), Γ analytic ionic forces
(`gpw-dftu-forces`, record in the case `REFERENCE.md`), Γ analytic cell stress
(`gpw-dftu-stress`), and multi-k analytic forces and stress
(`gpw-dftu-kpoint-forces`, `gpw-dftu-kpoint-stress`). The multi-k SCF energy
alone ([gpw-dftu-kpoints](#gpw-dftu-kpoints)) has no section of its own. All
paths are CPU only, PADE LSD. Build with `-DATV_BUILD_GPW=ON` (GPW needs
FFTW3) in the `buildenv-gcc/2026.03-mpich` login environment.

Case: `ctest -R '^gpw/dft-plus-u$'` — sources and oracle record in [`regressions/gpw/dft-plus-u/`](../../../regressions/gpw/dft-plus-u/). One section of the case per `##` heading below: `atv_regressions --case gpw/dft-plus-u --section <name>`.

## Common method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid set by the plane-wave cutoff [Ry];
GTH pseudopotentials.

**UKS / PADE LSD.** Separate \(P^\alpha\), \(P^\beta\) (per \(\mathbf k\) on the
multi-k path); spin-resolved Padé LDA; integer members `n_alpha` / `n_beta`
(not a multiplicity bag). The multi-k path uses fixed integer fillings (no
UKS + smearing with DFT+U). Monkhorst–Pack meshes follow the CP2K convention
\(\kappa_d=(2r-N_d-1)/(2N_d)\) (even \(N\) is Γ-avoiding); optional
time-reversal folding of \(\pm\mathbf k\) (collinear, no SOC) doubles the
weights of retained partners; explicit `k_points` + `k_point_weights`
(Σ weights = 1) override `k_mesh` when non-empty.

**Dudarev occupation, energy, potential.** Per element and angular momentum
\(L\), the occupation on the AO subspace of atom \(A\) with \(\ell=L\) is the
symmetrized Mulliken projection, BZ-averaged with weights \(w_k\) on the multi-k
path (Γ: a single term with \(w=1\)):

\[
n^{A\sigma}_{mm'} = \sum_k w_k\,\tfrac12\Big[
      (P_k^\sigma S_k)_{mm'} + (S_k P_k^\sigma)_{mm'}
    \Big]_{\text{Herm}}.
\]

The occupations are averaged over the BZ before the nonlinear Dudarev form —
partial per-rank occupations must not enter \(E_U\) under MPI k round-robin.
Energy (sum over atoms and spin; \(U\) in Hartree) and the potential added to
**every** spin Fock matrix \(H_k^\sigma\) on that subspace:

\[
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big],
\qquad
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big).
\]

This is the CP2K simplified rotationally invariant form (Dudarev et al.;
`PLUS_U_METHOD MULLIKEN`, `U_MINUS_J`). The SCF scatters the exact derivative
\(\partial E_U/\partial P = \tfrac12(\hat V S + S\hat V)\) (CP2K applies the same
S-weighted form), so the fixed point is stationary for the reported
\(E_U[n(P,S)]\) — the precondition for every analytic-vs-FD dual gate below.
The bare-\(\hat V\) scatter leaves the SCF non-stationary: analytic vs
multi-SCF FD forces disagree at O(1) Ha/Bohr and the +U stress dual gates
carry a ~9×10⁻⁴ Ha/Bohr³ residual. Implementation: `cpu/gpw/internal/dft_plus_u.h`
(`apply_dft_plus_u_k`, `build_dftu_potential_k`, `dft_plus_u_energy`; not
public). Empty `plus_u` leaves the plain UKS/RKS path unchanged. On
multi-function shells \(E_U\) can be negative (occupation-matrix eigenvalues
outside \([0,1]\)); multi-k gates assert \(|E_U|\) non-trivial and pin the
total energy, not \(E_U>0\).

**Optional AFM seed.** `atom_spin_guess` (per-atom, size 0 or \(N\)) biases
only the **initial** UKS density; solid AFM would use opposite signs on the
two Ni sites.

**Projector-model divergence from CP2K.** CP2K `PLUS_U_METHOD MULLIKEN`
projects on the on-site **net** AO block \(q_{ab} = P_{ab}S_{ab}\)
(elementwise; geometry-independent ⇒ no occupation force term), while ATV
projects on the symmetrized **gross** product \(\tfrac12(PS+SP)\) restricted
to the block (geometry-dependent). Energies are comparable; the +U **force**
contribution differs by model, not by bug (quantified in
[gpw-dftu-kpoint-forces](#gpw-dftu-kpoint-forces)). Aligning the projector, or
adding CP2K's as a second method, is an open decision.

**Analytic stress (all stress sections).** After one SCF, rebuild
\(P_k^\sigma\) and
\(W_k^\sigma=\sum_i^{n_\sigma}\varepsilon_i^\sigma c_i^\sigma(c_i^\sigma)^\dagger\)
(band energies include \(V_U\)); apply orthorhombic diagonal strain \(\pm h\) at
**fixed fractional** coordinates and take the central difference of the
**non-self-consistent** Lagrangian \(E_L[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon]\),
with \(E_U\) evaluated at fixed spin density and strained overlaps
(Hellmann–Feynman fixed-\(P\) + Pulay + \(E_U\)). Sign as CP2K `STRESS_TENSOR`:

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E_L}{\partial\varepsilon_{aa}}
\approx -\Omega^{-1}\frac{E_L(+h_a)-E_L(-h_a)}{2h},
\qquad a\in\{x,y,z\},
\]

Voigt order \((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\),
default strain step \(h = 2\times 10^{-3}\) (`stress_strain_h`), shear components
exactly 0 (orthorhombic only). `GpwDft::stress` returns eV/Å³ and writes Frame
`kStress` = σΩ [eV]; pins compare in Ha/Bohr³ (`last_stress_ha_bohr3()`, or
divide by `units::kHartreePerBohr3InEvPerAngstrom3`). The oracle twin
`gpw_stress_fd` fully re-converges the SCF at each \(\pm h\) strained cell — not
the product hot path. Fixed-\(P\)+Pulay stress lives in
`src/cpu/gpw/internal/gpw_stress.cpp`.

## Common systems

**NiO molecule** (`gpw-dftu-nio`, `gpw-dftu-stress`): a molecule, not solid
AFM rocksalt.

| Item | Value |
| ---- | ----- |
| System | NiO **molecule**, charge 0, UKS multiplicity 3 |
| Bond | \(R(\mathrm{Ni–O})=\) **1.668 Å** along *z*, **centered** |
| Coordinates (Å) | Ni (5, 5, 4.166), O (5, 5, 5.834) |
| Cell | **10 × 10 × 10 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT` / `nonperiodic = true`) |
| Basis | Ni `SZV-GTH-PADE-q18`, O `SZV-GTH-PADE` |
| PP | Ni `GTH-PADE-q18` (18e), O `GTH-PADE-q6` (6e) → **24** valence e⁻ |
| XC | PADE (LDA LSD) |
| DFT+U | Ni \(L=2\), \(U-J=6.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| Cutoff | **250 Ry** (ATV single density grid; CP2K `NGRIDS 4` / `REL_CUTOFF 40`) |
| Spin | \(N_\alpha=13\), \(N_\beta=11\) |
| SCF twin | CP2K: OT `FULL_SINGLE_INVERSE` + OUTER_SCF (diag+Broyden does not converge this twin within budget); ATV: DIIS-style mix 0.35, `tol=1e-5`, `max_iter=300` |

Offline CP2K 2026.1 ENERGY pin (`cp2k.psmp`, revision 85b8a9b49c;
`nio_mol_pade_u.inp`, OMP=8):

```text
  DFT+U energy:                                                 0.05230865222345
  Total energy:                                              -184.92466076474312
  Ideal and single determinant S**2 :                    2.000000       2.019198
       1     Ni       1         9.343192     8.165560     0.491248     1.177632
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.924660764743123
```

The geometry must be **centered** for MT: off-centre molecules destroy energy
parity. Rocksalt AFM NiO (2 f.u.) does **not** converge under SZV / Γ-only in
CP2K (diag+Broyden or no-U restart) within 200–300 steps; twin inputs
`nio_afm_pade_*.inp` are kept as exploration artifacts only.

**High-spin H₂ toy** (`gpw-dftu-kpoint-forces`, `gpw-dftu-kpoint-stress`):
H₂ (M = 3, \(N_\alpha=2\), \(N_\beta=0\)), compressed 1.0 Bohr bond along *x*
(H1 \((5.5, 6.0, 6.0)\), H2 \((6.5, 6.0, 6.0)\) Bohr), cubic **12 Bohr**
periodic cell, DZV-GTH-PADE / GTH-PADE-q1, PADE LSD, H \(L=0\), **100 Ry**,
`local_pp_on_grid`, MONKHORST-PACK **1×1×2** with `k_time_reversal`
(→ `nk=1`), SCF mix 0.35, `max_iter=120`.

## Oracle

Offline **CP2K 2026.1** Quickstep (`cp2k.psmp`, OMP=8), never live from ctest;
inputs under `tools/cp2k_oracle/inputs/<section>/`. CP2K runs multigrid + OT
while ATV runs a single density grid + DIIS mixing, so absolute energies agree
to the **5×10⁻² Ha** class, not bit-for-bit. Where CP2K cannot give a fair
absolute number (the H₂ toys: unphysical Mulliken charges; multi-k NiO: SCF
non-convergence) the decisive gates are internal analytic-vs-FD dual gates
and Γ-limit identities (complex \(k=\Gamma\) ≡ real Γ).

## Shared limitations

- **Molecule / toy footholds, not solid chemistry.** Rocksalt AFM NiO needs
  k-points and/or a larger basis before a production pin; SZV is a minimal
  PADE GTH basis, not a chemistry-quality NiO claim; chemistry-facing PLUS_U
  numbers stay on the Γ NiO molecule.
- **PADE LSD only** — UKS + PBE-GGA is unsupported for GPW UKS (same as
  [gpw-uks-o2-gamma](gpw-uks.md#gpw-uks-o2-gamma)), so no GGA +U.
- Liechtenstein full \(U,J\) tensor, orbital-dependent U ramping, GAPW / PAW
  projectors: out of scope. No OT + multi-k + DFT+U; multigrid / nonperiodic
  k-paths throw (`k_mesh` requires periodic single-grid `local_pp_on_grid`).
- Stress: orthorhombic diagonals only (no shear / triclinic).
- No MPI scaling claim; CPU only (no CUDA GPW).

## References

- V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991) — DFT+U
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified rotationally invariant \(U\)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele, M. Krack, F. Mohamed, M. Parrinello, T. Chassaing, J. Hutter, *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep GPW, forces
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- O. H. Nielsen, R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — stress theorem
- CP2K 2026.1 `src/dft_plus_u.F` — S-weighted KS contribution and `matrix_w` correction
- Related: UKS without U [gpw-uks-o2-gamma](gpw-uks.md#gpw-uks-o2-gamma);
  closed-shell analytic stress [gpw-analytic-stress](gpw-rks.md#gpw-analytic-stress);
  strain-FD stress [gpw-stress-water](gpw-rks.md#gpw-stress-water); plain k
  forces [gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces); multi-k UKS
  [gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks),
  [gpw-kpoint-uks-stress](gpw-uks.md#gpw-kpoint-uks-stress); narrative GPW
  [Ch. 9](../ch09-gpw.md); roadmap [`docs/roadmap/dft.md`](../../roadmap/dft.md)

## gpw-dftu-nio — Γ-UKS DFT+U energy, NiO molecule {#gpw-dftu-nio}

Γ-point unrestricted GPW with PLUS_U on the Ni \(d\) shell, run through
`GPWEvaluator` + `cpu::single_point`. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.21\) (≤10×).

```cpp
// frame: NiO, R = 1.668 Å along z, centred in a cubic L = 10.0 Å cell
cpu::gpw::GpwDft dft{{Ni, O}, 250.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 300; dft.tol = 1e-5; dft.mix = 0.35;
dft.nonperiodic = true;
dft.n_alpha = 13;  dft.n_beta = 11;      // multiplicity 3, 24 valence e⁻
dft.plus_u.push_back({molrs::Element::Ni, /*l=*/2, /*u_minus_j_ev=*/6.0});
cpu::gpw::GPWEvaluator eval(std::move(dft));

cpu::single_point(frame, eval);          // frame: Ni/O masses 58.693 / 15.999

const double e_ha = eval.last_energy_hartree();          // [Ha]
const double e_ev = frame.meta.get(kEnergy);             // [eV]
const cpu::gpw::GpwResult& r = eval.last_result();
const double e_u = r.e_dftu;                             // [Ha]
const double s2 = eval.last_s2();                        // Löwdin ⟨S²⟩
const double m_Ni = r.mulliken_spin.empty() ? 0.0 : r.mulliken_spin[0];
```

No unit test in the tree covers DFT+U. This section is
energy-only; +U forces and stress are the other sections.

### Validation (vs CP2K)

| Quantity | CP2K pin | Atomiverse | Tol (regression) | Notes |
| -------- | -------: | ---------: | ---------------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−184.92466076474312** | ≈ −184.94637 | **5×10⁻² Ha** | single-grid vs multigrid+OT |
| \(\lvert\Delta E\rvert\) vs CP2K | — | ≈ **2.2×10⁻²** Ha | ≤ 5×10⁻² | within gate |
| DFT+U energy [Ha] | **0.05230865222345** | \(>10^{-4}\) (asserted) | diagnostic pin | `r.e_dftu`; not equality-asserted |
| \(\langle S^2\rangle\) (single det.) | **2.019198** | ≈ 2.016 | **0.25** | ideal \(S(S+1)=2\) |
| Mulliken \(m_{\mathrm{Ni}}\) [e] | **1.177632** | ≈ 1.07 | **0.5** and \(m_{\mathrm{Ni}}>0.3\) | |
| Frame energy consistency | — | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}}\equiv E_{\mathrm{Ha}}\) | \(10^{-9}\) Ha | |
| SCF | converged | `r.converged` | — | ~56 iters (ATV) |

### Performance

Cutoff 250 Ry, PADE+U, MT, single rank, `OMP_NUM_THREADS=8`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| NiO mol PADE+U / SZV / MT, Γ | **15.79** | **3.29** | **0.21** | ✓ | CP2K OT+OUTER_SCF multigrid; ATV Release single density grid |

Passes without exemption. The ratio is for this SZV molecule; it does not
forecast solid NiO, multigrid parity, or OT vs DIIS on identical algorithms.

## gpw-dftu-stress — Γ-UKS DFT+U analytic cell stress, NiO molecule {#gpw-dftu-stress}

Hellmann–Feynman fixed-\(P\) cell stress (see **Common method**) on the same NiO
molecule as [gpw-dftu-nio](#gpw-dftu-nio), driven by `cpu::Evaluate` with
`GPWEvaluator` stress on. The decisive gate is internal: analytic vs multi-SCF
strain-FD (`gpw_stress_fd`) on the evaluator's own method; the energy reuses the
CP2K PLUS_U ENERGY pin as a soft gate. Absolute CP2K `STRESS_TENSOR NUMERICAL`
Voigt digits are not pinned: the CP2K module was unavailable on the pin host;
the twin input `nio_mol_pade_u_stress_num.inp` ships for an operator re-run.
Performance: **`perf_exempt`**.

```cpp
// same dft as gpw-dftu-nio (Ni/O SZV, 250 Ry, n_alpha 13 / n_beta 11, plus_u)
cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_forces = false;
eval.compute_stress = true;
eval.stress_use_fd = false;              // product analytic fixed-P
eval.method().stress_strain_h = 2e-3;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

const double e_u  = eval.last_result().e_dftu;           // [Ha]
const auto& s_ha  = eval.last_stress_ha_bohr3();         // Voigt [Ha/Bohr³]

// Dual-gate oracle (multi-SCF, regression only): [eV/Å³]
const auto sfd = cpu::gpw::gpw_stress_fd(
    frame, eval.method(), /*h=*/2e-3, /*axes*/ {true, true, true});
// compare s_ha[a] with sfd[a] / units::kHartreePerBohr3InEvPerAngstrom3, a = xx,yy,zz
```

### Validation

Settings: NiO molecule (**Common systems**), stress step \(h=2\times 10^{-3}\),
`compute_stress=true`, `stress_use_fd=false`.

| Quantity | Pin / expect | Tol (reg) | Notes |
| -------- | -----------: | --------- | ----- |
| \(E_{\mathrm{CP2K}}\) [Ha] | **−184.92466076474312** | soft | CP2K ENERGY pin (`nio_mol_pade_u.inp`) |
| \|E_ATV − E_CP2K\| | **2.15×10⁻²** Ha | **5×10⁻²** Ha | single-grid vs multigrid+OT |
| e_dftu [Ha] | **0.091249** | > **1×10⁻⁴** | positive U contribution |
| max\|σ_an − σ_fd\| diag | 2.8×10⁻⁶ (unit-level 8.4×10⁻⁸) | **2×10⁻⁴** Ha/Bohr³ | dual gate |
| Off-diagonal product | **0** | **1×10⁻⁸** | orthorhombic |
| max\|σ_diag\| floor | \(\sim 10^{-5}\) | > **1×10⁻⁸** | non-degenerate |
| ⟨S²⟩ | **2.0155** | diagnostic | ideal \(S(S+1)=2\) |
| Frame `kStress` units | matches Ha/Bohr³ × conversion | **1×10⁻⁹** | eV/Å³ |

ATV analytic stress components and e_dftu as recorded with the bare-\(\hat V\)
scatter (dual-gate residual **8.92×10⁻⁶** there):

| Component | Value [Ha/Bohr³] |
| --------- | ---------------: |
| \(\sigma_{xx}\) | \(2.10983351\times 10^{-7}\) |
| \(\sigma_{yy}\) | \(2.10983257\times 10^{-7}\) |
| \(\sigma_{zz}\) | \(-1.04886653\times 10^{-5}\) |

Bar → Ha/Bohr³ (CP2K internal table):
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\).

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| Fair NUMERICAL stress wall | n/a | product **12.0** (OMP=8) | **perf_exempt** | n/a | No CP2K binary on pin host for the NUMERICAL stress twin |
| Dual-gate total (product + FD) | — | **~37** (12.0 + 24.7) | n/a | n/a | FD twin = 6 SCFs; not product cost |
| Product-only vs ENERGY wall of `gpw-dftu-nio` | **15.79** | **12.0** | **≈0.76** | ✓ if reused | Context only; not the stress acceptance ratio |

`perf_exempt = true`; the target for a fair analytic-product vs NUMERICAL twin
on one host is \(r\le 10\).

## gpw-dftu-kpoint-forces — Multi-k analytic forces with DFT+U, H₂ toy + NiO Γ pin {#gpw-dftu-kpoint-forces}

Analytic ionic forces for multi-k collinear UKS GPW with Dudarev DFT+U. The
occupation depends on geometry through \(S_k\), so DFT+U adds an
**occupation-matrix force term** on top of the Hellmann–Feynman + Pulay
pieces:

\[
\frac{dE_U}{dR} = \sum_\sigma\sum_k w_k\,
  \mathrm{Re\,Tr}\big[M_k^\sigma\,\tfrac{dS_k}{dR}\big],
\qquad
M_k^\sigma = \tfrac12\big(\hat V^\sigma P_k^\sigma
  + P_k^\sigma \hat V^\sigma\big),
\]

folded as \(W_k \leftarrow W_k - M_k\) into the Pulay \(W\!\cdot\!dS_k\)
contraction on both the real Γ and complex multi-k paths
(`subtract_dftu_pulay[_k]` in `internal/dft_plus_u.h`), mirroring CP2K's own
`matrix_w` correction for its geometry-dependent Mulliken-charges +U
(`src/dft_plus_u.F`). The decisive gate is a full-6N multi-SCF FD dual gate on
the multi-k H₂ toy (6.4×10⁻⁶ Ha/Bohr); the external anchor is a CP2K 2026.1
`ENERGY_FORCE` pin on a periodic Γ NiO d-shell system.

```cpp
GpwDft m{{Kind{molrs::Element::H,
               GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
               GthPotential(molrs::Element::H, "GTH-PADE-q1")}}, 100.0, XcKind::PADE};
m.n_alpha = 2;  m.n_beta = 0;      // high-spin UKS
m.tol = 1e-8;   m.mix = 0.35;  m.max_iter = 120;
m.local_pp_on_grid = true;
m.k_mesh[0] = 1; m.k_mesh[1] = 1; m.k_mesh[2] = 2;   // MP 1×1×2, TR-folded
m.k_time_reversal = true;
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/2.0});

m.build(frame);
m.forces(frame);                                   // kAtom_fx/fy/fz [eV/Å]
const auto F_fd = gpw_forces_fd(frame, m, /*h=*/1e-3 * b2a);  // FD oracle [eV/Å]
```

U−J = 2 eV deliberately: at 4 eV this DZV s-shell toy sits in the
strong-overlap regime (CP2K itself warns "unphysical Mulliken charges") and the
displaced multi-SCF FD oracle hops between SCF branches — no dual gate holds on
a multi-valued surface. At 2 eV the SCF is single-branch. Both entry points
(`GpwDft::forces`, `gpw_forces_fd`) are public in `gpw_solver.h`; the assembly
is in `internal/gpw_forces.h`.

### Validation

**System A — multi-k dual gate (decisive).** H₂ toy (**Common systems**), H
L=0 U−J = 2 eV:

| Gate | Quantity | Measured | Tol |
| ---- | -------- | -------: | --- |
| A1 | E(k-UKS+U 1×1×2, 100 Ry) | −0.611940650579 Ha (e_dftu −0.10209345) | payload sanity |
| A2 | max\|F_an − F_fd\| (full-6N `gpw_forces_fd`, h=1e-3) | **6.4×10⁻⁶ Ha/Bohr** | 5×10⁻⁵ |
| A3 | \|ΣF\| / Newton-3 on the bond axis | 1.1×10⁻¹⁴ | 1×10⁻⁶ |
| A4 | max\|F(U) − F(U=0)\| (U-term activation) | 5.4×10⁻² Ha/Bohr | ≥ 1×10⁻⁴ |
| A5 | max\|F_kΓ − F_Γ\| (explicit k=Γ ≡ real Γ fold) | 6.0×10⁻⁸ Ha/Bohr | 5×10⁻⁶ |

Without the W-fold, A2 reads ~1.6 Ha/Bohr (wrong sign) — the occupation-matrix
term is decisive, four orders above the gate residual.

**System B — CP2K `ENERGY_FORCE` pin.** Periodic Γ NiO stretched molecule
(10 Å cubic PBC box, R = 1.8 Å), SZV-GTH-PADE(-q18) / GTH-PADE-q18/q6, PADE
LSD, Nα=13 / Nβ=11, PLUS_U Mulliken Ni d U−J = 6 eV. CP2K output (OT
FULL_SINGLE_INVERSE, multigrid 250 Ry / REL 40 / NGRIDS 4, wall 12.84 s):

```text
  *** SCF run converged in    14 steps ***
  DFT+U energy:                                                 0.04952309241684
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.929424563910374
 FORCES|      1  3.17871282E-07 -1.91561775E-06 -8.35837089E-03   8.35837111E-03
 FORCES|      2 -3.55176272E-08  2.19071254E-06 -7.27092413E-03   7.27092447E-03
```

| Check | Status | Value | Tol |
| ----- | ------ | ----: | --- |
| \|E_ATV − E_CP2K\| | **asserted** | 1.18×10⁻² Ha | 5×10⁻² (single grid vs multigrid+OT) |
| transverse F ≈ 0 (axial molecule) | **asserted** | ~1×10⁻⁸ Ha/Bohr | 5×10⁻⁴ |
| ΔF_z(Ni−O) vs CP2K | **diagnostic** | ATV +8.97×10⁻² vs CP2K −1.09×10⁻³ | projector-model difference |
| Γ d-shell FD, tol 1e-8, h=2×10⁻³ (offline) | diagnostic | \|F_an − F_fd\| = **3.7×10⁻⁴ Ha/Bohr** | outside the ctest budget |

Multi-k CP2K twin attempts (`nio_pbc_kpoint_force.inp`, MP 1×1×2, diag +
Broyden, MAX_SCF up to 600): the SCF oscillates and never converges — the
degenerate minority-spin d manifold defeats fixed-multiplicity
diagonalization, and CP2K has no OT with k-points. Multi-k force correctness is
therefore gated by A2 + the A5 Γ-limit identity, with the Γ d-shell pin as the
external anchor.

### Performance

Release, OMP=8.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| NiO Γ +U single point + analytic forces (B) | 12.84 | ≈ 13 | **≈ 1.0** | ✓ | ATV single grid 250 Ry vs CP2K multigrid+OT |
| Full regression A1–A5 + B (incl. full-6N FD) | — | 38.6 | n/a | — | FD oracle cost, not a product path |

### Limitations

- **Projector model differs from CP2K** on the +U force contribution; the force
  twin is diagnostic, the energy twin asserted.
- **Toy dual-gate system:** the multi-k gate is H₂ HS; the d-shell force check
  is Γ-only (offline FD at tol 1e-8). No converged multi-k CP2K force twin
  exists.

## gpw-dftu-kpoint-stress — Multi-k DFT+U analytic cell stress, H₂ toy {#gpw-dftu-kpoint-stress}

Monkhorst–Pack multi-k UKS GPW with Dudarev DFT+U and analytic
fixed-\(\{P_k^\sigma,W_k^\sigma\}\) cell stress (see **Common method**), with
BZ-averaged \(E_U[n^\sigma(\{P_k^\sigma\},\{S_k\})]\) at the strained overlaps.
It joins multi-k DFT+U energy ([gpw-dftu-kpoints](#gpw-dftu-kpoints)), Γ DFT+U
stress ([gpw-dftu-stress](#gpw-dftu-stress)) and multi-k UKS stress
([gpw-kpoint-uks-stress](gpw-uks.md#gpw-kpoint-uks-stress)). The regression
calls the method surface (`method.stress(frame)`); `GPWEvaluator` with
`compute_stress` is the outer Evaluate path for the same method. Decisive gates
are the internal dual gate and the Γ-limit identity; the CP2K twin is
provenance only (unphysical Mulliken / SCF abort on this toy). Ratio
**`perf_exempt`**.

```cpp
// H2 toy (Common systems): DZV-GTH-PADE / GTH-PADE-q1, 100 Ry
GpwDft m{h_kinds(), 100.0, XcKind::PADE};
m.n_alpha = 2; m.n_beta = 0;
m.max_iter = 120; m.tol = 1e-7; m.mix = 0.35;
m.local_pp_on_grid = true;
m.k_mesh[0] = 1; m.k_mesh[1] = 1; m.k_mesh[2] = 2;   // MP 1×1×2 (TR → nk=1)
m.k_time_reversal = true;
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});

m.build(frame);
const GpwResult& er = m.run(frame);   // er.e_total, er.e_dftu [Ha], er.C_occ_k / C_occ_k_beta

m.stress_strain_h = 2e-3;
const auto san = m.stress(frame);                  // product analytic [eV/Å³]
const auto sfd = gpw_stress_fd(frame, m, 2e-3);    // multi-SCF oracle [eV/Å³]
```

No unit test in the tree compares analytic and FD +U stress.

### Validation

Settings: H₂ toy (**Common systems**), H \(L=0\), \(U-J=4.0\) eV, SCF
`tol=1e-7`, stress step \(h=2\times 10^{-3}\). The CP2K twin input uses an
**8 Bohr** cheaper NUMERICAL-stress variant of the same method class
(PLUS_U + KPOINTS + `STRESS_TENSOR NUMERICAL`).

| Gate | Quantity | Pin / measured | Tol |
| ---- | -------- | -------------: | --- |
| A1 | \(E_U\) (`e_dftu`) | non-trivial (multi-function s shell, \(<0\)) | \(\lvert E_U\rvert>10^{-8}\) |
| A2 | \(\max\lvert\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\rvert\) diag | **3.0×10⁻⁹** Ha/Bohr³ | **1×10⁻⁴** Ha/Bohr³ (decisive) |
| A3 | off-diagonal product | 0 | **1×10⁻⁸** |
| A4 | \(\max\lvert\sigma_{\mathrm{diag}}\rvert\) floor | ≳ 3×10⁻⁴ | \(>10^{-8}\) |
| A5 | \(E_{\mathrm{tot}}\) [Ha] (k-UKS+U 1×1×2, 100 Ry) | **−0.734377348082** | **1×10⁻³** Ha soft |
| A6 | \(\lvert E(U)-E(0)\rvert\) [Ha] | **1.79×10⁻²** | \(\ge 10^{-4}\) |
| A7 | \(\max\lvert\sigma_{k\Gamma}-\sigma_\Gamma\rvert\) | **1.3×10⁻⁸** | **5×10⁻⁴** |

A7 shows the multi-k \(E_U\) Lagrangian path matches the Γ DFT+U stress path.
With the bare-\(\hat V\) scatter the same system gives
\(E_{\mathrm{tot}} =\) **−0.546250995860** Ha (~76 SCF iters, `nk=1`),
\(E_U =\) **−0.243601** Ha, analytic
σ_xx = −3.00982070×10⁻⁴, σ_yy = −6.59874084×10⁻⁵, σ_zz = −6.58998481×10⁻⁵
Ha/Bohr³ and a dual-gate residual of **9.33×10⁻⁴** Ha/Bohr³ (bond-axis xx
dominant, transverse ~1e-6) — which
would need a 1×10⁻³ tolerance; bare multi-k UKS sits at ~\(10^{-10}\).

**Offline CP2K twin (pathology — not an absolute pin):** CP2K **2026.1**
(`h2_uks_dftu_kpoint_stress.inp`) reports unphysical Mulliken charges, a large
negative DFT+U (\(\sim-175\) Ha), and the SCF fails within `MAX_SCF=100`:

```text
  DFT+U energy: ~ −175 Ha (unphysical)
  Total energy: ~ −174 Ha
  *** SCF run NOT converged ***
```

The literature-facing absolute PLUS_U energy remains the Γ NiO molecule pin
(CP2K \(E=\) −184.92466 Ha).

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| H₂ multi-k UKS+U stress (dual gate) | n/a (no fair twin) | **~2.4** (product+FD, OMP=8) | **n/a** | n/a | `perf_exempt`: CP2K H₂+U multi-k toy has unphysical Mulliken + SCF abort |

A fair multi-k solid DFT+U stress ratio needs a chemistry twin (e.g. AFM NiO /
larger basis) after absolute energy parity is established.

### Limitations

- **Toy H₂, not solid chemistry:** multi-function H \(s\) shell + large \(U\)
  gives negative \(E_U\); do not cite the CP2K H₂ run as energy/stress gold.
- Stress only here; multi-k +U forces are
  [gpw-dftu-kpoint-forces](#gpw-dftu-kpoint-forces).

## gpw-dftu-kpoints — Multi-k DFT+U SCF energy, H₂ toy {#gpw-dftu-kpoints}

*No section of the case runs this variant.*

Dudarev DFT+U on multi-k collinear UKS GPW (`GpwDft` members `k_mesh` /
`k_points` + `k_point_weights`, `k_time_reversal`, `n_alpha`/`n_beta`,
`plus_u`; method API `build(frame)` + `run(frame)`), between Γ-only PLUS_U
([gpw-dftu-nio](#gpw-dftu-nio)) and multi-k UKS without U
([gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks)). The k-summed occupations and
the same \(V^\sigma\) on **every** \(H_k^\sigma\) are in **Common method**
(`apply_dft_plus_u_k`; SCF wire-up `run_gpw_scf_kpoints` / `_uks` in
`src/cpu/gpw/gpw_solver.cpp`). The multi-k forces and stress sections exercise
this SCF. **`perf_exempt`**.

```cpp
// High-spin H2 toy, DZV-GTH-PADE, periodic
const double L = 8.0 * units::kBohrInAngstrom;   // 8 Bohr [Å]
const double d = 0.7 * units::kBohrInAngstrom;   // ±0.7 Bohr [Å]
GpwDft m{{cpu::gpw::Kind{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")}}, 120.0, XcKind::PADE};
m.n_alpha = 2; m.n_beta = 0;
m.max_iter = 120; m.tol = 1e-7; m.mix = 0.35;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;  // MP 2×2×2
m.k_time_reversal = true;                      // → n_kpts == 4
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});
m.build(frame);
const GpwResult r = m.run(frame);   // r.e_total, r.e_dftu [Ha], r.n_kpts; frame kEnergy [eV]
```

### Recorded validation

Setup: H₂ high-spin, H at \((0.5L\pm 0.7,\,0.5L,\,0.5L)\) Bohr, cell **8 × 8 × 8 Bohr**
(`PERIODIC XYZ`), DZV-GTH-PADE / GTH-PADE-q1, PADE LSD, H \(L=0\),
\(U-J=4.0\) eV; cutoff **100 Ry** (A1) / **120 Ry** (A2–A3); MP **2×2×2** with
time reversal → **4** k (A2–A3); `max_iter` 100–120. These ATV values were
recorded with the bare-\(\hat V\) scatter.

| Gate | Quantity | Recorded | Tol | Notes |
| ---- | -------- | -------: | --- | ----- |
| A1 | \|E(k=Γ complex+U) − E(real Γ-UKS+U)\| | ≈ \(4\times 10^{-7}\) Ha | **1×10⁻⁶ Ha** | explicit `k_points = {{0,0,0}}`, weight 1 |
| A1 | \|e_dftu(kΓ) − e_dftu(Γ)\| | ≲ \(1\times 10^{-4}\) Ha | **1×10⁻⁴ Ha** | decomposition soft vs total |
| A1 | \|e_dftu\| non-trivial | — | \(>10^{-8}\) Ha | PLUS_U active |
| A2 | \|E(U=4 eV) − E(U=0)\| on MP 2×2×2 | ≈ **9.5×10⁻²** Ha | **≥ 1×10⁻⁴ Ha** | activation floor |
| A3 | \(E_{\mathrm{tot}}\) (MP+TR+U, 120 Ry) | **−0.814170782662** Ha | **1×10⁻³ Ha** | ATV absolute |
| A3 | \(n_{\mathrm{kpts}}\) after TR fold | **4** | exact | MP 2×2×2 |
| A3 | \(E(U=0)\) diagnostic | −0.719059673672 Ha | — | not equality-asserted |
| A3 | e_dftu diagnostic | ≈ −0.177 Ha | \|e\| \(>10^{-8}\) | multi-function s shell |

**Offline CP2K twin (pathology — not an absolute energy gate;**
`h2_uks_dftu_kpoints.inp`, same H₂ / DZV / PADE / U=4 eV / MP 2×2×2 / 120 Ry):
SCF converged in **42** steps, \(E_U =\) **−158.76078199959053** Ha (unphysical
Mulliken), \(E_{\mathrm{tot}} =\) **−157.66686993943307** Ha, wall ≈ **9.4 s**
(OMP=8). ATV is never equality-asserted against these numbers.

```text
  *** SCF run converged in    42 steps ***
  DFT+U energy:                                              -158.76078199959053
  Total energy:                                              -157.66686993943307
 *** WARNING … DFT+U energy contribution is negative ***
 *** possibly due to unphysical Mulliken charges! ***
```

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| H₂ UKS+U MP 2×2×2 | ≈ 9.4 | ≈ **49** (A1–A3 suite) | **n/a** | **perf_exempt** | CP2K energy/Mulliken pathological; ATV wall is the full identity suite, not a single SCF |
