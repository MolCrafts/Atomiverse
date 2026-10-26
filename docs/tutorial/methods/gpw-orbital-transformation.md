# GPW orbital transformation (OT)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Orbital Transformation (OT) is a diagonalization-free SCF minimizer for GPW
Kohn–Sham DFT: occupied MO coefficients are optimized on the Stiefel-type
manifold \(C^\dagger S C = I\) instead of a dense generalized eigensolve +
Pulay DIIS every outer step. The only switch is the public member
`cpu::gpw::GpwDft::use_ot` (default `false`; no `*Config` bag, no factory). It
combines with the other `GpwDft` members — open-shell fillings `n_alpha` /
`n_beta`, Bloch sampling `k_mesh` / `k_points` + `k_point_weights` /
`k_time_reversal`, Dudarev `plus_u`, and metallic `smearing` / `degauss` [Ry] /
`extra_bands` / `mp_order`. The sections bind the configured method through
`cpu::gpw::GPWEvaluator(std::move(dft), …)` and run `cpu::single_point(frame,
eval)`; the no-section variants call the method directly (`build(frame)` →
`run(frame)`). Energy on the Frame is **eV** (`kEnergy`); the Hartree total,
\(\langle S^2\rangle\) and the DFT+U, Fermi-level and entropy pieces are on the
result (`last_energy_hartree()`, `last_s2()`, `GpwResult::{e_total, e_dftu,
e_fermi, e_entropy, n_elec_occ, s2, n_kpts}`).

The case runs three Γ-point sections, each pinned against a CP2K 2026.1 OT
twin and each also running the in-tree DIIS twin on the same grids:
`ot-minimizer-water` (closed-shell RKS), `ot-uks-o2` (collinear UKS) and
`ot-dftu` (UKS + DFT+U). The multi-k, smearing and combined variants below
have no section; their recorded values are kept for reference. Build with
`-DATV_BUILD_GPW=ON` (GPW needs FFTW3) in the `buildenv-gcc/2026.03-mpich`
login environment.

| Variant | Spin | k | DFT+U | Smearing | Section |
| ------- | ---- | - | ----- | -------- | ------- |
| [ot-minimizer-water](#ot-minimizer-water) | RKS | Γ | — | — | yes |
| [ot-uks-o2](#ot-uks-o2) | UKS | Γ | — | — | yes |
| [ot-dftu](#ot-dftu) | UKS | Γ | ✓ | — | yes |
| [ot-kpoints](#ot-kpoints) | RKS | multi-k | — | — | — |
| [ot-uks-kpoints](#ot-uks-kpoints) | UKS | multi-k | — | — | — |
| [ot-dftu-kpoints](#ot-dftu-kpoints) | RKS | multi-k | ✓ | — | — |
| [ot-uks-dftu](#ot-uks-dftu) | UKS | multi-k | ✓ | — | — |
| [ot-smearing](#ot-smearing) | RKS | multi-k | — | FD | — |
| [ot-uks-smearing](#ot-uks-smearing) | UKS | multi-k | — | FD | — |
| [ot-dftu-smearing](#ot-dftu-smearing) | RKS | multi-k | ✓ | FD | — |
| [ot-uks-dftu-smearing](#ot-uks-dftu-smearing) | UKS | multi-k | ✓ | FD | — |

Case: `ctest -R '^gpw/orbital-transformation$'` — sources and oracle record in [`regressions/gpw/orbital-transformation/`](../../../regressions/gpw/orbital-transformation/). One section of the case per `##` heading below: `atv_regressions --case gpw/orbital-transformation --section <name>`.

## Common method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid set by the plane-wave cutoff
\(E_{\mathrm{cut}}\) [Ry]; Hartree via FFT Poisson; XC on the grid; potential
integrated back into the AO basis; core electrons removed by analytic GTH
pseudopotentials.

**OT parameterization** (VandeVondele & Hutter 2003; Kühne et al. 2020 §VII C).
For reference occupied coefficients \(C_0\) with \(C_0^\dagger S C_0 = I\) and
an \(S\)-tangent step \(x\) (\(C_0^\dagger S x = 0\)):

\[
C(x) = C_0\,\cos U + x\,U^{-1}\sin U,\qquad
U = (x^\dagger S x)^{1/2},
\]

which preserves \(C(x)^\dagger S C(x) = I\) for any such \(x\). Closed-shell
RKS uses occupation 2, so the band-energy gradient consistent with
\(P = 2\,C C^\dagger\) is

\[
g = (I - S C_0 C_0^\dagger)\,(4 H C_0);
\]

open-shell UKS uses occupation **1** per spin channel
(\(P_\sigma = C_\sigma C_\sigma^\dagger\)) with the gradient formed per spin
manifold (`internal/ot.h`). Under metallic smearing the active subspace is
\(n_{\mathrm{occ}} + \max(\texttt{extra\_bands},1)\) (or the full AO spectrum,
below) with non-uniform occupations \(f_i\) and \(P = C\,\mathrm{diag}(f)\,C^\dagger\):

\[
g = (I - S C C^\dagger)\,(2 H C f)\ \ \text{(RKS, } f_i\in[0,2]\text{)},\qquad
g_\sigma = (I - S C_\sigma C_\sigma^\dagger)\,(H_\sigma C_\sigma f_\sigma)\ \ \text{(UKS, } f\in[0,1]\text{)}
\]

(column \(i\) of \(HC\) scaled by \(f_i\); `ot_gradient_k(..., f)`).

**Minimization.** Preconditioned conjugate gradient on the manifold. Dense
in-tree preconditioners (internal, not public API): FULL_KINETIC \(M = T + S\)
and FULL_SINGLE_INVERSE-style \(M = H + a S\) (the CP2K default class for gapped
systems, used by the twins). The outer SCF rebuilds \(H\) from the collocated
density until residual / energy criteria meet `tol`. Algorithm by combination:

- Γ RKS / UKS, multi-k RKS / UKS without U: joint Polak–Ribière+ CG with Armijo
  line search on the rebuilt total energy (one step per outer; joint over the
  concatenated spin × k variables on multi-k UKS).
- multi-k with `plus_u`, integer occupations: CP2K-style OUTER_SCF — rebuild
  every \(H_k^\sigma\) (including \(V_U^\sigma\)) from the k-summed density, then
  a **fixed-H** OT band minimize on each local \((k,\sigma)\)
  (`ot_minimize_fixed_k`; no Armijo one-step on this path).
- smearing (with or without U): free-energy PR+ with Armijo over the **full**
  AO spectrum per spin channel with fractional occupations.

All routes share the energy assembly (`build_k_energy` / `build_uks_k_energy`)
of the corresponding DIIS path.

**DIIS twin path.** `use_ot = false` (default): generalized diagonalization of
\((H_k^\sigma,S_k)\) + density mixing (`mix`, ignored when `use_ot`). For a
gapped ground state (or the same Mermin minimum under smearing) both paths
reach the same self-consistent fixed point on identical grids; every variant
gates \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\).

**k sampling.** Explicit `k_points` in fractional reciprocal coordinates (CP2K
`&KPOINTS SCHEME GENERAL` / `UNIT B_VECTOR`) with `k_point_weights`,
\(\sum w = 1\) (no implicit renormalization; a non-empty list overrides
`k_mesh`), or a Monkhorst–Pack `k_mesh = {N_x,N_y,N_z}` with the CP2K convention
\(\kappa_d = (2r - N_d - 1)/(2 N_d)\) (even \(N\) is Γ-avoiding) and optional
time-reversal folding (`k_time_reversal = true`, default; collinear, no SOC),
which doubles retained partners' weights. `{1,1,1}` or a single explicit Γ
keeps the real Γ path; multi-k uses complex Bloch \(H_k,S_k,C_k\). Multi-k OT is
periodic, single-grid, `local_pp_on_grid` only — multigrid and nonperiodic
k-paths throw at `build()`.

**Isolated electrostatics** (Γ sections). `nonperiodic = true` selects
Martyna–Tuckerman Poisson (`PERIODIC NONE` / `PSOLVER MT`). The molecule must
sit **centered** in the cell: nonperiodic MT grids cover \([0,L)\), and atoms
near an edge truncate Gaussian tails and destroy the energy.

**DFT+U.** `plus_u` is the Dudarev functional of
[gpw-dft-plus-u](gpw-dft-plus-u.md#gpw-dftu-nio) (symmetrized Mulliken
occupation, BZ-averaged on multi-k, \(U-J\) in eV); on the OT path \(V_U\) is
folded into the effective \(H\) that drives the manifold gradient — the same
\(E_U\) formula as the DIIS path. The recorded ATV energies of the multi-k +U
variants were taken with the bare-\(\hat V\) +U scatter; the code applies the
variational \(\tfrac12(\hat V S + S\hat V)\) form, and those values are not
re-measured.

**Metallic smearing (Mermin free energy).** With
\(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss` (Ry on the public
surface, converted to Ha inside the SCF):

| Kind | Spinless \(f(x)\) | Public enum |
| ---- | ----------------- | ----------- |
| Fixed integer | step fill of `n_occ` | `Smearing::none` |
| Fermi–Dirac | \(1/(1+e^{x})\) → closed-shell band \(f_b\in[0,2]\) | `Smearing::fermi_dirac` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

One chemical potential \(\mu\) enforces
\(N_{\mathrm{elec}} = \sum_{k,b} w_k\, f_b(\varepsilon_{kb},\mu,\sigma) = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}\)
(UKS: summed over \(\alpha+\beta\) with spin-orbital fillings \(f\in[0,1]\),
`spin_degen = 1` via the shared `cpu::pw::internal` occupation fill).
Fermi–Dirac entropy \(s = -[f\ln f + (1-f)\ln(1-f)]\):

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -\sigma\sum_{\sigma kb} w_k\, s_{\sigma kb},\qquad
\texttt{GpwResult::e\_total} = F,\quad
\texttt{e\_entropy} = -TS,\quad
\texttt{e\_fermi} = \mu .
\]

`extra_bands` adds empty MOs above \(n_{\mathrm{occ}}\) (API parity with CP2K
`ADDED_MOS` / PW `extra_bands`); under UKS smearing `n_alpha` / `n_beta` only
seed the initial density and the magnetization is free.

## Common systems

**H₂ toy** (all multi-k H₂ variants): periodic cubic **8 × 8 × 8 Bohr** cell, H at
\((L/2 \pm 0.7,\, L/2,\, L/2)\) = (3.3, 4.0, 4.0) and (4.7, 4.0, 4.0) Bohr
(Frame in Å via `units::kBohrInAngstrom`), DZV-GTH-PADE / GTH-PADE-q1, PADE,
**100 Ry** single density grid (`NGRIDS 1` on the twins). +U variants: H
\(L=0\), \(U-J=4.0\) eV, `PLUS_U_METHOD MULLIKEN`. Smearing variants:
Fermi–Dirac `degauss = 0.02` Ry (CP2K \(T_e = 3157.75\) K), `extra_bands = 2` /
`ADDED_MOS 2`, explicit \(\kappa = (1/4,0,0)\) with weight 1.

```cpp
constexpr double b2a = units::kBohrInAngstrom;  // Frame is Å
const double L = 8.0 * b2a;                     // 8 Bohr
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 1;
    frame.span(kAtom_x)[i] = 0.5 * L + (i == 0 ? -0.7 : 0.7) * b2a;
    frame.span(kAtom_y)[i] = 0.5 * L;
    frame.span(kAtom_z)[i] = 0.5 * L;
}
frame.box = cpu::Box::cubic(L);
const cpu::gpw::Kind H{molrs::Element::H,
                       cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
                       cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")};
```

**BCC Na** (OT + smearing variants): conventional cubic cell, \(a = 4.225\) Å,
2 Na at (0,0,0) and (½,½,½) = (2.1125, 2.1125, 2.1125) Å, `DZVP-MOLOPT-SR-GTH` /
`GTH-PBE-q9` (\(Z_{\mathrm{val}}=9\) → \(N_e=18\)), PADE, **150 Ry**,
Monkhorst–Pack \(2\times2\times2\) with time-reversal fold, Fermi–Dirac
`degauss = 0.02` Ry, `extra_bands = 4`, `tol=1e-6`, `eps_default=1e-7`,
`diis_depth = 6`, `mix=0.4`, OT `max_iter=200` (DIIS 120). CP2K 2026.1 DIIS +
FERMI_DIRAC twin (`na_ot_fermi_dirac.inp`; ADDED_MOS 4, \(T_e = 3157.75\) K):
example \(E_{\mathrm{tot}}\) **−92.76593904 Ha** (not an ATV pin), wall
**~89.5** s — same system class as
[gpw-metallic-smearing](gpw-rks.md#gpw-metallic-smearing) /
[gpw-uks-smearing](gpw-uks.md#gpw-uks-smearing). Grid / `EPS_DEFAULT` /
rel-cutoff defaults leave a ~0.6 Ha class offset, so the absolute free energy is
not a gate.

```cpp
const double a = 4.225; // Å
const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 11;
    frame.span(kAtom_x)[i] = frac[i][0] * a;
    frame.span(kAtom_y)[i] = frac[i][1] * a;
    frame.span(kAtom_z)[i] = frac[i][2] * a;
}
frame.box = cpu::Box::cubic(a);
GpwDft m{{cpu::gpw::Kind{molrs::Element::Na,
                         cpu::gpw::GaussianBasis(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH"),
                         cpu::gpw::GthPotential(molrs::Element::Na, "GTH-PBE-q9")}},
         150.0, cpu::gpw::XcKind::PADE};
m.max_iter = 200; m.tol = 1e-6; m.eps_default = 1e-7; m.diis_depth = 6; m.mix = 0.4;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;
m.smearing = Smearing::fermi_dirac;
m.degauss = 0.02;       // Ry
m.extra_bands = 4;
m.use_ot = true;
```

## Oracle

Offline **CP2K 2026.1** (`cp2k.psmp`), never live from ctest; inputs under
`tools/cp2k_oracle/inputs/`. The Γ sections have matched CP2K OT twins (`&OT`
`MINIMIZER DIIS`, `PRECONDITIONER FULL_SINGLE_INVERSE`, `OUTER_SCF`). CP2K
2026.1 hard-aborts OT with k-points (`OT not possible with kpoint
calculations`, `qs_scf_initialization.F`) and its OT keeps fixed integer
occupations (no OT+SMEAR). The multi-k and smearing variants are therefore
gated by in-tree identities (OT ≡ DIIS, complex \(k=\Gamma\) ≡ real Γ, U
activation, electron count) plus a frozen ATV absolute (1×10⁻³ Ha); their CP2K
inputs are DIIS / DIAGONALIZATION twins of the same fixed point, never a fair OT
wall, so those variants are `perf_exempt` unless a DIIS-class wall is quoted.
Fair walls report the **OT-only** time: the regression binaries also run the DIIS
twin and must not be ratioed against CP2K OT alone. The binding fair-wall
baseline (host, binary provenance, timer convention) is
`.claude/notes/gpw-perf-baseline.md`.

## Shared limitations

- Dense \(O(n_b^3)\) preconditioner inverse — no sparse / linear-scaling OT; the
  small-system ratios do not forecast it. No MPI multi-rank OT scaling gate.
- Forces / stress / BOMD under OT are not acceptance pins here.
- The PW backend has no OT switch (`docs/roadmap/dft.md` OT column: GPW ✓,
  PW ○, CUDA ○); no CUDA GPW OT / DFT+U / smear.
- No ROKS / spin projection; no cold / Marzari–Vanderbilt smearing enum; no
  non-collinear / SOC. Liechtenstein full \(U,J\) and GAPW / PAW projectors are
  out of scope.
- Γ pins are energy-class (milli-Ha) against CP2K multigrid (`NGRIDS 4` /
  `REL_CUTOFF 40`) from ATV's single density grid; H₂ / Na microcases are
  numerical fixtures, not chemistry claims.

## References

- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **118**, 4365 (2003),
  doi:10.1063/1.1543154 — original OT
- T. D. Kühne et al., *J. Chem. Phys.* **152**, 194103 (2020) §VII C,
  doi:10.1063/5.0007045 — CP2K / formulation followed in-tree
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — \(\langle S^2\rangle\) of unrestricted determinants
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified \(U\);
  V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991) — DFT+U
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965) — finite-\(T\) free energy;
  Methfessel & Paxton, *Phys. Rev. B* **40**, 3616 (1989)
- Related: DIIS UKS [gpw-uks-o2-gamma](gpw-uks.md#gpw-uks-o2-gamma); DIIS +U
  [gpw-dftu-nio](gpw-dft-plus-u.md#gpw-dftu-nio); multi-k UKS
  [gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks); PW metallic smearing
  [pw-smearing-al](pw-smearing.md#pw-smearing-al); narrative
  [Ch. 9 GPW](../ch09-gpw.md); roadmap [`docs/roadmap/dft.md`](../../roadmap/dft.md) §10

## ot-minimizer-water — Γ RKS OT, isolated H₂O / PBE {#ot-minimizer-water}

OT total energy for the CP2K getting-started isolated water (PBE /
DZVP-MOLOPT-GTH / GTH-PBE / Martyna–Tuckerman) against an offline CP2K OT twin,
plus OT vs in-tree DIIS on the same grids.

```cpp
// CP2K getting-started h2o [Å], 10 Å cubic cell (MT Poisson)
const int z[3] = {8, 1, 1};
const double x[3]  = {5.0000, 5.7586, 4.2414};
const double y[3]  = {5.0000, 5.0000, 5.0000};
const double zc[3] = {5.0000, 5.5043, 5.5043};
// ... fill kAtom_type / kAtom_x/y/z; frame.box = cpu::Box::cubic(10.0);

cpu::gpw::GpwDft dft{{O, H}, 250.0, cpu::gpw::XcKind::PBE};  // DZVP-MOLOPT-GTH, GTH-PBE-q6/q1
dft.max_iter = 200;
dft.tol = 1e-7;
dft.mix = 0.3;
dft.nonperiodic = true;
dft.use_ot = true;

cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/false);
cpu::single_point(frame, eval);
const double e_ha = eval.last_energy_hartree();   // [Hartree]
const bool ok = eval.last_result().converged;
```

The regression uses `250` Ry (budget); the tutorial class in
`examples/09_gpw.cpp` is `400` Ry. No unit test in the tree covers OT
(fixed-H OT, mini-H₂ OT≈DIIS, member toggle).

### Validation (vs CP2K)

Setup: isolated H₂O, closed shell; O (5, 5, 5), H (5.7586, 5, 5.5043),
H (4.2414, 5, 5.5043) Å; 10 Å cubic `PERIODIC NONE`, MT; DZVP-MOLOPT-GTH,
GTH-PBE-q6 (O) / GTH-PBE-q1 (H); PBE; **250 Ry**; `max_iter=200`, `tol=1e-7`.
CP2K 2026.1, 1 MPI rank.

| Quantity | Atomiverse | CP2K 2026.1 | Tol | Notes |
| -------- | ---------: | ----------: | --- | ----- |
| \(E_{\mathrm{tot}}\) OT [Ha] | −17.2118583480 | **−17.212848291913666** | **1.5×10⁻³ Ha** | primary pin |
| \(E_{\mathrm{tot}}\) DIIS (ATV) [Ha] | −17.2118583479 | — | — | same grids as OT |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{CP2K}}\rvert\) | \(9.90\times10^{-4}\) | — | ≤ 1.5×10⁻³ | |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\) (ATV) | \(\sim 4\times10^{-11}\) | — | ≤ 1×10⁻⁴ | fixed-point parity |

The milli-Ha tolerance at 250 Ry absorbs residual grid / multigrid defaults
(NGRIDS / REL_CUTOFF / collocation) that still differ from CP2K. At 400 Ry (not
the pin) CP2K OT gives −17.212337205347978 (DIAG −17.212337010579070) —
energy-class cross-check only. Matched NGRIDS / REL_CUTOFF and a higher cutoff
would be needed for sub-µHa parity.

### Performance

Cutoff 250 Ry, PBE, MT, single rank, `OMP_NUM_THREADS=4`.

| Twin | Wall [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | -------: | ----------------------------------------: | ---- | ------ |
| CP2K OT (`cp2k.psmp`) | **4.31** | — | — | primary twin |
| ATV OT-only (from reg timestamps) | **~39** | **~9.0** | ✓ | fair OT vs OT |
| ATV OT+DIIS (full regression binary) | 41.73 | — | — | **not** a fair twin |
| ATV OT+DIIS OMP=1 | 54.01 | — | — | e2e ≤60 s budget |

The GPW baseline note lists this twin at ATV 2.99 s (SCF body ~1.57) vs CP2K
2.496 s, ratio 1.20 (body ≤ 1.0), measured with OpenMP thread binding that it
flags as suspect for BLAS-bound walls.

## ot-uks-o2 — Γ UKS OT, triplet O₂ / PADE {#ot-uks-o2}

OT on Γ-point collinear UKS GPW: separate α/β occupied manifolds
(\(C_\sigma^\dagger S C_\sigma = I\)), spin-resolved PADE LSD, MT isolation.
Pins the triplet O₂ OT total energy and Löwdin \(\langle S^2\rangle\) (single
unrestricted determinant, not spin-projected; ideal \(S(S+1)=2\)) against a CP2K
OT twin at **250 Ry**, plus OT vs in-tree UKS DIIS. Valence bookkeeping: O
GTH-PBE-q6 → 12 valence e⁻, multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\) (not
all-electron \(9/7\)). The 400 Ry DIIS class is
[gpw-uks-o2-gamma](gpw-uks.md#gpw-uks-o2-gamma).

```cpp
// O₂, R = 1.2075 Å, centered in a cubic L = 12.0 Å cell (MT requires centering);
// kAtom_mass 15.999
cpu::gpw::GpwDft dft{{O}, 250.0, cpu::gpw::XcKind::PADE};  // DZVP-MOLOPT-GTH / GTH-PBE-q6
dft.max_iter = 200;
dft.tol = 1e-7;
dft.mix = 0.4;
dft.nonperiodic = true;
dft.use_ot = true;
dft.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
dft.n_beta = 5;
cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/false);

cpu::single_point(frame, eval);
const double e_ha = eval.last_energy_hartree();       // [Ha]
const double s2 = eval.last_s2();                     // Löwdin ⟨S²⟩
```

### Validation (vs CP2K)

CP2K **2026.1** (`cp2k.psmp`, git:5e54ba2), module `CP2K/2026.1` on Arrhenius;
`o2_uks_ot_pade.inp`. Geometry: O (6, 6, 5.39625), O (6, 6, 6.60375) Å, bond
along *z*, **12 × 12 × 12 Å** `PERIODIC NONE`, MT; DZVP-MOLOPT-GTH /
GTH-PBE-q6; PADE LSD; **250 Ry**.

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) OT [Ha] | **−31.770029989852723** | \(\lvert E_{\mathrm{ATV,OT}}-E_{\mathrm{CP2K}}\rvert\) | **2.5×10⁻³ Ha** | single grid vs CP2K multigrid |
| \(\langle S^2\rangle\) (single det.) | **2.004666** | \(\lvert\langle S^2\rangle_{\mathrm{ATV}}-\mathrm{pin}\rvert\) | **0.15** | UKS contamination accepted |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\) (ATV) | \(\sim 5\times10^{-10}\) Ha | asserted | **1×10⁻⁴ Ha** | same-grid fixed point |
| Frame energy consistency | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}} \equiv E_{\mathrm{Ha}}\) | asserted | \(10^{-9}\) Ha | `kEnergy` vs `last_energy_hartree()` |
| SCF | converged | `r.converged` | — | `max_iter=200`, `tol=1e-7` |

```text
  *** SCF run converged in    30 steps ***
  Total energy:                                               -31.77002998985272
  Ideal and single determinant S**2 :                    2.000000       2.004666
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.770029989852723
```

Cross-check: CP2K DIAG at **400 Ry** (centered O₂) gives −31.7698035 Ha — same
energy class. Offline re-run environment:
`module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1`,
`OMP_NUM_THREADS=4`, `mpirun -np 1`.

### Performance

Cutoff 250 Ry, PADE LSD, MT, single rank, `OMP_NUM_THREADS=4`,
`OMP_PROC_BIND=close`.

| Twin | Wall [s] | \(r\) | ≤10× | Caveat |
| ---- | -------: | ----: | ---- | ------ |
| CP2K OT (1 MPI × 4 OMP) | **8.68** | — | — | primary twin (`CP2K` timer MAXIMUM) |
| Atomiverse OT only | **~62** | **≈ 7.1** | ✓ | 180 DIIS+Armijo iters; FULL_SINGLE precond |
| Atomiverse DIIS (parity check) | **~11** | — | — | 30 DIIS iters; not the fair twin |
| Full regression binary (OT+DIIS) | ≈ 73 | — | — | not ratioed against CP2K OT |

Other case-local energy-OT measurements on this twin give \(r \approx 4.6\) and
~8.5×; the remaining gap is the OT outer count (84 vs CP2K 30; ATV DIIS+Armijo
vs CP2K MINIMIZER DIIS) plus per-iteration KS cost. The case `REFERENCE.md`
defers walls to the GPW baseline note, which has no row for this twin.

## ot-dftu — Γ UKS OT + DFT+U, NiO molecule {#ot-dftu}

OT combined with Dudarev DFT+U on Γ-point UKS GPW — the path for a correlated
open-shell system that needs both a diagonalization-free minimizer and a
Hubbard \(U_{\mathrm{eff}}\). `use_ot = true` with non-empty `plus_u` on UKS runs
OT with \(V_U\) in the effective \(H\); OT+U and DIIS+U converge to the same fixed
point within **1×10⁻³ Ha** on identical single-grid settings. Same chemistry as
[gpw-dftu-nio](gpw-dft-plus-u.md#gpw-dftu-nio), with OT on both sides (CP2K
always uses `&OT` for this NiO pin). Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.26\) (≤10×).

```cpp
// NiO molecule, R = 1.668 Å along z, centered in a 10 Å cubic cell (MT);
// kAtom_mass 58.693 / 15.999
constexpr double L = 10.0;
cpu::gpw::GpwDft dft{{Ni, O}, 250.0, cpu::gpw::XcKind::PADE};
dft.max_iter = 300;
dft.tol = 1e-5;
dft.mix = 0.35;               // DIIS path only; ignored when use_ot
dft.nonperiodic = true;
dft.use_ot = true;
dft.n_alpha = 13;             // multiplicity 3, 24 valence e⁻
dft.n_beta = 11;
dft.plus_u.push_back({molrs::Element::Ni, /*l=*/2, /*u_minus_j_ev=*/6.0});
cpu::gpw::GPWEvaluator eval(std::move(dft));

cpu::single_point(frame, eval);
const cpu::gpw::GpwResult& r = eval.last_result();
const double e_u = r.e_dftu;                             // [Ha]
const double m_Ni = r.mulliken_spin.empty() ? 0.0 : r.mulliken_spin[0];
```

### Validation (vs CP2K)

| Item | Value |
| ---- | ----- |
| System | NiO **molecule**, charge 0, UKS multiplicity 3 |
| Bond | \(R(\mathrm{Ni–O})=\) **1.668 Å** along *z*, **centered** |
| Coordinates (Å) | Ni (5, 5, 4.166), O (5, 5, 5.834) |
| Cell | **10 × 10 × 10 Å**, `PERIODIC NONE`, MT |
| Basis | Ni `SZV-GTH-PADE-q18`, O `SZV-GTH-PADE` |
| PP | Ni `GTH-PADE-q18` (18e), O `GTH-PADE-q6` (6e) → **24** valence e⁻ |
| XC | PADE (LDA LSD) |
| DFT+U | Ni \(L=2\), \(U-J=6.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| Cutoff | **250 Ry** (ATV single density grid; CP2K `NGRIDS 4` / `REL_CUTOFF 40`) |
| Spin | \(N_\alpha=13\), \(N_\beta=11\) |
| SCF twin | CP2K: OT `FULL_SINGLE_INVERSE` + OUTER_SCF (`nio_mol_pade_u_ot.inp`, OMP=8); ATV: `use_ot=true`, `tol=1e-5`, `max_iter=300` |

| Quantity | CP2K pin | Atomiverse | Tol (regression) | Notes |
| -------- | -------: | ---------: | ---------------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−184.92466076474310** | ≈ −184.94575 | **5×10⁻² Ha** | single-grid vs multigrid+OT |
| \(\lvert\Delta E\rvert\) vs CP2K | — | ≈ **2.1×10⁻²** Ha | ≤ 5×10⁻² | |
| \(\lvert E_{\mathrm{OT+U}}-E_{\mathrm{DIIS+U}}\rvert\) (ATV) | — | ≈ **3.8×10⁻⁴** Ha | **1×10⁻³ Ha** | fixed-point parity |
| DFT+U energy [Ha] | **0.05230865222344** | \(>10^{-4}\) (asserted) | diagnostic | `r.e_dftu`; not equality-asserted |
| \(\langle S^2\rangle\) (single det.) | **2.019198** | ≈ 2.015 | **0.25** | ideal \(S(S+1)=2\) |
| Mulliken \(m_{\mathrm{Ni}}\) [e] | **1.177632** | ≈ 1.00 | **0.5** and \(m_{\mathrm{Ni}}>0.3\) | spin localization |
| Frame energy consistency | — | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}}\equiv E_{\mathrm{Ha}}\) | \(10^{-9}\) Ha | |
| SCF | converged | `r.converged` | — | ~23 OT iters (ATV) |

```text
  DFT+U energy:                                                 0.05230865222344
  Ideal and single determinant S**2 :                    2.000000       2.019198
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.924660764743095
```

### Performance

Cutoff 250 Ry, PADE+U, MT, single rank, `OMP_NUM_THREADS=8`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| NiO mol OT+PADE+U / SZV / MT, Γ | **13.65** | **3.58** (OT+U) | **0.26** | ✓ | CP2K multigrid+OT; ATV Release single density grid |
| ATV DIIS+U (same binary) | — | **3.03** | — | — | dual-path check only |

The ratio is for this SZV molecule; it does not forecast solid NiO, multigrid
parity, or sparse OT.

### Limitations

- Molecule, not solid AFM rocksalt NiO; SZV-only minimal basis — the shared
  limitations of [gpw-dftu-nio](gpw-dft-plus-u.md#gpw-dftu-nio) apply.
- PADE LSD only (UKS + PBE-GGA unsupported on GPW UKS); Γ only for this
  section — multi-k OT+U is [ot-dftu-kpoints](#ot-dftu-kpoints) /
  [ot-uks-dftu](#ot-uks-dftu).

## ot-dftu-kpoints — Multi-k RKS OT + DFT+U, H₂ toy {#ot-dftu-kpoints}

*No section of the case runs this variant.*

Complex multi-k OT with Dudarev DFT+U — integer occupations, OUTER_SCF +
fixed-H OT per \(k\) (**Common method**). Neither Γ-only OT+U
([ot-dftu](#ot-dftu)) nor multi-k DIIS+U
([gpw-dftu-kpoints](gpw-dft-plus-u.md#gpw-dftu-kpoints)) covers it alone. The
CP2K input `h2_ot_dftu_kpoints.inp` is the DIAGONALIZATION + DIRECT_P_MIXING
multi-k PLUS_U twin of the same fixed point (OT+k would abort).
**`perf_exempt`**.

```cpp
// H2 toy (Common systems)
GpwDft m{{H}, 100.0, XcKind::PADE};
m.n_occ = 1;
m.max_iter = 200; m.tol = 1e-7; m.mix = 0.35;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;             // TR-fold → 4 k for 2×2×2
m.use_ot = true;
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});
m.build(frame);
const GpwResult& r = m.run(frame);    // r.e_total, r.e_dftu [Ha], r.n_kpts, r.iters

// A1 / A2: explicit single k instead of the mesh
m.k_points = {{0.25, 0.0, 0.0}};      // or {{0,0,0}} for Γ
m.k_point_weights = {1.0};
```

### Recorded validation

Closed-shell RKS (\(n_{\mathrm{occ}}=1\)); k: A1 explicit Γ, A2 explicit
\(\kappa=(1/4,0,0)\), A3 MP \(2\times2\times2\) + time reversal → **4** k.

| Quantity | Atomiverse | Twin / notes | Tol | Gate |
| -------- | ---------: | ------------ | --- | ---- |
| A1 \(E_{\mathrm{OT+U}}\) (Γ multi-k path) | −1.0867759 Ha | DIIS+U −1.0867757 Ha | \(\lvert\Delta\rvert\le 10^{-4}\) Ha | OT ≡ DIIS |
| A1 \(e_{\mathrm{dftu}}\) | non-trivial; OT vs DIIS | — | \(\lvert\Delta e_U\rvert\le 10^{-3}\) Ha | |
| A2 \(E_{\mathrm{OT+U}}\) (\(\kappa=\frac14\)) | **−1.081998145** Ha | DIIS+U −1.081997919 Ha (\(\lvert\Delta\rvert\sim 2.3\times10^{-7}\)) | \(10^{-4}\) Ha | OT ≡ DIIS |
| A3 \(E(U=0)\), OT MP\(2^3\) | −1.102850837 Ha | activation baseline | — | diagnostic |
| A3 \(E(U=4)\), OT MP\(2^3\) | **−1.077477969** Ha | frozen ATV absolute | \(10^{-3}\) Ha | absolute pin |
| A3 \(e_{\mathrm{dftu}}\) | \(\approx -0.0023\) Ha | — | \(\lvert e_U\rvert > 10^{-8}\) | non-trivial U |
| A3 \(\lvert E(U)-E(0)\rvert\) | \(\approx 0.0254\) Ha | — | \(\ge 10^{-4}\) Ha | activation |
| A3 \(n_{\mathrm{kpts}}\) | 4 | TR-folded \(2\times2\times2\) | exact | mesh |

ATV wall for A1–A3 (Release) ≈ **15–25** s.

### Limitations

- Toy H \(L=0\) U on DZV PADE — an SCF-combo fixture, not a correlated-solid pin.
- Multi-k UKS OT+U is [ot-uks-dftu](#ot-uks-dftu); with smearing,
  [ot-dftu-smearing](#ot-dftu-smearing).

## ot-dftu-smearing — Multi-k RKS OT + DFT+U + Fermi–Dirac, H₂ toy {#ot-dftu-smearing}

*No section of the case runs this variant.*

Closed-shell multi-k OT + Dudarev DFT+U + Fermi–Dirac occupations (Mermin free
energy): free-energy PR+ Armijo OT on the full AO spectrum with fractional
\(f\), \(V_U\) folded into each \(H_k\) via `build_k_energy`. Same fixed point as
multi-k DIIS + U + FD on identical grids. One explicit general
\(\kappa = (1/4,0,0)\), periodic cell (not MT). CP2K twin: multi-k DIIS +
FERMI_DIRAC + PLUS_U (`&DIAGONALIZATION` + `&SMEAR METHOD FERMI_DIRAC` +
`ADDED_MOS 2`). **`perf_exempt`**.

```cpp
// H2 toy (Common systems)
GpwDft m{{H}, 100.0, cpu::gpw::XcKind::PADE};
m.n_occ = 1;                 // closed-shell RKS, Ne = 2
m.max_iter = 220; m.tol = 1e-7; m.mix = 0.35;
m.k_points.push_back({0.25, 0.0, 0.0});
m.k_point_weights.push_back(1.0);
m.smearing = Smearing::fermi_dirac;
m.degauss = 0.02;            // Ry (CP2K Te ≈ 3157.75 K)
m.extra_bands = 2;
m.plus_u.push_back({molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});
m.use_ot = true;
m.build(frame);
const GpwResult& r = m.run(frame);   // r.e_total = F; r.e_dftu, r.e_fermi, r.e_entropy; r.n_elec_occ ≈ 2
```

### Recorded validation

SCF `tol=1e-7`, `max_iter` 160–220, mix 0.35 (DIIS twin).

| Quantity | Atomiverse | Twin / notes | Tol | Gate |
| -------- | ---------: | ------------ | --- | ---- |
| \(F_{\mathrm{OT}}\) (U=4, FD) | **−1.081998158198 Ha** | frozen ATV absolute | \(10^{-3}\) Ha | A6 |
| \(F_{\mathrm{DIIS}}\) same | −1.081997918765 Ha | same grids as OT | — | A1 |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **2.4×10⁻⁷ Ha** | fixed-point parity | \(\le 10^{-4}\) Ha | A2 |
| \(E_U\) (OT) | ≈ −8.57×10⁻⁴ Ha | non-trivial; OT vs DIIS \(E_U\) | \(\le 10^{-3}\) Ha | A3 |
| \(N_e=\sum w f\) | **2.0** | valence count | \(10^{-5}\) | A4 |
| \(F(U=0)\) OT+FD | −1.108020857737 Ha | activation baseline | — | A5 |
| \(\lvert F(U)-F(0)\rvert\) | ≈ **0.026 Ha** | U activation floor | \(\ge 10^{-4}\) Ha | A5 |
| OT / DIIS iters | 11 / 42 | diagnostic | — | — |

Walls: OT-only **~0.169** s (Release, OMP=4); full check A0–A7 **≈ 0.4** s.

## ot-kpoints — Multi-k RKS OT, compact H₂ {#ot-kpoints}

*No section of the case runs this variant.*

Closed-shell multi-k OT on complex Bloch \(C_k\) (\(C_k^\dagger S_k C_k = I\)
per \(k\), Hermitian \(H_k,S_k\), k-summed density, joint CG + Armijo). Multi-k OT
energy vs in-tree multi-k DIIS on identical grids, plus a frozen ATV absolute.
The CP2K OT+KPOINTS input ships but cannot run on 2026.1; fair wall
**`perf_exempt`**. With U, smearing or UKS see the sibling variants.

```cpp
// H2 toy (Common systems); explicit single k = (1/4, 0, 0) — CP2K GENERAL twin
GpwDft ot{{H}, 100.0, cpu::gpw::XcKind::PADE};
ot.n_occ = 1;
ot.max_iter = 200; ot.tol = 1e-8; ot.eps_default = 1e-7;
ot.use_ot = true;
ot.k_points.push_back({0.25, 0.0, 0.0});
ot.k_point_weights.push_back(1.0);
ot.build(frame);
const auto& ro = ot.run(frame);       // ro.e_total [Ha]; ro.n_kpts; ro.k_frac

// Monkhorst–Pack 2×1×1 with time reversal (A3)
GpwDft mp{{H}, 100.0, cpu::gpw::XcKind::PADE};
mp.n_occ = 1; mp.max_iter = 200; mp.tol = 1e-8; mp.use_ot = true;
mp.k_mesh[0] = 2; mp.k_mesh[1] = 1; mp.k_mesh[2] = 1;
mp.k_time_reversal = true;
```

### Recorded validation

| Quantity [Ha] | Atomiverse | Tol | Notes |
| ------------- | ---------: | --- | ----- |
| \(E_{\mathrm{tot}}\) multi-k OT (\(\kappa=1/4\)) | **−1.1080204833** | 1e-3 (A5, frozen) | 13 OT iters |
| \(E_{\mathrm{tot}}\) multi-k DIIS (same grids) | −1.1080204833 | — | 5 DIIS iters |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\) (A2) | \(6.2\times10^{-11}\) | ≤ 1×10⁻⁴ | identity |
| real Γ OT | −1.1123413990 | — | real-path baseline |
| k-path Γ OT | −1.1123413989 | ≤ 1×10⁻⁶ vs real (A1) | explicit Γ via k-list |
| MP \(2\times1\times1\) OT ≡ DIIS (A3) | — | 1e-4 | TR mesh |

SCF `max_iter` up to 200, `tol=1e-8`. CP2K OT settings in the shipped input:
`&OT` `MINIMIZER DIIS`, `PRECONDITIONER FULL_SINGLE_INVERSE`, outer SCF. The
recorded check runs several OT+DIIS constructions (A1–A5) and is not a
single-path fair twin.

## ot-smearing — Multi-k RKS OT + Fermi–Dirac, BCC Na {#ot-smearing}

*No section of the case runs this variant.*

OT with metallic occupations on multi-k closed-shell RKS: the manifold grows
to \(n_{\mathrm{occ}} + \max(\texttt{extra\_bands},1)\), occupations are FD/MP
(shared with the PW path) and the projected gradient uses non-uniform band
occupations (**Common method**). Gate: free energy OT+FD ≡ DIIS+FD, \(N_e=18\),
frozen \(F\). The CP2K twin is DIIS + FERMI_DIRAC (stock CP2K OT does not combine
with SMEAR). Fair OT-only wall ratio \(r \approx 0.20\) (≤10×).

```cpp
// BCC Na (Common systems)
m.n_occ = 9;            // Ne/2 for GTH-PBE-q9 × 2
// r.e_total = F; r.e_fermi [Ha]; r.n_elec_occ ≈ 18; r.e_entropy (−TS); r.n_kpts > 1
```

DIIS twin: same members with `use_ot = false`.

### Recorded validation

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| Free energy \(F\) (OT) | **−92.1263304935 Ha** | **\(10^{-3}\) Ha** | frozen ATV absolute (A5) |
| Free energy \(F\) (DIIS twin) | −92.1263306084 Ha | — | same grids |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.15×10⁻⁷ Ha** | ≤ \(10^{-4}\) Ha | identity (A3) |
| \(N_{\mathrm{occ}}=\sum w f\) | **18.0** | \(10^{-5}\) | valence count (A4) |
| OT iters | 12 | — | diagnostic |

Further gates: A1 `build` accepts RKS OT + FD; A2 OT+FD converges with finite
\(E_F\), entropy and multi-k active.

### Performance

Cutoff 150 Ry, PADE, multi-k \(2^3\), single rank, `OMP_NUM_THREADS=4`.

| Twin | Wall [s] | \(r\) | ≤10× | Caveat |
| ---- | -------: | ----: | ---- | ------ |
| CP2K DIIS+FD (`na_ot_fermi_dirac.inp`) | **~89.5** | — | — | DIIS+SMEAR, not OT+SMEAR |
| ATV OT+FD only (fair) | **~18.2** | **~0.20** | ✓ | primary ratio |

\(r \approx 18.2 / 89.5 \approx 0.20 \ll 10\) — dual-gate throughput against a
DIIS twin, not a same-algorithm OT race. Methfessel–Paxton is available via the
members and unit tests; the recorded gate is Fermi–Dirac.

## ot-uks-dftu — Multi-k UKS OT + DFT+U, high-spin H₂ {#ot-uks-dftu}

*No section of the case runs this variant.*

Multi-k collinear UKS OT with Dudarev DFT+U (integer occupations): OT applied
per \((k,\sigma)\) with OUTER_SCF + fixed-H OT (`ot_minimize_fixed_k`); without
`plus_u`, multi-k UKS OT uses the dual-spin loop of
[ot-uks-kpoints](#ot-uks-kpoints). High-spin H₂ (\(N_\alpha=2\), \(N_\beta=0\),
two valence e⁻ on GTH-q1 H). CP2K twin: multi-k UKS DIAGONALIZATION +
DIRECT_P_MIXING ALPHA 0.35 + PLUS_U (fixed-point documentation only).
**`perf_exempt`**.

```cpp
// H2 toy (Common systems); A3-class product: MP 2×1×1, time reversal, OT + U
GpwDft m{{H}, 100.0, XcKind::PADE};
m.n_alpha = 2; m.n_beta = 0;
m.max_iter = 200; m.tol = 1e-7; m.mix = 0.35;
m.k_mesh[0] = 2; m.k_mesh[1] = 1; m.k_mesh[2] = 1;
m.k_time_reversal = true;
m.use_ot = true;
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});
m.build(frame);
const GpwResult& r = m.run(frame);   // r.e_total, r.e_dftu [Ha], r.s2, r.n_kpts
// A2 class: k_points {{0.25, 0, 0}}, weights {1.0} (overrides k_mesh)
```

### Recorded validation

k: A1 explicit Γ (complex path at the Γ limit), A2 \(\kappa = (¼, 0, 0)\),
A3 MP **2×1×1** with time reversal. ATV: `tol=1e-7`, `max_iter=200`.

| Quantity | Atomiverse | Twin | Tol | Notes |
| -------- | ---------: | ---- | --- | ----- |
| A1 \(E\) OT+U @ Γ | −0.7802796 Ha | DIIS+U −0.7802792 Ha | **1×10⁻⁴ Ha** | multi-k UKS path at Γ |
| A1 \(\lvert\Delta E\rvert\) OT−DIIS | ≈ \(4\times10^{-7}\) | — | ≤ 1×10⁻⁴ | identity |
| A1 \(e_{\mathrm{dftu}}\) | non-trivial; OT≡DIIS | — | 1×10⁻³ Ha on \(e_U\) | |
| A2 \(E\) OT+U (κ=¼) | **−0.802825557 Ha** | DIIS+U −0.802825142 Ha | **1×10⁻⁴ Ha** | 32 OT outer / 66 DIIS |
| A2 \(\lvert\Delta E\rvert\) | ≈ \(4.1\times10^{-7}\) Ha | — | ≤ 1×10⁻⁴ | identity |
| A2 \(e_{\mathrm{dftu}}\), \(\langle S^2\rangle\) | OT ≡ DIIS | — | \(e_U\) 1×10⁻³; \(S^2\) 1×10⁻⁶ | |
| A3 \(E(U=0)\) OT MP 2×1×1 | −0.706134617 Ha | — | — | no-U baseline |
| A3 \(E(U=4)\) OT MP 2×1×1 | **−0.802825557 Ha** | frozen ATV absolute | **1×10⁻³ Ha** | |
| A3 \(e_{\mathrm{dftu}}\) | ≈ −0.1803 Ha | — | \(\lvert e_U\rvert>10^{-8}\) | diagnostic |
| A3 \(\lvert E(U)-E(0)\rvert\) | ≈ 0.0967 Ha | — | ≥ **1×10⁻⁴ Ha** | activation floor |

ATV wall (A1–A3, OMP=8, Release) **≈ 0.48 s**.

## ot-uks-dftu-smearing — Multi-k UKS OT + DFT+U + Fermi–Dirac, H₂ toy {#ot-uks-dftu-smearing}

*No section of the case runs this variant.*

The full stack on multi-k collinear UKS: OT, Dudarev DFT+U and Fermi–Dirac
occupations. The U + smear path uses free-energy PR+ Armijo OT on the full AO
spectrum per spin with one \(\mu\) over \(\alpha+\beta\) (\(f_\sigma\in[0,1]\)) and
\(V_U^\sigma\) folded into \(H_k^\sigma\) (`build_uks_k_energy`). Equal-spin seed
(\(n_\alpha=n_\beta=1\), CP2K `MULTIPLICITY 1`) keeps DIIS and free-energy OT on
the same Mermin minimum; a high-spin integer seed (\(n_\alpha=2\), \(n_\beta=0\))
can leave DIIS metastable while OT finds the lower closed-shell-like minimum.
CP2K twin: multi-k UKS DIIS + FERMI_DIRAC + PLUS_U (`DIAGONALIZATION` +
`DIRECT_P_MIXING` ALPHA 0.35); `&OT` under the same setup aborts.
**`perf_exempt`**.

```cpp
// H2 toy (Common systems)
GpwDft m{{H}, 100.0, cpu::gpw::XcKind::PADE};
m.n_occ = 0;
m.n_alpha = 1;   // equal-spin seed under FD
m.n_beta = 1;
m.max_iter = 220; m.tol = 1e-7; m.mix = 0.35;
m.k_points.push_back({0.25, 0.0, 0.0});
m.k_point_weights.push_back(1.0);
m.smearing = cpu::gpw::Smearing::fermi_dirac;
m.degauss = 0.02;   // Ry
m.extra_bands = 2;
m.plus_u.push_back({molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});
m.use_ot = true;
m.build(frame);
const GpwResult& r = m.run(frame);   // r.e_total = F; r.e_dftu; r.n_elec_occ ≈ 2
```

### Recorded validation

ATV: `tol=1e-7`, `max_iter=160` (DIIS) / `220` (OT), mix 0.35; gates A0–A6.

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(F_{\mathrm{OT}}\) (U=4, FD) [Ha] | **−1.081998158198** | **1×10⁻³** (A6 frozen) | primary absolute |
| \(F_{\mathrm{DIIS}}\) same [Ha] | −1.081997968639 | — | same grids |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.9×10⁻⁷** | ≤ **1×10⁻⁴** (A2) | decisive fixed point |
| \(E_U\) (OT) [Ha] | ≈ −8.57×10⁻⁴ | non-trivial; OT vs DIIS ≤ 1×10⁻³ (A3) | `r.e_dftu` |
| \(N_e\) | **2.0** | 1×10⁻⁵ (A4) | valence count |
| \(F(U=0)\) OT+FD [Ha] | −1.108020857737 | — | activation baseline |
| \(\lvert F(U)-F(0)\rvert\) | ≈ 0.026 | ≥ **1×10⁻⁴** (A5) | U activation floor |
| OT / DIIS iters | 11 / 43 | converged (A1) | |

OT-only wall ≈ **0.325** s (OMP=4, Release), indicative only.

## ot-uks-kpoints — Multi-k UKS OT, high-spin H₂ {#ot-uks-kpoints}

*No section of the case runs this variant.*

Multi-k collinear UKS OT on the complex Bloch path with dual spin manifolds
(occupation 1 per spin; joint PR+ CG over spin × k with Armijo on the rebuilt
total energy), between multi-k RKS OT ([ot-kpoints](#ot-kpoints)) and Γ UKS OT
([ot-uks-o2](#ot-uks-o2)). Gates: OT ≡ DIIS, real Γ ≡ k-path Γ, frozen ATV
absolute. The CP2K input documents a DIAG multi-k UKS twin only.
**`perf_exempt`**.

```cpp
// H2 toy (Common systems); multi-k collinear UKS OT at κ = (1/4, 0, 0)
cpu::gpw::GpwDft ot{{H}, 100.0, cpu::gpw::XcKind::PADE};
ot.n_alpha = 2;   // high-spin H₂: Nα=2, Nβ=0
ot.n_beta = 0;
ot.n_occ = 0;     // UKS path when n_alpha+n_beta > 0
ot.max_iter = 200; ot.tol = 1e-8; ot.eps_default = 1e-7;
ot.use_ot = true;
ot.k_points.push_back({0.25, 0.0, 0.0});
ot.k_point_weights.push_back(1.0);
ot.build(frame);
const auto& res = ot.run(frame);      // res.e_total [Ha]; res.s2; res.n_kpts; res.k_frac
// A3: k_mesh {2,1,1} + k_time_reversal; A1: k_points {{0,0,0}} vs real Γ
```

### Recorded validation

SCF `max_iter=120` (DIIS) / `200` (OT), `tol=1e-8`.

| Quantity [Ha] | Atomiverse | Tol | Notes |
| ------------- | ---------: | --- | ----- |
| multi-k UKS OT \(E\) (κ=¼) | **−0.7061343842** | 1e-3 vs frozen (A4) | |
| multi-k UKS DIIS (same grids) | −0.7061343842 | — | 6 DIIS iters |
| \(\lvert E_{\mathrm{OT}}-E_{\mathrm{DIIS}}\rvert\) | \(4.8\times10^{-12}\) | ≤ **1×10⁻⁴**; \(S^2\) 1e-6 (A2) | fixed point |
| real Γ UKS OT | −0.6687052609 | — | real Γ path |
| k-path Γ UKS OT | −0.6687052608 | ≤ **1×10⁻⁶** vs real Γ (A1) | |
| MP 2×1×1 OT ≡ DIIS (A3) | — | ≤ **1×10⁻⁴** | TR mesh |

Compact high-spin H₂ multi-k UKS DIAG on CP2K does not converge reliably
(outer-SCF energy oscillation), so it is not a digit pin. The toy is the same
class as [gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks); multi-k UKS forces and
stress are [gpw-kpoint-uks-forces](gpw-uks.md#gpw-kpoint-uks-forces) /
[gpw-kpoint-uks-stress](gpw-uks.md#gpw-kpoint-uks-stress) (DIIS).

## ot-uks-smearing — Multi-k UKS OT + Fermi–Dirac, BCC Na {#ot-uks-smearing}

*No section of the case runs this variant.*

Collinear UKS OT with Fermi–Dirac occupations on a multi-k mesh: fractional
spin-orbital \(f_{\sigma kb}\in[0,1]\), one \(\mu\), Mermin free energy, OT
manifold expanded to the full independent AO spectrum. Seed integers
\(n_\alpha=10\), \(n_\beta=8\) set the cold-start ferro density only; BCC Na is
non-magnetic and the seed collapses under smear (residual magnetization is not
an observable here). Gate: OT+FD ≡ DIIS+FD, \(N_e=18\), frozen \(F\). CP2K twin:
DIIS + LSD + FERMI_DIRAC. Fair OT-only wall ratio \(r \approx 0.12\) (OMP=4).

```cpp
// BCC Na (Common systems), UKS seed
m.n_occ = 0;
m.n_alpha = 10; // ferro seed; free under smear
m.n_beta = 8;
// r.e_total = F; r.e_fermi [Ha]; r.n_elec_occ ≈ 18; r.e_entropy
```

### Recorded validation

| Quantity | Atomiverse | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(F\) OT [Ha] | **−92.1263304935** | \(10^{-3}\) Ha | frozen ATV absolute (A5) |
| \(F\) DIIS twin (ATV) [Ha] | −92.1263306080 | — | same grids |
| \(\lvert F_{\mathrm{OT}}-F_{\mathrm{DIIS}}\rvert\) | **1.15×10⁻⁷** | ≤ \(10^{-4}\) (A3) | identity |
| \(N_{\mathrm{occ}}\) | **18.0** | \(10^{-5}\) (A4) | electron count |
| OT iters | 12 | — | |

Further gates: A1 `build` accepts UKS OT + FD; A2 OT+FD and DIIS+FD converge
with multi-k active; A6 finite \(E_F\) and entropy term.

### Performance

Cutoff 150 Ry, PADE, multi-k \(2^3\), single rank, `OMP_NUM_THREADS=4`.

| Twin | Wall [s] | \(r\) | ≤10× | Caveat |
| ---- | -------: | ----: | ---- | ------ |
| CP2K DIIS+LSD+FD twin | **~89.5** | — | — | no OT+kpoints twin exists |
| ATV OT+FD only | **~10.3** | **~0.115** | ✓ | fair OT-only wall |
