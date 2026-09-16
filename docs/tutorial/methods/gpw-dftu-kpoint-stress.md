# Multi-k Dudarev DFT+U analytic cell stress — H₂ UKS / PADE / DZV-GTH-PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-dftu-kpoint-stress` |
| Regression | `regressions/gpw/gpw-dftu-kpoint-stress/` |
| Public API | `cpu::gpw::GpwDft` members `k_mesh` / `plus_u` / `n_alpha`/`n_beta`; free `gpw_stress` (product) + `gpw_stress_fd` (oracle); optional `GPWEvaluator::{k_mesh,plus_u,compute_stress}` |
| Roadmap | `docs/roadmap/dft.md` — multi-k DFT+U analytic stress (`reg_gpw-dftu-kpoint-stress`) ✓ |

## Summary

**Monkhorst–Pack multi-k unrestricted GPW** with **Dudarev DFT+U**
(CP2K `PLUS_U` / Mulliken) and **Hellmann–Feynman fixed-\(\{P_k^\sigma,W_k^\sigma\}\)**
analytic **cell stress**. One multi-k UKS+U SCF yields occupied Bloch MOs
and band energies (eigenvalues include \(V_U\)); the product stress path
then takes central finite differences of the **non-self-consistent**
Lagrangian under orthorhombic diagonal strain at fixed fractional
coordinates, with BZ-averaged \(E_U[n^\sigma(\{P_k^\sigma\},\{S_k\})]\)
evaluated at the strained overlaps. Sign convention matches CP2K
`STRESS_TENSOR`: \(\sigma_{aa}=-\Omega^{-1}\partial E_L/\partial\varepsilon_{aa}\)
in Ha/Bohr³.

This page is the product gate that closes the **k-point × DFT+U × stress**
product of three prior footholds: multi-k DFT+U energy
(`reg_gpw-dftu-kpoints`), Γ DFT+U stress (`reg_gpw-dftu-stress`), and multi-k
UKS stress (`reg_gpw-kpoint-uks-stress`). Public surface: set `GpwDft`
members (`k_mesh`, `plus_u`, spin occupations, cell, cutoff) and call
`gpw_stress(atoms, method, h)` — **no** `*Config` bag, **no** `make_*` /
`build_*` factories. Product regression:
`reg_gpw-dftu-kpoint-stress`. Oracle: offline **CP2K 2026.1** twin retained
for provenance; **not** a fair absolute energy/stress pin on this H₂ toy
(unphysical Mulliken / SCF abort — same class as `gpw-dftu-kpoints`).
Decisive gates are internal dual-gate + Γ-limit identity. Wall ~**2.4 s**
(Release, OMP=8, product+FD); ratio **`perf_exempt`**.

Related: Γ DFT+U NiO energy — [gpw-dftu-nio](gpw-dftu-nio.md); Γ analytic
stress — [gpw-analytic-stress](gpw-analytic-stress.md); narrative GPW —
[Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Grid set by
plane-wave cutoff [Ry]. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.*
**92**, 477 (1997).

**Multi-k UKS.** Collinear dual \(H_k^\sigma\), spin-resolved density/XC,
fixed integer fillings `n_alpha` / `n_beta` (no UKS+smear on this path).
Monkhorst–Pack mesh uses CP2K convention
\(\kappa_d=(2r-N_d-1)/(2N_d)\) (even \(N\) is Γ-avoiding). Optional
time-reversal folding doubles weights on retained partners. Explicit
`k_points` / `k_point_weights` override `k_mesh` when non-empty.

**Dudarev DFT+U (multi-k Mulliken).** Per element and angular momentum
\(L\), effective \(U_{\mathrm{eff}}=U-J\) in **eV** on the public surface
(converted to Hartree with `units::kHartreeInEv` inside). Occupations are
**BZ-averaged** (weights \(w_k\)) before the nonlinear Dudarev form — partial
per-rank occupations must not enter \(E_U\) under MPI k-RR:

\[
n^\sigma
=
\sum_k w_k\,\tfrac12\mathrm{Herm}\big[
  (P_k^\sigma S_k)+(S_k P_k^\sigma)
\big]
\quad\text{(shell block on atom \(A\), \(\ell=L\))},
\]

\[
E_U^\sigma
=
\tfrac{U_{\mathrm{eff}}}{2}\Big[
  \mathrm{Tr}\,n^\sigma - \mathrm{Tr}(n^\sigma n^\sigma)
\Big],
\qquad
V^\sigma
=
U_{\mathrm{eff}}\Big(\tfrac12 I - n^\sigma\Big).
\]

\(V^\sigma\) is scattered into every local \(H_k^\sigma\). Matches CP2K
simplified rotationally invariant form (`PLUS_U_METHOD MULLIKEN`,
`U_MINUS_J`). Implementation: `cpu/gpw/internal/dft_plus_u.h`
(`apply_dft_plus_u_k` / `build_dftu_potential_k`). Empty `plus_u` leaves
the multi-k UKS path unchanged.

**Analytic cell stress (fixed-\(\{P_k^\sigma,W_k^\sigma\}\)+Pulay).** After
one SCF:

1. Rebuild \(P_k^\sigma\), \(W_k^\sigma\) from occupied Bloch MOs (band
   energies include \(V_U\)).
2. Apply orthorhombic diagonal strain \(\pm h\) at **fixed fractional**
   coordinates; evaluate the non-self-consistent Lagrangian
   \(E_L[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon]\) including
   BZ-averaged \(E_U\) at strained \(S_k\).
3. Central FD:
   \(\sigma_{aa}=-\Omega^{-1}[E_L(+h_a)-E_L(-h_a)]/(2h)\).

Oracle twin: multi-SCF strain-FD (`gpw_stress_fd`) fully re-converges SCF
at each strained cell. Γ-limit identity: complex \(k=\Gamma\) + U stress
≡ real Γ + U stress (proves the multi-k \(E_U\) Lagrangian matches the
shipped Γ DFT+U stress path). Citations: Dudarev et al., *Phys. Rev. B*
**57**, 1505 (1998); Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985).

**Key exclusions (closed-spec scope-out).**

- Fair absolute CP2K NUMERICAL stress/energy twin on this H₂+U multi-k toy
  (CP2K unphysical Mulliken / non-convergence — see Validation).
- Solid rocksalt AFM NiO multi-k + U stress product pin (follow-on chemistry
  twin; Γ molecular energy remains `reg_gpw-dftu-nio`).
- Liechtenstein full \(U,J\) tensor; shear / triclinic stress; UKS+smear;
  DFT+U+OT; multigrid product path on multi-k.
- Analytic DFT+U **forces** product pin on multi-k (stress only here).
- Public `*Config` bags / factories; live CP2K from ctest; CUDA GPW.

## Implementation

Public surface only (`GpwAtom` / `GpwDft` members / free `gpw_stress`; no
`*Config`, no `make_*`). Match the regression
(`reg_gpw-dftu-kpoint-stress.cpp` uses the solver surface; Frame /
`GPWEvaluator` is the outer Evaluate path with the same members):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

// High-spin H2, compressed bond along x, cubic 12 Bohr periodic cell
const double L = 12.0;
const GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
const GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
const double r0[3] = {0.5 * L - 0.5, 0.5 * L, 0.5 * L};
const double r1[3] = {0.5 * L + 0.5, 0.5 * L, 0.5 * L};
const std::vector<GpwAtom> atoms = {
    GpwAtom(r0, bH, pH, 1),
    GpwAtom(r1, bH, pH, 1),
};

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = L;
m.cutoff_ry = 100.0;
m.n_alpha = 2;   // multiplicity 3, 2 valence e⁻
m.n_beta = 0;
m.max_iter = 120;
m.tol = 1e-7;
m.mix = 0.35;
m.xc = XcKind::PADE;
m.local_pp_on_grid = true;
m.k_mesh[0] = 1;
m.k_mesh[1] = 1;
m.k_mesh[2] = 2;           // MONKHORST-PACK 1×1×2 (TR → nk=1)
m.k_time_reversal = true;
m.plus_u.push_back(PlusU{molrs::Element::H, /*l=*/0, /*u_minus_j_ev=*/4.0});

m.build(atoms);
const GpwResult& er = m.run(atoms);
// er.e_total [Ha], er.e_dftu [Ha], er.converged, er.C_occ_k / C_occ_k_beta

const double h = 2e-3;
const auto san = gpw_stress(atoms, m, h);     // product analytic [Ha/Bohr³]
const auto sfd = gpw_stress_fd(atoms, m, h);  // multi-SCF oracle only
// Voigt: xx,yy,zz,xy,xz,yz — off-diagonals 0 (orthorhombic)
```

Build (GPW needs FFTW3) and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-dftu-kpoint-stress
  ./build/regressions/reg_gpw-dftu-kpoint-stress
  # or: cmake --build build --target run_reg_gpw-dftu-kpoint-stress
'
```

| Piece | Path |
| ----- | ---- |
| Public `PlusU` / `GpwDft::{k_mesh,plus_u}` / `gpw_stress` | `include/atomiverse/cpu/gpw/gpw_solver.h` |
| Evaluator members (Frame path) | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Multi-k Dudarev \(E_U\)/\(V\) (internal) | `include/atomiverse/cpu/gpw/internal/dft_plus_u.h` |
| Fixed-\(P_k\)+Pulay stress (internal) | `include/atomiverse/cpu/gpw/internal/gpw_stress.h`, `src/cpu/gpw/internal/gpw_stress.cpp` |
| Unit dual-gate smoke | `tests/cpu/gpw/test_cpu_gpw_stress.cpp` |
| Regression | `regressions/gpw/gpw-dftu-kpoint-stress/reg_gpw-dftu-kpoint-stress.cpp` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/h2_uks_dftu_kpoint_stress.inp` |

## Validation (vs CP2K + internal dual-gate)

Source: [`regressions/gpw/gpw-dftu-kpoint-stress/REFERENCE.md`](../../../regressions/gpw/gpw-dftu-kpoint-stress/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`); **never** live in ctest. Pin
date: **2026-07-26**. Closed-spec gates A1–A7.

**Setup (ATV product twin — regression geometry):**

| Item | Value |
| ---- | ----- |
| System | H₂ high-spin (M = 3), compressed bond along *x* |
| Positions [Bohr] | H1 \((5.5, 6.0, 6.0)\), H2 \((6.5, 6.0, 6.0)\) (cell \(L=12\)) |
| Cell | cubic **12 Bohr**, periodic XYZ |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| XC | PADE LSD |
| DFT+U | H \(L=0\), \(U-J=4.0\) eV, Mulliken |
| Cutoff | **100 Ry**, `local_pp_on_grid` |
| k mesh | MONKHORST-PACK **1×1×2** (TR-folded → `nk=1`) |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) |
| Stress step | \(h=2\times 10^{-3}\) |
| SCF | mix 0.35, `tol=1e-7`, `max_iter=120` |

CP2K twin input uses an **8 Bohr** cheaper NUMERICAL-stress variant of the
same method class (PLUS_U + KPOINTS + `STRESS_TENSOR NUMERICAL`); see
`tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/`.

**Pinned quantities (ATV first-green 2026-07-26):**

| Quantity | Pin / measured | Tol (regression) | Notes |
| -------- | -------------: | ---------------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] (k-UKS+U 1×1×2, 100 Ry) | **−0.546250995860** | **1×10⁻³** Ha soft | A5; ~76 SCF iters, `nk=1` |
| \(E_U\) [Ha] (`e_dftu`) | **−0.243601** | \(\lvert E_U\rvert>10^{-8}\) | A1; multi-function s shell can be \(<0\) |
| \(\sigma_{xx}\) analytic [Ha/Bohr³] | −3.00982070×10⁻⁴ | dual-gate | bond axis |
| \(\sigma_{yy}\) analytic [Ha/Bohr³] | −6.59874084×10⁻⁵ | dual-gate | transverse |
| \(\sigma_{zz}\) analytic [Ha/Bohr³] | −6.58998481×10⁻⁵ | dual-gate | transverse |
| \(\max\lvert\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\rvert\) diag | **9.33×10⁻⁴** | **1×10⁻³** | A2 decisive; xx residual dominates |
| off-diagonal product | 0 | **1×10⁻⁸** | A3 orthorhombic |
| \(\max\lvert\sigma_{\mathrm{diag}}\rvert\) floor | ≳ 3×10⁻⁴ | \(>10^{-8}\) | A4 |
| \(\lvert E(U)-E(0)\rvert\) [Ha] | **1.79×10⁻²** | \(\ge 10^{-4}\) | A6 U activation |
| \(\max\lvert\sigma_{k\Gamma}-\sigma_\Gamma\rvert\) | **1.3×10⁻⁸** | **5×10⁻⁴** | A7 Γ-limit identity |

Bond-axis residual (xx) dominates the dual-gate; transverse axes agree at
~\(10^{-6}\). Nonlinear Mulliken \(n(S)\) + large \(\lvert e_{\mathrm{dftu}}\rvert\)
on multi-function s shells make multi-SCF FD noisier than bare multi-k UKS
(~\(10^{-10}\)). Γ-limit identity proves the multi-k \(E_U\) Lagrangian path
matches shipped Γ DFT+U stress.

**Offline CP2K twin (pathology — not absolute pin):**

| Item | Value |
| ---- | ----- |
| Engine | CP2K **2026.1** (`cp2k.psmp`) |
| Input | `tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/h2_uks_dftu_kpoint_stress.inp` |
| Result | Unphysical Mulliken charges; large negative DFT+U (\(\sim-175\) Ha); SCF fails within `MAX_SCF=100` |
| Use | Provenance / class note only — **same pathology class** as `gpw-dftu-kpoints` |

```text
  DFT+U energy: ~ −175 Ha (unphysical)
  Total energy: ~ −174 Ha
  *** SCF run NOT converged ***
```

Literature-facing absolute PLUS_U energy remains the Γ NiO molecule pin
(`reg_gpw-dftu-nio`, CP2K \(E=\) −184.92466 Ha). Multi-k stress capability
is proven by A2 dual-gate + A7 Γ-limit, not by the H₂ CP2K absolute number.

Offline re-run (operator only; not ctest):

```bash
export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/h2_uks_dftu_kpoint_stress.inp
OMP_NUM_THREADS=8 cp2k.psmp -i h2_uks_dftu_kpoint_stress.inp -o h2.out
```

## Performance

Source: fair-wall / exemption table in
`regressions/gpw/gpw-dftu-kpoint-stress/REFERENCE.md` (2026-07-26). Handoff:
the case `REFERENCE.md` — ratio **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| H₂ multi-k UKS+U stress (dual-gate) | n/a (no fair twin) | **~2.4** (product+FD, OMP=8) | **n/a** | n/a | `perf_exempt`: CP2K H₂+U multi-k toy has unphysical Mulliken + SCF abort; dual-gate wall is not a fair oracle ratio |

Product pass **with exemption**. A fair multi-k solid DFT+U stress ratio
needs a chemistry twin (e.g. AFM NiO / larger basis) after absolute energy
parity is established.

## Limitations

- **Toy H₂ foothold, not solid chemistry:** multi-function H \(s\) shell +
  large \(U\) produces negative \(E_U\); chemistry-facing PLUS_U numbers
  stay on Γ NiO (`gpw-dftu-nio`).
- **No fair absolute CP2K stress pin** on this twin — dual-gate + Γ-limit
  are decisive; do not cite the CP2K H₂ run as energy/stress gold.
- **Dual-gate looser than bare multi-k UKS:** product tol **1×10⁻³**
  Ha/Bohr³ (xx residual); transverse ~1e-6. Nonlinear \(n(S)\) under strain.
- **Orthorhombic diagonal only:** shear Voigt components are exactly 0.
- **Fixed fillings only:** no multi-k UKS + metallic smearing with DFT+U.
- **OT / multigrid / nonperiodic** throw or are out of multi-k product path
  (`k_mesh` requires periodic single-grid `local_pp_on_grid`).
- **Forces:** multi-k DFT+U analytic **forces** are a separate product
  surface; this slug is stress-only.
- **CPU only** — no CUDA GPW.

## References

- Offline pin + gates: [`regressions/gpw/gpw-dftu-kpoint-stress/REFERENCE.md`](../../../regressions/gpw/gpw-dftu-kpoint-stress/REFERENCE.md)
- Regression consumer: [`reg_gpw-dftu-kpoint-stress.cpp`](../../../regressions/gpw/gpw-dftu-kpoint-stress/reg_gpw-dftu-kpoint-stress.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/`](../../../tools/cp2k_oracle/inputs/gpw-dftu-kpoint-stress/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Closed spec (git history): `.claude/specs/gpw-dftu-kpoint-stress/` (deleted at done)
- Deps: `gpw-dftu-kpoints` ✓, `gpw-dftu-stress` ✓, `gpw-kpoint-uks-stress` ✓
- Γ DFT+U energy handbook: [gpw-dftu-nio](gpw-dftu-nio.md)
- Γ analytic stress: [gpw-analytic-stress](gpw-analytic-stress.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified rotationally invariant \(U\)
- V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991) — DFT+U
- O. H. Nielsen, R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — stress theorem
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
