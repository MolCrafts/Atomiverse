# GPW analytic cell stress + Dudarev DFT+U — NiO molecule / PADE / SZV-GTH-PADE (MT)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-dftu-stress` |
| Regression | `regressions/gpw/gpw-dftu-stress/` |
| Public API | `cpu::gpw::GPWEvaluator` (`plus_u`, `n_alpha`/`n_beta`, `compute_stress`, `stress_use_fd=false`); free `gpw_stress` / `gpw_stress_fd`; Frame `kStress` |
| Roadmap | [`docs/roadmap/dft.md`](../../roadmap/dft.md) — DFT+U analytic stress (`reg_gpw-dftu-stress`) ✓ |

## Summary

**Hellmann–Feynman fixed-\(P\) cell stress** for **Γ-point UKS GPW** with
**Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken) on the product path. Twin is the
same **NiO molecule** as [gpw-dftu-nio](gpw-dftu-nio.md): PADE LSD, SZV GTH
bases, Martyna–Tuckerman isolation, Ni \(d\) shell \(L=2\), \(U-J=6.0\) eV.

After one UKS+U SCF, the non-self-consistent Lagrangian (including \(E_U\) at
fixed spin density and strained overlap) is differentiated under orthorhombic
diagonal strain. Product gate is an **internal dual-gate** analytic vs multi-SCF
strain-FD (`gpw_stress_fd`); soft energy reuses the offline **CP2K 2026.1**
PLUS_U ENERGY pin from `gpw-dftu-nio`. Absolute CP2K NUMERICAL Voigt stress
digits are deferred until a fair oracle host re-run (twin input already
shipped). Public energy on Frame is **eV** (`kEnergy`); pin channel for stress
is `last_stress_ha_bohr3()` [Ha/Bohr³]. Performance: **`perf_exempt`**
(no fair NUMERICAL stress wall twin at pin time).

Related: energy foothold [gpw-dftu-nio](gpw-dftu-nio.md); closed-shell analytic
stress [gpw-analytic-stress](gpw-analytic-stress.md); strain-FD history
[gpw-stress-water](gpw-stress-water.md); narrative GPW [Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Citation:
Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997); VandeVondele et
al., *Comput. Phys. Commun.* **167**, 103 (2005).

**UKS / PADE LSD + occupations.** Separate \(P^\alpha\), \(P^\beta\); spin
Padé LDA. Integer members `n_alpha` / `n_beta` (no multiplicity bag). Pin:
24 valence e⁻ (Ni GTH-q18 + O GTH-q6), multiplicity 3 ⇒ \(N_\alpha=13\),
\(N_\beta=11\).

**Isolated electrostatics.** `nonperiodic = true` ⇔ CP2K `PERIODIC NONE` /
`PSOLVER MT`. Geometry must be **centered** in the cell (Martyna & Tuckerman,
*J. Chem. Phys.* **110**, 2810 (1999)).

**Dudarev DFT+U (energy / potential).** Public \(U_{\mathrm{eff}}=U-J\) in eV
on `PlusU{element, l, u_minus_j_ev}`; converted to Hartree inside. Symmetrized
Mulliken occupation on the AO subspace of atom \(A\) with \(\ell=L\):

\[
n^{A\sigma}_{mm'} = \tfrac12\big[(P^\sigma S)_{mm'} + (S P^\sigma)_{mm'}\big].
\]

\[
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big],
\qquad
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big).
\]

Matches CP2K simplified rotationally invariant form (`PLUS_U_METHOD MULLIKEN`,
`U_MINUS_J`). SCF energy contribution is `GpwResult::e_dftu` [Ha].

**Product cell stress (fixed-\(P\) + Pulay + \(E_U\)).**

1. One UKS+DFT+U SCF at the unstrained cell → \(P_\alpha,P_\beta\) and
   \(W_\sigma=\sum_i^{n_\sigma}\varepsilon_i^\sigma c_i^\sigma(c_i^\sigma)^\top\)
   (eigenvalues include \(V_U\)).
2. Central finite difference of the **non-self-consistent** Lagrangian
   \(E_L[P_\alpha,P_\beta;W_\alpha,W_\beta;\varepsilon]\) under orthorhombic
   diagonal strain at fixed fractional coordinates, with
   \(E_U[n(P_\sigma,S)]\) evaluated at fixed spin density and strained
   overlap (`internal/dft_plus_u.h` `dft_plus_u_energy`).
3. Sign (CP2K `STRESS_TENSOR`):

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E_L}{\partial\varepsilon_{aa}},
\qquad a\in\{x,y,z\},
\]

with \(E\) in Hartree and \(\Omega\) in Bohr³. Voigt order
\((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\).
Default strain step \(h = 2\times 10^{-3}\) (`stress_strain_h`). Shear
components are **0** in v1 (orthorhombic cell only).

**Oracle twin (dual-gate).** Multi-SCF strain-FD (`gpw_stress_fd`) on the same
UKS+U method fully re-converges SCF at each \(\pm h\) strained cell — not the
product hot path. Absolute offline CP2K twin uses `STRESS_TENSOR NUMERICAL`
(`nio_mol_pade_u_stress_num.inp`); absolute Voigt digits not yet pinned
(host lacked CP2K at first-green).

**Key exclusions**

- Multi-k DFT+U stress (explicitly unimplemented on `gpw_stress`).
- Liechtenstein full \(U,J\); GAPW/PAW projectors.
- Shear / triclinic strain; CP2K `STRESS_TENSOR ANALYTICAL` absolute pin on MT.
- Solid AFM rocksalt NiO product pin (molecule foothold only).
- Public `*Config` bags / `make_*` factories; live CP2K in ctest; CUDA GPW.

## Implementation

Public surface only (`cpu::Frame`, `PlusU` + evaluator members; **no**
`*Config` bag, **no** `make_*` / `build_*`). Matches the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// NiO molecule, R = 1.668 Å along z, centered in 10 Å cubic cell (MT)
constexpr double L = 10.0;
constexpr double R = 1.668;
cpu::Frame frame(2);
frame.span(kAtom_type)[0] = 28;
frame.span(kAtom_type)[1] = 8;
frame.span(kAtom_mass)[0] = 58.693;
frame.span(kAtom_mass)[1] = 15.999;
frame.span(kAtom_x)[0] = 0.5 * L;
frame.span(kAtom_y)[0] = 0.5 * L;
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_x)[1] = 0.5 * L;
frame.span(kAtom_y)[1] = 0.5 * L;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
frame.box = cpu::Box::cubic(L);

const cpu::gpw::Kind Ni{
    molrs::Element::Ni,
    cpu::gpw::GaussianBasis(molrs::Element::Ni, "SZV-GTH-PADE-q18"),
    cpu::gpw::GthPotential(molrs::Element::Ni, "GTH-PADE-q18")};
const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "SZV-GTH-PADE"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PADE-q6")};

cpu::gpw::GPWEvaluator eval({Ni, O});
eval.cutoff_ry = 250.0;
eval.max_iter = 300;
eval.tol = 1e-5;
eval.mix = 0.35;
eval.xc = cpu::gpw::XcKind::PADE;
eval.nonperiodic = true;
eval.n_alpha = 13;  // multiplicity 3, 24 valence e⁻
eval.n_beta = 11;
eval.plus_u.push_back({molrs::Element::Ni, /*l=*/2, /*u_minus_j_ev=*/6.0});
eval.compute_forces = false;
eval.compute_stress = true;
eval.stress_use_fd = false;   // product analytic fixed-P
eval.stress_strain_h = 2e-3;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

const double e_ha = eval.last_energy_hartree();          // [Ha]
const double e_ev = frame.meta.get(kEnergy);             // [eV]
const double e_u  = eval.last_result().e_dftu;           // [Ha]
const auto& s_ha  = eval.last_stress_ha_bohr3();         // Voigt [Ha/Bohr³]
// frame.meta.get(kStress) is Voigt [eV/Å³]
```

Dual-gate oracle (regression only; multi-SCF, not product hot path):

```cpp
// After method.build(atoms) with the same plus_u / UKS members:
const auto sfd = cpu::gpw::gpw_stress_fd(
    atoms, method, /*h=*/2e-3, /*axes*/ {true, true, true},
    /*warm C*/ &eval.last_result().C);
// Assert max |s_ha[a] - sfd[a]| for a = xx,yy,zz
```

Build (GPW needs FFTW3) and run:

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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-dftu-stress
  ./build/regressions/reg_gpw-dftu-stress
'
```

| Piece | Path |
| ----- | ---- |
| Public `PlusU` / `gpw_stress` / `gpw_stress_fd` | `include/atomiverse/cpu/gpw/gpw_solver.h` |
| Evaluator `plus_u`, `compute_stress`, `stress_use_fd` | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Dudarev E/V + fixed-\(P\) \(E_U\) (internal) | `include/atomiverse/cpu/gpw/internal/dft_plus_u.h` |
| Fixed-\(P\)+Pulay stress implementation | `src/cpu/gpw/internal/gpw_stress.cpp` |
| Energy foothold regression | `regressions/gpw/gpw-dftu-nio/` |
| Stress product gate | `regressions/gpw/gpw-dftu-stress/reg_gpw-dftu-stress.cpp` |
| Offline pins | `regressions/gpw/gpw-dftu-stress/REFERENCE.md` |
| CP2K NUMERICAL twin input | `tools/cp2k_oracle/inputs/gpw-dftu-stress/nio_mol_pade_u_stress_num.inp` |
| CP2K ENERGY twin (soft) | `tools/cp2k_oracle/inputs/gpw-dftu-nio/nio_mol_pade_u.inp` |

## Validation (vs CP2K)

Source:
[`regressions/gpw/gpw-dftu-stress/REFERENCE.md`](../../../regressions/gpw/gpw-dftu-stress/REFERENCE.md).
Oracle: offline **CP2K 2026.1** Quickstep; **never** live in ctest. Pin date
**2026-07-26** (first green). Decisive product gate is **analytic vs multi-SCF
FD**; energy soft twin reuses `gpw-dftu-nio`.

**Setup (must match twin):**

| Item | Value |
| ---- | ----- |
| System | NiO **molecule**, charge 0, UKS multiplicity 3 |
| Bond | \(R(\mathrm{Ni–O})=\) **1.668 Å** along *z*, **centered** |
| Coordinates (Å) | Ni (5, 5, 4.166), O (5, 5, 5.834) |
| Cell | **10 × 10 × 10 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT` / `nonperiodic`) |
| Basis | Ni `SZV-GTH-PADE-q18`, O `SZV-GTH-PADE` |
| PP | Ni `GTH-PADE-q18` (18e), O `GTH-PADE-q6` (6e) → **24** valence e⁻ |
| XC | PADE (LDA LSD) |
| DFT+U | Ni \(L=2\), \(U-J=6.0\) eV, `PLUS_U_METHOD MULLIKEN` |
| Cutoff | **250 Ry** (ATV single density grid; CP2K `NGRIDS 4` / `REL_CUTOFF 40`) |
| Spin | \(N_\alpha=13\), \(N_\beta=11\) |
| Stress step | \(h=2\times 10^{-3}\) |
| Product path | `compute_stress=true`, `stress_use_fd=false` |
| Oracle twin | `gpw_stress_fd` multi-SCF; CP2K `STRESS_TENSOR NUMERICAL` input shipped |

**Hard-coded regression gates:**

| Quantity | Pin / expect | Tol (reg) | Notes |
| -------- | -----------: | --------- | ----- |
| \(E_{\mathrm{CP2K}}\) [Ha] | **−184.92466076474312** | soft | Offline ENERGY pin (`nio_mol_pade_u.inp`) |
| \|E_ATV − E_CP2K\| | first-green **2.15×10⁻²** Ha | **5×10⁻²** Ha | single-grid vs multigrid+OT |
| e_dftu [Ha] | first-green **0.091249** | > **1×10⁻⁴** | positive U contribution |
| max\|σ_an − σ_fd\| diag | first-green **8.92×10⁻⁶** | **2×10⁻⁴** Ha/Bohr³ | dual-gate |
| Off-diagonal product | **0** | **1×10⁻⁸** | orthorhombic v1 |
| max\|σ_diag\| floor | first-green \(\sim 10^{-5}\) | > **1×10⁻⁸** | non-degenerate |
| ⟨S²⟩ | first-green **2.0155** | diagnostic | ideal \(S(S+1)=2\) |
| Frame `kStress` units | matches Ha/Bohr³ × conversion | **1×10⁻⁹** | eV/Å³ |

**First-green ATV stress components (REFERENCE, analytic product):**

| Component | Value [Ha/Bohr³] |
| --------- | ---------------: |
| \(\sigma_{xx}\) | \(2.10983351\times 10^{-7}\) |
| \(\sigma_{yy}\) | \(2.10983257\times 10^{-7}\) |
| \(\sigma_{zz}\) | \(-1.04886653\times 10^{-5}\) |

Absolute CP2K NUMERICAL Voigt digits: **not invented** (CP2K module unavailable
on pin host 2026-07-26). Twin input ready for operator re-run:

```bash
export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-dftu-stress/nio_mol_pade_u_stress_num.inp
OMP_NUM_THREADS=8 cp2k.psmp -i nio_mol_pade_u_stress_num.inp -o nio_stress.out
```

Bar → Ha/Bohr³ (CP2K internal table):
\(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\).

## Performance

Source: `regressions/gpw/gpw-dftu-stress/REFERENCE.md` (2026-07-26). Handoff:
the case `REFERENCE.md`
— ratio **`perf_exempt`**.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| Fair NUMERICAL stress wall | n/a | product **12.0** (OMP=8) | **perf_exempt** | n/a | No CP2K binary on pin host for NUMERICAL stress twin |
| Dual-gate total (product + FD) | — | **~37** (12.0 + 24.7) | n/a | n/a | FD twin = 6 SCFs; not product cost |
| Product-only vs sibling ENERGY wall (`gpw-dftu-nio`) | **15.79** | **12.0** | **≈0.76** | ✓ if reused | Context only; not the stress acceptance ratio |

`perf_exempt = true`. When CP2K is available, record product Evaluate wall vs
fair ENERGY_FORCE / NUMERICAL twin on the same host and replace this section
(target \(r\le 10\) for analytic product vs fair twin).

## Limitations

- **Molecule foothold, not solid AFM NiO** — same scope as `gpw-dftu-nio`;
  periodic rocksalt AFM needs k-points / larger basis before a production pin.
- **Absolute CP2K NUMERICAL stress digits deferred** — dual-gate is binding;
  replace when oracle host re-runs `nio_mol_pade_u_stress_num.inp`.
- **Γ only** — multi-k DFT+U stress is not implemented (`gpw_stress` throws /
  excludes multi-k U).
- **Orthorhombic diagonals only** — shear Voigt components hard-zero in v1.
- **Grid / SCF twin, not bit-identical** — ATV single density grid + DIIS mix
  vs CP2K multigrid + OT; energy soft tol **5×10⁻²** Ha (same class as
  `gpw-dftu-nio`).
- **SZV-only pin** — not a chemistry-quality electronic-structure claim for
  NiO solids.
- **UKS + PADE** on this path — GGA UKS+U stress is a separate matrix.
- Frame stress is eV/Å³; literature / pin compare uses Ha/Bohr³ via
  `last_stress_ha_bohr3()`.
- Unit tests under `tests/cpu/gpw/` only for local dual-gates; e2e CP2K
  energy soft-pin + dual-gate lives exclusively in
  `reg_gpw-dftu-stress`.

## References

- Offline pin + dual-gate walls:
  [`regressions/gpw/gpw-dftu-stress/REFERENCE.md`](../../../regressions/gpw/gpw-dftu-stress/REFERENCE.md)
- Regression consumer:
  [`reg_gpw-dftu-stress.cpp`](../../../regressions/gpw/gpw-dftu-stress/reg_gpw-dftu-stress.cpp)
- CP2K NUMERICAL twin:
  [`tools/cp2k_oracle/inputs/gpw-dftu-stress/`](../../../tools/cp2k_oracle/inputs/gpw-dftu-stress/)
- Energy twin / digits:
  [`regressions/gpw/gpw-dftu-nio/REFERENCE.md`](../../../regressions/gpw/gpw-dftu-nio/REFERENCE.md)
- Oracle contract:
  [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Spec history (closed, deleted at done): `.claude/specs/gpw-dftu-stress/`
  (git log under `wf/gpw-dftu-stress` / merge into `dev`)
- Related tutorials: [gpw-dftu-nio](gpw-dftu-nio.md),
  [gpw-analytic-stress](gpw-analytic-stress.md),
  [gpw-stress-water](gpw-stress-water.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified
  rotationally invariant \(U\)
- V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991)
  — DFT+U
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) — stress formalism context
