# Dudarev DFT+U (GPW UKS) — NiO molecule / PADE / SZV-GTH-PADE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-dftu-nio` |
| Regression | `regressions/gpw/gpw-dftu-nio/` |
| Public API | `cpu::gpw::GPWEvaluator` + `std::vector<PlusU> plus_u`; `n_alpha`/`n_beta`; `cpu::single_point` |
| Roadmap | `docs/roadmap/dft.md` — GPW Dudarev DFT+U (`plus_u`; `reg_gpw-dftu-nio`) ✓ |

## Summary

**Dudarev DFT+U** (CP2K `PLUS_U` / Mulliken occupation) on **Γ-point unrestricted
GPW** corrects self-interaction on partially filled correlated shells (here Ni
\(d\), \(L=2\)) by adding an effective Hubbard \(U_{\mathrm{eff}}=U-J\) energy and
potential. This page is the product foothold for that path: a **NiO molecule**
(not solid AFM rocksalt — see Limitations) with PADE LSD, SZV GTH bases, and
Martyna–Tuckerman isolation, pinned offline against **CP2K 2026.1** Quickstep.

Public surface: construct `cpu::gpw::GPWEvaluator` with `Kind` rows (lookup
`GaussianBasis` / `GthPotential`), set UKS occupations and
`plus_u.push_back({Ni, 2, 6.0})` with \(U-J\) in **eV**, run
`cpu::single_point(frame, eval)`. Total energy on Frame is **eV** (`kEnergy`);
Hartree total, DFT+U contribution, \(\langle S^2\rangle\), and Mulliken spin via
`last_energy_hartree()` / `last_result().{e_dftu,s2,mulliken_spin}`. Product
gate: `reg_gpw-dftu-nio`. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.21\) (≤10×).

Related: UKS without U — [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md); narrative
GPW — [Ch. 9](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in contracted Gaussians; density /
Hartree / XC on a uniform real-space grid; GTH pseudopotentials. Grid set by
plane-wave cutoff [Ry]. Citation: Lippert, Hutter, Parrinello, *Mol. Phys.*
**92**, 477 (1997).

**UKS / PADE LSD.** Separate \(P^\alpha\), \(P^\beta\); spin-resolved Padé LDA;
integer members `n_alpha` / `n_beta` (not a multiplicity bag). For this pin:
24 valence e⁻ (Ni GTH-q18 + O GTH-q6), multiplicity 3 ⇒ \(N_\alpha=13\),
\(N_\beta=11\). Citation: Goedecker, Teter, Hutter, *Phys. Rev. B* **54**, 1703
(1996).

**Isolated electrostatics.** `nonperiodic = true` ⇔ CP2K `PERIODIC NONE` /
`PSOLVER MT`. Geometry must be **centered** in the cell. Citation: Martyna &
Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999).

**Dudarev DFT+U (binding, closed spec).** Per element and angular momentum
\(L\), effective \(U_{\mathrm{eff}}\) in eV on the public surface (converted to
Hartree with `units::kHartreeInEv` inside). Occupation on the AO subspace of
atom \(A\) with \(\ell=L\) (symmetrized Mulliken, CP2K `PLUS_U_METHOD MULLIKEN`):

\[
n^{A\sigma}_{mm'} = \tfrac12\big[(P^\sigma S)_{mm'} + (S P^\sigma)_{mm'}\big].
\]

Energy (sum over atoms and spin channels; \(U\) in Hartree):

\[
E_U = \sum_{A\sigma}\frac{U_{\mathrm{eff}}}{2}\Big[\mathrm{Tr}\,n^{A\sigma}
  - \mathrm{Tr}(n^{A\sigma} n^{A\sigma})\Big].
\]

Potential added to the spin Fock matrix on that subspace:

\[
V^{A\sigma}_{mm'} = U_{\mathrm{eff}}\Big(\tfrac12\delta_{mm'} - n^{A\sigma}_{mm'}\Big).
\]

Matches CP2K simplified rotationally invariant form (Dudarev et al.; CP2K
`U_MINUS_J`). Implementation lives in `cpu/gpw/internal/dft_plus_u.h` (not
public). Empty `plus_u` leaves the legacy UKS/RKS path unchanged.

**Optional AFM seed.** `atom_spin_guess` (per-atom, size 0 or \(N\)) biases only
the **initial** UKS density. This molecular pin uses occupations \((13,11)\)
alone; solid AFM would use opposite signs on the two Ni sites.

**Key exclusions (closed-spec scope-out).**

- Liechtenstein full \(U,J\) tensor; orbital-dependent U ramping productization.
- k-point DFT+U; UKS + PBE-GGA; analytic DFT+U forces / stress.
- GAPW / PAW projectors; MPI scaling claim.
- Solid rocksalt AFM NiO product pin (follow-on after k-points / larger basis).

## Implementation

Public surface only (`cpu::Frame`, `PlusU` + evaluator members; **no**
`*Config` bag, **no** `make_*` / `build_*`):

```cpp
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/single_point.h"
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

cpu::gpw::Kind Ni{molrs::Element::Ni,
                  cpu::gpw::GaussianBasis(molrs::Element::Ni, "SZV-GTH-PADE-q18"),
                  cpu::gpw::GthPotential(molrs::Element::Ni, "GTH-PADE-q18")};
cpu::gpw::Kind O{molrs::Element::O,
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

cpu::single_point(frame, eval);

const double e_ha = eval.last_energy_hartree();          // [Ha]
const double e_ev = frame.meta.get(kEnergy);             // [eV]
const cpu::gpw::GpwResult& r = eval.last_result();
const double e_u = r.e_dftu;                             // [Ha]
const double s2 = eval.last_s2();                        // Löwdin ⟨S²⟩
const double m_Ni = r.mulliken_spin.empty() ? 0.0 : r.mulliken_spin[0];
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
  cmake --build build --target reg_gpw-dftu-nio -j$(nproc)
  ./build/regressions/reg_gpw-dftu-nio
'
```

| Piece | Path |
| ----- | ---- |
| Public `PlusU` / `GpwResult::{e_dftu,mulliken_spin}` | `include/atomiverse/cpu/gpw/gpw_solver.h` |
| Evaluator member `plus_u` | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Dudarev E/V (internal) | `include/atomiverse/cpu/gpw/internal/dft_plus_u.h` |
| SCF wire-up | `src/cpu/gpw/gpw_solver.cpp` |
| Ni/O PADE SZV + GTH tables | `include/atomiverse/cpu/gpw/basis_data/gpw_tables.h` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_dftu.cpp` |
| Regression | `regressions/gpw/gpw-dftu-nio/reg_gpw-dftu-nio.cpp` |
| CP2K twin inputs | `tools/cp2k_oracle/inputs/gpw-dftu-nio/` |

## Validation (vs CP2K)

Source: [`regressions/gpw/gpw-dftu-nio/REFERENCE.md`](../../../regressions/gpw/gpw-dftu-nio/REFERENCE.md).
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`, revision 85b8a9b49c); **never** live
in ctest. Pin date: **2026-07-24**. Closed-spec gates A3–A5 / A7.

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
| SCF twin | CP2K: OT `FULL_SINGLE_INVERSE` + OUTER_SCF (diag+Broyden did not converge this twin within budget); ATV: DIIS-style mix 0.35, `tol=1e-5`, `max_iter=300` |

**Pinned quantities:**

| Quantity | CP2K pin | Atomiverse (REFERENCE) | Tol (regression) | Notes |
| -------- | -------: | ---------------------: | ---------------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−184.92466076474312** | ≈ −184.94637 | **5×10⁻² Ha** | A4 first-green: single-grid vs multigrid+OT |
| \(\lvert\Delta E\rvert\) vs CP2K | — | ≈ **2.2×10⁻²** Ha | ≤ 5×10⁻² | within gate |
| DFT+U energy [Ha] | **0.05230865222345** | \(>10^{-4}\) (asserted) | diagnostic pin | `r.e_dftu`; not equality-asserted |
| \(\langle S^2\rangle\) (single det.) | **2.019198** | ≈ 2.016 | **0.25** | ideal \(S(S+1)=2\) |
| Mulliken \(m_{\mathrm{Ni}}\) [e] | **1.177632** | ≈ 1.07 | **0.5** and \(m_{\mathrm{Ni}}>0.3\) | A5 |
| Frame energy consistency | — | \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}}\equiv E_{\mathrm{Ha}}\) | \(10^{-9}\) Ha | reg assert |
| SCF | converged | `r.converged` | — | ~56 iters (ATV, REFERENCE) |

Key CP2K log lines (from REFERENCE):

```text
  DFT+U energy:                                                 0.05230865222345
  Total energy:                                              -184.92466076474312
  Ideal and single determinant S**2 :                    2.000000       2.019198
       1     Ni       1         9.343192     8.165560     0.491248     1.177632
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.924660764743123
```

Offline re-run (operator only; not ctest):

```bash
export LD_LIBRARY_PATH=$CP2K_ROOT/lib:$LD_LIBRARY_PATH
export CP2K_DATA_DIR=$CP2K_ROOT/share/cp2k/data
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_SET,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-dftu-nio/nio_mol_pade_u.inp
OMP_NUM_THREADS=8 cp2k.psmp -i nio_mol_pade_u.inp -o nio_mol.out
```

**Solid AFM note (spec / REFERENCE).** Rocksalt AFM NiO (2 f.u.) did **not**
converge under SZV / Γ-only in CP2K (diag+Broyden or no-U restart) within
200–300 steps. Twin inputs `nio_afm_pade_*.inp` are retained as exploration
artifacts; the **molecule** is the first-green PLUS_U foothold.

## Performance

Source: fair wall table in
`regressions/gpw/gpw-dftu-nio/REFERENCE.md` (2026-07-24). Handoff:
the case `REFERENCE.md` ratio **0.21**. Gate \(r\le 10\) or
`perf_exempt` (closed-spec A8).

Comparable settings: cutoff 250 Ry, PADE+U, MT, single rank, `OMP_NUM_THREADS=8`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| NiO mol PADE+U / SZV / MT, Γ | **15.79** | **3.29** | **0.21** | ✓ | CP2K OT+OUTER_SCF multigrid; ATV Release single density grid |

Product pass without exemption. Ratio is for this microcase (SZV molecule);
it does not forecast solid NiO, multigrid parity, or OT vs DIIS apples-to-apples
on identical algorithms.

## Limitations

- **Molecule foothold, not solid AFM:** periodic rocksalt AFM NiO is out of
  this regression; needs k-points and/or larger basis before a production pin
  (closed-spec follow-on).
- **Grid / SCF twin, not bit-identical:** ATV single density grid + DIIS mix vs
  CP2K multigrid + OT; energy tolerance is deliberately **5×10⁻² Ha** for
  first-green (A4). Tighter agreement requires multigrid parity work.
- **SZV-only pin:** minimal PADE GTH basis; not a chemistry-quality NiO
  electronic-structure claim.
- **No analytic DFT+U forces / stress** on this path; forces remain separate
  closed-shell GPW regressions.
- **UKS + PBE-GGA** unsupported for GPW UKS (same as `gpw-uks-o2-gamma`); DFT+U
  here is PADE LSD only.
- **Γ only** — no k-point DFT+U product path yet (`gpw-kpoints-si` incomplete
  at handbook write time).
- **Liechtenstein full \(U,J\)** and PAW/GAPW projectors out of scope.
- **Geometry hygiene:** off-center molecules under MT destroy energy parity;
  always center for isolated pins.

## References

- Offline pin + fair wall: [`regressions/gpw/gpw-dftu-nio/REFERENCE.md`](../../../regressions/gpw/gpw-dftu-nio/REFERENCE.md)
- Regression consumer: [`reg_gpw-dftu-nio.cpp`](../../../regressions/gpw/gpw-dftu-nio/reg_gpw-dftu-nio.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-dftu-nio/`](../../../tools/cp2k_oracle/inputs/gpw-dftu-nio/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Closed spec (git history): `.claude/specs/gpw-dftu-nio/` (deleted at done;
  commit `feat(gpw): Dudarev DFT+U for NiO (gpw-dftu-nio)`)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- V. I. Anisimov, J. Zaanen, O. K. Andersen, *Phys. Rev. B* **44**, 943 (1991) — DFT+U
- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — simplified rotationally invariant \(U\)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
